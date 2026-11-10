# Code review: DSPy.RLM

**No experiment was re-run for this review.** No model, server or network call was made. Every check is static or
uses mocks.

## Scope

| Tree | Files |
|---|---|
| Original (the code that ran, checksummed in `agents/dspy_rlm/original/MANIFEST.json`) | `original/RLM/rag/{__init__,agent,signatures,tools}.py`, `examples/browsecomp_eval_dspy_rlm.py`, `examples/obliq_eval_dspy_rlm.py`, `scripts/run_obliq_dspy_shard.sh`, `pyproject.toml` |
| Refactored | `agents/dspy_rlm/src/dspy_rlm/*.py`, `src/run_browsecomp.py`, `src/run_obliq.py`, `experiments/{obliq_shard,browsecomp}.sh`, `configs/example.env` |
| Shared code used | `agents/shared/rlm_shared/{browsecomp_data,obliq_score_advice,obliq_data,paths}.py` |
| Library behaviour relied on | installed DSPy 3.3.1, `dspy/predict/rlm.py` (`RLM.forward`, `_make_llm_tools`, `_extract_fallback`) and `dspy/primitives/python_interpreter.py` |

Out of scope:
- the retrieval servers (`benchmarks/retrieval/`);
- the earlier campaign's result files. The original workbook `docs/rlm_results_summary.xlsx` was read only to
  summarise it in the README (§11);
- DSPy itself, beyond the functions named above.

DSPy.RLM has no row in `results/original_baselines/Experiments.xlsx`. No reported number in this repository depends
on it.

## Method specification vs code

| Specified behaviour (README §4) | Where in the code | Agrees? |
|---|---|---|
| One root-LM call per REPL turn; code runs in a sandbox; SUBMIT ends the run; at `max_iters` an extract call produces the answer | `dspy.RLM.forward` (DSPy 3.3.1); the harness passes `max_iters` (`agent.py` `build_rlm_agent`, `run_obliq.py` `_build_agent_with_sig`) | yes (inspected) |
| Prompt = `_REPL_TOOLS + _REPL_RULES + strategy + _ORCHESTRATION`; v3 states the real budget; v9 is hand-built | `signatures.py` `make_rlm_signature`, `browsecomp_minimal_budget_sig`, `BrowseCompSubtractiveSignature` | yes (tested) |
| `search_index`: top-k clamp → per-turn cap → hard budget stub → cache → request → snippet cut | `agent.py` `search_index` closure | yes (tested, order included) |
| `get_document` prints documents up to `GETDOC_INLINE_CHARS`; its docstring follows the cap | `agent.py` `get_document` + docstring block | yes (tested) |
| Sub-LM for `llm_query` is a separate Azure deployment, with calls counted | `agent.py` `_sub_kwargs` / `_CountedLM`; `run_obliq.py` `_build_sub_lm` | yes (tested for BrowseComp model choice) |
| Root LM is Azure, or a local OpenAI-compatible server when `LOCAL_LLM_BASE_URL` is set | `agent.py` `build_lm` | yes (tested: local branch returns `openai/…`) |
| `llm_query` is capped at 50 calls per run | dspy default `max_llm_calls=50`; the harness never sets it | yes (inspected) |
| BrowseComp scoring = substring shortcut, else Azure LLM judge | `rlm_shared/browsecomp_data.py` `judge_match` | yes; the shortcut is defective (D2) |
| OBLIQ scoring = NDCG/Recall@k on trajectory, answer and tail-fill rankings | `run_obliq.py` `run_one`, `ndcg_at`, `recall_at`, `extract_doc_ids_from_answer` | yes (metric functions tested) |
| Stated per-turn budget in the v1 prompt | `_BROWSECOMP_STRATEGY`: "after turn 12 … out of 16" | **only when `--max-iters 16`**: the text is fixed. Prompt text, so unchanged; recorded in README §13 |
| Snippet length stated to the model | prompts and docstrings say "≤2000 chars"; the harness cuts to 500 | **no** (D13) |

## Checks performed

1. **AST equality, package.** For each of `__init__.py`, `agent.py`, `signatures.py` and `tools.py`, the AST of
   `src/dspy_rlm/<f>` equals the AST of `original/RLM/rag/<f>` once the module docstring is removed. Comments are not
   in the AST. So the four files differ only in documentation (`tests/test_dspy_rlm_static.py`).
2. **AST equality, drivers.** The original driver's source has only the allowed changes undone:
   - `rag` → `dspy_rlm`, `examples.` → `rlm_shared.`;
   - `sys.path.insert` lines dropped;
   - for OBLIQ, `OBLIQ_ROOT /` → `_obliq_root() /`, with the helper and its import removed on the new side.

   After that, its AST equals the refactored driver's.
3. **Signature equality.** The original package is imported under its own name `rag` from `original/RLM`, isolated
   from `sys.modules` and without writing bytecode. Then:
   - the set of `dspy.Signature` classes is the same;
   - for every class, `instructions`, `__name__` and both field descriptions are equal;
   - every private string constant (the prompt blocks) is equal;
   - `make_rlm_signature` (3 argument sets) and `browsecomp_minimal_budget_sig` (6 budget/`BROWSECOMP_COMMIT_TURN`
     cases, including the clamp to [2, max_iters]) give identical instructions.

   Every class has `question → answer` fields and builds a `dspy.RLM` without an LM call.
4. **Budget behaviour** (fake retriever, recording `dspy.RLM`):
   - hard budget: after `max_search_calls`, the exact "SEARCH DISABLED" stub is returned and the retriever is not
     called; the stub comes before the cache;
   - `MAX_SEARCH_PER_TURN`: the call over the cap returns the exact "SEARCH-PER-TURN LIMIT" stub and is not counted
     as a search; a gap over 5 s resets the counter; cached repeats count toward the per-turn cap, and the per-turn
     check runs before the hard budget;
   - `MAX_TOP_K` clamp, 500-char truncation, cache, `sys.exit` on an unreachable server.

   A scripted sequence of searches and fetches on the refactored and the original `build_rlm_agent` gives identical
   returns, retriever calls, counters, tool docstrings, `max_iters` and sub-LM model.
5. **Wiring.** `build_rlm_agent` passes `max_iters`, `verbose`, the signature and `[search_index, get_document]` to
   `dspy.RLM`. `DISABLE_GET_DOCUMENT` and `RLM_MAX_OUTPUT_CHARS` are honoured. One real `dspy.RLM` is constructed and
   holds `max_iters` and both tools.
6. **Retriever payloads.** `RemoteRetriever` URLs, JSON bodies (with and without `index` / `OBLIQ_INDEX`) and timeouts
   (`RETRIEVER_TIMEOUT`), with `requests.post` monkeypatched. The original produces byte-identical requests.
7. **Static checks.** `py_compile` on every `src/` file; `bash -n` on both launchers.
8. **Diff review.** `diff -u` of every original/refactored pair; each hunk was classified under REFACTOR_RULES 1–5,
   and none falls outside them. `preserve_originals.py --check`: all copies match their manifest.

Result: `pytest agents/dspy_rlm/tests`: **98 passed** (49 signatures, 18 agent, 14 static, 9 tools, 8 drivers).

## Findings

The IDs are those of [CHANGES_FROM_ORIGINAL.md](../../agents/dspy_rlm/CHANGES_FROM_ORIGINAL.md), which gives
file:line for both trees. They are cited as Dn in [KNOWN_ISSUES.md](KNOWN_ISSUES.md). None was fixed (REFACTOR_RULES).

| ID | Finding | Evidence | KNOWN_ISSUES |
|---|---|---|---|
| D1 | `--timeout > 0`: the worker's signature map lacks v13–v17 and v20; they fall back to v1 | inspected; pinned by a test that reads the worker source | K20 |
| D2 | `heuristic_match` (`agents/shared/rlm_shared/browsecomp_data.py:112-114`): an empty prediction, or any substring of the gold, is correct before the LLM judge runs | **demonstrated** (`heuristic_match("Queen Arwa University", "") is True`) | K17 |
| D3 | a local root LM still requires Azure for the BrowseComp sub-LM; OBLIQ too unless `LOCAL_SUB_LM=1` | **demonstrated** for BrowseComp (`KeyError` without `AZURE_OPENAI_ENDPOINT` while `build_lm` is local); OBLIQ inspected | K20 |
| D4 | OBLIQ records (and the BrowseComp TXT header) log `AZURE_OPENAI_MODEL` for local-model runs | inspected | K20 |
| D5 | the OBLIQ `judge_model` field's default (`gpt-5.4-mini`) differs from the sub-LM fallback (`AZURE_OPENAI_MODEL`) | inspected | K20 |
| D6 | `--subset wildchat-v7` is an argparse choice with no signature, which raises `KeyError` | **demonstrated** that the key is missing; the crash itself is inspected | K20 |
| D7 | `--max-iters` help says 16, the default is 12 (OBLIQ); the BrowseComp TXT header hard-codes 25 / 5; `--sig` help covers only v1/v2 | inspected | K20 |
| D8 | an OBLIQ timeout abandons a daemon thread that keeps running and keeps its class-level timing patch | inspected | K20 |
| D9 | BrowseComp timeout worker joins before `q.get()`: a large result (RunMetrics carries fetched document text) can be killed as TIMEOUT; a worker that dies without `put()` hangs the parent | inspected, from the documented `multiprocessing` behaviour; not reproduced | K20 |
| D10 | OBLIQ excluded ids are filtered after the server's top-k | inspected | K20 |
| D11 | OBLIQ loaders and metrics duplicate `rlm_shared.obliq_data`, with differences (wildchat regex gate; pool fallback) | inspected (the two sources compared) | K20, K23 |
| D12 | dead code: `max_depth`/`depth`, `delegation_calls`, `fetched_docs` never written, `trajectory` never filled for BrowseComp, `make_embedding_fn`, the undefined `FaissRetriever` annotation, the missing `scripts/update_baseline_summary.py` (deleted in commit 6948bf7), unused imports | inspected; the AST tests show these are unchanged from the original | — |
| D13 | prompts and tool docstrings promise "≤2000-char" snippets, but the agent gets 500 characters by default. The retrieval servers do cut at 2000 (benchmarks/README.md), and the harness then cuts to 500 | inspected | — |

Further observations, recorded in the README but not given an ID:
- The v1 prompt's turn deadline is fixed at "12 of 16".
- `load_dotenv()` resolves `.env` relative to the source file, so the move to `src/` changes which `.env` is found
  (CHANGES_FROM_ORIGINAL.md; see K15).
- dspy's action and extract prompts depend on the installed DSPy version, and the earlier campaign's version is not
  recorded here.

## Demonstrated vs only inspected

**Demonstrated by executed tests (mocks only):**
- prompt and signature identity with the original;
- code identity of the package (AST) and of the drivers (AST after the allowed renames);
- search budget, per-turn cap, clamp, cache, truncation and exit behaviour, with the same sequence matching the
  original;
- tool wiring into `dspy.RLM`; sub-LM model selection; the local-root-LM branch of `build_lm`;
- `RemoteRetriever` request format;
- OBLIQ loaders reading `OBLIQ_BENCH_ROOT`, and the NDCG/recall/id-parser functions;
- D2 and D3 (BrowseComp), and the missing key behind D6;
- `py_compile`, `bash -n`.

**Only inspected (read, not executed):**
- the `dspy.RLM` loop with a real LM: SUBMIT, the extract fallback, `max_llm_calls`, output truncation;
- the Deno/Pyodide sandbox;
- the judge call;
- `run_eval` and `run_one` end to end, including both timeout paths (D1, D8, D9);
- the OBLIQ tool closures and the nested `rlm_query` reader;
- `DSPY_FORMAT_FIX`;
- the launchers beyond syntax;
- the earlier-campaign numbers in the README, which are copied from the original workbook and were not recomputed.

## Statement

No experiment was re-run. No result in this repository or in the original workbook was regenerated or re-graded as
part of this review, and no LLM, retrieval server or network service was contacted.
