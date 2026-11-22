# DSPy.RLM: changes from the original code

`original/RLM/` is the code exactly as it ran (SHA-256 in `original/MANIFEST.json`); `src/` is the readable,
runnable copy. Only the changes allowed by `docs/architecture/REFACTOR_RULES.md` were made: renames, import wiring,
path parametrisation, documentation, launchers. Every bug found is listed under "Issues found, not fixed" and left in
place.

DSPy.RLM was evaluated in an earlier campaign (results in the original repository's `docs/rlm_results_summary.xlsx`).
It has **no row in the final results table** (`docs/Experiments.xlsx`), so there are no per-experiment configs;
`experiments/browsecomp.sh` is illustrative and `experiments/obliq_shard.sh` is the earlier campaign's OBLIQ launcher.

## Renames

| Original (`original/RLM/…`)            | New (`agents/dspy_rlm/…`)            |
|----------------------------------------|------------------------------------------|
| `rag/` (package `rag`)                 | `src/dspy_rlm/` (package `dspy_rlm`)     |
| `rag/__init__.py`                      | `src/dspy_rlm/__init__.py`               |
| `rag/agent.py`                         | `src/dspy_rlm/agent.py`                  |
| `rag/signatures.py`                    | `src/dspy_rlm/signatures.py`             |
| `rag/tools.py`                         | `src/dspy_rlm/tools.py`                  |
| `examples/browsecomp_eval_dspy_rlm.py` | `src/run_browsecomp.py`                  |
| `examples/obliq_eval_dspy_rlm.py`      | `src/run_obliq.py`                       |
| `scripts/run_obliq_dspy_shard.sh`      | `experiments/obliq_shard.sh`             |
| (none)                                 | `experiments/browsecomp.sh` (illustrative launcher, new) |
| `examples/browsecomp_data.py`          | `agents/shared/rlm_shared/browsecomp_data.py` (imported, not copied) |
| `examples/obliq_score_advice.py`       | `agents/shared/rlm_shared/obliq_score_advice.py` (imported, not copied) |
| `examples/obliq_data.py` (used by the shard script) | `agents/shared/rlm_shared/obliq_data.py` (imported, not copied) |
| `pyproject.toml`                       | not copied: dependencies are the repository's (`dspy` 3.3.1, `requests`, `python-dotenv`, `datasets`, `openai`, `transformers` only for `SEARCH_SNIPPET_TOKENS`); Deno is needed by `dspy.RLM`'s sandboxed REPL |

Public names are unchanged: `dspy_rlm` exports `RemoteRetriever`, `FaissRetriever`, `make_embedding_fn`,
`RunMetrics`, `build_lm`, `build_rlm_agent`, `run_question`, and every signature class keeps its name.

## Per-file diff summary

All `+` lines not listed here are comments (`# Refactor note: …`) or module docstrings.

- **`src/dspy_rlm/__init__.py`, `agent.py`, `signatures.py`, `tools.py`**: documentation only. Module docstrings
  extended (origin, role, how it is run); in `agent.py` the docstring's usage example now says `from dspy_rlm import`.
  Comments mark dead code and stale text. No code line changed: `tests/test_dspy_rlm_static.py` checks that each
  file's AST equals the original's once the module docstring is removed. The `rag.agent` relative imports
  (`from .signatures`, `from .tools`) needed no change.
- **`src/run_browsecomp.py`**:
  - `sys.path.insert(0, <RLM root>)` → two inserts computed from `__file__`: `src/` (for `dspy_rlm`) and `<repo>/agents/shared`
    (for `rlm_shared`). Rule 2.
  - `from rag…` → `from dspy_rlm…` (including the in-worker imports and the `__import__("rag.signatures", …)`
    calls → `__import__("dspy_rlm.signatures", …)`); `from examples.browsecomp_data` → `from rlm_shared.browsecomp_data`.
  - Docstring and comments. `tests/test_dspy_rlm_static.py` checks that the AST equals the original's after undoing
    exactly these renames and dropping `sys.path.insert` lines.
- **`src/run_obliq.py`**:
  - Same `sys.path` and `rag` → `dspy_rlm` wiring; `from examples.obliq_score_advice` → `from rlm_shared.obliq_score_advice`.
  - `OBLIQ_ROOT = Path(__file__).parent.parent.parent / "OBLIQ-Bench"` (the OBLIQ-Bench checkout beside the RLM
    repository) → `_obliq_root()`, which returns `rlm_shared.paths.path("OBLIQ_BENCH_ROOT")` when used (raises naming the
    variable if unset). The four uses (`load_queries`, `load_qrels` ×2, `load_excluded_ids`) call it. Rule 3.
  - The file's own copies of `ndcg_at`, `recall_at`, `extract_doc_ids_from_answer`, `load_queries`, `load_qrels` are
    **kept**, not replaced by `rlm_shared.obliq_data` (they differ in detail; see issues).
  - Docstring and comments. AST-checked like `run_browsecomp.py`, with the `OBLIQ_ROOT` substitution undone.
- **`experiments/obliq_shard.sh`** (from `scripts/run_obliq_dspy_shard.sh`): positional arguments, `OBLIQ_TASK/SUBSET/INDEX`
  defaults, per-shard environment (`EMBEDDING_SERVER_URL`, `OBLIQ_INDEX`, `LOCAL_LLM_BASE_URL`, `LOCAL_LLM_MODEL`,
  `LOCAL_LLM_EFFORT=medium`, `LOCAL_LM_TIMEOUT=1800`, `PYTHONUNBUFFERED=1`) and driver flags
  (`--max-iters 12 --max-search 8 --top-k 20 --timeout 0`) are unchanged. Changed: sources `configs/paths.env` and
  exports `PYTHONPATH=<repo>/agents/shared:<algo>/src`; the interpreter path → `${PYTHON:?}`; `HF_HOME` → `${HF_HOME:?}` from
  paths.env; Deno's `<home>/.deno/bin` → `${DENO_INSTALL:-$HOME/.deno}/bin` (same directory on the original machine);
  `cd <RLM root>` → `cd <repo root>` (relative `OUT` prefixes now resolve against the repository root); the id→index
  mapping heredoc imports `rlm_shared.obliq_data` instead of `sys.path.insert(<RLM root>); from examples import
  obliq_data`, and checks `OBLIQ_BENCH_ROOT` up front; driver path → `src/run_obliq.py`. `/tmp/obliq_dspy_shards` kept.
- **`experiments/browsecomp.sh`** (new): the original had no wrapper for this driver. Sources `configs/paths.env`, sets
  `PYTHONPATH`, puts Deno on `PATH`, and runs `src/run_browsecomp.py` with the driver's own argparse defaults written
  out (`--start 0 --end 50 --out logs/run --max-iters 16 --max-search 35 --max-depth 5 --top-k 10 --sig v1 --timeout
  ${BROWSECOMP_TIMEOUT:-0}`); extra arguments are appended and override them. Marked illustrative: no final-table experiment.
- **`configs/example.env`** (new): every environment variable the code reads, with its default and meaning. Optional
  variables are commented out, because an empty value is not the same as unset (`int("")` raises).

Hunks outside rules 1–5: **none**.

One consequence of the move that is not a code change: `load_dotenv()` (agent.py:45, run_browsecomp.py:81,
run_obliq.py:88) searches for `.env` upward from the calling file's directory. In the original layout that found the
RLM repository's `.env` (Azure credentials); here it searches `src/dspy_rlm/`, `src/`, `agents/dspy_rlm/`,
`agents/`, the repository root and above. Export `AZURE_OPENAI_*` or place a `.env` accordingly.

## Issues found, not fixed

Line numbers are `src/…` (original in parentheses).

1. **v13–v20 fall back to v1 when `--timeout > 0`.** `run_browsecomp.py:109-117,124` (orig 98-105,113): the worker's
   `_SIGS` map has v1–v11 only; `_SIGS.get(sig_key)` returns `None` for v13/v14/v15/v16/v17/v20 and `run_question`
   then uses `BrowseCompSignature` (v1). Without a timeout the right signature is used.
2. **`heuristic_match` counts an empty prediction as correct.** `agents/shared/rlm_shared/browsecomp_data.py:112-114`
   (orig `examples/browsecomp_data.py`, identical): `g in r or r in g` with `r == ""` is `True` for every gold answer,
   and `judge_match` (`browsecomp_data.py:135`) returns before calling the LLM judge. More generally any prediction that
   is a substring of the gold (e.g. "1" for "1990") is correct. Used by `run_browsecomp.py:211` (orig 192).
3. **The local-LLM path still needs Azure for the sub-LM.** `agent.py:477-502` (orig 459-484): `build_rlm_agent`
   always builds the `llm_query` sub-LM as `azure/<SUB_LM_MODEL…>` and reads `os.environ["AZURE_OPENAI_ENDPOINT"]` /
   `["AZURE_OPENAI_API_KEY"]` (agent.py:489-490), so `LOCAL_LLM_BASE_URL` (agent.py:190) switches only the root LM;
   without Azure settings it raises `KeyError`. The BrowseComp judge is Azure too. In OBLIQ, `run_obliq.py:288-306`
   (orig 262-280) uses the local model only with `LOCAL_SUB_LM=1`, which `obliq_shard.sh` does not set, so unless the
   caller exported it the Qwen shard runs sent `llm_query` to Azure.
4. **OBLIQ records log `AZURE_OPENAI_MODEL` even for Qwen runs.** `run_obliq.py:812` (orig 777) writes `"model":
   AZURE_OPENAI_MODEL` and `:845` (orig 810) prints it in the .txt, also when `LOCAL_LLM_BASE_URL`/`LOCAL_LLM_MODEL`
   chose the root LM. Same in `run_browsecomp.py:365` (orig 344, "… (Azure)").
5. **OBLIQ `judge_model` record does not match the sub-LM used.** `run_obliq.py:816` (orig 781) records
   `AZURE_OPENAI_JUDGE_MODEL` with default `gpt-5.4-mini`, but `_build_sub_lm` (`run_obliq.py:281-282`, orig 258-259)
   falls back to `AZURE_OPENAI_MODEL`.
6. **`--subset wildchat-v7` raises `KeyError`.** It is an argparse choice (`run_obliq.py:935`, orig 900) with no entry
   in `SUBSET_SIGNATURES`, looked up at `run_obliq.py:703` (orig 671).
7. **`--max-iters` help/default mismatch.** `run_obliq.py:946-947` (orig 910-911): default 12, help "default 16".
   In `run_browsecomp.py` the CLI default is 16 (`:423`, orig 400) while `run_one`/`run_eval` default to 25 (`:143,265`,
   orig 130,246), and the .txt header hard-codes "Max iters : 25   Max depth: 5" (`:368`, orig 346) regardless of the
   values used. The `--sig` help (`:434`, orig 411) describes only v1/v2.
8. **The OBLIQ timeout leaves the daemon thread running.** `run_obliq.py:582-590` (orig 553-561): after
   `t.join(timeout)` the question is recorded as TIMEOUT but the thread keeps calling the LM and retriever while later
   questions run, and its class-level `__call__` timing patch (`:461-484`, orig 434-457) stays installed, so the next
   question's patch wraps it and LM latency can be charged to the wrong question.
9. **BrowseComp timeout worker: join-before-get.** `run_browsecomp.py:174-182` (orig 157-165): `p.join(timeout)` runs
   before `q.get()`. A child whose pickled `(pred, RunMetrics)` exceeds the pipe buffer (RunMetrics carries the full text
   of every fetched document in `fetched_docs`) cannot exit until it is read, so it is killed and recorded as TIMEOUT
   (Python docs, "Joining processes that use queues"). A child that dies without `put()` (e.g. `sys.exit(1)` in
   `search_index`, agent.py:342) leaves `q.get()` blocking forever.
10. **`_FilteredRetriever` filters after top-k.** `run_obliq.py:553-556` (orig 524-527): excluded ids are removed from
    the server's top-k, so a search returns fewer than `top_k` results (possibly none) instead of the next-best
    non-excluded documents.
11. **Duplicate OBLIQ loaders/metrics.** `run_obliq.py:107-217` (orig 84-194) duplicate `rlm_shared/obliq_data.py`, with
    differences: the wildchat id regex is gated on a fixed tuple (`run_obliq.py:141`, orig 118) that omits
    `wildchat-v7`, vs `subset.startswith("wildchat")` (`obliq_data.py:52`); the `qrels_pool.tsv` fallback applies only
    when `pool=True` and prints a warning (`run_obliq.py:206`, orig 183) vs always (`obliq_data.py:65`). Kept, not
    replaced, to preserve behaviour.
12. **Dead code** (kept, marked with comments):
    - `max_depth` / `depth`: parameters of `build_rlm_agent` (`agent.py:232-233`, orig 219-220) and `run_question`,
      passed through from `--max-depth` (`run_browsecomp.py:426`, orig 403) but never read.
    - `RunMetrics.delegation_calls` (`agent.py:70`, orig 58): never incremented, yet printed and recorded as `deleg=`.
    - `RunMetrics.fetched_docs` (`agent.py:76`, orig 64) is filled but never written out; `RunMetrics.trajectory`
      (`agent.py:81`, orig 68) is never filled by `run_question`, so BrowseComp records have no transcript.
    - `make_embedding_fn` (`tools.py:65`, orig 59): exported, never called.
    - `FaissRetriever` annotation in `run_browsecomp.py:140` (orig 127): undefined in that module; harmless only because
      of `from __future__ import annotations`. The `agent.py` usage docstring (`:18`, orig 10) shows a stale
      `FaissRetriever(index_pattern, embedding_fn)` constructor.
    - `scripts/update_baseline_summary.py` (`run_browsecomp.py:344-348`, orig 324-327): run when `--out` contains
      "baseline", but the script does not exist (deleted from the original repository in commit 6948bf7); `check=False`
      hides the failure.
    - Unused imports: `re`, `signal`, `decrypt`, `heuristic_match` in `run_browsecomp.py:20-21,40-41`; `re`, `signal`
      in `run_obliq.py:46-47`; a redundant local `import os` in `run_obliq.py:275`.
13. **"≤2000 chars" docstrings vs 500-char snippets.** The model-visible `search_index` docstrings (`agent.py:269`, orig
    253; `run_obliq.py:328`, orig 301) and the shared prompt blocks (`signatures.py:38,78,2371`, orig 32,72,2365) promise
    ≤2000-char / "2000-char" snippets, but both harnesses truncate to `SEARCH_SNIPPET_LEN=500` characters by default
    (`agent.py`, orig 344; `run_obliq.py`, orig 334). The prompt's `top_k: int = 20` also differs from the BrowseComp
    default of 10. Prompt text, so unchanged.

## How verified

- `python -m py_compile` on every `.py` in `src/`; `bash -n` on both launchers (also run as tests).
- `diff -u original/<file> src/<file>` for every file, each hunk reviewed against rules 1–5 (summary above).
- `pytest agents/dspy_rlm/tests` (run with `.venv/bin/python -m pytest`, which uses the vllm conda env with
  dspy 3.3.1): **98 passed**, mocks only, no network or LM call:
  - `test_dspy_rlm_signatures.py` (49): every signature class has `question → answer` fields, non-empty instructions,
    its own `__name__`, and builds a `dspy.RLM`; the full set of classes, their instructions, names and field
    descriptions, and every private prompt constant equal the original `rag.signatures` (imported from
    `original/RLM` under its own name); `make_rlm_signature` and `browsecomp_minimal_budget_sig` (several budgets, with
    and without `BROWSECOMP_COMMIT_TURN`) produce identical instructions.
  - `test_dspy_rlm_tools.py` (9): `RemoteRetriever` URLs, JSON payloads (with/without `index`, `OBLIQ_INDEX`) and
    timeouts (`RETRIEVER_TIMEOUT`), with `requests.post` monkeypatched; byte-identical requests to the original.
  - `test_dspy_rlm_agent.py` (18): `build_rlm_agent` passes `max_iters`, `verbose`, the signature and the
    `[search_index, get_document]` tools to `dspy.RLM` (recorder mock, plus one real construction);
    `DISABLE_GET_DOCUMENT`, `RLM_MAX_OUTPUT_CHARS`, sub-LM model selection, tool docstrings; `search_index` snippet
    truncation, caching, `MAX_TOP_K`, the per-turn cap (`MAX_SEARCH_PER_TURN`, 5 s turn gap) and the hard
    `max_search_calls` stub, with a fake retriever; the same call sequence gives identical results on the original
    `rag.agent`; `run_question` with a fake RLM.
  - `test_dspy_rlm_drivers.py` (8): signature maps, shared-module wiring, `OBLIQ_BENCH_ROOT` loaders on a temp tree,
    metrics/id parser; pins issues 1, 2 and 6.
  - `test_dspy_rlm_static.py` (14): py_compile, `bash -n`, AST identity of the package with `rag/`, and of the drivers
    after undoing only the allowed renames.
- No experiment, server, network or LLM call was run.

## Later edit (review, 2026-10-30)

- `src/run_obliq.py`: the `--model` help text named a personal Azure deployment; it now reads "an Azure deployment
  name" (rule 4, documentation; not seen by the model). `tests/test_dspy_rlm_static.py` blanks that one help string on
  both sides before comparing the AST, so every other line is still checked against the original.

## Second snapshot (2026-11-11): the DSA-3 / DSA-5 ablation

The DSPy.RLM ablation of 2026-10-31 (arms 3 and 5, docs/experiments/dspy_rlm/DSA-*.md) ran on files changed after the
first snapshot. They are preserved in `original/RLM_2026-11-11/` (own `MANIFEST.json`); `original/RLM/` is untouched.

| File (original repository) | 2026-10-30 → 2026-10-31 | In `src/` |
|---|---|---|
| `rag/signatures.py` | signature v21 `BrowseCompSnippetPromptSignature` (the snippet REPL agent's task text) appended | yes, verbatim (`src/dspy_rlm/signatures.py`, end of file) |
| `rag/agent.py` | `DSPY_QUIET_GETDOC=1` (no reading advice in the get_document line) and `DSPY_MAX_LLM_CALLS` (lifts dspy.RLM's 50-call sub-LLM cap); both opt-in, defaults unchanged | yes, verbatim (`src/dspy_rlm/agent.py:398-403, 535-537`) |
| `examples/browsecomp_eval_dspy_rlm.py` | `v21` in the signature table and `--sig` choices; **fix of the result-queue deadlock** in `run_one` (read the queue before `join`; a result larger than the 64 KB pipe buffer otherwise blocked until the timeout and was recorded as TIMEOUT) | v21 only (`src/run_browsecomp.py:73, 434`). The fix is a behaviour change of the default path (`--timeout > 0`), so `src/` keeps the 2026-10-30 behaviour (rule: no bug fixes in `src/`) |
| `rag/__init__.py`, `rag/tools.py`, `examples/browsecomp_data.py` | unchanged | copied so the snapshot runs on its own |
| `scripts/run5d/dspy_ablation_arm{3,5}.sh` | new | `experiments/dsa_arm{3,5}.sh` |
| `scripts/run5d/dspy_ablation_report.py` | new | `src/ablation_report.py` |
| `scripts/run5d/dspy_ablation_strata.py`, `dspy_vs_snip.py` | new (drew `sample100.json`) | not copied: their input is lost (below) |
| `logs/dspy_ablation/sample100.json` | the 100 questions (qid, index, stratum; no text) | `experiments/dsa_sample100.json` (byte-identical) |

Arm 4 (`dspy_ablation_arm4.sh`, stopped after 7 questions) is not migrated.

**The DSA launchers run the snapshot, not `src/`.** `experiments/dsa_arm{3,5}.sh` start
`original/RLM_2026-11-11/RLM/examples/browsecomp_eval_dspy_rlm.py`, because only that file has the queue fix the arms
ran with. Their arguments and environment are those of the original arm scripts; changed only: locations from
`configs/paths.env`, the root deployment from `AZURE_DEPLOYMENT_GPT5` (original: a personal deployment name), `AZ_ENV`
required (original default: a Claude Code scratchpad file), outputs under `$RUNS_DIR/dspy_ablation/arm{3,5}`.
`rag/agent.py` calls `load_dotenv()`, which searches upwards from the snapshot directory; the original found the
original repository's `.env` (KNOWN_ISSUES K15). Only the in-run `correct` field depends on it.

`src/ablation_report.py`: docstring (rule 4), the judge import from `agents/shared/rlm_shared/grading/bcp_judge.py`
(rule 2), `LOGS_ROOT` for the original machine's paths (rule 3). Nothing else.

### Issues found, not fixed (second snapshot)
16. **(F-F) The result-queue deadlock remains in `src/`.** `src/run_browsecomp.py:175` `run_one` (orig. 2026-10-30
    `examples/browsecomp_eval_dspy_rlm.py:157`) joins the worker before reading its queue. Fixed in the
    2026-10-31 file (`original/RLM_2026-11-11/RLM/examples/browsecomp_eval_dspy_rlm.py:158-188`), which is what DSA-3/5 ran.
17. **(F-G) The sample cannot be redrawn.** `original/RLM_2026-11-11/RLM/scripts/run5d/dspy_vs_snip.py:8` reads
    `per_question.json` from a Claude Code scratchpad that no longer exists; `dspy_ablation_strata.py:4` `exec`s that
    file. The sample is preserved (`experiments/dsa_sample100.json`), the code to redraw it is not runnable.
18. **The report names every arm `arm3_*`.** `src/ablation_report.py:74-92` (orig. 56-74): the keys of `report.json`
    say `arm3` also for arm 5 (the file was written for arm 3 and reused).
19. **The arm-5 header says "NOT YET RUN".** `original/RLM_2026-11-11/RLM/scripts/run5d/dspy_ablation_arm5.sh:2`; the
    arm was run on 2026-10-31/11-01 (`logs/dspy_ablation/arm5/`). Kept in the launcher's quoted header.

Verified: `tests/test_dspy_rlm_static.py` (agent.py and signatures.py are AST-identical to the second snapshot;
run_browsecomp.py equals the 2026-10-30 driver plus the two v21 lines; the second snapshot's driver differs from the
first only by v21 and the queue fix), `tests/test_dspy_rlm_second_snapshot.py` (v21 text, both switches on and off,
the launchers), `tests/test_ablation_report.py` (identical `report.json` on synthetic records).
