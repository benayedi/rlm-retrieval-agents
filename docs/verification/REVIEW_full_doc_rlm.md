# Scientific code review: Full-Doc RLM

## 1. Scope

**Reviewed:**
- the code in `algorithms/full_doc_rlm/src/` and the exact originals in `algorithms/full_doc_rlm/original/`;
- the launchers in `algorithms/full_doc_rlm/experiments/` and the 11 row configurations in `algorithms/full_doc_rlm/configs/`;
- the parts of the `rlm` library (github.com/alexzhang13/rlm @ 854e688) that the harness depends on: the turn loop,
  the forced final answer, output truncation, the REPL builtins, environment keyword handling and the Azure client;
- read-only checks against the stored run records of the rows that are on this host:
  - BCP-25, BCP-27, BCP-29, BCP-31, FR-11, FR-13: one record header per run directory;
  - MAT/TWI/CON-11: every trajectory;
  - OBLIQ `_provenance.json` launch arguments.

**Not reviewed:**
- the retrieval servers outside this folder (BM25, Gemini, the Math proxy; see `benchmarks/`);
- the gpt-4.1 grading pipeline that produced the workbook numbers (see [RESULT_INTEGRITY.md](RESULT_INTEGRITY.md));
- the run records of BCP-21 and BCP-23, which are not on this host.

No experiment, model call, retrieval server or network call was made for this review.

## 2. Method specification vs. what the code does

The specification is the method as stated in the project documents: "search returns whole documents into a `DOCS`
variable in the REPL; the model reads slices itself; client-side caps of 10 searches and 30 turns, plus a forced
final answer; no sub-LLM".

| Specified | What the code does | Agreement |
|---|---|---|
| Search returns whole documents into `DOCS` | Design B sends `mode: "full"`; every hit carries the complete `text`; hits are merged into `DOCS`, and a later snippet-only hit never replaces a full text (`tools.py` `search`) | yes |
| The model reads slices itself | Only `print` output reaches the model; each code block's output is cut to 20,000 characters (`rlm/utils/parsing.py` `format_iteration`, called at `rlm/core/rlm.py:456`) | yes |
| Results per search | k is chosen by the model and clamped to 1..50 (`MAX_K`); `--default-k 5` is only the fallback. OBLIQ pins k = 20 with `--fixed-k` | **differs** from "top-5" in earlier descriptions (K1) |
| 10-search cap | Enforced client-side in `ToolBox.search`, in the same process as the model's code: over-cap calls return `[]` and are counted in `search_refused`. Failed calls also consume the cap. The server counts independently, and `judge.py` flags excess | yes, with caveats (K21; BCP-23 shows up to 19 searches, K5) |
| 30 turns + forced final answer | Library loop of `max_iterations`, then `_default_answer` makes one more call, logged as an iteration, so `n_iterations` = 31 at the cap | yes; the count includes the forced call (K12) |
| No sub-LLM | `--no-delegation` reaches `LocalREPL` as an unused keyword argument; `llm_query`, `llm_query_batched`, `rlm_query` and `rlm_query_batched` stay installed. Only the solo prompt omits them | **prompt-only** (K9) |
| Tools described to the model | The description also advertises `read_docs` (not installed) and `llm_query` for reading, and calls the retriever "a dense semantic index (one embedding per document)" in the ColBERT rows. The BM25 rows rewrite these sentences in memory (`run_bm25.py`) | **differs** (K10) |
| Reasoning effort as labelled | The stock Azure client sends only model and messages. Effort reaches Azure only with the `azure_sampling` hook (part of `run5d`: FR-11, OBLIQ). The OpenAI-compatible client (Qwen3.6 rows) forwards it | **differs** for BCP-25; BCP-21/23 unverified (K3) |
| Seed context, grounding gate, sandbox | `--seed-context` spends search 1 on the question verbatim. The gate holds a submission once if its `Exact Answer:` string is in no `DOCS` text. Imports, `open` and DNS are restricted; a post-hoc token scan flags runs | yes (one edge case, §4 F-new-1) |

## 3. Checks performed and their evidence

| # | Check | Method | Result |
|---|---|---|---|
| C1 | Prompt identity | `tests/test_prompts_identical.py` loads `src/prompts.py` and `original/FULLDOC/src/prompts.py` side by side and compares: every module-level string; `build_prologue()` for all 128 switch/budget combinations; `ToolBox.descriptions()` for five design/cap/k settings; the BM25 rewrite of `run_bm25.py` against the original wrapper's output, for FRAMES and BrowseComp-Plus | all equal |
| C2 | Prompt id against the run records | `prologue_id()` of `src/prompts.py`, without and with the BM25 rewrite, and with each OBLIQ prompt file in `GEPA_PROMPTS` | `94f31e91f6b7` (BCP-27/29 records), `758e43ed5b95` (BCP-25/31, FR-11/13), `f92043bfd120` / `299e86ce3e1e` / `891e1319f80c` (MAT/TWI/CON-11): each equals the `prompt_id` in the corresponding records |
| C3 | Result-cache keys | `tests/test_cache_keys.py` imports each driver in a fresh interpreter with the environment its wrapper set and prints `FINGERPRINT` and `ckey(seed)` | `9e6ce99e` / `e925f0f6c7` (BCP-27), `29587b4e` / `ee5ea32a1c` (BCP-29), `aa1d1029` / `11d0c5bc6b` (BCP-31): equal to the original progress logs, so the refactored driver addresses the original caches |
| C4 | ToolBox behaviour | `tests/test_toolbox.py`, with a fake retrieval server patched into `urllib.request.urlopen` | cap of 10 with 2 refusals on 12 calls, none sent to the server; `search_many` counts per query; k clamps (None→5, 30→30, 100→50, 0→1, "x"→5); `--fixed-k` and design A ignore k; DOCS keeps full text over a later snippet; docids de-duplicated in order; unread tracking; `read_docs` advertised but not installed |
| C5 | Record building | `tests/test_run_one_record.py` runs `run_one.main()` with a fake agent that drives the real `RLMLogger` and calls the library's own `RLM._default_answer` | `n_iterations` = 31 after 30 turns, 5 after 4; `served_models` = `None`; on `TimeoutExceededError` with a partial answer, status `error` and empty output; `k_settable_by_model` true by default, false with `--fixed-k` |
| C6 | Library behaviour | Source reading plus `tests/test_documented_issues.py` | `LocalREPL(no_delegation=True)` still exposes the four sub-LLM functions; `prologue_id()` is unchanged when the solo prompt, `LEAN_CLAUSE` or `NO_DELEGATION_CLAUSE` change; adapter snippet mode returns full text |
| C7 | Refactor fidelity | `diff -u` of every original/refactored pair; flag sets of each launcher vs. its original (`tests/test_static.py`); `preserve_originals.py --check` | only renames, import wiring, path/credential parametrisation and documentation; launcher flags identical; originals unmodified |
| C8 | Row configurations | Head of one record per run directory (`config` block, first 200 KB); OBLIQ `_provenance.json`; tails of records for `n_iterations`, `served_models`, search counts | configs match. BCP-25: 154/830 records at `n_iterations` 31, `served_models` null in all, max 10 searches. FR-11: 48/824 at 31. CON-11: all 254 used exactly 10 searches |
| C9 | Sub-LLM and tool use in OBLIQ trajectories | Every `code_blocks[].code` of MAT/TWI/CON-11 (151/281/254 trajectories) scanned for calls | no call to `llm_query`, `llm_query_batched`, `rlm_query` or `rlm_query_batched`. Calls to the advertised `read_docs` appear in 33/151, 17/281 and 37/254 runs; each of those runs records `NameError: name 'read_docs' is not defined` |

Test totals: 89 passed with `rlm` importable; 82 passed and 7 skipped without it.

## 4. Findings

Ids refer to [KNOWN_ISSUES.md](KNOWN_ISSUES.md); file:line references are in
[CHANGES_FROM_ORIGINAL.md](../../algorithms/full_doc_rlm/CHANGES_FROM_ORIGINAL.md).

| Id | Finding | Effect on reported results |
|---|---|---|
| K1 | k is model-chosen up to 50, not 5; only OBLIQ fixes it (20) | The method description must state it. Per-search document counts are much larger than a top-5 agent's |
| K2 | Full-Doc vs Snippet rows also differ in search cap, turn limit, time limit and ColBERT index length | A pair compares packages, not reading strategies |
| K3 | The effort setting is dropped on Azure without the `azure_sampling` hook | BCP-25 ran at the deployment default; BCP-21/23 unverified |
| K5 | BCP-21/23 ran on an unpreserved patched `rlm` build; BCP-21 launchers lost; BCP-23 shows up to 19 searches with a cap of 10 | These two rows cannot be reproduced exactly from this repository |
| K7 | Unfinished questions are scored wrong; a partial answer is dropped on timeout | Convention; Qwen3.6 rows measure accuracy within the time budget |
| K9 | "No sub-LLM" is prompt-only | No call found in the OBLIQ trajectories (C9); BrowseComp-Plus/FRAMES trajectories not scanned |
| K10 | The tool text advertises `read_docs`, `llm_query` for reading, and a dense retriever for ColBERT runs | Part of the evaluated prompt. In OBLIQ the model called `read_docs` and received a `NameError` in 33 MAT-11, 17 TWI-11 and 37 CON-11 runs (C9), each costing a turn |
| K12 | `n_iterations` includes the forced final-answer call | Averages of iterations include it; the workbook footnote states this |
| K21 | Hygiene: `served_models` always `None`; `prompt_id` blind to the solo prompt; process-wide `json.dump` patch in `run_bm25.py`; temp prompt files kept; failed searches use the cap | Provenance and robustness only |
| K22 | Missing helpers: `assert_served_model.py`, `report.py`, `budget.py` | BCP-23 launchers abort at the served-model gate |

Additional observations from this review, not yet in KNOWN_ISSUES.md:

- **F-new-1.** The grounding gate searches DOCS only for answers of at least two characters, but its hold condition
  does not check length. A one-character `Exact Answer:` (e.g. `7`) is therefore always held once, whether or not it
  occurs in a source (`run_one.py` `install_grounding_gate`).
- **F-new-2.** The cap sentence in the tool description says over-cap calls "return an error instead of results";
  the code returns `[]` with a printed notice.
- **F-new-3.** `ToolBox` sends `retriever: "dense"` with every design-B search, including to BM25 and ColBERT
  servers. No preserved server reads the field.
- **F-new-4.** `judge.py`'s one-configuration check compares only model, effort, `default_k`, `k_settable_by_model`,
  `prompt_id`, budget and orchestrator. It does not compare the search cap, `fixed_k`, the server, the prompt
  switches or the timeout.
- **F-new-5.** Design D remains in `ToolBox`, but no preserved server implements its `passages` mode. No row uses it.

## 5. Demonstrated vs. only inspected

**Demonstrated** (executed in tests or on stored records):
- prompt and tool-text identity with the originals (C1);
- prompt ids and cache keys equal to the recorded values (C2, C3);
- ToolBox cap, refusal, k and DOCS behaviour against a fake server (C4);
- record fields `n_iterations`, `served_models` and the timeout path through the real logger and the library's
  forced-answer code (C5);
- `no_delegation` ignored by `LocalREPL` (C6);
- configuration values and counts read from stored records, and the OBLIQ trajectory scan (C8, C9).

**Inspected only** (source reading, not executed):
- the sandbox (`harden_sandbox`, resolver restriction, token scan);
- the grounding gate on real model output;
- the 20,000-character truncation in a live loop;
- the Azure client dropping the effort setting;
- the retrieval servers, `run_experiment.py` scheduling, resume and timeout stubs;
- `judge.py` voting and caching;
- the GEPA optimisation path of `split_runner.py`;
- the launchers beyond `bash -n` and flag comparison;
- `provenance.py`.

**Not checked:**
- BCP-21/23 behaviour;
- sub-LLM use in BrowseComp-Plus/FRAMES trajectories;
- whether vLLM's Qwen3.6 template honours `reasoning_effort`.

## 6. Equivalence statement

Full experimental equivalence between `src/` and the code that produced the results was **not** re-established.
No question was re-run, so no outputs, accuracies or costs were compared. The review establishes four things:
- the refactor preserves every model-visible text and the configuration-dependent identifiers (prompt ids, cache keys);
- it changes only file names, import wiring, paths, credentials and documentation;
- the behaviours above hold on mocks;
- the stored records agree with the configurations.

Behavioural equivalence on real runs is an open question. It is in any case out of reach for BCP-21 and BCP-23,
whose library build and (for BCP-21) launchers were not preserved.
