# Full-Doc RLM: changes from the original code

`original/` holds the exact files that produced the results (`original/MANIFEST.json`, SHA-256 checked by
`scripts/provenance/preserve_originals.py --check`). `src/` is the same code made runnable from this repository under
`docs/architecture/REFACTOR_RULES.md`: renames, import wiring, path/credential parametrisation, documentation and
launchers. No prompt string, tool description, default, cap, budget, retry or scoring rule was changed; the bugs
found are listed under [Issues found, not fixed](#issues-found-not-fixed).

The algorithm in one paragraph: `run_one.py` answers one question with the `rlm` library (github.com/alexzhang13/rlm
@ 854e688, from `RLMLIB_PATH`). Its `ToolBox` (`tools.py`) installs `search` in the REPL; in design B a search
returns whole documents, which also accumulate in the REPL variable `DOCS`, and the model reads slices of them in
its own code. The client caps searches (`--max-searches 10`) and turns (`--max-iterations 30`); after the last turn
the library makes one more call that forces a final answer.

## Rename table

| Original | New |
|---|---|
| `original/FULLDOC/src/{run_one, tools, prompts, fence_tolerant, rep_guard, obliq_adapter, retrieval_server, retrieval_server_colbert, encoder, hybrid, build_bm25, prepare_data, run_experiment, judge, grade_local, score_obliq, provenance}.py` | `src/` under the same names |
| `original/RLM/scripts/frames_sup_run_one.py` | `src/run_bm25.py` |
| `original/RLM/scripts/gepa_bcp.py` | `src/split_runner.py` (loaded by the two wrappers below as module `split_runner`, was `gepa_bcp`) |
| `original/RLM/scripts/run5d/bcp_fd_qwen_dense.py` | `src/bcp_qwen_dense.py` |
| `original/RLM/scripts/bcp_bm25_test.py` | `src/bcp_bm25_split.py` |
| `original/FULLDOC/src/apply_*_patch.py` (9 files) | not copied: historical in-place patchers, already applied; see `src/PATCH_HISTORY.md` |
| `original/RLM/scripts/bcp_gpt5_sup.sh` | `experiments/bcp_gpt5_colbert.sh` |
| `original/RLM/scripts/bcp_gpt5_bm25.sh` | `experiments/bcp_gpt5_bm25.sh` |
| `original/RLM/scripts/run_frames_sup.sh` | `experiments/frames_qwen.sh` |
| `original/RLM/scripts/run5d/frames_fd_gpt5.sh` | `experiments/frames_gpt5.sh` |
| full-doc half (`fd` arm) of the original repository's `scripts/run5d/obliq_gemini.sh` | `experiments/obliq_fd.sh` |
| `original/FULLDOC/src/{final830_run, final830_complete, launch_high400, launch_high100b, launch_high100c, launch_high_rest230}.sh` | `experiments/bcp_gpt5_final830/` under the same names |
| (new) | `experiments/_common.sh`, `experiments/bcp_gpt5_final830/_azure.sh`, `configs/<ROW>.env` (11 rows), `src/PATCH_HISTORY.md`, `tests/` |

## Configuration this algorithm needs

From `configs/paths.example.env`: `RLMLIB_PATH`, `FULLDOC_ROOT` (data: `data/prepared`, `data/indexes`, `data/models`;
also the working directory the workers run in), `RUNS_DIR` (all outputs), `FRAMES_DATA_DIR`, `AZ_ENV`,
`AZURE_DEPLOYMENT_GPT5`, `AZURE_DEPLOYMENT_GPT52`, `PYTHON`, `PYLATE_PYTHON`, `HF_HOME`.

Not yet in `configs/paths.example.env` (used here, each fails with a message naming it when missing):

| Variable | Used by | Points at (original) |
|---|---|---|
| `RLM_DATA_DIR` | `obliq_fd.sh`, `bcp_bm25_split.py`, default of `GEPA_SPLIT` in `split_runner.py` | `data/` of the original repository: `obliq_math/`, `obliq_congress/`, `obliq_fd/`, `obliq_fd_gemini/`, `bcp_gepa_splits/` |
| `HOOKS_DIR` | `bcp_gpt5_colbert.sh`, `bcp_gpt5_bm25.sh`, `frames_gpt5.sh`, `obliq_fd.sh` | the startup-hook directories (`retry429/`, `run5d/`, ...) of the Snippet RLM; default `algorithms/snippet_rlm/src/hooks` |
| `AZURE2_API_VERSION` | `bcp_gpt5_final830/_azure.sh` | the API version in the original working directory's `.env` (value not recorded) |
| `FRAMES_GRADE_SUP` | last step of `frames_qwen.sh` | `scripts/frames_grade_sup.py` of the original repository |
| `BCP_POLICY_COMPARE` | comparison step of `bcp_bm25_split.py` | `scripts/bcp_policy_compare.py` of the original repository |
| `WAIT_PID` (optional) | `launch_high400.sh`, `launch_high100c.sh` | a predecessor PID that was hard-coded |

## Per-file diff summary

Every hunk of `diff -u original/... src/...` falls under rules 1 to 5. Lines marked "rule 3" turn an absolute path
into a configured location and, where the original value was not personal, keep it in a comment.

| File | Changes |
|---|---|
| `run_one.py` | Docstring. `ROOT="/home/.../suprlm"` + `sys.path.insert(ROOT/src)` replaced by: this directory on `sys.path` (rule 2); `rlm_shared/paths.py` loaded **by file**, not put on `sys.path`, so the set of modules REPL code can import is the same as in the original worker (rule 2); `load_dotenv(<FULLDOC_ROOT>/.env)` only when `FULLDOC_ROOT` is set (a missing file is a no-op); `load_query` default path `<FULLDOC_ROOT>/data/prepared/queries.jsonl` via `path()` (rule 3). Sandbox (`BLOCKED_MODULES`, guarded import/open/resolver), record building, prompts: unchanged. |
| `tools.py`, `prompts.py`, `fence_tolerant.py`, `rep_guard.py`, `hybrid.py` | Docstring paragraph only (rule 4). |
| `obliq_adapter.py` | Docstring rewritten neutrally (it described the two harnesses by person); code unchanged. |
| `retrieval_server.py` | Docstring; `ROOT` = `path("FULLDOC_ROOT")` with `shared/` on `sys.path` (rules 2, 3). |
| `retrieval_server_colbert.py` | Docstring; `ROOT` was `dirname(dirname(__file__))` (the working directory), now `path("FULLDOC_ROOT")` (rule 3); `import sys` for the wiring. |
| `encoder.py` | Docstring; `DEFAULT_MODEL` = `<FULLDOC_ROOT>/data/models/Qwen3-Embedding-8B` (rule 3). |
| `build_bm25.py`, `prepare_data.py` | Docstring; `ROOT` = `path("FULLDOC_ROOT")` (rule 3). |
| `run_experiment.py` | Docstring; `ROOT` (data) = `FULLDOC_ROOT`; run directories `<root>/runs/<name>` -> `$RUNS_DIR/<name>`; `<root>/src/*.py` -> this directory; worker interpreter `<root>/.venv/bin/python` -> `$PYTHON`, with a clear error when unset (rule 3). Comment at the `--expect-served-model` gate (the scripts it calls are missing, below). |
| `judge.py` | Docstring; data paths via `path("FULLDOC_ROOT")`; default `--cache` `$RUNS_DIR/_judge_cache.json` (was `<root>/runs/...`); `load_dotenv` guarded as in `run_one.py` (rule 3). |
| `grade_local.py`, `score_obliq.py` | Docstrings (neutral wording); `sys.path.insert("/home/.../RLM")` + `examples.*` -> `rlm_shared.browsecomp_data` / `rlm_shared.obliq_data` (rule 2; the functions used are byte-identical to `examples/`). |
| `provenance.py` | Docstring; the original's one `ROOT` held both code and data. Code is still hashed under `ROOT` (= this algorithm folder, parent of `src/`); data, index, `.env` and `rlm_upstream` are read under `DATA_ROOT = FULLDOC_ROOT`, via a `root` argument to `tree()` (rule 3). `code_digest` values therefore differ from the original runs (different files). |
| `run_bm25.py` | Docstring (neutral wording, new usage line); `SUP_SRC` = this directory (rule 2). All replacements and assertions unchanged. |
| `split_runner.py` | Docstring documents both modes (GEPA driver, `--seed-split` runner) and the cache key. `R`/`SUP` constants removed: `shared/` and this directory on `sys.path`, `examples.browsecomp_data` -> `rlm_shared.browsecomp_data` (rule 2); `GEPA_OUT` default and its relative base `R` -> `RUNS_DIR`; smoke/dry-run dirs under `RUNS_DIR`; `GEPA_SPLIT` default only from `RLM_DATA_DIR`, else a clear error at use (`_split_file()`); rollout interpreter `PYV` from `PYTHON` (checked at use); `CLAUDE_BIN` default `claude` on `PATH`; rollout `PYTHONPATH` = `RLMLIB_PATH` (was the same checkout, still no hooks), `HF_HOME` from the environment, `cwd` = `FULLDOC_ROOT`; `preload_seed`/dry-run read `<FULLDOC_ROOT>/runs/...` (rule 3). `BASE_ARGS`, `FINGERPRINT`, `ckey`, scoring, critic: unchanged; tests reproduce the three original cache keys. |
| `bcp_qwen_dense.py`, `bcp_bm25_split.py` | Docstrings; load `split_runner.py` by file (was `gepa_bcp.py`); `OUT` defaults under `RUNS_DIR`; queries from `FULLDOC_ROOT`; split file from `RLM_DATA_DIR`; rollout env as in `split_runner.py`; comparison script from `BCP_POLICY_COMPARE` (rules 2, 3). Fingerprint suffixes unchanged. |
| launchers (`experiments/`) | Source `configs/paths.env` through `_common.sh`; `/home/...` locations, the vllm interpreter, `HF_HOME`, the personal Azure deployment names and the session-scratch default of `AZ_ENV` replaced by configured values; `PYTHONPATH` keeps the original structure (`<hook dir>:<rlm>`, or `<rlm>` alone). Every CLI flag and every exported/unset behaviour variable is unchanged (`tests/test_static.py` compares the flag sets). `launch_high400.sh`/`launch_high100c.sh`: the hard-coded predecessor PIDs became optional `WAIT_PID`. `final830_*.sh`/`launch_high*.sh`: `runs/` -> `$RUNS_DIR`, `src/` -> `$SRC`; `_azure.sh` provides the credentials the original read from `<root>/.env` (`AZURE2_*`). `obliq_fd.sh`: the `snip` arm prints that it belongs to the Snippet RLM; the `fd` arm is the original command. `frames_qwen.sh`: the final grading call takes its script from `FRAMES_GRADE_SUP`. |

No hunk falls outside rules 1 to 5.

## Issues found, not fixed

Paths are `src/` of this algorithm unless prefixed; `rlm/...` is `RLMLIB_PATH` (rlm @ 854e688).

1. **k is chosen by the model, up to 50, not 5.** `tools.py:33` (`MAX_K = 50`), `tools.py:86` (design B sets
   `allow_k_override=True` unless `--fixed-k`), `tools.py:148-154` (k clamped to 1..50); `run_one.py:344`
   (`--default-k 5` is only the fallback when the model passes no k). The description tells the model "k is yours to
   choose per call, from 1 to 50" and that "k=30 returns the same first five plus ranks 6-30" (`tools.py:331-333`,
   `357-359`). All BrowseComp+ and FRAMES rows ran design B without `--fixed-k` (records: `k_settable_by_model: true`);
   the run summaries of BCP-21/23 report median k 30 (range 5-50). Experiments.xlsx describes several of these rows
   as "top-5" / "top-k 30". Only the OBLIQ rows used a fixed k (20).
2. **`--no-delegation` is ignored by the stock library.** `run_one.py:253` and `:296` pass
   `environment_kwargs={"no_delegation": ...}`; `rlm/environments/local_repl.py:165` accepts it as `**kwargs` and
   forwards it to `BaseEnv`, which only stores it (`rlm/environments/base_env.py:216-222`); `LocalREPL.setup()` installs
   `llm_query`, `llm_query_batched`, `rlm_query`, `rlm_query_batched` unconditionally
   (`rlm/environments/local_repl.py:224-227`). The sub-LLM functions therefore work in every row that ran on
   `RLMLIB_PATH` (BCP-25/27/29/31, FR-11/13, MAT/TWI/CON-11); `NO_DELEGATION_CLAUSE` (`prompts.py:271`) says they raise.
   *Correction (review, 2026-10-30):* an earlier draft said model code calls them in 37 CON-11 and 17 TWI-11 runs.
   A scan of every `response` and `code_blocks` field of all MAT/TWI/CON-11 trajectories for
   `(llm_query|llm_query_batched|rlm_query|rlm_query_batched)\s*\(` finds **no** call (0/151, 0/281, 0/254); the
   names occur only in the prompts. The functions are available, but these runs did not use them. BCP-21/23 ran on a different, patched rlm build whose summaries say these functions raised (issue 12).
3. **The reasoning-effort setting is dropped on Azure without the `azure_sampling` hook.** `run_one.py:270-271`,
   `:317-318` pass `sampling_args={"reasoning_effort": ...}`; `rlm/clients/azure_openai.py:86-89` calls
   `chat.completions.create(model=..., messages=...)` only. Only the `run5d` hook (which includes `azure_sampling`)
   sends it: FR-11 and the OBLIQ rows. BCP-25 (`retry429` hook only) recorded `reasoning_effort: "high"` but sent none;
   gpt-5 ran at the deployment default. `experiments/bcp_gpt5_colbert.sh` has the same property. The OpenAI-compatible
   client (`--backend openai`, Qwen rows) does forward it.
4. **`n_iterations` is 31 when the turn cap is hit.** The library logs the extra forced-answer call as an iteration
   (`rlm/core/rlm.py:361` loop of 30, then `:472` `_default_answer`, which logs at `:687`); `run_one.py:482` records
   `len(traj["iterations"])`. 154 of 830 BCP-25 records and 48 of 824 FR-11 records have 31. Averages of
   `n_iterations` include that call.
5. **`served_models` is always `None` with the stock library.** `run_one.py:510-517` imports `SERVED_MODELS` from
   `rlm.clients.{openai,azure_openai}`, which does not exist at 854e688; the `ImportError` is swallowed. All records of
   BCP-25, FR-11 and CON-11 checked have `null`. The served build is therefore not recorded for any `RLMLIB_PATH` row.
6. **A partial answer is discarded on timeout.** On `TimeoutExceededError` (raised with `partial_answer`,
   `rlm/core/rlm.py:505`) `run_one.py:485-489` writes status `error` and an empty output; `e.partial_answer` is never
   read. The judge scores the question wrong without looking at it.
7. **`prompt_id` does not identify the prompt.** `prompts.py:69` hashes `QUERY_TEMPLATE + CONTEXT_NOTE + USER_PROLOGUE`
   and the design markers only: not `SOLO_SYSTEM_PROMPT`, `LEAN_CLAUSE`, `NO_DELEGATION_CLAUSE`, the other clauses, the
   assembled prologue, or the tool descriptions. A GEPA override of `SOLO_SYSTEM_PROMPT` or `LEAN_CLAUSE` leaves it
   unchanged. `globals().get('DESIGN_F', '')` refers to a constant that does not exist.
8. **The tool text describes tools and a retriever that are not there.** Every search description advertises
   `read_docs(docs, question)` (`tools.py:367`), which is never installed (`tools.py:418-455`). The design-B text tells
   the model to "slice or delegate them to llm_query" (`:326`), the fixed-k text (OBLIQ) to delegate reading to
   `llm_query` / `llm_query_batched` (`:288`), and the sandbox note lists `llm_query, llm_query_batched` as tools
   (`:354`), while `--solo-prompt` says "There are no sub-agents, no helper models" (`prompts.py:115`). The design-B text
   states "The retriever is a dense semantic index (one embedding per document)" and that "the index embedded only the
   first part of each document" (`:298`, `:336`): false for the ColBERT rows (BCP-23, BCP-29; the Design M patch that
   would have changed it was never applied, `src/PATCH_HISTORY.md`); for BM25 rows `run_bm25.py` rewrites it.
9. **`obliq_adapter.py` snippet mode returns the full text.** `obliq_adapter.py:71-72`: any mode other than `ids` and
   `full` returns the whole document under the key `snippet`; `--snippet-max-tokens` is parsed (`:36`) but unused. Its
   arguments are parsed at import time (`:37`).
10. **The search cap is client-side.** `tools.py:136-144` refuses in the same process as model-written REPL code; the
    request carries no cap and no server enforces one. `judge.py:336` only flags runs whose server count exceeds the
    cap. The cap text says over-cap calls "return an error instead of results" (`tools.py:376`), but they return `[]`
    with a printed notice. A search that fails at the server still consumes one unit of the cap (`tools.py:144` runs
    before the request).
11. **The `retriever` request field is meaningless.** `tools.py:165` sends `"dense"` (design B) to every server,
    including BM25 and ColBERT; no preserved server reads it (`apply_hybrid_patch.py` is not applied).
12. **BCP-21 and BCP-23 cannot be reproduced exactly from what was preserved.** They ran on the original working
    directory's patched rlm build (`rlm_upstream`, which `provenance.py` hashes; not preserved). Per their run
    summaries that build recorded the served model and made the delegation functions raise. On `RLMLIB_PATH` @ 854e688 the same flags give issues 2, 3 and 5. Their run
    records are not on this host; the per-question summaries are. The BCP-21 launchers were not preserved
    (`experiments/bcp_gpt5_final830/final830_*.sh` is a different run, Design G at high effort, which per
    `launch_high400.sh` was served gpt-5.4). The BCP-23 summaries say "search CALL cap: not recorded" and report up to
    19 searches in `colbert8192_high400`, although its launcher passes `--max-searches 10`: whether the cap bound is
    UNKNOWN.
13. **Missing helper scripts.** `run_experiment.py:126-139` calls `assert_openai_compat_model.py` /
    `assert_served_model.py` for `--expect-served-model`; neither was preserved, so every BCP-23 launcher aborts at that
    gate. `final830_*.sh` call `report.py` and `budget.py` (not preserved). The BM25 search servers (`:8041`, `:8043`),
    the Gemini-Embedding-2 servers, the Math proxy, `frames_grade_sup.py` and `bcp_policy_compare.py` are outside this
    algorithm.
14. **The judge's one-configuration check ignores most of the configuration.** `judge.py:193-195` compares only root/sub
    model, effort, `default_k`, `k_settable_by_model`, `prompt_id`, budget and orchestrator: not the search cap,
    `fixed_k`, server, prompt switches or timeout (the patches that extended it were not applied, see
    `src/PATCH_HISTORY.md`).
15. **`run_bm25.py:115-125` patches `json.dump` process-wide** with `default=repr`, so a record can contain `repr()`
    strings where `run_one.py` alone would crash; record files of the BM25 rows are not strictly the same format.
16. **`split_runner.py:302`** writes each candidate's prompt file to the system temp directory and never deletes it.

## How verified

- `diff -u` of every file in `original/` against its `src/` / `experiments/` counterpart, read hunk by hunk.
- `python -m py_compile` on every `src/*.py`; `bash -n` on every launcher and config file.
- `pytest algorithms/full_doc_rlm/tests` (mocks only): **89 passed** with `rlm` importable; **82 passed, 7 skipped**
  without it. The tests cover: search cap 10 and refusal counter against a fake retrieval server (`urllib` patched);
  `search_many` against the cap; model-chosen k clamped to 1..50 in design B, ignored with `--fixed-k` and design A;
  DOCS merging (full text kept over a snippet-only hit), docid dedup, SEARCH_LOG, unread tracking; every prompt string,
  every `build_prologue` combination and every tool description equal to the original; the BM25 rewrite of
  `run_bm25.py` equal to the original wrapper's output for FRAMES and BrowseComp+; `run_one.main()` with a fake agent
  that drives the real `RLMLogger` and the library's own `_default_answer`: `n_iterations` = 31 at the cap, `served_models`
  `None`, partial answer dropped on timeout; `no_delegation` ignored by `LocalREPL`; `prompt_id` blind to the solo
  prompt; the adapter's snippet mode; every config parses and names its row; launcher flag sets equal the originals.
- `prompt_id` of `src/prompts.py` reproduces the values in the run records: `94f31e91f6b7` (dense/ColBERT), `758e43ed5b95`
  (BM25 wrapper), and with the OBLIQ prompt files `f92043bfd120` / `299e86ce3e1e` / `891e1319f80c` (MAT/TWI/CON-11).
- `split_runner.py` with each wrapper's environment reproduces the original fingerprints and cache keys:
  9e6ce99e / e925f0f6c7 (BCP-27), 29587b4e / ee5ea32a1c (BCP-29), aa1d1029 / 11d0c5bc6b (BCP-31).
- Configs: the `config` header of one record per run directory was read (first 200 KB only); OBLIQ launch argv from
  each run's `_provenance.json`; n_iterations / served_models / search counts from record heads and tails.
- `scripts/provenance/preserve_originals.py --check`: all copies match their manifest. Nothing outside this folder
  was modified; no experiment, server, network or model call was run.
