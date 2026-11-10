# Snippet REPL agent: changes from the original

`original/RLM/` holds the files that produced the run5d rows (BCP-13/15/17/19, FR-7/9, MAT-9/TWI-9/CON-9), copied from
the working tree of the original repository (`MANIFEST.json` has the SHA-256 of each). `original/RLM_git_7685538/` holds
the committed September harness; it is reference only. `src/` is the refactored copy. The research logic in `src/` is
identical to `original/RLM/`; every hunk of `diff -u original/<file> src/<file>` falls under rules 1-5 of
`docs/architecture/REFACTOR_RULES.md`. **Hunks outside the allowed rules: none.**

## Renames

| Original (`original/RLM/...`) | New (`agents/snippet_repl/...`) |
|---|---|
| `examples/browsecomp_eval_official_rlm.py` | `src/harness.py` |
| `scripts/run_bcp_snippet.py` | `src/run_browsecomp.py` |
| `examples/frames_eval_rlm.py` | `src/run_frames.py` |
| `scripts/run_obliq_mine.py` | `src/run_obliq.py` |
| `scripts/official_prompt_modules.py` | `src/prompt_modules.py` |
| `scripts/hooks/{retry429,azure_sampling,run5d,cap_ranking,retry429_cap_ranking}/sitecustomize.py` | `src/hooks/<same>/sitecustomize.py` |
| `examples/browsecomp_data.py`, `examples/obliq_data.py`, `examples/obliq_score_advice.py` | not copied: `agents/shared/rlm_shared/{browsecomp_data,obliq_data,obliq_score_advice}.py` |
| `rag` package (DSPy RLM; only `rag.agent._get_snippet_tokenizer` is used) | `agents/dspy_rlm/src/dspy_rlm` (`dspy_rlm.agent`) |
| `scripts/run5d/bcp_snip.sh` | `experiments/bcp_snip.sh` |
| `scripts/run5d/bcp_snip2.sh` | `experiments/bcp_snip2.sh` |
| `scripts/run5d/frames_snip.sh` | `experiments/frames_snip.sh` |
| `scripts/run_obliq_shard.sh` | `experiments/obliq_shard.sh` |
| `scripts/run5d/obliq_gemini.sh` (the `snip` arm only) | `experiments/obliq_snip.sh` |
| `scripts/run5d/qwen_bcp_snip_queue2.sh` | `experiments/qwen_bcp_snip_queue.sh` |
| `scripts/run5d/question_watchdog.sh`, `scripts/run5d/budget_guard.sh` | `experiments/question_watchdog.sh`, `experiments/budget_guard.sh` |
| `scripts/run_qwen_full.sh` (BCP-9, rlmfix32) | `experiments/september/bcp_snip_qwen_embed.sh` |
| `scripts/run_no_orchestrator.sh` (BCP-11, rlmfix24) | `experiments/september/bcp_snip_no_orchestrator.sh` |
| `scripts/run5d/night_finisher.sh`, `scripts/run5d/bcp_snip_gpt5_bm25_finish.sh` | `experiments/ops/` (same names) |
| `<original repo>/data/obliq_{math,congress}/{task,strategy}_snippet.txt`, `data/obliq_math/all151.json`, `data/obliq_fd/all281.json`, `data/obliq_congress/all254.json` | `configs/obliq/<task>/` (byte-identical copies; SHA-256 checked) |

New files with no original: `experiments/_common.sh` (layout wiring sourced by every launcher),
`experiments/run_config.sh` (runs one `configs/<ROW>.env`), `configs/*.env` (one per workbook row), `tests/`.

## Diff hunks per file

### `src/harness.py`
1. **Docs (rule 4)**: a refactor paragraph added at the top of the existing module docstring.
2. **Import wiring (rule 2)**: `sys.path.insert(0, <original repo root>)` (computed from `__file__`) becomes two
   `__file__`-relative entries: `<repo>/agents/shared` (inserted at 0) and `<repo>/agents/dspy_rlm/src` (appended).
3. **Import wiring (rule 2)**: `from examples.browsecomp_data import ...` -> `from rlm_shared.browsecomp_data import ...`
   (same five names).
4. **Import wiring (rules 1-2)**: inside `_truncate`, `from rag.agent import _get_snippet_tokenizer` ->
   `from dspy_rlm.agent import _get_snippet_tokenizer`, plus a comment.

### `src/run_browsecomp.py`
1. Docs: refactor paragraph in the docstring.
2. Import wiring: `sys.path.insert(0, "/home/extstu_00/RLM")` -> `__file__`-relative `src/` and `agents/shared/`.
3. Import wiring: `from examples import browsecomp_eval_official_rlm as H` -> `import harness as H` (same module object
   the other code patches; attribute names unchanged).
4. Import wiring: the `spec_from_file_location("_opm", ...)` path names `prompt_modules.py` (same directory, as before).

### `src/run_frames.py`
1. Docs. 2. Import wiring: repo-root insert -> `src/` + `agents/shared/`; `import examples.browsecomp_eval_official_rlm as H`
-> `import harness as H`; `from examples.browsecomp_data import ...` -> `rlm_shared.browsecomp_data`.
3. **Path parametrisation (rule 3)**: the constant `ROWS = "/home/extstu_00/RLM/data/frames/frames_rows.json"` is replaced
   by `_rows_path()` = `$FRAMES_DATA_DIR/frames_rows.json`, read in `main()` where `ROWS` was read.
4. Import wiring: the prompt-modules path was `<repo>/scripts/official_prompt_modules.py`, now `src/prompt_modules.py`.

### `src/run_obliq.py`
1. Docs. 2. Import wiring: repo-root insert -> `src/` + `agents/shared/`; `examples.browsecomp_data` / `examples.obliq_data`
/ `examples.obliq_score_advice` -> `rlm_shared.*`; `from examples import browsecomp_eval_official_rlm as H` ->
`import harness as H`; prompt-modules path renamed. The module-name-dependent patching still works: `BD` is
`rlm_shared.browsecomp_data`, the module the harness imports its accessors from, and it is rebound before the harness
is imported, exactly as before.

### `src/prompt_modules.py`, `src/hooks/*/sitecustomize.py`
Docs only: a refactor paragraph prepended to the module docstring. Code is AST-identical (tested). The hooks still
locate each other through `os.path.dirname(__file__)`, so the directory-per-hook layout is kept.

### Launchers (`experiments/`)
All hunk categories, per launcher, are: (a) docs header; (b) `. _common.sh`, which sources `configs/paths.env`;
(c) `/home/extstu_00/miniconda3/envs/vllm/bin/python` and `./.venv/bin/python` -> `$PYTHON`;
(d) `AZ_ENV=${AZ_ENV:-/tmp/claude-.../azure.env}` -> `${AZ_ENV:?}` from paths.env (no default, rule 3);
(e) `PYTHONPATH=<repo>/scripts/hooks/run5d:/home/extstu_00/rlmrepo` -> `src/hooks/run5d:$RLMLIB_PATH:shared:dspy_rlm/src`
(hook first, as before); (f) `HF_HOME=/home/extstu_00/.cache/huggingface` -> `${HF_HOME:?}`;
(g) Azure deployment names (personal deployment names of the original account) -> `$AZURE_DEPLOYMENT_GPT5` / `$AZURE_DEPLOYMENT_GPT52`;
(h) script paths -> `$SRC/<new name>`, with `SRC` added to the `export` line that feeds the `xargs sh -c` body;
(i) `cd /home/extstu_00/RLM` (or `/home/extstu_00/master`) -> `cd $RUNS_DIR` where the original changed directory, so
relative paths (e.g. obliq_shard's default `RLM_TRAJ_DIR=logs/rlm_traj_obliq_sweep`, the September `logs/rlmfix*`)
land under `$RUNS_DIR`; (j) fail-fast `${VAR:?}` checks for the variables a launcher needs. Arguments, defaults, caps,
timeouts and every behaviour-selecting variable are unchanged. Per-launcher specifics:
- `obliq_snip.sh`: only the `snip` arm of `obliq_gemini.sh`; the `fd` arm (and its `S`, `FD_URL`, `QF` variables) is
  the document REPL agent's and lives in `agents/document_repl`. Prompt files come from `configs/obliq/<task>/`.
  `_common.sh` exports `configs/paths.env` (incl. `OBLIQ_BENCH_ROOT`, which the shared `obliq_data` now reads).
- `obliq_shard.sh`: `/tmp/obliq_shards` kept as is (portable scratch path).
- `qwen_bcp_snip_queue.sh`: output dirs `logs/run5d/full/...` -> `$RUNS_DIR/run5d/full/...` (`mkdir -p` added).
- `question_watchdog.sh`: process pattern `run_bcp_snippet.py` -> `run_browsecomp.py` (consequence of the rename).
- `september/*.sh`: the original sourced `/home/extstu_00/master/.env` (defining `AZURE_OPENAI_*` directly, silently if
  missing); now `$AZ_ENV` plus `AZURE_OPENAI_API_KEY/ENDPOINT/MODEL` derived from it. Whatever else that `.env` held is
  UNKNOWN (the directory no longer exists). No startup hook, as in the original. `/tmp/noorch.txt` -> `$NOORCH_QIDS_FILE`.
  Side effect of `_common.sh`: `HF_HOME` (and the other paths.env variables) are exported, which the September
  launchers did not do.
- `ops/night_finisher.sh`: process patterns follow the renames (`^bash .*experiments/qwen_bcp_snip_queue.sh`,
  `^bash .*experiments/frames_snip.sh qwen`, `run_browsecomp.py`); the document REPL pattern is `$FD_DENSE_PATTERN`; grading
  scripts (not part of this agent) are looked up in `$GRADERS_DIR`; log paths under `$RUNS_DIR/run5d`.
- `ops/bcp_snip_gpt5_bm25_finish.sh`: same treatment (`$GRADERS_DIR`, `$RUNS_DIR`).

Two variables used by the launchers are not in `configs/paths.example.env` (outside this folder):
`NOORCH_QIDS_FILE` (September rlmfix24 only) and `GRADERS_DIR` / `FD_DENSE_PATTERN` (ops scripts only).

## Issues found, not fixed

Line numbers: `O:` = `original/RLM/...`, `S:` = `src/...` / `experiments/...`.

1. **`iterations` and `input_tokens` are wrong whenever llm_query ran.** `rlm` `core/lm_handler.py:237-251` merges the
   per-client usage dicts with `dict.update()` keyed by model name; the root client and the depth-1 sub-LM client
   (created by `sub_sampling_args`) share the model name, so the sub-LM's entry overwrites the root's. The harness then
   reads `total_input_tokens`/`total_calls` from it (O: `examples/browsecomp_eval_official_rlm.py:1127-1133`,
   S: `harness.py:1145-1151`) and derives `"iterations": lm_calls - llm_query_calls` (O:1168, S:1186), which is
   meaningless for the same reason. `output_tokens`, `cost_usd`, `lm_calls` are affected alike. Only trajectory counts
   and `RLM_USAGE_LOG` (azure_sampling hook) are reliable; the workbook already uses those.
2. **429s, timeouts and kills score as wrong, silently.** Any exception in `rlm.completion` (exhausted 429 retries,
   `APITimeoutError` after the 300 s library default or `LOCAL_LM_TIMEOUT`) sets `predicted="ERROR"`, `correct=False`
   (O:1134-1150, S:1152-1168); OBLIQ's `_score` is then never called, so the record has no NDCG (0 in the means). The launchers' `timeout 7200`/`3600` kill a question
   with no record at all, and `question_watchdog.sh` SIGTERMs over-budget questions (BCP-13: q212); graders count
   missing questions as wrong. Measured: BCP-13 828/830 records, 7 with `error`; BCP-19 587/830 (243 killed at 1 h);
   FR-9 751/824; OBLIQ errors 11 (math), 26 (twitter), 17 (congress).
3. **The BrowseComp+ format gate stays on in the OBLIQ rows.** `obliq_gemini.sh:34-41` (S: `obliq_snip.sh`) does not set
   `OBLIQ_NO_FORMAT_GATE=1` (O: `scripts/run_obliq_mine.py:274-276`, S: `run_obliq.py:285-287`), so every OBLIQ
   doc-id-only submission is refused up to 3 times by the "Explanation:/Exact Answer:" gate (O: harness 265-297).
   Measured in the original `all.json`: at least one rejection on 140/151 (math), 279/281 (twitter), 237/254
   (congress) questions. Also, the comment at O: `run_obliq_mine.py:272` ("a module global, so it stops enforcing after
   3 rejections per shard process") is wrong: `run_one` resets the counter per question (O: harness 999, S: 1017).
4. **`reasoning_effort` reaches Azure only with the `azure_sampling` or `run5d` hook loaded.** `rlm`
   `clients/azure_openai.py:74-90` builds requests from model + messages only (see
   `src/hooks/azure_sampling/sitecustomize.py`). The run5d launchers load `hooks/run5d`; `obliq_shard.sh` loads a
   hook only through `EXTRA_PYTHONPATH` (set by `obliq_snip.sh`); the September launchers (`run_qwen_full.sh`,
   `run_no_orchestrator.sh`, so BCP-9 and BCP-11) load none, so those GPT-5 runs used the deployment default (measured
   medium) although they request high/medium. Local (vLLM) runs go through `OpenAIClient`, which forwards them.
5. **Duplicated fence-tolerance / repl-cue / search-cap / LM-timeout code.** `_tolerant_find_code_blocks`: O:
   `run_bcp_snippet.py:45-71`, `frames_eval_rlm.py:28-55`, `run_obliq_mine.py:153-186` (the OBLIQ copy also counts
   rescues); the repl-fence cue: O: `run_bcp_snippet.py:87-101`, `frames_eval_rlm.py:114-128`,
   `run_obliq_mine.py:233-239`; the search cap + prompt sentence: O: `run_bcp_snippet.py:122-151`,
   `frames_eval_rlm.py:83-112`; the per-call timeout is injected two different ways (backend_kwargs in
   `run_bcp_snippet.py:24-43` and `run_obliq_mine.py:82-101`, `BaseLM.__init__` in `frames_eval_rlm.py:134-144`), and in
   `run_bcp_snippet.py:30` only when `LOCAL_LLM_BASE_URL` is set (GPT-5 calls keep the library's 300 s).
6. **`OBLIQ_INDEX` is read by nothing.** Set by `obliq_gemini.sh:38` and defaulted in `run_obliq_shard.sh:23`; no Python
   file of this agent reads it (the retriever is chosen by `EMBEDDING_SERVER_URL`).
7. **GPT-5 and Qwen3.6 rows differ in caps, turns and timeouts.** BrowseComp+: GPT-5 100 turns, no search cap, 7200 s per
   question, no output cap, effort high; Qwen3.6 50 turns, 25 searches, 3600 s, 8192 output tokens per call, effort
   medium, plus presence penalty 1.5, fence tolerance, repl cue and prose-final submission (`bcp_snip.sh:24-35`,
   `bcp_snip2.sh:30-44`). FRAMES: both 50 turns, cap 25, 7200 s; Qwen adds fence tolerance, cue and 1800 s per call.
   The September GPT-5 rows (BCP-9/11) differ again from BCP-13: no compaction, no per-question timeout, no hook.
8. **`bcp_snip.sh` header says GPT runs are "without" compaction, but `COMPACT=--compaction` is passed for both models**
   (O: `scripts/run5d/bcp_snip.sh:9` vs `:21,43`), so BCP-13 ran with compaction and BCP-9/11 without.
9. **Implicit `.env`.** The harness's first `search_index` call imports `rag.agent` (now `dspy_rlm.agent`), whose
   module-level `load_dotenv()` fills variables missing from the environment from the `.env` found upward from that
   file (original: `<original repo>/.env`). For the run5d BrowseComp+ rows this supplied `AZURE_OPENAI_JUDGE_MODEL`
   (gpt-5.4-mini) for the in-run `correct` field. In this repository no `.env` sits above `dspy_rlm/agent.py`, so the
   configs set `AZURE_OPENAI_JUDGE_MODEL` explicitly. (Only the in-run field; reported numbers are gpt-4.1 re-grades.)
10. **OBLIQ shard merge double-counts.** `run_obliq_shard.sh:38` globs `${OUT}_s*.json`, which also matches
    `${OUT}_s<k>_checkpoint.json`; the merged `all.json` of MAT/TWI/CON-9 holds every question twice (302/562/508 rows
    for 151/281/254 ids). The printed means are unaffected because each question appears exactly twice.
11. **The September rows were not produced by any preserved harness version.** The `rlmfix24..32` records lack the
    `child_*` fields that commit 7685538's harness (`original/RLM_git_7685538/`) already writes. With the switches they
    ran with, 7685538 and `src/harness.py` build identical prompts and tools (the only differences are the opt-in
    `OBLIQ_NO_GETDOC` and `RLM_DEBUG_TB`), but the exact file is not recoverable. Their launchers for rlmfix25/29/30/31
    and 83 of rlmfix24's questions are not in the repository, and `/tmp/noorch.txt` is lost (configs say UNKNOWN).
12. **Empty prediction scores correct in-run.** `agents/shared/rlm_shared/browsecomp_data.py:112-114` `heuristic_match` returns
    True for an empty response (`"" in gold`); the harness and `run_frames.judge_match` call it first. In-run field only.
13. **Watchdog / guard pricing.** `question_watchdog.sh:27` prices every model at GPT-5 rates; `budget_guard.sh:13`
    prices by the substring "5.2" in the model name. The BCP-13 budget passed to the guard is not recorded.
14. **`examples/obliq_score_advice.py` is missing from `original/RLM/`** (MANIFEST.json omits it although
    `run_obliq_mine.py` imports it); `agents/shared/rlm_shared/obliq_score_advice.py` is byte-identical to the original
    repository's file (checked with `diff`).
15. **Prose-final keeps a closing fence.** With `QWEN_PROSE_FINAL=1`, an answer template written inside a bare ```` ``` ````
    fence is submitted with the closing ```` ``` ```` inside `answer["content"]` (O: `run_bcp_snippet.py:103-119`;
    shown in `tests/test_snippet_repl_behaviour.py`). Harmless for the judge.
16. **Cross-agent dependency.** `_truncate` needs `dspy_rlm.agent` (DSPy RLM package), which imports `dspy` and runs
    `load_dotenv()`, only to load the Qwen3-0.6B tokenizer.

## How verified

Interpreter: `/home/extstu_00/miniconda3/envs/vllm/bin/python` (compile) and `.venv/bin/python` (pytest; inherits the
conda env), `RLMLIB_PATH=/home/extstu_00/rlmrepo` (854e688, clean). No experiment, server, network or LLM call was run.

- `python -m py_compile` on every `src/**/*.py`: all compile.
- `bash -n` on every `experiments/**/*.sh` and `configs/*.env`: all parse.
- `diff -u original/... src/...` for every renamed file and launcher: every hunk is listed above (rules 1-5).
- `pytest agents/snippet_repl/tests`: **78 passed, 0 failed, 0 skipped** (18 skipped with a reason when `rlm` is not
  importable). The tests:
  - *equivalence* (`test_snippet_repl_equivalence.py`): 9 scenarios (harness defaults and every prompt switch; BCP-13;
    BCP-15/17/19; FR-7; FR-9; TWI-9; MAT-9 prompt files plus the remaining OBLIQ switches) import the original and the
    refactored entry point in separate interpreters with a fake retriever and a fake agent, and compare the system
    prompt, addendum, task text, tool specs, the full first message list, the turn-0 message, the search-cap refusal,
    code-block parsing of 10 synthetic turns, the format gate, the `run_one` record, the `RLM(...)` kwargs and the
    retriever requests: all identical. A mutation check (one word changed in a tool description) made 8 tests fail.
  - *static*: string constants of every src file equal the original's except the listed wiring strings; hooks and
    prompt_modules AST-identical apart from the docstring; no `examples.*`/`rag` imports or machine paths left;
    `bash -n` per launcher; every config parses, names an existing executable launcher, and uses UNKNOWN only with a
    comment; MAT-9 prompt-file lengths equal the lengths the run logged (1571 / 932 chars).
  - *behaviour*: recall on a synthetic record (gold 2/4, evidence 1/2), FRAMES/OBLIQ recall and NDCG, the format gate
    (3 refusals, then accepts), fence tolerance and prose-final extraction, `_final_answer_span`, and the
    `dspy_rlm.agent` tokenizer wiring of `_truncate` (mocked tokenizer).
- Original logs read (head lines only) to fill the configs: `logs/run5d/full/*.out`, `*.watchdog.log`, `*.guard.log`,
  `night_finisher.log`, `obliq_snip_*/all_s0.log`, and the record keys of `logs/rlmfix{24,31,32}_q*.json`.
