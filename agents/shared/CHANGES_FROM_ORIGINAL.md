# Shared code: changes from the original

This covers the cross-agent code: the graders (`agents/shared/rlm_shared/grading/`), the retrieval servers and index
builders (`benchmarks/retrieval/`) and the reporting scripts (`scripts/reporting/`). The originals are kept byte for
byte in `agents/shared/original/RLM/`, listed with their SHA-256 in `agents/shared/original/MANIFEST.json`. The rules are in
`docs/architecture/REFACTOR_RULES.md`.

Every change below is one of four kinds:
- a rename;
- import wiring;
- a path, credential file or deployment name moved into an environment variable;
- documentation.

No prompt, default, retry, threshold, score or output format changed, and no bug was fixed.

## Renames

| original (`agents/shared/original/RLM/…`) | new |
|---|---|
| `scripts/bcp_judge.py` | `agents/shared/rlm_shared/grading/bcp_judge.py` |
| `scripts/run5d/bcp_snip_grade.py` | `agents/shared/rlm_shared/grading/grade_bcp_snippet.py` |
| `scripts/run5d/bcp_fd_grade.py` | `agents/shared/rlm_shared/grading/grade_bcp_fulldoc.py` |
| `scripts/bcp_rejudge_gpt41.py` | `agents/shared/rlm_shared/grading/rejudge_bcp.py` |
| `scripts/frames_judge_gpt41.py` | `agents/shared/rlm_shared/grading/grade_frames.py` |
| `scripts/run5d/obliq_report.py` | `agents/shared/rlm_shared/grading/obliq_report.py` |
| `scripts/fusion_proxy.py` | `benchmarks/retrieval/fusion_proxy.py` |
| `scripts/frames_search_server.py` | `benchmarks/retrieval/frames_search_server.py` |
| `scripts/multi_search_server.py` | `benchmarks/retrieval/multi_search_server.py` |
| `scripts/search_server.py` | `benchmarks/retrieval/search_server.py` |
| `scripts/obliq_math_search.py` | `benchmarks/retrieval/obliq_math_search.py` |
| `scripts/gemini/{search_server,vertex,build_index}.py` | `benchmarks/retrieval/gemini/{search_server,vertex,build_index}.py` |
| `scripts/frames_build_bm25.py` | `benchmarks/retrieval/frames_build_bm25.py` |
| `scripts/bcp_build_bm25_default.py` | `benchmarks/retrieval/bcp_build_bm25_default.py` |
| `scripts/build_bm25_index.py` | `benchmarks/retrieval/build_bm25_index.py` |
| `scripts/build_obliq_index.py` | `benchmarks/retrieval/build_obliq_index.py` |
| `scripts/build_bc_rmcb_dl8192.sh` | `benchmarks/retrieval/build_bc_rmcb_dl8192.sh` |
| `scripts/reindex_rmcb_8192.py` | `benchmarks/retrieval/reindex_rmcb_8192.py` |
| `scripts/failure_analysis_{extract,turns,turns_merge,build,findings,extras,turns_sheet,eval,inject}.py` | `scripts/reporting/` (same names) |
| `scripts/run5d/iteration_counts.py`, `scripts/run5d/token_columns.py` | `scripts/reporting/` (same names) |

The server and builder files keep their original names, because the run launchers, logs and documents of the original
runs refer to them by those names.

## New configuration variables

These were added to `agents/shared/rlm_shared/paths.py` (`VARIABLES`) and `configs/paths.example.env`:

| variable | replaces |
|---|---|
| `LOGS_ROOT` | root of the original run logs (logs/ of the original repository) |
| `ORIGINAL_REPO_ROOT` | the original repository's root, used for `docs/supervisor_artifact.html`, `docs/gpt5_ivfsq8_experiment.txt` and `scripts/workbook/tokmaps/` |
| `RETRIEVAL_EVAL_ROOT` | `cd <home>/master` in `build_bc_rmcb_dl8192.sh` (the checkout with `scripts/retrieval_eval.py`) |
| `GCP_VERTEX_ENV` | the hard-coded `.env` path in `gemini/vertex.py` (Vertex AI service-account JSON) |

These existing variables are now also used here: `AZ_ENV`, `AZURE_DEPLOYMENT_JUDGE`, `FULLDOC_ROOT`,
`FRAMES_DATA_DIR`, `BCP_BM25_INDEX_DIR`, `OBLIQ_BENCH_ROOT`, `PYLATE_PYTHON` and `GEMINI_PYTHON`.

## Per-file diff summary

To list the code changes, compare original and new with docstrings removed (see "How verified"). Every file also got
a new module docstring, or a comment header for the exec()'d fragments. Each header says the origin, how the file is
run and what changed. The original docstring is kept below it.

**`grading/bcp_judge.py`**
- The `sys.path` inserts of `<home>/suprlm/src` and `<home>/RLM` became `<repo>/agents/document_repl/src` and
  `<repo>/agents/shared`, computed from `__file__`.
- `from judge import GRADER_TEMPLATE, OFFICIAL_TEMPLATE, parse_judge` is unchanged. It now resolves to the document REPL
  harness's `agents/document_repl/src/judge.py`. This is a declared dependency. A test checks that file's templates
  and `parse_judge` against the original.
- `examples.browsecomp_data` became `rlm_shared.browsecomp_data`, which is byte-identical in its functions.
- `open(os.environ["AZ_ENV"])` became `open(path("AZ_ENV"))`.
- `MODEL = os.environ.get("JUDGE_MODEL", "gpt-4.1")` became
  `os.environ["JUDGE_MODEL"] if "JUDGE_MODEL" in os.environ else setting("AZURE_DEPLOYMENT_JUDGE")`. With
  `AZURE_DEPLOYMENT_JUDGE=gpt-4.1` (the example env) the value is the same, and so are the cache keys.
- The side effects at import are kept and documented. Importing the module reads the AZ_ENV file, calls
  `load_dataset("Tevatron/browsecomp-plus", split="test")`, decrypts it into `GT` and builds the AzureOpenAI client.
  `judge.py` also runs its own import-time code.

**`grading/grade_bcp_snippet.py`, `grade_bcp_fulldoc.py`**
- Import wiring only: `rlm_shared.grading.bcp_judge`.

**`grading/rejudge_bcp.py`**
- Import wiring.
- `R` now comes from `ORIGINAL_REPO_ROOT` and is used only for `docs/`. `L` comes from `LOGS_ROOT`.
- `<home>/suprlm/runs/_graded_cache.json` became `$FULLDOC_ROOT/runs/_graded_cache.json`.

**`grading/grade_frames.py`**
- Import wiring.
- `frames_rows.json` is read from `FRAMES_DATA_DIR`.

**`grading/obliq_report.py`**
- Import wiring: `rlm_shared.obliq_data`. That module differs from `examples/obliq_data.py` only in `OBLIQ_ROOT`, which
  comes from `OBLIQ_BENCH_ROOT`.
- `import os` was added for the wiring.
- The document REPL runs directory is `$FULLDOC_ROOT/runs`.

**`benchmarks/retrieval/frames_search_server.py`**
- `DATA`, `CORPUS` and the Qwen3-0.6B tokenizer path are read from `FRAMES_DATA_DIR`, `BCP_BM25_INDEX_DIR` and
  `FULLDOC_ROOT`.
- The FRAMES `DATA` assignment moved from before `argparse` into the `else` branch, where it is used. Only the
  variable of the selected `--dataset` is then required.
- `import os, sys` was added for the wiring.

**`benchmarks/retrieval/obliq_math_search.py`**
- `examples.obliq_data.OBLIQ_ROOT` became `rlm_shared.obliq_data.OBLIQ_ROOT`, with `import os` added.

**`benchmarks/retrieval/gemini/vertex.py`**
- The constant `ENV` (absolute `.env` path) was removed. `_sa_info` opens `path("GCP_VERTEX_ENV")` only when
  `NNEXT_PROVIDERS_GCP_VERTEX_SERVICE_ACCOUNT_JSON` is not set, which is the same branch as before.

**`benchmarks/retrieval/frames_build_bm25.py`, `bcp_build_bm25_default.py`**
- `TFDS`, `OUT` and `CORPUS` come from `FRAMES_DATA_DIR`, `BCP_BM25_INDEX_DIR` and `FULLDOC_ROOT`.

**`benchmarks/retrieval/build_bc_rmcb_dl8192.sh`**
- The script sources `configs/paths.env`.
- `cd <home>/master` became `cd "${RETRIEVAL_EVAL_ROOT:?}"`.
- The absolute `embed_server` conda interpreter became `${PYLATE_PYTHON}`, with a `:?` check before use.
- Arguments are unchanged.

**Documentation only (no code line changed)**
- `fusion_proxy.py`, `search_server.py`, `multi_search_server.py`, `build_obliq_index.py`, `reindex_rmcb_8192.py`,
  `build_bm25_index.py`, `gemini/search_server.py` and `gemini/build_index.py`. The original usage lines in the gemini
  docstrings now name `${GEMINI_PYTHON}` and `$OBLIQ_BENCH_ROOT` instead of personal paths.
- In `scripts/reporting/`: `failure_analysis_{build,eval,inject,turns}.py` and the exec()'d fragments
  `failure_analysis_{findings,extras,turns_sheet}.py`. The fragments got a comment header, not a docstring, because
  `exec()` of a module docstring would overwrite `build.py`'s `__doc__`.

**`scripts/reporting/failure_analysis_extract.py`, `failure_analysis_turns_merge.py`, `iteration_counts.py`,
`token_columns.py`**
- `R = "<home>/RLM"` became `ORIGINAL_REPO_ROOT`, and `<R>/logs` became `LOGS_ROOT`.
- `<R>/data/frames` became `FRAMES_DATA_DIR`, and `<home>/suprlm` became `FULLDOC_ROOT`.
- `examples.*` became `rlm_shared.*`.
- `<repo>/agents/shared` is put on `sys.path` from `__file__`.
- No `/tmp` scratchpad path occurs in the copied reporting scripts. The workbook text written by `build.py` ("Built …
  by scratchpad scripts … (Claude Code session)") is output content and was left unchanged.

**Hunks outside rules 1–5: none.**

## Issues found, not fixed

File:line is given as original → new. These are the current behaviour, and the tests pin it where noted.

1. **A content-filtered verdict counts as correct.**
   - `run5d/bcp_snip_grade.py:24` → `grading/grade_bcp_snippet.py:37`, `"correct": bool(ok[q])`: `bcp_judge` returns
     the string `"filtered"` when Azure's content filter blocks the judge prompt, and `bool("filtered")` is `True`.
     The question is then written as correct in `graded_gpt41.json` and counted in the printed accuracy.
   - The same happens in `run5d/bcp_fd_grade.py:44` → `grading/grade_bcp_fulldoc.py:54`, `bool(ok.get(q))`.
   - `rejudge_bcp.py` handles it: it keeps the old verdict. `grade_frames.py` handles it too: `res.get(q) is True`, so
     filtered counts wrong.
   - Pinned by `tests/test_shared_grading.py::test_grade_bcp_snippet_filtered_counts_as_correct` and
     `…fulldoc…`.

2. **The recall denominators differ between graders.**
   - `grade_bcp_snippet.py:45` (orig. `:32`) averages `gold_recall` over the questions that have a result file. A
     missing question is not counted. Accuracy is over all 830.
   - `grade_bcp_fulldoc.py:62` (orig. `:52`) averages over the questions with a rollout file, using `cache.json`
     recall. A rollout with no cache entry counts 0.
   - `grade_frames.py:97–98` (orig. `:81–82`) sums recall over the results and divides by **824**, so missing
     questions count 0.
   - Searches are averaged over the results in all three. BrowseComp+ and FRAMES recall columns are therefore not
     computed the same way.

3. **The FRAMES grader has no retry round, and `eval()` parses the gold ids.**
   - In `frames_judge_gpt41.py:62` → `grade_frames.py:78`, a question whose votes all fail returns `None`. The caller,
     at orig. `:77` → `:93`, only reports it as "no vote" and counts it wrong.
   - `bcp_judge.grade_outputs` instead retries such questions in up to 5 rounds and raises if any are still unjudged.
   - `grade_frames.py:52` (orig. `:36`) runs `eval(ROWS[q]["gold_ids"])` on a string field of `frames_rows.json`.
   - Content-filter detection matches only `"content management policy"` (`:75`, orig. `:59`). `bcp_judge.py:89` also
     matches `"content_filter"`.
   - Tries per vote are 4 here (`:67`, orig. `:51`) and 3 in `bcp_judge` (`:81`, orig. `:49`).
   - Pinned by `test_frames_grade_four_tries_per_vote_and_no_retry_round` and
     `test_frames_content_filter_detection_differs_from_bcp_judge`.

4. **The voting loops are duplicated.**
   - `bcp_judge._grade_one` (`bcp_judge.py:69–97`, orig. `:39–65`) and `grade_frames.grade` (`grade_frames.py:57–83`,
     orig. `:41–67`) are near-copies. They differ in the cache-key prefix, tries per vote and filter detection.
   - The document REPL harness's own `judge.py` (`agents/document_repl/original/FULLDOC/src/judge.py:219–298`) has a
     third loop with different rules:
     - the default is 1 vote;
     - an all-failed `_call` returns `"correct": "no"` (`:236`);
     - the extra votes are taken only after a parsed first vote.
   - The same question can therefore get different verdicts depending on which grader ran.

5. **Ties and single votes.** With 3 votes of 3 tries each (`bcp_judge.py:80–94`), a vote whose tries all fail is
   dropped. Two parsed votes that disagree give `1*2 > 2 == False`, so a tie counts wrong. One parsed vote decides
   alone. Pinned by `test_tie_of_two_parsed_votes_counts_wrong`.

6. **Head-parsed rollouts (`grade_bcp_fulldoc.py:29–39, 46`, orig. `:19–29, 36`).**
   - `field()` reads only the first 400,000 characters. If `"result"` starts beyond that, or is cut by it,
     `raw_decode` fails and the answer becomes `""`, which is graded wrong.
   - `field()` also takes the first occurrence of `"<name>":`, so a nested key of the same name earlier in the record
     would be read instead.
   - Neither effect was checked on the real rollouts.

7. **Empty-input crashes in the summary lines.**
   - `grade_bcp_snippet.py:44–47` divides by `n` (questions with a result) and calls `statistics.mean` on them. It
     raises if no result file exists.
   - `grade_bcp_fulldoc.py:65` raises `StatisticsError` when no rollout completed (`tin` is empty).

8. **Error tests are inconsistent.**
   - `obliq_report.py:39` (orig. `:26`) treats any truthy `error` as an error.
   - The BrowseComp+ and FRAMES graders treat `error not in (None, "None")` as an error.
   - A string `"None"` error is therefore an error in the OBLIQ report and not elsewhere.

9. **`gemini/vertex.py:32` (orig. `:20`).** If the service-account variable is neither exported nor present in the
   file, `v` is `None` and `v[:1]` raises `TypeError` instead of naming the missing setting.

10. **`fusion_proxy.py:66` (orig. `:61`).** `--trust-remote-code` is `store_true` with `default=True`, so it cannot be
    turned off.

11. **A FRAMES server input has no builder.** `frames_search_server.py:60` (orig. `:45`) reads
    `$FRAMES_DATA_DIR/corpus_meta.npz` (`offsets`). `frames_build_bm25.py` does not write it, and no copied source
    does.

12. **Two builders depend on code outside this repository.**
    - `build_bc_rmcb_dl8192.sh` runs `scripts/retrieval_eval.py` of another checkout.
    - The :8001 Qwen3-Embedding-8B server of the snippet REPL rows (`embed_server_new.py`) is not among the copied
      sources.
    - The builders of `bc_reason_mcb` (512) and of the :8001 IVF-SQ8 index are not among the copied sources either.

13. **`build_bm25_index.py`** uses the bm25s defaults (k1 1.5, b 0.75), not the k1 0.9 / b 0.4 of the Experiments
    BM25 rows. The output path `~/bm25_index` is fixed. It is the fusion proxy's lexical arm, not a sheet row.

14. **Minor.**
    - `obliq_math_search.py:102` (orig. `:94`) increments `STATS["refused_untagged"]` outside the lock.
    - `gemini/search_server.py:41` raises a bare `StopIteration` when no index exists under `--index-root`.
    - `token_columns.py:25` (orig. `:18`) puts `scripts/workbook` on `sys.path`, but nothing is imported from it.
    - `failure_analysis_build.py:427` (orig. `:420`) `exec()`s every file named on its command line. This is by design.
    - `rejudge_bcp.py:77` (orig. `:61`) reads predictions from fixed character columns `[41:81]` of a text table.
      `STORED` (`:52`, orig. `:36`) is hard-coded from an earlier sheet.

## How verified

- **Code differences.** Each original/new pair was parsed with `ast`. Module, class and function docstrings were
  removed, the files were unparsed, and the results were diffed (a scratch script, not committed). 134 changed code
  lines remain in total. Every one is listed in the per-file summary above: an import, a `sys.path` insert, or a
  path/variable read. The documentation-only files show 0 changed code lines. `diff -u` of
  `build_bc_rmcb_dl8192.sh` shows only the header comment, the sourcing of `configs/paths.env`, the `cd`, the
  `PYLATE_PYTHON` check and the two interpreter substitutions.
- **The reporting copies started identical.** `scripts/reporting/` was checked with `cmp` against
  `agents/shared/original/RLM/scripts/` before editing.
- **Compilation.**
  - `python -m py_compile` passes on every file written: the grading package, `benchmarks/retrieval/**.py`,
    `scripts/reporting/*.py`, `paths.py` and the tests.
  - `bash -n` passes on `build_bc_rmcb_dl8192.sh`.
  - Both checks are also tests in `tests/test_shared_retrieval.py`.
- **Tests:** `.venv/bin/python -m pytest tests/test_shared_grading.py tests/test_shared_retrieval.py` gives 121 passed
  and 4 skipped. The skips need a model, an index or Vertex AI. They use mocks only: no network, no LLM, no dataset
  download, and no server is started. They cover:
  - `parse_judge` on synthetic replies;
  - the import side effects and wiring of `bcp_judge`;
  - majority voting, retries within a vote, tie handling, the 5-round retry and its final `RuntimeError`, `filtered`,
    and the cache key;
  - the filtered→True issue in both BrowseComp+ run graders;
  - FRAMES loading (`eval()` of `gold_ids`), voting and no-retry;
  - the OBLIQ report;
  - the fusion proxy (chunk collapse, 2,000-character cap, RRF, passthrough, mv-only);
  - the BM25 server on a three-document index (2,000-character cut, `top_k` clamp 1–100, both datasets, missing
    variable);
  - the OBLIQ Math front-end (exclusions, both formats, the untagged-request refusal);
  - `vertex._sa_info` (nothing printed);
  - a static check that every model server cuts `/search` text to 2,000 characters;
  - a check that no string literal in code names `/home/` or `/tmp/claude`.
- The full suite (`pytest` with the `pyproject.toml` test paths) gives 316 passed and 22 skipped.

## Second snapshot (2026-11-11): follow-up studies

`original/RLM_2026-11-11/` (own `MANIFEST.json`; `original/RLM/` is untouched) holds files written or changed after the
first snapshot of 2026-10-30.

| Original | New | What |
|---|---|---|
| `scripts/bcp_rejudge_gpt41_rq4.py` (written 2026-11-10) | `rlm_shared/grading/rejudge_bcp_rq4.py` | gpt-4.1 re-judge of SP-2 (reranker only) and GEPA-2 (run-2 winner) on test200. Changed only: import wiring (`bcp_judge.py` of this package, same judge and template), `LOGS_ROOT` / `ORIGINAL_REPO_ROOT` for the original machine's paths, the split file from `studies/prompt_optimisation/splits/bcp_gepa_splits/split.json` (byte-identical copy), documentation |
| `scripts/serve_qwen36_moe.sh` | `benchmarks/models/serve_qwen36.sh` | the vLLM launcher of every Qwen3.6 run: native 262,144-token context, 16,384-token output cap per call (via `GEN_OVERRIDE`). Changed only: `ENV_BIN` from `VLLM_BIN`, `HF_HOME` from `configs/paths.env`, documentation (including the non-default arguments the saved vLLM logs show) |
| `scripts/failure_analysis_{extract,build,findings,extras,turns_merge,turns_sheet}.py` (changed 2026-10-31 13:16–13:20) | not copied to `scripts/reporting/` | see F-A below |

New configuration variable: `VLLM_BIN` (absolute path of the vllm executable; `configs/paths.example.env`).

### Issues found, not fixed (second snapshot)
- **F-A. `Failure_Analysis.xlsx` in this repository is the 2026-10-30 state.** The six `failure_analysis_*` scripts were
  changed on 2026-10-31, after the snapshot, to add the completed Qwen3.6 snippet REPL rows (BCP-15 dense, BCP-17
  ColBERT, BCP-19 BM25 on all 830) and the Qwen3.6 snippet-vs-document pairs; the original repository's
  `docs/Failure_Analysis.xlsx` was rebuilt from them at 13:21. `scripts/reporting/` and
  `results/original_baselines/Failure_Analysis.xlsx` are both the earlier state (BCP-19 partial, BCP-15/17 absent), so
  they agree with each other, but the workbook does not cover the final Qwen3.6 snippet rows. The later scripts are
  preserved in `original/RLM_2026-11-11/`; the later workbook is not migrated (owner's decision pending).
- **G-RQ4-1. Content-filtered verdicts count wrong.** `rlm_shared/grading/rejudge_bcp_rq4.py` maps a `"filtered"`
  verdict to wrong (`x is True`) and lists the question ids in the summary (none in either arm). This differs from
  `rejudge_bcp.py`, which keeps the old verdict for a filtered prompt (EXPERIMENTS_STATUS §5); stated in its summary field.
