# Prompt optimisation (GEPA): changes from the original code

`original/RLM/` holds the exact files of the five GEPA studies, copied byte for byte from the original repository on
2026-11-11 (`original/MANIFEST.json`: source path, SHA-256, size, modification time, git state;
`scripts/provenance/preserve_originals.py --check`). `src/`, `obliq_twitter/src/` and `hover/src/` are the same code
made runnable from this repository under `docs/architecture/REFACTOR_RULES.md`: renames, import wiring, path and
interpreter parametrisation, one documentation paragraph per module, launchers. No prompt string, critic text,
feedback text, objective (correct + 0.5 x gold recall; NDCG@10 + 0.5 x gold recall on OBLIQ), budget, threshold, cache
key or fingerprint input was changed. Bugs and provenance gaps are listed under
[Issues found, not fixed](#issues-found-not-fixed).

| Study | What | Driver |
|---|---|---|
| GEPA-1 | BrowseComp-Plus run 1, proportional D_pareto 100, minibatch 3, budget 800 | `agents/document_repl/src/split_runner.py` (original `scripts/gepa_bcp.py`, already in this repository) |
| GEPA-2 | BrowseComp-Plus run 2, enriched D_pareto, minibatch 8, budget 1200; test200 confirmation | same + `src/bcp_confirm.py` |
| GEPA-3 | FRAMES, budget 800; confirmation on the 324 unused questions | `src/gepa_frames.py`, `src/gepa_frames_confirm.py` |
| GEPA-4 | OBLIQ-Twitter, snippet REPL agent, Qwen3.8-27B, four runs | `obliq_twitter/src/` |
| GEPA-5 | HoVer sanity check (the GEPA paper's program) | `hover/src/hover_gepa.py` |

Names: the original text calls the document REPL agent "supervisor harness" / "full-doc RLM" and the snippet REPL
agent "official RLM harness" / "our harness" (`run_obliq_mine.py`). Each module's new paragraph says so; the original
wording is left in place.

## Rename table

| Original (`original/RLM/...`) | New (`studies/prompt_optimisation/...`) |
|---|---|
| `scripts/gepa_frames.py`, `gepa_frames_confirm.py`, `bcp_confirm.py`, `validate_bcp.py`, `bcp_trace_health.py`, `bcp_search_ndcg.py`, `bcp_recover_misfiled.py` | `src/` under the same names |
| `scripts/bcp_gepa_split.py` | `src/make_bcp_splits.py` |
| `scripts/bcp_enrich_split.py` | `src/make_bcp_enriched_split.py` |
| `scripts/gepa_official_obliq.py` (run 1), `gepa_official_obliq_v2.py` (run 2), `gepa_night.py` (paper-shaped, valset 50), `gepa_paper.py` (run 3), `gepa_wide.py` (run 4), `validate_wide.py`, `trace_check.py` | `obliq_twitter/src/` under the same names |
| `scripts/gepa_night.py.bak-leak` | not copied (the version before the leak fix; `original/` only) |
| `scripts/hover_retriever.py`, `hover_gepa.py` | `hover/src/` under the same names |
| `scripts/hover_build_index.py`, `hover_make_splits.py`, `hover_baseline.py` | `hover/src/build_index.py`, `make_splits.py`, `baseline.py` |
| `scripts/gepa_bcp.py` | not copied again: `agents/document_repl/src/split_runner.py`. Its version before 2026-10-12 20:28 is `agents/document_repl/original/RLM_2026-11-11/RLM/logs/gepa_bcp.py.bak_1012` (PO-I1) |
| `scripts/run_gepa_bcp.sh` | `experiments/gepa1_bcp_run1.sh` |
| `logs/run_gepa_bcp_frozen.sh`, `logs/gepa_bcp/restart_after_c36.sh` | not copied (provenance; differences in the launcher header and PO-I2) |
| `scripts/run_gepa_bcp2.sh` (frozen copy `logs/run_gepa_bcp2_frozen.sh`: wait pattern only) | `experiments/gepa2_bcp_run2.sh` |
| `logs/gepa_bcp2/continue_after_fix.sh` | `experiments/gepa2_bcp_run2_continue.sh` |
| usage line of `scripts/bcp_confirm.py` (no wrapper existed) | `experiments/gepa2_confirm_test200.sh` (reconstructed) |
| `scripts/frames_gepa_chain.sh`, `frames_gepa_chain_1gpu.sh` (the one that produced the run) | `experiments/gepa3_frames.sh`, `gepa3_frames_1gpu.sh` |
| usage line of `scripts/gepa_frames_confirm.py` | `experiments/gepa3_frames_confirm.sh` (reconstructed) |
| none for run 1 / run 2 | `obliq_twitter/experiments/run1_gepa_official.sh`, `run2_gepa_official_v2.sh` (reconstructed from the drivers' defaults) |
| `scripts/night_chain.sh` lines 10-11, 79-80 | `obliq_twitter/experiments/run3a_gepa_night.sh` |
| `scripts/overnight2.sh` lines 34-38 | `obliq_twitter/experiments/run3_gepa_paper.sh` |
| `scripts/overnight3.sh` 27-30, `night5.sh` 7-10, `night6.sh` 7-10, `night7_newmodel.sh` 17 + 25-28 (also `logs/night7/chain_frozen.sh`), `night8_gepa27_finish.sh` 18 + 37-38 (also `logs/night8_frozen.sh`) | `obliq_twitter/experiments/run4_gepa_wide.sh` (`STAGE=` selects the chain) |
| `scripts/night8_gepa27_finish.sh` line 61 | `obliq_twitter/experiments/run4_validate.sh` |
| `scripts/on_gepa_finish.sh`, `watchdog.sh`, `logs/night7/watchdog_frozen.sh` | not copied (monitoring / an earlier held-out test of run 3; `original/` only) |
| `scripts/run_hover_gepa.sh` (= `logs/run_hover_gepa_frozen.sh`) | `hover/experiments/run_hover_gepa.sh` |
| `data/{bcp_gepa_splits,bcp_full830,frames/gepa_splits,obliq_splits}/...` (10 files) | `splits/` (byte-exact, own `MANIFEST.json` and `README.md`) |
| (new) | `experiments/_common.sh`, `tests/` |

Which file produced which run directory (from the run logs):

| Run directory | Driver | Started by |
|---|---|---|
| `logs/gepa_official` (2026-09-28) | `gepa_official_obliq.py` | by hand (no launcher survives) |
| `logs/gepa_official_v2/progress_w12_aborted.txt`, `progress_leaked.txt`, `candidates_leaked.json` (09-29 to 10-03 05:36) | `gepa_official_obliq_v2.py` (09-29), then `gepa_night.py.bak-leak` (`night_chain.sh` PHASE E, 10-02 23:47, W=16) | by hand; `night_chain.sh` |
| `logs/gepa_official_v2/progress.txt` (10-03 05:55, restarted 06:28:54) | `gepa_night.py` (final version, 06:28:05) | by hand (arguments not recorded) |
| `logs/gepa_paper` (10-04 05:05) | `gepa_paper.py` | `overnight2.sh` line 37 |
| `logs/gepa_wide` (10-04 10:07 to 10-05 13:12) | `gepa_wide.py` (final version from the 23:33 restart on; PO-I6) | `overnight3.sh`, `night5.sh`, `night6.sh`, `night7_newmodel.sh`, `night8_gepa27_finish.sh` |
| `logs/gepa_frames` (10-15 15:34) | `gepa_frames.py` | `frames_gepa_chain_1gpu.sh` |
| `logs/gepa_frames_confirm` (10-17 07:18 / 07:45) | `gepa_frames_confirm.py` | by hand |
| `logs/hover_gepa` (10-10) | `hover_gepa.py` | `run_hover_gepa.sh` (frozen copy) |

## Configuration

From `configs/paths.example.env`: `RUNS_DIR` (the original `logs/`: outputs, seed runs, preloaded caches),
`RLMLIB_PATH`, `PYTHON` (interpreter of the rollouts; original: the vllm conda env), `FULLDOC_ROOT` (the document REPL
agent's working directory and its `runs/_graded_cache.json`), `FRAMES_DATA_DIR` (`frames_rows.json`,
`frames_queries.jsonl`), `OBLIQ_BENCH_ROOT` (read by `rlm_shared.obliq_data` at import of every GEPA-4 driver),
`HOVER_DATA_DIR` (HoVer data and BM25 index; `hover_retriever.py` reads it at import, so `hover_gepa.py`, which only
uses its HTTP client, needs it set too), `ORIGINAL_REPO_ROOT` (only `gepa_wide.py`, PO-I7), `PYLATE_PYTHON`, `HF_HOME`,
`FRAMES_GRADE_SUP` (GEPA-3 chain step 1). Splits are read from `splits/` (computed from `__file__`).

Not yet in `configs/paths.example.env` (each launcher stops with a message naming the variable when it needs it):

| Variable | Used by | What |
|---|---|---|
| `SERVE_QWEN36`, `SERVE_QWEN27B` | `experiments/gepa1_bcp_run1.sh` (only if :8030 is down) | the vLLM launchers (original `scripts/serve_qwen36_moe.sh`, `serve_qwen27b_bf16.sh`) |
| `BCP_FORMAT_CHECK` | `experiments/gepa1_bcp_run1.sh` (smoke step) | the original repository's `scripts/bcp_format_check.py` (not part of this repository) |
| `CLAUDE_BIN` | `split_runner.py` (critic) | already read by `split_runner.py`; default `claude` |

## Per-file diff summary

Every Python file: (1) one paragraph added after the first line of the module docstring (origin, how it is run, what
changed); one-line docstrings (`build_index.py`) are reopened for it. (2) A wiring block replaces
`R = "<original repo>"` / `SUP = ...` / `sys.path.insert(0, "<abs>")`: `_REPO` from `__file__`, `agents/shared` on
`sys.path`, `RUNS = path("RUNS_DIR")` (the original `{R}/logs`), `SPLITS` (the original `{R}/data`), `DOC_SRC`.
(3) `import gepa_bcp as G` -> `import split_runner as G` (same module, renamed in the first migration);
`examples.browsecomp_data` / `examples.obliq_data` -> `rlm_shared.*`; `examples.frames_eval_rlm.judge_match` ->
`agents/snippet_repl/src/run_frames.judge_match`. (4) Interpreter, `PYTHONPATH` and `HF_HOME` of the rollouts from
the environment, with a `SystemExit` naming `PYTHON` when it is unset (added immediately before the rollouts start).
Nothing else differs; `tests/test_static.py` checks that every removed original line is one of these and that every
string of 40+ characters of the original is still present.

| File | Specifics |
|---|---|
| `src/gepa_frames.py` | `gepa_bcp` is still loaded under that module name, now from `agents/document_repl/src/split_runner.py`; rollout entry `run_bm25.py` (was `scripts/frames_sup_run_one.py`; `seed_blocks()` still execs its text between the same two anchors, tested); rollouts run with cwd `FULLDOC_ROOT` (was `SUP`); the dry-run file-name match uses the path `split_runner.py` opens (`FULLDOC_ROOT/runs/_graded_cache.json`). |
| `src/gepa_frames_confirm.py` | loads `gepa_frames.py` from its own directory. |
| `src/bcp_confirm.py`, `validate_bcp.py`, `bcp_recover_misfiled.py` | wiring only; default `--split-file` = `splits/bcp_gepa_splits/split.json`. |
| `src/make_bcp_splits.py`, `make_bcp_enriched_split.py` | input/output locations; they rewrite `splits/bcp_gepa_splits/` as the originals rewrote `data/bcp_gepa_splits/`. |
| `src/bcp_trace_health.py`, `bcp_search_ndcg.py` | default run directory `RUNS_DIR/gepa_bcp`. |
| `obliq_twitter/src/*` | the rollouts run `agents/snippet_repl/src/run_obliq.py` (was `scripts/run_obliq_mine.py`) with cwd `RUNS_DIR` (was the original repository root); `RLM_TRAJ_DIR` defaults `logs/rlm_traj_gepa*` -> `RUNS_DIR/rlm_traj_gepa*`; the seed STRATEGY is still parsed from that file's text (tested: 970 chars, sha fef4457968); `gepa_official_obliq.py` loads `agents/snippet_repl/src/prompt_modules.py` (was `scripts/official_prompt_modules.py`); preload globs and split files as above. |
| `obliq_twitter/src/trace_check.py` | docstring paragraph only (no path in the file). |
| `hover/src/*` | data from `HOVER_DATA_DIR`; outputs `RUNS_DIR/hover_gepa`; `hover_gepa.py` imports `split_runner` for the critic. |
| launchers | absolute code paths (`$DOC_SRC`, `$SRC`, `$OBLIQ_SRC`, `$HOVER_SRC`), `cd $RUNS_DIR` instead of `cd <original repo>`, `logs/X` -> `$RUNS_DIR/X`, `data/X` -> `$SPLITS/X`; process-wait patterns name the new files (`split_runner[.]py`, `frames_qwen[.]sh`); flags, budgets, worker counts and behaviour variables unchanged (`tests/test_launchers.py`). |

## Issues found, not fixed

Line numbers are `src` (original in parentheses).

1. **PO-I1 (F-B). GEPA-1 and GEPA-2 ran on an earlier `gepa_bcp.py`.** It was edited 2026-10-12 20:28 (two lines,
   `_grade` and `feedback`, original 234 and 507): the `queries` field now tolerates `search_log` entries that are not
   dicts (`str(x.get("query", "") if isinstance(x, dict) else x)`). GEPA-1 (10-07/08), GEPA-2 (10-08/09) and its
   test200 confirmation (10-09/10) ran the earlier version (`agents/document_repl/original/RLM_2026-11-11/`); this
   repository's `split_runner.py` is the later one. For dict entries both give the same field; with a non-dict entry
   the earlier `_grade` raises `AttributeError`. A sample of 120 of the 2,225 rollouts under `logs/gepa_bcp*/rollouts`
   holds only dict entries (1,045), so no record field is expected to differ.
2. **PO-I2. The GEPA-1 launcher is the later state of `run_gepa_bcp.sh`.** The frozen copy that was started on
   2026-10-07 did not export `SUP_FENCE_TOLERANT/SUP_MAX_BLOCKS/SUP_PRESENCE_PENALTY` in its qwen36 branch and always
   started a second Qwen3.6 server on GPU 1 (UTIL 0.70); the driver was then resumed by `restart_after_c36.sh` with all
   three switches, `GEPA_BCP_SERVERS=:8030` only and 20 workers. The run's cache key (ee5ea32a1c) contains the switches,
   so the bulk of run 1 ran with them; which rollouts ran before the resume, and with what, is only in the logs.
3. **PO-I3 (F-E). Killed rollouts leave the paired comparison, although both docstrings say they count wrong.**
   `src/gepa_frames_confirm.py:68` (orig 50) keeps only records whose status is not `None`/`killed`;
   `src/bcp_confirm.py:64` (orig 45) drops pairs where either arm is `killed` (timeouts and empty submissions do count
   wrong). GEPA-3 confirmation: 324 unused questions, 323 completed + 1 killed (q529; cache status `killed`), so the
   script reports n=323, which is the n in `~/TUMlatex/evidence/EVIDENCE.md`. Under the documented rule n would be 324.
   Not resolved here.
4. **PO-I4. A relative `GEPA_OUT` resolves against different bases.** `split_runner.py` joins it to `RUNS_DIR`;
   `src/validate_bcp.py:35` (orig 16) and `src/bcp_trace_health.py:36` (orig 19) use it as given (relative to the
   working directory). In the original both bases were the original repository root (the chains ran there), so they
   agreed. The launchers pass absolute paths.
5. **PO-I5. GEPA-4 ran on intermediate, unpreserved versions of the snippet REPL agent's OBLIQ runner.**
   `agents/snippet_repl/original/RLM/scripts/run_obliq_mine.py` is the 2026-10-27 state (SHA-256 b6cc74af...). Two
   backups exist in the original repository: `.bak-taskhook` (09-28 06:43; before the `OBLIQ_TASK_FILE` hook that run 4
   needs) and `.bak-cue` (09-29 09:52). From `.bak-cue` to the snapshot, 136 lines were added and none removed except
   two moved into a per-question checkpoint/resume (10-13): the `OBLIQ_PRESENCE_PENALTY`, `OBLIQ_FENCE_TOLERANT`,
   `OBLIQ_REPL_CUE`, `GPTOSS_FINAL_CUE`, `OBLIQ_NO_FORMAT_GATE`, `OBLIQ_TAG_SEARCH` switches (all opt-in; the GEPA-4
   drivers set none of them) and an unconditional import of `obliq_score_advice` (inactive unless
   `OBLIQ_RELATIVE_SCORES=1`). The seed STRATEGY text is unchanged (sha fef4457968 reproduces). Changes to the harness
   module (`browsecomp_eval_official_rlm.py`) between September and the snapshot were not compared. The backups were
   not copied.
6. **PO-I6. Drivers changed while their runs were going on.** `gepa_wide.py` in `original/` is the 2026-10-04 20:38
   version; the `overnight3` (10:07), `night5` (13:16, `GEPA_SHARD_WALL_CAP`, which the preserved version no longer
   reads) and `night6` (20:19) stages ran earlier versions; the final version ran from the night7 restart (23:33) to the
   end, so the `logs/gepa_wide` cache mixes records of several versions (the cache key carries prompt text only).
   `gepa_night.py` was edited at 06:28:05 during its run and restarted at 06:28:54. `validate_bcp.py` (10-09 18:35, counts
   timeouts wrong) is later than run 1's validation in `logs/gepa_bcp2/run1_validate.log` (10-08).
   `logs/gepa_official_v2` holds the output of two different drivers.
7. **PO-I7. `gepa_wide.py` reads its seed answer contract from a file outside this repository.**
   `obliq_twitter/src/gepa_wide.py:369` (orig 349) opens `<original repo>/experiments/rank_ab/A_control.txt`
   (`ORIGINAL_REPO_ROOT`). The file is byte-identical to the `_TASK` literal of `agents/snippet_repl/src/run_obliq.py`
   (checked), so the seed key de68174356 reproduces from this repository (`tests/test_cache_keys.py`), but running the
   driver needs the file.
8. **PO-I8. Helpers the original chains call are not in this repository:** `scripts/bcp_format_check.py` (GEPA-1
   smoke), `scripts/frames_grade_sup.py` (GEPA-3 chain step 1; the interim Qwen3.6 judge), `scripts/serve_qwen36_moe.sh`
   and `serve_qwen27b_bf16.sh` (vLLM; the first is in `agents/shared/original/RLM_2026-11-11/`), `scripts/dual_probe.py`
   (night_chain throughput probe; the chosen W=16 is recorded in `logs/night/chain.log`), `scripts/score_pair.py`
   (night3/night8 held-out tests, never reached for run 4), and the GEPA paper's HoVer program copy named in
   `hover_gepa.py`'s docstring (`data/hover/gepa_artifact_hover_reference.py`).
9. **PO-I9. Reconstructed launchers.** No wrapper survives for GEPA-4 runs 1 and 2 (the drivers' defaults are used;
   run 1's `progress.txt` matches them: val 12, minibatch 6; workers unknown), for the 10-03 05:55/06:28 relaunches of
   `gepa_night.py`, for `bcp_confirm.py` and for `gepa_frames_confirm.py` (their usage lines are used).
10. **PO-I10. `gepa_official_obliq.py` writes its progress log to a fixed directory.**
    `obliq_twitter/src/gepa_official_obliq.py:143` (orig 121): `PROG` is `RUNS_DIR/gepa_official/progress.txt`
    whatever `--out` says.
11. **PO-I11. Temporary files are never removed.** The OBLIQ drivers write strategy/task files and (runs 1-3) shard
    directories under `tempfile.gettempdir()` and leave them (`gepa_official_obliq.py:67-69`, `gepa_official_obliq_v2.py:124-126`,
    `gepa_night.py:150-152`, `gepa_paper.py:149-151`, `gepa_wide.py:168-170`).
12. **PO-I12. Split files carry machine paths.** `splits/bcp_gepa_splits/split.json`, `splits/bcp_full830/split_rest300.json`
    (`source`) and `splits/frames/gepa_splits/split.json` (`_meta.seed_run`) contain the original machine's input path;
    they are byte-exact copies and were left so.

## How verified

- `python -m py_compile` on every new `.py`; `bash -n` on every new `.sh`.
- `pytest studies/prompt_optimisation/tests` (mocks only; no model, server, network or `claude` call): 141 passed,
  3 skipped; with `LOGS_ROOT` and `FULLDOC_ROOT` pointing at the original records, 144 passed. The tests cover:
  - static checks (`test_static.py`);
  - cache keys reproduced from code: GEPA-1/2 seed ee5ea32a1c and winner d911ebbe23, GEPA-3 seed 1b5846380c,
    best 598a758e3a and confirmed candidate 8ba345b481, GEPA-4 seeds fef4457968 and de68174356 (`test_cache_keys.py`);
  - the split files and builders, including `make_bcp_splits.py` reproducing `split.json` from the original grades
    (`test_splits.py`);
  - the confirmation and validation scoring rules on synthetic caches (`test_scoring.py`);
  - the HoVer helpers and the critic path with a stubbed `claude -p` (`test_hover_and_critic.py`);
  - launcher flags and behaviour variables against the original wrappers and chain lines (`test_launchers.py`).
- `diff -u original/RLM/scripts/<file> <new file>` for every file: every hunk is one of the kinds in the per-file
  summary.
