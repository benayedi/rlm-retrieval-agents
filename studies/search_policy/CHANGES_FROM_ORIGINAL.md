# Search-policy study: changes from the original code

`original/` holds the exact files of the original research repository that ran or analysed the search-policy runs
(`original/MANIFEST.json`, SHA-256 checked by `scripts/provenance/preserve_originals.py --check`; copied 2026-11-11).
`src/` and `analysis/` are the same code made runnable from this repository under
`docs/architecture/REFACTOR_RULES.md`: renames, import wiring, locations from `configs/paths.env`, documentation. No
default, threshold, policy rule, reranker prompt (`PREFIX`, `SUFFIX`, `INSTR`) or scoring rule was changed; the bugs
found are listed under [Issues found, not fixed](#issues-found-not-fixed). The launchers in `experiments/` (and the
agent-side launchers added under `agents/`) reproduce the original command lines; most of those were typed by hand and
are **reconstructed** from the session transcripts and run records (SP-I18).

The study in one paragraph: `src/rerank_proxy.py` sits where an agent's retrieval server would be. It asks the real
retriever for the top `DEPTH` documents, optionally drops the documents this run (tag) was already shown (novelty),
scores the rest with Qwen3-Reranker-4B against "<full question>\nCurrent search: <query>", and returns the top k. The
agent, its prompt, model and search cap are unchanged; only the search URL it is given changes.

## Rename table

| Original | New |
|---|---|
| `original/RLM/scripts/bcp_rerank_proxy.py` | `src/rerank_proxy.py` (one file for BrowseComp-Plus, FRAMES and OBLIQ; the FRAMES and OBLIQ uses are environment settings, documented in its docstring, not separate files) |
| `original/RLM/logs/restart_1013/bcp_rerank_proxy.v3.py` | not copied: the 2026-10-13 09:17 backup; differs from the proxy by `WIRE=mine` (`search_mine`), `GET /health`, `POST /document`; the `WIRE=sup` `/search` path is the same code (`search(req)` vs `search(req, pre=None)` with `pre` unset) |
| `original/RLM/scripts/bcp_policy_replay.py` | `analysis/policy_replay.py` |
| `original/RLM/scripts/bcp_policy_cache.py` | `analysis/policy_cache.py` |
| `original/RLM/scripts/bcp_policy_eval.py` | `analysis/policy_eval.py` |
| `original/RLM/scripts/bcp_novelty_eval.py` | `analysis/novelty_eval.py` |
| `original/RLM/scripts/bcp_rerank_eval.py` | `analysis/rerank_eval.py` |
| `original/RLM/scripts/frames_rerank_eval.py` | `analysis/frames_rerank_eval.py` |
| `original/RLM/scripts/bcp_policy_compare.py` | `analysis/compare_bcp_split.py` |
| `original/RLM/scripts/bcp_full830_compare.py` | `analysis/compare_bcp_830.py` (SP-1 vs BCP-29) |
| `original/RLM/scripts/frames_rr_compare.py` | `analysis/compare_frames_test200.py` (the comparison of the dropped run SP-5; kept because it is in `original/`) |
| `original/RLM/scripts/bcp_pilot_grade.py` | `analysis/grade_gpt5_pilot.py` (SP-3) |
| `original/RLM/scripts/obliq_rr_compare.py` | `analysis/compare_obliq_document_repl.py` (SP-7 vs TWQ-2) |
| `original/RLM/scripts/obliq_mine_rr_compare.py` | `analysis/compare_obliq_snippet_dev100.py` (SP-6 vs TWQ-1, dev100) |
| `original/RLM/scripts/obliq_q36_next_report.py` | `analysis/compare_obliq_snippet_281.py` (SP-6 vs TWQ-1, all 281) |
| `original/RLM/scripts/frames_s25_rr_chain.sh` | `experiments/sp4_frames_qwen36_cap25_rerank.sh` |
| `original/RLM/logs/restart_1013/restart_bcp_rr_servers.sh`, `logs/bcp_full830_watch.sh`, `logs/obliq_rr/overnight.sh`, `logs/obliq_mine_rr/watch.sh` | not copied: server restarts and watchers; read as sources of the SP-1 / SP-6 / SP-7 settings (SP-I18) |
| `agents/snippet_repl/original/RLM_2026-11-11/RLM/scripts/obliq_best_full.py` | `agents/snippet_repl/src/obliq_pick_best.py` (TWQ-1: pick the version on dev100, report all 281) |
| `agents/snippet_repl/original/RLM_2026-11-11/RLM/scripts/obliq_best_chain.sh` | its SNIP branch (the one taken on 2026-10-20) in `agents/snippet_repl/experiments/obliq_twitter_qwen36.sh ARM=ctrl SLICE=rest181`; the document REPL branches (versions A, B) are not migrated |
| (new) | `experiments/_common.sh`, `experiments/sp1_bcp_qwen36_rerank.sh`, `sp2_bcp_qwen36_rerank_only.sh`, `sp3_bcp_gpt5_rerank_pilot.sh`, `sp6_obliq_snippet_qwen36_rerank.sh`, `sp7_obliq_document_qwen36_rerank.sh` (reconstructed), `tests/` |
| (new, agent side) | `agents/document_repl/experiments/bcp_qwen36_split.sh` (BCP-29 / SP-1 / SP-2: `src/split_runner.py --seed-split`), `agents/document_repl/experiments/obliq_twitter_qwen36.sh` (TWQ-2 / SP-7), `agents/snippet_repl/experiments/obliq_twitter_qwen36.sh` (TWQ-1 / SP-6), all reconstructed |

Run names in the code and outputs (`bcp_policy_e2e`, `bcp_testrest_rr`, `obliq_q36_rr`, `obliq_mine_rr`, ...) are kept.
Docstrings written for this repository use the paper's names (document REPL agent, snippet REPL agent, search cap);
the original docstrings, kept below each new paragraph, say "supervisor harness" / "full-doc RLM" (document REPL agent)
and "our harness" / "snippet RLM" (snippet REPL agent).

## Configuration used

From `configs/paths.example.env`: `RUNS_DIR`, `LOGS_ROOT` (the original run logs the analysis reads), `RLM_DATA_DIR`
(`bcp_policy/`, `bcp_gpt5_rr/`, `obliq_fd/`), `FULLDOC_ROOT` (the document REPL agent's runs and
`data/obliq_twitter/{queries.jsonl, prompts_obliq.json}`), `FRAMES_DATA_DIR` (`frames_rows.json`, `frames_queries.jsonl`,
`gepa_splits/all824_shuf_gpu{0,1}.txt`), `OBLIQ_BENCH_ROOT` (qrels, through `rlm_shared.obliq_data`), `AZ_ENV`
(`analysis/grade_gpt5_pilot.py` only), `PYTHON`, `RLMLIB_PATH`, `HF_HOME`. Split files are read from the repository
copies in `studies/prompt_optimisation/splits/` (`bcp_gepa_splits/split.json`, `bcp_full830/split_rest300.json`,
`frames/gepa_splits/split.json`, `obliq_splits/twitter_dev.json`); `GEPA_SPLIT` overrides the BrowseComp-Plus one.

Not yet in `configs/paths.example.env` (each stops with a message naming it when missing):

| Variable | Used by | Original value |
|---|---|---|
| `BM25_INDEX_DIR` | `analysis/policy_cache.py` | `~/bm25_index` (bm25s defaults; built by `benchmarks/retrieval/build_bm25_index.py`) |
| `FRAMES_TITLE_COMPARE` (optional) | `experiments/sp4_frames_qwen36_cap25_rerank.sh` (final step) | `scripts/frames_title_compare.py` of the original repository (not migrated) |
| `FRAMES_GRADE_SUP` | `agents/document_repl/experiments/frames_qwen.sh`, called by SP-4 | already documented there |
| `IDS_FILE` / `IDS_DIR` (optional overrides) | SP-3, TWQ-1/SP-6 rest181, TWQ-2/SP-7 | default under `RLM_DATA_DIR` / `LOGS_ROOT` |

## Per-file diff summary

Every hunk of `diff -u original/RLM/scripts/<file> <new>` is one of (checked by
`tests/test_search_policy_static.py::test_diff_touches_only_paths_imports_docs`, which allows a removed original line
only when it holds a location, an import / `sys.path` line or the first docstring line):

1. **Docs (rule 4):** a new first paragraph in the module docstring (study run, original file, how to run, what it
   reads and writes); the original docstring follows unchanged after "Original docstring follows."
2. **Import wiring (rule 2):** `sys.path.insert(0, "/home/<user>/RLM")` becomes the `agents/shared` directory located from
   `__file__`; `from examples.browsecomp_data import ...` → `from rlm_shared.browsecomp_data import ...` (same names);
   `from examples import obliq_data` → `from rlm_shared import obliq_data`; `sys.path.insert(0, f"{R}/scripts"); from
   bcp_judge import ...` → `from rlm_shared.grading.bcp_judge import ...` (grade_gpt5_pilot). `import os` / `sys` added
   where the wiring needs them.
3. **Locations (rule 3):** `R = "/home/<user>/RLM"` and `/home/<user>/suprlm` paths become `path("LOGS_ROOT")`,
   `path("RLM_DATA_DIR")`, `path("FULLDOC_ROOT")`, `path("FRAMES_DATA_DIR")`, `path("RUNS_DIR")` (outputs:
   `frames_rerank_eval.py --out`) or the repository split copies; the hard-coded bm25s index becomes `BM25_INDEX_DIR`.
   In `compare_bcp_830.py` the cache paths keep their original form `"logs/<run>/cache.json"` and `arm()` reads them as
   `<LOGS_ROOT>/<run>/cache.json`. `rerank_proxy.py` keeps its URLs (`UPSTREAM` default `http://127.0.0.1:8766`);
   the localhost URLs of the analysis scripts (`:8766`, `:8041`) are unchanged.

`tests/test_search_policy_analysis.py` runs nine of the scripts twice on the same synthetic inputs: the original
(with its hard-coded roots pointed into a temporary tree) and the copy (the same tree through the variables), and
asserts identical output (`compare_bcp_split`, `compare_bcp_830`, `novelty_eval` incl. the `hist.json` it writes,
`policy_eval`, `compare_obliq_snippet_dev100`, `compare_obliq_snippet_281` with and without `--final`,
`compare_frames_test200`, `compare_obliq_document_repl`, and `obliq_pick_best` `pick` / `report`). The scripts that
need a GPU, the Hugging Face hub or the judge (`rerank_eval`, `frames_rerank_eval`, `policy_cache`, `policy_replay`,
`grade_gpt5_pilot`) are checked statically only.

## Launchers

| Launcher | Reproduces | Source of every flag and variable |
|---|---|---|
| `experiments/sp1_bcp_qwen36_rerank.sh <slice>` | SP-1 proxy + agent, slices test200 (:8768), test_rest, pareto, feedback, rest300 (:8772) | transcript 8ba31816: 2026-10-10 11:19 / 11:28, 10-11 08:47, 10-12 20:29, 10-14 12:22 and 14:47; `logs/obliq_rr/overnight.sh` (10-13 resume); `logs/restart_1013/restart_bcp_rr_servers.sh`; record configs (`retrieval_server` :8768 / :8772, `prompt_id` 94f31e91f6b7) |
| `experiments/sp2_bcp_qwen36_rerank_only.sh` | SP-2 (:8769, `POLICY=rerank`) | transcript 8ba31816, 2026-10-10 18:16 |
| `experiments/sp3_bcp_gpt5_rerank_pilot.sh` | SP-3 proxy (:8772) + `agents/document_repl/experiments/bcp_gpt5_colbert.sh` (the original wrapper `bcp_gpt5_sup.sh`) | transcript 2a1318b5, 2026-10-26 22:28 and 22:35 (second start, `RERANK_MEM_FRACTION=0.5`); `logs/bcp_gpt5_rr/run_rr.log` (W 6, then 4) |
| `experiments/sp4_frames_qwen36_cap25_rerank.sh` | SP-4 | `original/RLM/scripts/frames_s25_rr_chain.sh`, arguments and variables unchanged (`tests/...::test_sp4_matches_original_chain`) |
| `experiments/sp6_obliq_snippet_qwen36_rerank.sh dev100\|rest181` | SP-6 proxy (:8775, `WIRE=mine`) + snippet REPL agent | transcript 8ba31816 2026-10-13 09:18 / 09:19 (dev100); transcript 1a0da3c7 2026-10-21 12:34 (rest181) |
| `experiments/sp7_obliq_document_qwen36_rerank.sh` | SP-7 adapter (:8767) + proxy (:8774) + document REPL agent | transcript 8ba31816, 2026-10-12 20:55; `<FULLDOC_ROOT>/runs/obliq_q36_rr/_provenance.json` `launch_argv` |
| `agents/document_repl/experiments/bcp_qwen36_split.sh` | the agent side of SP-1, SP-2 (and BCP-29) | the same command lines (identical variable set, `tests/...::test_split_launcher_env`) |
| `agents/document_repl/experiments/obliq_twitter_qwen36.sh ARM=ctrl\|rr` | TWQ-2 / SP-7 agent | transcript 8ba31816 2026-10-12 20:55 (variables); `_provenance.json` `launch_argv` (arguments, `test_obliq_document_launcher_argv`); prompt file checked: `GEPA_PROMPTS=<FULLDOC_ROOT>/data/obliq_twitter/prompts_obliq.json` gives `prompt_id` 00a9b02c72d6, the value in the records |
| `agents/snippet_repl/experiments/obliq_twitter_qwen36.sh ARM SLICE` | TWQ-1 / SP-6 agent | dev100: transcript 8ba31816 2026-10-13 09:19 (both arms); ctrl rest181: `obliq_best_chain.sh` `snip_run 12` (original wrapper; `EMBEDDING_SERVER_URL` is set explicitly to the `:8010` the shard script defaulted to); rr rest181: transcript 1a0da3c7 2026-10-21 12:34 |

Differences of every launcher from the original command lines: locations from `configs/paths.env`; run outputs under
`$RUNS_DIR` (original `logs/` of the original repository or `<FULLDOC_ROOT>/runs/`); `$PYTHON` (original: the vllm conda
env's python); the agent runs in the foreground (originals: `setsid nohup ... &`); `start_proxy` leaves a proxy that
already answers on the port alone (as the original FRAMES chain did). One addition: `RERANK_XBATCH=0` for the
2026-10-10 proxies (SP-1 test200, SP-2), see SP-I1.

## Issues found, not fixed

Line numbers are `src/…` or `analysis/…` (original `original/RLM/scripts/…` in parentheses).

- **SP-I1 (F-C) — the proxy changed during the runs; the first version is not preserved.** SP-2 and the test200 slice
  of SP-1 ran on 2026-10-10 on a proxy that scored one search at a time under a GPU lock. Cross-request batching
  (`_worker`, `RERANK_TOK_BUDGET`) was added on 2026-10-11 for test_rest; `WIRE=mine`, `/health` and `/document` on
  2026-10-13 (backup `v3`, 09:17). Only the 2026-10-13 states survive. The kept one-search-at-a-time path
  (`RERANK_XBATCH=0`, `src/rerank_proxy.py:150`, orig 117) is the closest available code for the 10-10 runs; whether it is
  identical to what ran then cannot be verified. Batching changes which documents share a padded forward pass, so
  bf16 scores, and with them rankings near ties, can differ between the versions. Which proxy process (and so which
  file state) served the pareto / feedback / rest300 slices after the 10-13 changes and the 10-15 reboot is not
  recorded; the `WIRE=sup` `/search` code is the same in every preserved state.
- **SP-I2 — the reranker is skipped silently when the question is unknown.** `src/rerank_proxy.py:203` (orig 170):
  `qid in QUESTIONS` is required; when a tag's prefix is not a known query id (wrong `QUERIES_FILE`, other tag format)
  the retriever's order is returned and no counter records it (`tests/...::test_rerank_skipped_when_question_unknown`).
- **SP-I3 — snippet reshaping counts characters, not tokens.** `src/rerank_proxy.py:212` (orig 179): for a non-`full`
  request under a non-passthrough policy the snippet is `text[:snippet_max_tokens * 4]`, not the upstream's
  token-based snippet. The study's runs request `mode="full"` (design B) or go through `search_mine` (`mode="full"`).
- **SP-I4 — statistics counters updated outside the lock.** `src/rerank_proxy.py:200-207` (orig 167-174): `STATS`
  fields other than `searches` are incremented without `LOCK` under the threading server; `/stats` counts only.
- **SP-I5 — dead code.** `src/rerank_proxy.py:232` (orig 199): `up = upstream("/stats", {}) if False else {}`.
- **SP-I6 — the preserved pilot grader is not the version that wrote the stored report.**
  `analysis/grade_gpt5_pilot.py:77` (orig 61) writes `{"rr_gpt41", "stored_rejudged_gpt41", "stored_gpt41",
  "estimate_830"}`; the stored `<LOGS_ROOT>/bcp_gpt5_rr/pilot_report.json` has `{"rr_gpt4o", "stored_gpt4o",
  "stored_gpt41", "estimate_830"}` (an earlier gpt-4o version). Its docstring and labels say "gpt-4.1 (ours)" vs
  "gpt-4.1". The reported SP-3 verdicts are `logs/bcp_rejudge_gpt41/pilot_rr.json` (`rlm_shared.grading.rejudge_bcp`),
  not this script's output. Running it overwrites `pilot_report.json` in the run directory, as the original did.
- **SP-I7 — rerank_eval's docstring and default model.** `analysis/rerank_eval.py:86` (orig 71) computes three arms;
  the docstring lists a fourth, `rr(question)`. The default `--model` is `Qwen/Qwen3-Reranker-0.6B` and `--device
  cuda:1` (`:46`, orig 31); the 4B figures in GEPA_BROWSECOMP_PLAN §7d need `--model Qwen/Qwen3-Reranker-4B`
  (`data/bcp_policy/rerank4b_val.log` suggests so); the invocation is not recorded.
- **SP-I8 — the offline BM25 arm uses the bm25s-defaults index.** `analysis/policy_cache.py:74` (orig 50) loads
  `~/bm25_index` (now `BM25_INDEX_DIR`; k1 1.5, b 0.75, KNOWN_ISSUES K25), not the k1 0.9 / b 0.4 index of BCP-31. The
  "BM25 fusion: no reliable gain" reading of the offline replay rests on that index.
- **SP-I9 — `frames_rerank_eval.py` does not create its output directory.** `analysis/frames_rerank_eval.py:154`
  (orig 139); the original's `logs/frames/` existed.
- **SP-I10 — compare_bcp_830 pairs only questions the treatment finished.** `analysis/compare_bcp_830.py:55`
  (orig 34): `... for q in qs if t[q]`, so a missing treatment record is dropped, not counted wrong as the docstring
  says; a missing control record counts wrong. The reported SP-1 numbers are the gpt-4.1 re-judge over all 830.
- **SP-I11 — compare_bcp_split drops killed runs from the pairing.** `analysis/compare_bcp_split.py:49` (orig 30): `ok`
  excludes `status == "killed"` in either arm from `both`, while the docstring says kills count as WRONG (timeouts do
  count wrong: their `correct` is false).
- **SP-I12 — digit-only ranking parser.** `analysis/compare_obliq_document_repl.py:48` (orig 33) and
  `agents/snippet_repl/src/obliq_pick_best.py:41` (orig 25): a ranking line counts only if `line.strip().isdigit()`;
  ids written with a prefix or punctuation are dropped, and NDCG@10 is scored only for `status == "completed"`.
- **SP-I13 — empty report line.** `analysis/compare_obliq_snippet_281.py:92` (orig 76): `print(f"..." if ls else "")`
  prints an empty line instead of the arm's figures when no latency is recorded (the conditional binds to the whole
  f-string).
- **SP-I14 — SP-3's first start was discarded.** The first proxy start of 2026-10-26 ran out of GPU memory
  (`logs/bcp_gpt5_rr/proxy8772_oom.log`); the `rr` directory was deleted and the run restarted (transcript 2a1318b5,
  22:35). The 100 records are from the second start. Documented, not an issue of the code.
- **SP-I15 — the slices of a run differ in settings.** SP-1's slices ran on two proxies (:8768 on GPU 0, 2026-10-10;
  :8772 on GPU 1 from 10-11) with different memory settings and worker counts, and partly while another run shared
  vLLM :8030 (10-14 12:22-14:25, HANDOFF); TWQ-1 / SP-6 dev100 (10-13, both arms together, one LLM server) and rest181
  (10-20 / 10-21, two LLM servers) ran on different days. Only wall-clock-bound outcomes (timeouts) can depend on it.
- **SP-I16 — the OBLIQ document REPL prompt file has a `.bak` sibling.** `<FULLDOC_ROOT>/data/obliq_twitter/
  prompts_obliq.json.bak` gives `prompt_id` 669457b02612; the current file gives 00a9b02c72d6, the value recorded in
  every TWQ-2 / SP-7 record, so the current file is the one that ran. It is read from `FULLDOC_ROOT`, not copied.
- **SP-I17 — hard-coded machine paths in the originals.** Every original here hard-codes `R = "/home/<user>/RLM"`
  (the proxy also puts it on `sys.path`) and `/home/<user>/suprlm`; `policy_cache.py` hard-codes `~/bm25_index`. Replaced
  under rule 3 (above).
- **SP-I18 (F-D) — the BrowseComp-Plus and OBLIQ arms had no launcher file.** SP-1, SP-2, the SP-3 proxy, SP-6, SP-7,
  TWQ-2 and the TWQ-1 dev100 slice were started as command lines in Claude Code sessions (8ba31816, 1a0da3c7,
  2a1318b5); the launchers are rebuilt from those lines, the run logs and the record configs (table above). SP-4 and the
  TWQ-1 rest181 slice have original wrappers.

## How verified

- `python -m py_compile` on every `.py` in `src/`, `analysis/` and the new `agents/snippet_repl/src/obliq_pick_best.py`;
  `bash -n` on every launcher (`tests/test_search_policy_static.py`).
- `diff -u` of every original/new pair: only the hunks listed above (`test_diff_touches_only_paths_imports_docs`).
- `pytest studies/search_policy/tests` (mocks only): the proxy with a mocked retriever and reranker (passthrough
  byte-identical, novelty filter and fallback, rerank order and scores, per-tag counts, snippet reshaping, `WIRE=mine`
  translation, batched and unbatched `rerank_scores` keep the order), original-vs-copy equivalence of nine analysis
  scripts, launcher settings against the recorded command lines.
- No experiment, server, GPU job, network call or LLM call was run.
