# Initial codebase inventory

Status: Phase 1 output, written 2026-10-30. Nothing in the original repository was modified to produce it. Source:
`/home/extstu_00/RLM` at commit `7685538` (branch `obliq-dataset-experiment`, 336 uncommitted / untracked paths), plus
the two external trees the code depends on: `/home/extstu_00/rlmrepo` and `/home/extstu_00/suprlm`.

Every statement below is from reading code, configs, logs and the three results workbooks. File references are
`path:line` in the original trees.

---

## 1. Overview of the original project

A research project comparing **recursive-language-model (RLM) agents** that answer by writing Python in a persistent
REPL, on retrieval-heavy benchmarks. The project grew in three campaigns:

| Period | Main harness | Judge | Results record |
|---|---|---|---|
| Aug–Sep 2026 | DSPy.RLM (`rag/`), then the official `rlm` library ("snippet REPL agent") | gpt-5.4-mini (BrowseComp+), NDCG (OBLIQ) | `docs/rlm_results_summary.xlsx`, `docs/browsecomp_final_results.xlsx` |
| Oct 2026 | Snippet REPL agent vs document REPL agent; GEPA prompt optimisation; Qwen3.6 local models | mixed | `docs/rlm_results_summary.xlsx` (later rows), `docs/GEPA_*.md` |
| 27–30 Oct 2026 ("run5d") | Snippet REPL agent vs document REPL agent, final matrix | **gpt-4.1** (3 votes) for BrowseComp+ / FRAMES; NDCG@10 for OBLIQ | `docs/Experiments.xlsx` (the current headline table) |

The repository is a working lab notebook, not a package: 216 scripts in `scripts/` (most are one-off chains, probes and
report generators), 8 example drivers, a small `rag/` package, and 4,142 log entries.

**Size.** `.git` 1.4 TB (logs are tracked: 3,802 of 3,908 tracked files are under `logs/`), `logs/` 431 GB, `data/`
89 GB (83 GB is the FRAMES Wikipedia corpus), `.pylate_idx/` 43 GB. None of this can go into a GitHub repository.

**Still running (do not disturb):** the Qwen3.6 snippet REPL agent BrowseComp+ queue (`scripts/run5d/qwen_bcp_snip_queue2.sh`):
dense row in progress (~ 30 Oct 18:00 UTC), ColBERT row queued (~ 31 Oct). Two vLLM servers (:8030, :8031) and the
search servers belong to it.

## 2. The three agents

| Name | What it is | Code | Agent loop from |
|---|---|---|---|
| **DSPy.RLM** | An agent built on `dspy.RLM` (DSPy 3.3.1). Typed DSPy signatures with task-specific strategy text; `search_index` / `get_document` tools; an Azure sub-LM; SUBMIT or the iteration cap ends a run. | `rag/agent.py`, `rag/signatures.py` (2,378 lines, ~40 signature variants), `rag/tools.py`; drivers `examples/browsecomp_eval_dspy_rlm.py`, `examples/obliq_eval_dspy_rlm.py` | `dspy.predict.rlm` (pip package) |
| **Snippet REPL agent** | The RLM paper authors' reference implementation, driven by our harness. Search returns top-k results, each cut to min(2,000 chars, 512 tokens); the model can open whole documents (`get_document`) and read through `llm_query` sub-LLM calls; history compaction; a format gate on the final answer. Our changes are runtime monkey-patches, not library edits. | `examples/browsecomp_eval_official_rlm.py` (core harness, imported by every runner as `H`), `scripts/run_bcp_snippet.py`, `examples/frames_eval_rlm.py`, `scripts/run_obliq_mine.py`, `scripts/official_prompt_modules.py`, `scripts/hooks/*/sitecustomize.py` | `/home/extstu_00/rlmrepo` = github.com/alexzhang13/rlm @ `854e688` (v0.1.3, MIT), via `PYTHONPATH` only |
| **Document REPL agent** | The document REPL harness (directory `suprlm`). Search puts **whole documents** into a `DOCS` variable; the model reads slices itself; client-side search cap (10); 30 turns plus a forced final answer; optional pre-search (`--seed-context`) and grounding gate. | `/home/extstu_00/suprlm/src/run_one.py`, `tools.py`, `prompts.py`, many `apply_*_patch.py` in-place patchers; wrappers in this repo (`scripts/bcp_gpt5_sup.sh`, `bcp_gpt5_bm25.sh`, `frames_sup_run_one.py`, `run_frames_sup.sh`, `run5d/frames_fd_gpt5.sh`, `run5d/bcp_fd_qwen_dense.py`, `gepa_bcp.py`, `obliq_fd_chain.sh`) | the same `rlmrepo` clone |

Two important corrections to the current written description of document REPL agent (`docs/SNIPPET_VS_FULLDOC.md`):
- **k is model-chosen, not 5.** Design B sets `allow_k_override=True` with `MAX_K = 50` (`suprlm/src/tools.py:29,82`),
  and no wrapper passes `--fixed-k` (OBLIQ excepted). Measured from the existing logs: mean 24.5 documents per search
  (BrowseComp+ BM25, 553 searches sampled, max 50) and 13.2 (FRAMES GPT-5, 398 searches).
- **Sub-LLM functions are still installed.** `--no-delegation` is passed but ignored by the stock library, which always
  installs `llm_query`/`rlm_query` in the REPL (`rlmrepo/rlm/environments/local_repl.py:224-227`, per the explorer;
  to be re-checked). "No sub-LLM" holds only through the prompt.

## 3. Benchmarks discovered

| Benchmark | Size | Corpus | Metric | Loader | Licence status |
|---|---|---|---|---|---|
| BrowseComp-Plus | 830 q | 100,195 web docs (HF `Tevatron/browsecomp-plus-corpus`) | accuracy (LLM judge), gold/evidence recall | `examples/browsecomp_data.py` (HF `Tevatron/browsecomp-plus`, canary-XOR decryption) | **answers are distributed encrypted; decrypted gold must not be published** |
| FRAMES | 824 q | Wikipedia `20230601.en` (TFDS), BM25 (bm25s) | accuracy (LLM judge), gold-article recall | `data/frames/test.tsv`, `data/frames/frames_rows.json` | not recorded |
| OBLIQ-Bench | Math 151, Twitter 281, Congress 254, WildChat 40, Writing 512 | per-task corpora, `/home/extstu_00/OBLIQ-Bench` | NDCG@10/@50, Recall@10/@50 (gold and pooled qrels) | `examples/obliq_data.py` (and a duplicate inside `obliq_eval_dspy_rlm.py`) | not recorded |
| HoVer | splits in `data/hover` (4.8 GB) | 2017 Wikipedia abstracts | claim-verification accuracy | `scripts/hover_*.py` | not recorded; used only in a GEPA side experiment, **not with any of the three RLMs** |

## 4. Experiments discovered

An "experiment" here is a configuration with a reported result in one of the three results workbooks. Pilots, smoke
tests and probes in `logs/` are listed separately (Appendix A of the final inventory) but get no individual document.

| Source | Agent | BrowseComp+ | FRAMES | OBLIQ Math | Twitter | Congress | WildChat | Writing | Count |
|---|---|---|---|---|---|---|---|---|---|
| `Experiments.xlsx` (run5d, final) | Snippet REPL agent | 5 (1 partial, 1 running) | 2 | 1 | 1 | 1 | — | — | 11 |
| | Document REPL agent | 6 | 2 | 1 | 1 | 1 | — | — | 11 |
| `rlm_results_summary.xlsx` (earlier) | DSPy.RLM | 5 | — | 11 | 9 | 9 | 7 | 1 | 42 |
| | Snippet REPL agent | 2 | — | 4 | 3 | — | — | — | 9 |
| | Document REPL agent | 9 | — | 2 | 2 | — | — | — | 13 |
| | other (OBLIQ multi-hop re-implementation) | — | — | 2 | 2 | — | — | — | 4 |
| `browsecomp_final_results.xlsx` (100-question ablation arms) | Snippet REPL agent | 17 | — | — | — | — | — | — | 17 |
| GEPA prompt-optimisation runs (`docs/GEPA_*.md`, GEPA sheet) | Snippet REPL / document REPL / DSPy.RLM | ~3 | ~2 | — | ~3 | — | — | — | ~8 |

About 115 experiments, of which some appear in two workbooks (the same BrowseComp+ 830-question runs re-judged by
gpt-4.1 in `Experiments.xlsx`); these must be deduplicated into one document with both judgements. **DSPy.RLM has no
row in the final `Experiments.xlsx`** (`docs/EXPERIMENTS_STATUS.md:16,19`: removed 2026-10-28, not run) and was never
run on FRAMES.

Experiment-to-implementation mapping for the final matrix (full mapping for all ~115 goes into the experiment registry):

| Exp ID | Agent | Model | Retriever | Runner | Results |
|---|---|---|---|---|---|
| BCP-9 | Snippet REPL | GPT-5 | Qwen3-Embed-8B | `examples/browsecomp_eval_official_rlm.py` (Sept, `rlmfix32`) | `logs/rlmfix32_q*.json`, `logs/bcp_rejudge_gpt41/10.json` |
| BCP-11 | Snippet REPL | GPT-5 | Reason-ModernColBERT | same (`rlmfix24,25,29,30,31`) | `logs/rlmfix{..}_q*.json`, `bcp_rejudge_gpt41/9.json` |
| BCP-13 / -15 / -17 / -19 | Snippet REPL | GPT-5 / Qwen3.6 | BM25 / dense / ColBERT / BM25 | `scripts/run5d/bcp_snip.sh`, `bcp_snip2.sh` → `scripts/run_bcp_snippet.py` | `logs/run5d/full/bcp_snip_*` |
| BCP-21 / -23 | Document REPL | GPT-5 | dense / ColBERT | earlier full runs | `docs/supervisor_artifact.html` (trace JSON); raw logs not in this repo |
| BCP-25 | Document REPL | GPT-5 | BM25 | `scripts/bcp_gpt5_bm25.sh` | `logs/bcp_gpt5_bm25/runs` |
| BCP-27 / -29 / -31 | Document REPL | Qwen3.6 | dense / ColBERT / BM25 | `scripts/run5d/bcp_fd_qwen_dense.py`, `scripts/gepa_bcp.py --seed-split`, `scripts/bcp_bm25_test.py` | `logs/run5d/bcp_fd_qwen_dense`, `logs/{gepa_bcp2,bcp_testrest_ctrl,gepa_bcp}`, `logs/bcp_bm25_*` (`cache.json`) |
| FR-7 / -9 | Snippet REPL | GPT-5 / Qwen3.6 | BM25 | `scripts/run5d/frames_snip.sh` → `examples/frames_eval_rlm.py` | `logs/run5d/full/frames_snip_*` |
| FR-11 / -13 | Document REPL | GPT-5 / Qwen3.6 | BM25 | `scripts/run5d/frames_fd_gpt5.sh`, `scripts/run_frames_sup.sh` | `logs/run5d/full/frames_fd_gpt5`, `logs/frames/sup_q36_bm25f` |
| MAT/TWI/CON-9 | Snippet REPL | GPT-5.2 | Gemini-Embedding-2 | `scripts/run5d/obliq_gemini.sh snip` → `run_obliq_shard.sh` → `run_obliq_mine.py` | `logs/run5d/full/obliq_snip_*/all.json` |
| MAT/TWI/CON-11 | Document REPL | GPT-5.2 | Gemini-Embedding-2 | `scripts/run5d/obliq_gemini.sh fd` | `/home/extstu_00/suprlm/runs/obliq_fd_*` |

## 5. Important implementation files

- Snippet REPL agent core: `examples/browsecomp_eval_official_rlm.py` (1,216 lines; runtime patches at 43-61, 271-659;
  task text 850-888; tools 904-935; recall 1183-1184).
- Snippet REPL runners: `scripts/run_bcp_snippet.py` (Qwen opt-ins), `examples/frames_eval_rlm.py`, `scripts/run_obliq_mine.py`,
  `scripts/official_prompt_modules.py`.
- Document REPL agent: `suprlm/src/run_one.py` (records 397-502), `suprlm/src/tools.py` (ToolBox, caps 132-139),
  `suprlm/src/prompts.py`.
- DSPy.RLM: `rag/agent.py` (`build_lm` 165-214, `build_rlm_agent` 217-519), `rag/signatures.py`, `rag/tools.py`.
- Retrieval servers: `scripts/fusion_proxy.py` (ColBERT), `scripts/frames_search_server.py` (BM25, FRAMES and BCP),
  `scripts/multi_search_server.py` / `search_server.py` (Qwen3-Embed-8B FAISS), `scripts/gemini/search_server.py`
  (Vertex Gemini-Embedding-2), `scripts/obliq_math_search.py`, `suprlm/src/retrieval_server*.py`, `suprlm/src/obliq_adapter.py`.
- Grading: `scripts/bcp_judge.py`, `scripts/run5d/bcp_snip_grade.py`, `scripts/run5d/bcp_fd_grade.py`,
  `scripts/bcp_rejudge_gpt41.py`, `scripts/frames_judge_gpt41.py`, `scripts/run5d/obliq_report.py`.
- Reporting: `scripts/run5d/iteration_counts.py`, `token_columns.py`, `scripts/workbook/*`, `scripts/failure_analysis_*.py`.

## 6. Configuration

There are no configuration files. Configuration lives in (a) CLI defaults of the drivers, (b) environment variables
read throughout the harnesses (snippet REPL agent alone reads ~40: `ORCHESTRATOR`, `LIGHTON_TOOL_DOCS`, `SEARCH_SNIPPET_TOKENS`,
`OFFICIAL_TOP_K`, `QWEN_*`, `FRAMES_*`, `OBLIQ_*`, `BCP_SEARCH_CAP`, …), and (c) the shell wrappers that set them.
The effective configuration of a finished run is recoverable only from the wrapper script plus the record header
(Document REPL records store their `config`; snippet REPL records do not).

## 7. Dataset references

HF `Tevatron/browsecomp-plus` (+ `-corpus`); `data/frames/*` and TFDS `wikipedia/20230601.en`;
`/home/extstu_00/OBLIQ-Bench/<task>/<subset>/{queries.jsonl,qrels.tsv}`; `data/hover/*`; indexes in `.pylate_idx/`,
`data/bcp_bm25_default/`, `suprlm/data/indexes/`.

## 8. Existing result locations

Workbooks (`docs/Experiments.xlsx`, `docs/rlm_results_summary.xlsx`, `docs/browsecomp_final_results.xlsx`,
`docs/Failure_Analysis.xlsx`), per-question JSON under `logs/` (431 GB including trajectories), `suprlm/runs/` (38 GB),
and the results page with an embedded trace, `docs/supervisor_artifact.html`. Graded verdicts: `logs/bcp_rejudge_gpt41/*.json`,
`logs/run5d/**/graded_gpt41.json`.

## 9. Dependencies

Python 3.12 conda env `vllm` (dspy 3.3.1, openpyxl, pandas, datasets, faiss, bm25s, vllm); env `pylate` (ColBERT);
venv `gemini` (Vertex AI); Deno (dspy's default interpreter); the `rlmrepo` clone; vLLM-served Qwen3.6-35B-A3B-FP8;
Azure OpenAI deployments (GPT-5, GPT-5.2, gpt-4.1 judge). `pyproject.toml` declares only the `rag/` package's 8
dependencies and none of the above. There is no lock file and no record of exact versions per run.

## 10. Existing testing infrastructure

None. No `tests/`, no pytest configuration, no CI. Validation was done by ad-hoc scripts (`scripts/validate_*.py`,
`assert_*.py`, `trace_check.py`) that call live servers.

## 11. Code quality observations

- Three agent implementations share nothing but the retrieval wire format; each has its own copy of data loading,
  judging and recall computation.
- Behaviour is switched by environment variables and monkey-patches (snippet REPL agent) or by in-place source patch scripts
  (Document REPL agent: `apply_*_patch.py` rewrite `tools.py`/`run_one.py`), so the code that ran is not the code on disk
  without replaying the patches.
- Duplicated logic: fence tolerance ×3, search cap ×2, Qwen REPL cue ×3, LM-timeout patching ×2, OBLIQ loaders ×2,
  judge voting loops ×2.
- 216 scripts, many one-off chains (`night*.sh`, `frames_chain{2..6}.sh`); `.bak` copies in `examples/`, `rag/`,
  `scripts/`, `logs/`; a tracked `.bak.xlsx`; `server.pid` in the root.
- Hard-coded absolute paths (`/home/extstu_00/...`) in roughly 100 files; credentials loaded from a session scratchpad
  file under `/tmp/claude-1003/...`; an Azure deployment name containing a person's name.

## 12. Potential scientific-correctness issues (to be verified in Phase 6)

| # | Issue | Where | Possible effect on reported results |
|---|---|---|---|
| C1 | Document REPL k is model-chosen (≤ 50), documented as 5 | `suprlm/src/tools.py:29,82` | Methodology text wrong; comparison with snippet REPL's k = 5 is not like-for-like |
| C2 | Document REPL "no sub-LLM" is prompt-only | `rlmrepo` `local_repl.py:224-227` | Needs a log scan for `llm_query(` calls in document REPL trajectories |
| C3 | `reasoning_effort` dropped by the stock Azure client unless the `azure_sampling` hook is loaded; BCP document REPL GPT-5 wrappers load only `retry429`; all Azure runs before 27 Oct likely ran at the deployment default | `scripts/hooks/*`, `bcp_gpt5_sup.sh:21`, `bcp_gpt5_bm25.sh:26` | Effort labels ("high") may be wrong for BCP-9/-11/-21/-23/-25 and every earlier Azure row; `Experiments.xlsx` already corrects BCP-25 only |
| C4 | `bcp_snip_grade.py:24` turns a content-filtered verdict (`"filtered"`) into correct; FRAMES counts it wrong; the re-judge keeps the old verdict | `scripts/run5d/bcp_snip_grade.py`, `frames_judge_gpt41.py:78`, `bcp_rejudge_gpt41.py:149` | Small; count of affected questions to be measured |
| C5 | Recall denominators differ (questions with a result vs all questions) | `bcp_snip_grade.py` vs `frames_judge_gpt41.py` | Recall columns not comparable across rows |
| C6 | Snippet REPL `iterations` and `input_tokens` fields are wrong (usage dict overwritten per model) | harness 1168 | Already worked around (trajectory counts); must stay documented |
| C7 | 429s, timeouts and killed runs score as wrong, silently | harness 1134-1150; wrappers' `timeout` | Lowers accuracy of affected rows; counts recoverable from logs |
| C8 | `heuristic_match` counts an empty prediction as correct (`r in g`) | `examples/browsecomp_data.py:101-103` | Only the in-run `correct` field; final numbers use gpt-4.1 verdicts. To be verified |
| C9 | OBLIQ snippet runs keep the BrowseComp format gate on, rejecting valid doc-id answers | `obliq_gemini.sh`, harness 999 | Likely contributes to OBLIQ O1 "no usable ranking" (9-15% of questions) |
| C10 | DSPy.RLM: signatures v13-v20 silently fall back to v1 when `--timeout > 0`; OBLIQ timeout leaves the thread running; OBLIQ records log the wrong model name for Qwen runs | `browsecomp_eval_dspy_rlm.py:98-105`, `obliq_eval_dspy_rlm.py:534-559, 777` | Some DSPy rows may not have run the signature their label says |
| C11 | Document REPL `n_iterations` counts the forced-answer call (cap hit = 31); `served_models` always `None`; partial answer discarded on harness timeout | `suprlm/src/run_one.py:457, 486-492` | Turn counts off by one at the cap; provenance gap |
| C12 | Two judges across campaigns (gpt-5.4-mini vs gpt-4.1) and different turn / search / timeout budgets between GPT-5 and Qwen3.6 rows | workbooks, wrappers | Cross-campaign and cross-model comparisons are confounded |

## 13. Missing information

- Raw logs for BCP-21 / BCP-23 (only the embedded trace of the results page).
- Exact library versions per run; the `rlmrepo` commit used by September runs (assumed `854e688`, no record).
- Random seeds: none are set or recorded anywhere (API models at temperature 1.0 or provider default).
- Licences for FRAMES, OBLIQ-Bench and HoVer redistribution.
- Which `apply_*_patch.py` patches were applied at the time of each document REPL run (the code has been patched in place).

## 14. Unresolved ambiguities at the time of the inventory (resolved later; see the note below)

1. **Inclusion of the document REPL harness code.** Resolved: included as part of this repository (`agents/document_repl/`).
2. **Scope of "experiment".** ~115 reported configurations across three workbooks, plus hundreds of pilots in `logs/`.
3. **Results to publish.** Per-question records are tens of GB with trajectories; BrowseComp+ gold answers and
   OBLIQ/FRAMES corpora have redistribution limits.
4. **Which rows are paper rows.** `Experiments.xlsx` is the current headline table; the older DSPy.RLM rows use other
   judges and are not part of it.

**Resolutions (2026-10-30, by the repository owner):**
1. the document REPL code is included;
2. per-experiment documents cover the rows of Experiments.xlsx only;
3. results are published as the Excel workbooks only (BrowseComp-Plus text removed).
