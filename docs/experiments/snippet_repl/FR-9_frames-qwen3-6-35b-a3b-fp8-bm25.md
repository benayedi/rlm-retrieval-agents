# FR-9: Snippet REPL agent · FRAMES · Qwen3.6-35B-A3B-FP8 · BM25 (bm25s

Status: complete. Row 9 of the `FRAMES` sheet of [Experiments.xlsx](../../../results/original_baselines/Experiments.xlsx).

## 1. Experiment overview

|  |  |
|---|---|
| Identifier | `FR-9` (sheet `FRAMES`, row 9) |
| Agent | [Snippet REPL agent](../../../agents/snippet_repl/README.md) |
| Benchmark | FRAMES (824 questions, Wikipedia 20230601.en) |
| Model | Qwen3.6-35B-A3B-FP8 (vLLM, local) |
| Retriever | BM25 (bm25s, k1 0.9, b 0.4) over Wikipedia |
| Research question | Snippet REPL agent on FRAMES with an open-weight model. |
| Original execution | 2026-10-27 to 2026-10-29 |
| Run name in the logs | `frames_snip_qwen` |

## 2. Research motivation

Snippet REPL agent on FRAMES with an open-weight model. This row is one cell of the comparison matrix in Experiments.xlsx: the same benchmark is run with each agent × model × retriever combination, so this row is read against its neighbours (section 8). See [AGENT_COMPARISON.md](../AGENT_COMPARISON.md) for the matrix and [research_overview.md](../../research_overview.md) for the research questions.

## 3. Experimental methodology

The snippet REPL agent answers each question by writing Python in a persistent REPL, one turn at a time. Its tools return search results as short snippets (the first tokens of each document) into a variable; it can open a whole document and pass text to a sub-LLM (`llm_query`) to read it. A run ends when the model sets a final answer or the turn limit is reached.
 Agent details: [agents/snippet_repl/README.md](../../../agents/snippet_repl/README.md).

Per question: the agent runs on one benchmark question with the budgets in section 4; the run record (answer, retrieved document ids, tool counts, turns, time, tokens) is written to the run directory.
 Correctness is judged afterwards by gpt-4.1 (Azure deployment gpt-4.1, served gpt-4.1-2025-04-14), official BrowseComp-Plus grader template, 3 votes, majority; a question without a completed answer is wrong. Recall is the share of the question's gold documents retrieved at any point of the run (definitions per row: [RESULT_INTEGRITY.md](../../verification/RESULT_INTEGRITY.md) §R1).

## 4. Configuration

| Setting | Value |
|---|---|
| Model | Qwen3.6-35B-A3B-FP8 (vLLM, local) |
| Reasoning effort requested | medium |
| Reasoning effort actually sent | as requested |
| Retriever / index | BM25 (bm25s, k1 0.9, b 0.4) over Wikipedia |
| Turns | 50 |
| Searches | 25 |
| Results per search | 5 (default; the model may request more) |
| Snippet | min(2,000 chars, 512 tokens) |
| Timeout | 7,200 s per question |
| Compaction | on |
| Random seed | none set (API / vLLM sampling defaults); runs are not bit-reproducible |


Exact environment and arguments ([agents/snippet_repl/configs/FR-9.env](../../../agents/snippet_repl/configs/FR-9.env)):

```bash
# FR-9  snippet REPL agent | Qwen3.6-35B-A3B-FP8 (vLLM :8030/:8031, effort medium) | BM25 over Wikipedia 2023-06-01 on :8040 | FRAMES
# Budgets: 50 turns, depth 1, --compaction, search cap 25 (FRAMES_SEARCH_CAP), top_k default 5, 512-token snippets,
#          7200 s per question, per-LM-call timeout 1800 s, 8 workers; FRAMES_REPL_CUE=1 + FRAMES_FENCE_TOLERANT=1; no
#          output-token cap and no prose-final rule (those are BrowseComp+-only QWEN_* switches). In-run judge: local Qwen3.6.
# Launch line (logs/run5d/full/frames_snip_qwen.out): 2026-10-27 18:27:04 q=[0,824) workers=8; done 2026-10-29 13:17:35 with
#          751 results (73 missing count as wrong); graded by experiments/ops/night_finisher.sh -> frames_snip_qwen.grade.txt.
# Original outputs: <original repo>/logs/run5d/full/frames_snip_qwen/
# Use: experiments/run_config.sh configs/FR-9.env
ROW=FR-9
LAUNCHER=experiments/frames_snip.sh
LAUNCHER_ARGS=(qwen "$RUNS_DIR/run5d/full/frames_snip_qwen" 8)
# --- effective environment of every question process (set by the launcher; listed for the record) ---
export EMBEDDING_SERVER_URL=http://localhost:8040 FRAMES_SEARCH_CAP=25 RETRIEVER_LINE=0 FRAMES_BM25_LINE=1
export ORCHESTRATOR=0 LIGHTON_TOOL_DOCS=1 LIGHTON_TASK_TEXT=1 SEARCH_SNIPPET_TOKENS=512 OFFICIAL_TOP_K=5
export AZURE_OPENAI_API_VERSION=2025-04-01-preview LOCAL_JUDGE_BASE_URL=http://localhost:8030/v1 LOCAL_JUDGE_MODEL=qwen36-35b-a3b
export LOCAL_LLM_MODEL=qwen36-35b-a3b LOCAL_LM_TIMEOUT=1800 FRAMES_REPL_CUE=1 FRAMES_FENCE_TOLERANT=1
export AZURE_OPENAI_REASONING_EFFORT=medium SUB_LM_REASONING_EFFORT=medium
# LOCAL_LLM_BASE_URL=http://localhost:$((8030 + Q % 2))/v1 per question (re-balanced at start by src/hooks/run5d).
#   CLI per question: run_frames.py --start Q --end Q+1 --max-iterations 50 --compaction --out <out>/qNNN   (timeout 7200)
```

## 5. Execution instructions (reproduction, not executed for this documentation)

These commands launch the full experiment: hundreds of model calls and hours of GPU or API time. They were **not** run while preparing this repository.

```bash
set -a; . configs/paths.env; . agents/snippet_repl/configs/FR-9.env; set +a
# then the launcher; its header documents its arguments:
agents/snippet_repl/experiments/frames_snip.sh
```

Launcher: [agents/snippet_repl/experiments/frames_snip.sh](../../../agents/snippet_repl/experiments/frames_snip.sh). It reads the benchmark questions, talks to the retrieval server named in the config and writes one record per question (plus trajectories) under `$RUNS_DIR`. Grade afterwards with the matching script in [agents/shared/rlm_shared/grading/](../../../agents/shared/rlm_shared/grading).

## 6. Implementation references

- Agent source: [harness.py](../../../agents/snippet_repl/src/harness.py), [sitecustomize.py](../../../agents/snippet_repl/src/hooks/azure_sampling/sitecustomize.py), [sitecustomize.py](../../../agents/snippet_repl/src/hooks/cap_ranking/sitecustomize.py), [sitecustomize.py](../../../agents/snippet_repl/src/hooks/retry429/sitecustomize.py), [sitecustomize.py](../../../agents/snippet_repl/src/hooks/retry429_cap_ranking/sitecustomize.py), [sitecustomize.py](../../../agents/snippet_repl/src/hooks/run5d/sitecustomize.py), [obliq_pick_best.py](../../../agents/snippet_repl/src/obliq_pick_best.py), [prompt_modules.py](../../../agents/snippet_repl/src/prompt_modules.py), [run_browsecomp.py](../../../agents/snippet_repl/src/run_browsecomp.py), [run_frames.py](../../../agents/snippet_repl/src/run_frames.py), [run_obliq.py](../../../agents/snippet_repl/src/run_obliq.py)
- Original code that produced the result (verbatim, SHA-256 manifest): [agents/snippet_repl/original/](../../../agents/snippet_repl/original/MANIFEST.json)
- Configuration: [agents/snippet_repl/configs/FR-9.env](../../../agents/snippet_repl/configs/FR-9.env) · Launcher: [agents/snippet_repl/experiments/frames_snip.sh](../../../agents/snippet_repl/experiments/frames_snip.sh)
- Data loaders: [agents/shared/rlm_shared/](../../../agents/shared/rlm_shared/__init__.py) (`browsecomp_data.py`, `obliq_data.py`)
- Grading: [agents/shared/rlm_shared/grading/](../../../agents/shared/rlm_shared/grading) · Metrics: [metrics.py](../../../agents/shared/rlm_shared/metrics.py)
- Tests: [agents/snippet_repl/tests/](../../../agents/snippet_repl/tests), [tests/](../../../tests)
- **Per-question outputs in this repository** (Git LFS): [trajectories/snippet_repl/FR-9/](../../../trajectories/snippet_repl/FR-9/MANIFEST.json). 2,403 files, 3.78 GB as written by the run; extract with `python scripts/provenance/unpack_trajectories.py FR-9 OUT_DIR` ([format notes](../../../trajectories/README.md))
- Original locations on the machine that ran the experiments: `logs/run5d/full/frames_snip_qwen/`

## 7. Existing experimental results

Reported in Experiments.xlsx (every non-empty column of the row):

| Metric | Value |
|---|---|
| Accuracy | 0.6201 |
| Recall | 0.5752 |
| NDCG | — |
| Avg. Search Calls Per Query | 12.85 |
| Avg. Iterations Per Query | 10.9 |
| Time Per Query | 903 s (15.1 min) |
| Cached Cost Per Query | local GPU (no API cost) |
| Uncached Cost Per Query | local GPU (no API cost) |
| Input Tokens Per Query | not logged |
| Output Tokens Per Query | not logged |


**Failure breakdown** (from [Failure_Analysis.xlsx](../../../results/original_baselines/Failure_Analysis.xlsx), QA Summary; first matching category wins):

| Category | Questions | Share of failures |
|---|---|---|
| F1 did not finish | 75 | 24.0% |
| F2 no / abstained | 1 | 0.3% |
| F3 retrieval miss | 26 | 8.3% |
| F4 partial evidence | 146 | 46.6% |
| F5 all gold, wrong | 65 | 20.8% |
| **Failed in total** | 313 |  |


**Accuracy given the retrieved evidence** (By Recall sheet):

| Evidence retrieved | Questions | Accuracy |
|---|---|---|
| no gold retrieved | 76 | 65.8% |
| some gold | 423 | 65.5% |
| all gold | 250 | 73.6% |


**Turns** (By Turns sheet): limit 50, mean 10.9 (correct 7.9, not correct 15.8); 3 questions used every turn (0.4%), accuracy when answered before the limit 68.4%.

## 8. Results interpretation

**Paired comparisons on the same questions** (Pairwise sheet; exact McNemar test on the discordant pairs):

| Comparison | Other row | Only FR-9 right | Only other right | p | Reading |
|---|---|---|---|---|---|
| Snippet REPL vs document REPL, Qwen3.6, BM25 (FRAMES) | FR-13 | 89 | 81 | 0.59 | not significant |
| GPT-5 vs Qwen3.6, snippet (FRAMES) | FR-7 | 50 | 168 | 4.1e-16 | significant at 0.05 |


- **Largest failure mode:** partial evidence (146 of 313 failures, 46.6%). Retrieval-side failures (F3 + F4) are 55.0% of failures; reading-side (F5) 20.8%. *(Measured.)*
- **With all gold evidence retrieved** the run is right 73.6% of the time (250 questions); with none, 65.8%. The gap between the two bounds how much better retrieval alone could help. *(Measured; causal reading is a hypothesis.)*
- Statements about *why* a configuration wins or loses are hypotheses unless the comparison isolates one variable; see the confounds in section 10.

## 9. Verification status

| Check | Result |
|---|---|
| Experiment previously executed | yes |
| Original outputs found | yes (on the original machine; paths in section 6) |
| Configuration identified | yes |
| Implementation reviewed | yes ([agent review](../../verification/REVIEW_snippet_repl.md)) |
| Lightweight tests | unit tests with mocks ([TESTING_REPORT.md](../../verification/TESTING_REPORT.md)); the experiment itself was not re-run |
| Numbers independently recomputed | **matches the workbook** (accuracy, recall, searches): accuracy 0.6201 from 751 result records ([audit](../../verification/RESULT_INTEGRITY.md)) |
| Reproduced by re-running | **no**: no experiment was re-run; full experimental equivalence of the refactored code is not established |


## 10. Limitations

- No random seed; model sampling is stochastic, so a re-run gives different per-question results (two identical 830-question snippet REPL agent runs disagreed on 17.8% of questions in the earlier campaign).
- Reasoning effort actually sent: as requested.
- The judge (gpt-4.1) is an LLM; its verdicts were not audited by hand.
- Comparisons with other rows can differ in more than one variable (search cap, turn limit, time limit, effort, index document length); see [AGENT_COMPARISON.md](../AGENT_COMPARISON.md).

## 11. Related files

- [Experiments.xlsx](../../../results/original_baselines/Experiments.xlsx) · [Failure_Analysis.xlsx](../../../results/original_baselines/Failure_Analysis.xlsx) · [registry.yaml](../../../experiments/registry.yaml)
- [Known issues](../../verification/KNOWN_ISSUES.md) · [Result integrity](../../verification/RESULT_INTEGRITY.md) · [Changes from the original code](../../../agents/snippet_repl/CHANGES_FROM_ORIGINAL.md)
