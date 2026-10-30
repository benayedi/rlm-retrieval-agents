# BCP-15: Snippet RLM · BrowseComp-Plus · Qwen3.6-35B-A3B-FP8 · Qwen3-Embedding-8B

Status: **running** at the time of writing (2026-10-30); numbers below are partial. Row 15 of the `BrowseComp+` sheet of [Experiments.xlsx](../../../results/original_baselines/Experiments.xlsx).

## 1. Experiment overview

|  |  |
|---|---|
| Identifier | `BCP-15` (sheet `BrowseComp+`, row 15) |
| Algorithm | [Snippet RLM](../../../algorithms/snippet_rlm/README.md) |
| Benchmark | BrowseComp-Plus (830 questions, 100,195 documents) |
| Model | Qwen3.6-35B-A3B-FP8 (vLLM, local) |
| Retriever | Qwen3-Embedding-8B, single-vector dense |
| Research question | Snippet RLM with an open-weight model: dense retrieval. |
| Original execution | started 2026-10-30 02:27, running at the time of writing |
| Run name in the logs | `bcp_snip_qwen_dense` |

## 2. Research motivation

Snippet RLM with an open-weight model: dense retrieval. This row is one cell of the comparison matrix in Experiments.xlsx: the same benchmark is run with each algorithm × model × retriever combination, so this row is read against its neighbours (section 8). See [ALGORITHM_COMPARISON.md](../ALGORITHM_COMPARISON.md) for the matrix and [research_overview.md](../../research_overview.md) for the research questions.

## 3. Experimental methodology

The Snippet RLM answers each question by writing Python in a persistent REPL, one turn at a time. Its tools return search results as short snippets (the first tokens of each document) into a variable; it can open a whole document and pass text to a sub-LLM (`llm_query`) to read it. A run ends when the model sets a final answer or the turn limit is reached.
 Algorithm details: [algorithms/snippet_rlm/README.md](../../../algorithms/snippet_rlm/README.md).

Per question: the agent runs on one benchmark question with the budgets in section 4; the run record (answer, retrieved document ids, tool counts, turns, time, tokens) is written to the run directory.
 Correctness is judged afterwards by gpt-4.1 (Azure deployment gpt-4.1, served gpt-4.1-2025-04-14), official BrowseComp-Plus grader template, 3 votes, majority; a question without a completed answer is wrong. Recall is the share of the question's gold documents retrieved at any point of the run (definitions per row: [RESULT_INTEGRITY.md](../../verification/RESULT_INTEGRITY.md) §R1).

## 4. Configuration

| Setting | Value |
|---|---|
| Model | Qwen3.6-35B-A3B-FP8 (vLLM, local) |
| Reasoning effort requested | medium |
| Reasoning effort actually sent | as requested (OpenAI-compatible client forwards it) |
| Retriever / index | Qwen3-Embedding-8B, single-vector dense |
| Turns | 50 |
| Searches | 25 |
| Results per search | 5 (default; the model may request more) |
| Snippet | min(2,000 chars, 512 tokens) |
| Timeout | 3,600 s per question |
| Max output tokens | 8192 |
| Compaction | on |
| Random seed | none set (API / vLLM sampling defaults); runs are not bit-reproducible |


Exact environment and arguments ([algorithms/snippet_rlm/configs/BCP-15.env](../../../algorithms/snippet_rlm/configs/BCP-15.env)):

```bash
# BCP-15  Snippet RLM | Qwen3.6-35B-A3B-FP8 (vLLM :8030/:8031, effort medium) | Qwen3-Embedding-8B (FAISS IVFSQ8) on :8001 | BrowseComp+ 830 q
# Budgets: 50 turns (MAX_ITERS_QWEN default), depth 1, --compaction, search cap 25 (BCP_SEARCH_CAP), QWEN_MAX_TOKENS 8192
#          per call, 3600 s per question (QTIMEOUT_QWEN default), per-LM-call timeout 1800 s, 8 workers.
# Status: RUNNING at refactor time. Launch line (logs/run5d/full/bcp_snip_qwen_dense.out): 2026-10-30 02:27:00
#          model=qwen retriever=http://localhost:8001 q=[0,830) workers=8 iters=50 timeout=3600s max_tokens=8192; raised to
#          20 workers at 02:27:03 by experiments/ops/night_finisher.sh. Second run of qwen_bcp_snip_queue.sh.
# Original outputs: <original repo>/logs/run5d/full/bcp_snip_qwen_dense/
# Use: experiments/run_config.sh configs/BCP-15.env   (or the whole queue: experiments/qwen_bcp_snip_queue.sh)
ROW=BCP-15
LAUNCHER=experiments/bcp_snip2.sh
LAUNCHER_ARGS=(qwen http://localhost:8001 "$RUNS_DIR/run5d/full/bcp_snip_qwen_dense" 8)
# --- effective environment of every question process (set by the launcher; listed for the record) ---
export EMBEDDING_SERVER_URL=http://localhost:8001 ORCHESTRATOR=0 LIGHTON_TOOL_DOCS=1 LIGHTON_TASK_TEXT=1
export SEARCH_SNIPPET_TOKENS=512 OFFICIAL_TOP_K=5 AZURE_OPENAI_API_VERSION=2025-04-01-preview
export LOCAL_LLM_MODEL=qwen36-35b-a3b LOCAL_LLM_API_KEY=EMPTY LOCAL_LM_TIMEOUT=1800
export AZURE_OPENAI_REASONING_EFFORT=medium SUB_LM_REASONING_EFFORT=medium QWEN_PRESENCE_PENALTY=1.5 QWEN_FENCE_TOLERANT=1
export QWEN_REPL_CUE=1 QWEN_PROSE_FINAL=1 QWEN_MAX_TOKENS=8192 BCP_SEARCH_CAP=25
# LOCAL_LLM_BASE_URL=http://localhost:$((8030 + Q % 2))/v1 per question, re-pointed at start to the less loaded server
#   by the balance_local part of src/hooks/run5d. CLI per question:
#   run_browsecomp.py --start Q --end Q+1 --max-iterations 50 --max-depth 1 --compaction --out <out>/qQ   (timeout 3600)
# --- implicit environment (see BCP-13.env): in-run judge of the `correct` field, from <original repo>/.env ---
export AZURE_OPENAI_JUDGE_MODEL=gpt-5.4-mini
```

## 5. Execution instructions (reproduction, not executed for this documentation)

These commands launch the full experiment: hundreds of model calls and hours of GPU or API time. They were **not** run while preparing this repository.

```bash
set -a; . configs/paths.env; . algorithms/snippet_rlm/configs/BCP-15.env; set +a
# then the launcher; its header documents its arguments:
algorithms/snippet_rlm/experiments/bcp_snip2.sh
```

Launcher: [algorithms/snippet_rlm/experiments/bcp_snip2.sh](../../../algorithms/snippet_rlm/experiments/bcp_snip2.sh). It reads the benchmark questions, talks to the retrieval server named in the config and writes one record per question (plus trajectories) under `$RUNS_DIR`. Grade afterwards with the matching script in [shared/rlm_shared/grading/](../../../shared/rlm_shared/grading).

## 6. Implementation references

- Algorithm source: [harness.py](../../../algorithms/snippet_rlm/src/harness.py), [sitecustomize.py](../../../algorithms/snippet_rlm/src/hooks/azure_sampling/sitecustomize.py), [sitecustomize.py](../../../algorithms/snippet_rlm/src/hooks/cap_ranking/sitecustomize.py), [sitecustomize.py](../../../algorithms/snippet_rlm/src/hooks/retry429/sitecustomize.py), [sitecustomize.py](../../../algorithms/snippet_rlm/src/hooks/retry429_cap_ranking/sitecustomize.py), [sitecustomize.py](../../../algorithms/snippet_rlm/src/hooks/run5d/sitecustomize.py), [prompt_modules.py](../../../algorithms/snippet_rlm/src/prompt_modules.py), [run_browsecomp.py](../../../algorithms/snippet_rlm/src/run_browsecomp.py), [run_frames.py](../../../algorithms/snippet_rlm/src/run_frames.py), [run_obliq.py](../../../algorithms/snippet_rlm/src/run_obliq.py)
- Original code that produced the result (verbatim, SHA-256 manifest): [algorithms/snippet_rlm/original/](../../../algorithms/snippet_rlm/original/MANIFEST.json)
- Configuration: [algorithms/snippet_rlm/configs/BCP-15.env](../../../algorithms/snippet_rlm/configs/BCP-15.env) · Launcher: [algorithms/snippet_rlm/experiments/bcp_snip2.sh](../../../algorithms/snippet_rlm/experiments/bcp_snip2.sh)
- Data loaders: [shared/rlm_shared/](../../../shared/rlm_shared/__init__.py) (`browsecomp_data.py`, `obliq_data.py`)
- Grading: [shared/rlm_shared/grading/](../../../shared/rlm_shared/grading) · Metrics: [metrics.py](../../../shared/rlm_shared/metrics.py)
- Tests: [algorithms/snippet_rlm/tests/](../../../algorithms/snippet_rlm/tests), [tests/](../../../tests)
- Per-question outputs: **not in this repository** (see [trajectories/README.md](../../../trajectories/README.md) for why)
- Original locations on the machine that ran the experiments: `logs/run5d/full/bcp_snip_qwen_dense/`

## 7. Existing experimental results

Reported in Experiments.xlsx (every non-empty column of the row):

| Metric | Value |
|---|---|
| Accuracy | 0.380 † (276/830 done) |
| Recall | 0.4979 |
| Avg. Search Calls Per Query | 19.23 |
| Avg. Iterations Per Query | 14.6 † |
| Time Per Query | 874 s (14.6 min) † |
| Cached Cost Per Query | local GPU (no API cost) |
| Uncached Cost Per Query | local GPU (no API cost) |
| Input Tokens Per Query | not logged |
| Output Tokens Per Query | not logged |


Workbook footnotes that apply to this row:

> Input / Output Tokens Per Query: all model calls of a question (root + sub-LLM), reasoning tokens included in output; these do not depend on how many runs shared a server, unlike Time Per Query. Exact from logged per-call usage, except values marked ≈ (Sept. 2026 snippet runs, whose logs overwrote the root model's usage with the sub-LLM's: rebuilt from the trajectories, about 10% low when the same method is checked on a run with exact usage). Uncached cost = input x list input price + output x list output price. Avg. Iterations Per Query: root-model turns per question, counted from the trajectory logs (snippet RLM) or the run records (full-doc RLM); sub-LLM calls are not counted. A run that reaches its turn cap adds one forced final-answer turn.

> † Still running (status 2026-10-30 08:40). Accuracy = gpt-4.1-correct answers over the questions finished so far; questions killed at the time limit count as wrong. Time is the mean over questions that returned a result, with 8-16 questions sharing each GPU. Qwen3.6 snippet RLM on BrowseComp+: 25-search cap, 50 turns, 8,192 output tokens per call, 1 h per question; on FRAMES: 25-search cap, 50 turns, 2 h per question. Snippet-run tokens are not logged for local runs (the harness keeps only the sub-LLM's count). Qwen3.6 snippet BM25 row finished 2026-10-30 02:27: 243 of 830 questions hit the 1 h limit and count as wrong.

## 8. Results interpretation

Not interpreted: the run is not complete.

## 9. Verification status

| Check | Result |
|---|---|
| Experiment previously executed | running |
| Original outputs found | yes (on the original machine; paths in section 6) |
| Configuration identified | yes |
| Implementation reviewed | yes ([algorithm review](../../verification/REVIEW_snippet_rlm.md)) |
| Lightweight tests | unit tests with mocks ([TESTING_REPORT.md](../../verification/TESTING_REPORT.md)); the experiment itself was not re-run |
| Numbers independently recomputed | partial / see audit: accuracy 0.1265 from 220 result records ([audit](../../verification/RESULT_INTEGRITY.md)) |
| Reproduced by re-running | **no**: no experiment was re-run; full experimental equivalence of the refactored code is not established |


## 10. Limitations

- No random seed; model sampling is stochastic, so a re-run gives different per-question results (two identical 830-question Snippet RLM runs disagreed on 17.8% of questions in the earlier campaign).
- Reasoning effort actually sent: as requested (OpenAI-compatible client forwards it).
- The judge (gpt-4.1) is an LLM; its verdicts were not audited by hand.
- Comparisons with other rows can differ in more than one variable (search cap, turn limit, time limit, effort, index document length); see [ALGORITHM_COMPARISON.md](../ALGORITHM_COMPARISON.md).

## 11. Related files

- [Experiments.xlsx](../../../results/original_baselines/Experiments.xlsx) · [Failure_Analysis.xlsx](../../../results/original_baselines/Failure_Analysis.xlsx) · [registry.yaml](../../../experiments/registry.yaml)
- [Known issues](../../verification/KNOWN_ISSUES.md) · [Result integrity](../../verification/RESULT_INTEGRITY.md) · [Changes from the original code](../../../algorithms/snippet_rlm/CHANGES_FROM_ORIGINAL.md)
