# BCP-15: Snippet REPL agent · BrowseComp-Plus · Qwen3.6-35B-A3B-FP8 · Qwen3-Embedding-8B

Status: complete. Row 15 of the `BrowseComp+` sheet of [Experiments.xlsx](../../../results/original_baselines/Experiments.xlsx).

## 1. Experiment overview

|  |  |
|---|---|
| Identifier | `BCP-15` (sheet `BrowseComp+`, row 15) |
| Agent | [Snippet REPL agent](../../../agents/snippet_repl/README.md) |
| Benchmark | BrowseComp-Plus (830 questions, 100,195 documents) |
| Model | Qwen3.6-35B-A3B-FP8 (vLLM, local) |
| Retriever | Qwen3-Embedding-8B, single-vector dense |
| Research question | Snippet REPL agent with an open-weight model: dense retrieval. |
| Original execution | 2026-10-30 02:27 to 2026-10-30 20:45 |
| Run name in the logs | `bcp_snip_qwen_dense` |

## 2. Research motivation

Snippet REPL agent with an open-weight model: dense retrieval. This row is one cell of the comparison matrix in Experiments.xlsx: the same benchmark is run with each agent × model × retriever combination, so this row is read against its neighbours (section 8). See [AGENT_COMPARISON.md](../AGENT_COMPARISON.md) for the matrix and [research_overview.md](../../research_overview.md) for the research questions.

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


Exact environment and arguments ([agents/snippet_repl/configs/BCP-15.env](../../../agents/snippet_repl/configs/BCP-15.env)):

```bash
# BCP-15  snippet REPL agent | Qwen3.6-35B-A3B-FP8 (vLLM :8030/:8031, effort medium) | Qwen3-Embedding-8B (FAISS IVFSQ8) on :8001 | BrowseComp+ 830 q
# Budgets: 50 turns (MAX_ITERS_QWEN default), depth 1, --compaction, search cap 25 (BCP_SEARCH_CAP), QWEN_MAX_TOKENS 8192
#          per call, 3600 s per question (QTIMEOUT_QWEN default), per-LM-call timeout 1800 s, 8 workers.
# Launch line (logs/run5d/full/bcp_snip_qwen_dense.out): 2026-10-30 02:27:00
#          model=qwen retriever=http://localhost:8001 q=[0,830) workers=8 iters=50 timeout=3600s max_tokens=8192; raised to
#          20 workers at 02:27:03 by experiments/ops/night_finisher.sh; done 2026-10-30 20:45:30 with 640 result files
#          (190 questions hit the 1 h limit and count as wrong). Second run of qwen_bcp_snip_queue.sh.
#          (Running when this repository was first prepared; final results migrated 2026-11-22, KNOWN_ISSUES K45.)
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
set -a; . configs/paths.env; . agents/snippet_repl/configs/BCP-15.env; set +a
# then the launcher; its header documents its arguments:
agents/snippet_repl/experiments/bcp_snip2.sh
```

Launcher: [agents/snippet_repl/experiments/bcp_snip2.sh](../../../agents/snippet_repl/experiments/bcp_snip2.sh). It reads the benchmark questions, talks to the retrieval server named in the config and writes one record per question (plus trajectories) under `$RUNS_DIR`. Grade afterwards with the matching script in [agents/shared/rlm_shared/grading/](../../../agents/shared/rlm_shared/grading).

## 6. Implementation references

- Agent source: [harness.py](../../../agents/snippet_repl/src/harness.py), [sitecustomize.py](../../../agents/snippet_repl/src/hooks/azure_sampling/sitecustomize.py), [sitecustomize.py](../../../agents/snippet_repl/src/hooks/cap_ranking/sitecustomize.py), [sitecustomize.py](../../../agents/snippet_repl/src/hooks/retry429/sitecustomize.py), [sitecustomize.py](../../../agents/snippet_repl/src/hooks/retry429_cap_ranking/sitecustomize.py), [sitecustomize.py](../../../agents/snippet_repl/src/hooks/run5d/sitecustomize.py), [obliq_pick_best.py](../../../agents/snippet_repl/src/obliq_pick_best.py), [prompt_modules.py](../../../agents/snippet_repl/src/prompt_modules.py), [run_browsecomp.py](../../../agents/snippet_repl/src/run_browsecomp.py), [run_frames.py](../../../agents/snippet_repl/src/run_frames.py), [run_obliq.py](../../../agents/snippet_repl/src/run_obliq.py)
- Original code that produced the result (verbatim, SHA-256 manifest): [agents/snippet_repl/original/](../../../agents/snippet_repl/original/MANIFEST.json)
- Configuration: [agents/snippet_repl/configs/BCP-15.env](../../../agents/snippet_repl/configs/BCP-15.env) · Launcher: [agents/snippet_repl/experiments/bcp_snip2.sh](../../../agents/snippet_repl/experiments/bcp_snip2.sh)
- Data loaders: [agents/shared/rlm_shared/](../../../agents/shared/rlm_shared/__init__.py) (`browsecomp_data.py`, `obliq_data.py`)
- Grading: [agents/shared/rlm_shared/grading/](../../../agents/shared/rlm_shared/grading) · Metrics: [metrics.py](../../../agents/shared/rlm_shared/metrics.py)
- Tests: [agents/snippet_repl/tests/](../../../agents/snippet_repl/tests), [tests/](../../../tests)
- **Per-question outputs in this repository** (Git LFS): [trajectories/snippet_repl/BCP-15/](../../../trajectories/snippet_repl/BCP-15/MANIFEST.json). 2,303 files, 3.85 GB as written by the run, encrypted with the BrowseComp canary key; extract with `python scripts/provenance/unpack_trajectories.py BCP-15 OUT_DIR` ([format notes](../../../trajectories/README.md))
- Original locations on the machine that ran the experiments: `logs/run5d/full/bcp_snip_qwen_dense/`

## 7. Existing experimental results

Reported in Experiments.xlsx (every non-empty column of the row):

| Metric | Value |
|---|---|
| Accuracy | 0.3675 |
| Recall | 0.4806 |
| NDCG | — |
| Avg. Search Calls Per Query | 18.99 |
| Avg. Iterations Per Query | 14.9 |
| Time Per Query | 943 s (15.7 min) |
| Cached Cost Per Query | local GPU (no API cost) |
| Uncached Cost Per Query | local GPU (no API cost) |
| Input Tokens Per Query | not logged |
| Output Tokens Per Query | not logged |


Workbook footnotes that apply to this row:

> Input / Output Tokens Per Query: all model calls of a question (root + sub-LLM), reasoning tokens included in output; these do not depend on how many runs shared a server, unlike Time Per Query. Exact from logged per-call usage, except values marked ≈ (Sept. 2026 snippet runs, whose logs overwrote the root model's usage with the sub-LLM's: rebuilt from the trajectories, about 10% low when the same method is checked on a run with exact usage). Uncached cost = input x list input price + output x list output price. Avg. Iterations Per Query: root-model turns per question, counted from the trajectory logs (snippet RLM) or the run records (document REPL agent); sub-LLM calls are not counted. A run that reaches its turn cap adds one forced final-answer turn.


**Failure breakdown** (from [Failure_Analysis.xlsx](../../../results/original_baselines/Failure_Analysis.xlsx), QA Summary; first matching category wins):

| Category | Questions | Share of failures |
|---|---|---|
| F1 did not finish | 190 | 36.2% |
| F2 no / abstained | 36 | 6.9% |
| F3 retrieval miss | 217 | 41.3% |
| F4 partial evidence | 48 | 9.1% |
| F5 all gold, wrong | 34 | 6.5% |
| **Failed in total** | 525 |  |


**Accuracy given the retrieved evidence** (By Recall sheet):

| Evidence retrieved | Questions | Accuracy |
|---|---|---|
| no gold retrieved | 260 | 3.8% |
| some gold | 150 | 66.7% |
| all gold | 230 | 84.8% |


**Turns** (By Turns sheet): limit 50, mean 14.8 (correct 8.0, not correct 18.7); 20 questions used every turn (2.4%), accuracy when answered before the limit 49.0%.

## 8. Results interpretation

**Paired comparisons on the same questions** (Pairwise sheet; exact McNemar test on the discordant pairs):

| Comparison | Other row | Only BCP-15 right | Only other right | p | Reading |
|---|---|---|---|---|---|
| GPT-5 vs Qwen3.6, snippet, Qwen3-Embed-8B | BCP-9 | 18 | 372 | 3.8e-87 | significant at 0.05 |
| Snippet REPL vs document REPL, Qwen3.6, Qwen3-Embed-8B | BCP-27 | 75 | 137 | 2.5e-05 | significant at 0.05 |
| ColBERT vs dense, snippet, Qwen3.6 | BCP-17 | 61 | 255 | 2.4e-29 | significant at 0.05 |


- **Largest failure mode:** never retrieving a gold document (217 of 525 failures, 41.3%). Retrieval-side failures (F3 + F4) are 50.5% of failures; reading-side (F5) 6.5%. *(Measured.)*
- **With all gold evidence retrieved** the run is right 84.8% of the time (230 questions); with none, 3.8%. The gap between the two bounds how much better retrieval alone could help. *(Measured; causal reading is a hypothesis.)*
- Statements about *why* a configuration wins or loses are hypotheses unless the comparison isolates one variable; see the confounds in section 10.

## 9. Verification status

| Check | Result |
|---|---|
| Experiment previously executed | yes |
| Original outputs found | yes (on the original machine; paths in section 6) |
| Configuration identified | yes |
| Implementation reviewed | yes ([agent review](../../verification/REVIEW_snippet_repl.md)) |
| Lightweight tests | unit tests with mocks ([TESTING_REPORT.md](../../verification/TESTING_REPORT.md)); the experiment itself was not re-run |
| Numbers independently recomputed | **matches the workbook** (accuracy, recall, searches): accuracy 0.3675 from 640 result records ([audit](../../verification/RESULT_INTEGRITY.md)) |
| Reproduced by re-running | **no**: no experiment was re-run; full experimental equivalence of the refactored code is not established |


## 10. Limitations

- No random seed; model sampling is stochastic, so a re-run gives different per-question results (two identical 830-question snippet REPL agent runs disagreed on 17.8% of questions in the earlier campaign).
- Reasoning effort actually sent: as requested (OpenAI-compatible client forwards it).
- The judge (gpt-4.1) is an LLM; its verdicts were not audited by hand.
- Comparisons with other rows can differ in more than one variable (search cap, turn limit, time limit, effort, index document length); see [AGENT_COMPARISON.md](../AGENT_COMPARISON.md).

## 11. Related files

- [Experiments.xlsx](../../../results/original_baselines/Experiments.xlsx) · [Failure_Analysis.xlsx](../../../results/original_baselines/Failure_Analysis.xlsx) · [registry.yaml](../../../experiments/registry.yaml)
- [Known issues](../../verification/KNOWN_ISSUES.md) · [Result integrity](../../verification/RESULT_INTEGRITY.md) · [Changes from the original code](../../../agents/snippet_repl/CHANGES_FROM_ORIGINAL.md)
