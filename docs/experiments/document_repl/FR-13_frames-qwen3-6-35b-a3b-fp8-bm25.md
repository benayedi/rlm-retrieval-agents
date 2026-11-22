# FR-13: Document REPL agent · FRAMES · Qwen3.6-35B-A3B-FP8 · BM25 (bm25s) over Wikipedia

Status: complete. Row 13 of the `FRAMES` sheet of [Experiments.xlsx](../../../results/original_baselines/Experiments.xlsx).

## 1. Experiment overview

|  |  |
|---|---|
| Identifier | `FR-13` (sheet `FRAMES`, row 13) |
| Agent | [Document REPL agent](../../../agents/document_repl/README.md) |
| Benchmark | FRAMES (824 questions, Wikipedia 20230601.en) |
| Model | Qwen3.6-35B-A3B-FP8 (vLLM, local) |
| Retriever | BM25 (bm25s) over Wikipedia |
| Research question | Document REPL agent on FRAMES with an open-weight model: the counterpart of FR-9. |
| Original execution | 2026-10-15 (re-judged with gpt-4.1 on 2026-10-27) |
| Run name in the logs | `sup_q36_bm25f` |

## 2. Research motivation

Document REPL agent on FRAMES with an open-weight model: the counterpart of FR-9. This row is one cell of the comparison matrix in Experiments.xlsx: the same benchmark is run with each agent × model × retriever combination, so this row is read against its neighbours (section 8). See [AGENT_COMPARISON.md](../AGENT_COMPARISON.md) for the matrix and [research_overview.md](../../research_overview.md) for the research questions.

## 3. Experimental methodology

The document REPL agent answers each question by writing Python in a persistent REPL. Each search puts the **whole** retrieved documents into a `DOCS` variable; the model reads slices of them itself (prints are capped per code block). Searches are capped client-side; at the turn limit the harness forces a final answer.
 Agent details: [agents/document_repl/README.md](../../../agents/document_repl/README.md).

Per question: the agent runs on one benchmark question with the budgets in section 4; the run record (answer, retrieved document ids, tool counts, turns, time, tokens) is written to the run directory.
 Correctness is judged afterwards by gpt-4.1 (Azure deployment gpt-4.1, served gpt-4.1-2025-04-14), official BrowseComp-Plus grader template, 3 votes, majority; a question without a completed answer is wrong. Recall is the share of the question's gold documents retrieved at any point of the run (definitions per row: [RESULT_INTEGRITY.md](../../verification/RESULT_INTEGRITY.md) §R1).

## 4. Configuration

| Setting | Value |
|---|---|
| Model | Qwen3.6-35B-A3B-FP8 (vLLM, local) |
| Reasoning effort requested | medium |
| Reasoning effort actually sent | as requested |
| Retriever / index | BM25 (bm25s) over Wikipedia |
| Turns | 30 + forced final answer |
| Searches | 10 |
| Results per search | model-chosen k (≤ 50) |
| Documents | whole documents into DOCS |
| Timeout | 5,400 s |
| Random seed | none set (API / vLLM sampling defaults); runs are not bit-reproducible |


Exact environment and arguments ([agents/document_repl/configs/FR-13.env](../../../agents/document_repl/configs/FR-13.env)):

```bash
# FR-13 | FRAMES (824) | Document REPL agent | Qwen3.6-35B-A3B (medium) | BM25 over Wikipedia 20230601.en on :8041
# model: qwen36-35b-a3b served by vLLM (BASE_URL default :8030; the URL(s) actually used are UNKNOWN), --backend openai
# effort requested: medium; sent as reasoning_effort=medium (OpenAI client forwards sampling_args); presence_penalty 1.5
# retriever: FRAMES BM25 search server (not part of this agent); BM25_PROMPT=1 -> src/run_bm25.py (record prompt_id
#   758e43ed5b95 = the BM25-patched prompt)
# caps: --max-searches 10 (client-side), --max-iterations 30 (+1 forced final answer), k model-chosen 1..50 with
#   fallback 5, --max-timeout 5400, --lm-timeout 1800, xargs timeout 6300; SUP_MAX_BLOCKS=5
# original output: <original repo>/logs/frames/sup_q36_bm25f (818 records; config read from 000.json); W, START/END,
#   REVERSE and IDS_FILE of the invocations are UNKNOWN
ROW_ID=FR-13
BENCHMARK=frames
ENTRY=src/run_bm25.py
LAUNCHER="BM25_PROMPT=1 OUT=<RUNS_DIR>/frames/sup_q36_bm25f START=0 END=824 experiments/frames_qwen.sh"
WORKERS=UNKNOWN
BM25_PROMPT=1
MODEL=qwen36-35b-a3b
ROOT_MODEL=qwen36-35b-a3b
SUB_MODEL=qwen36-35b-a3b
REASONING_EFFORT=medium
SUB_REASONING_EFFORT=medium
EFFORT_SENT=medium
RETRIEVER="BM25 over Wikipedia 20230601.en (FRAMES_DATA_DIR)"
SERVER=http://127.0.0.1:8041
MAX_SEARCHES=10
MAX_ITERATIONS=30
MAX_TIMEOUT=5400
SUP_FENCE_TOLERANT=1
SUP_MAX_BLOCKS=5
SUP_PRESENCE_PENALTY=1.5
RUN_ARGS='--base-url ${BASE_URL} --design B --server http://127.0.0.1:8041 --backend openai --api-key-env ALIBABA_TOKEN --root-model qwen36-35b-a3b --sub-model qwen36-35b-a3b --reasoning-effort medium --sub-reasoning-effort medium --max-iterations 30 --max-searches 10 --seed-context --compaction --grounding-gate --lean-prologue --no-delegation --solo-prompt --no-orchestrator --compaction-threshold-pct 0.85 --max-concurrent-subcalls 4 --max-timeout 5400 --lm-timeout 1800 --save-trajectory'
ENV_SET='PYTHONPATH=<RLMLIB_PATH> ALIBABA_TOKEN=EMPTY QUERIES_FILE=<FRAMES_DATA_DIR>/frames_queries.jsonl'
ORIGINAL_OUTPUT="<original repo>/logs/frames/sup_q36_bm25f"
```

## 5. Execution instructions (reproduction, not executed for this documentation)

These commands launch the full experiment: hundreds of model calls and hours of GPU or API time. They were **not** run while preparing this repository.

```bash
set -a; . configs/paths.env; . agents/document_repl/configs/FR-13.env; set +a
# then the launcher; its header documents its arguments:
agents/document_repl/experiments/frames_qwen.sh
```

Launcher: [agents/document_repl/experiments/frames_qwen.sh](../../../agents/document_repl/experiments/frames_qwen.sh). It reads the benchmark questions, talks to the retrieval server named in the config and writes one record per question (plus trajectories) under `$RUNS_DIR`. Grade afterwards with the matching script in [agents/shared/rlm_shared/grading/](../../../agents/shared/rlm_shared/grading).

## 6. Implementation references

- Agent source: [bcp_bm25_split.py](../../../agents/document_repl/src/bcp_bm25_split.py), [bcp_qwen_dense.py](../../../agents/document_repl/src/bcp_qwen_dense.py), [build_bm25.py](../../../agents/document_repl/src/build_bm25.py), [encoder.py](../../../agents/document_repl/src/encoder.py), [fence_tolerant.py](../../../agents/document_repl/src/fence_tolerant.py), [frames_cap25_report.py](../../../agents/document_repl/src/frames_cap25_report.py), [grade_local.py](../../../agents/document_repl/src/grade_local.py), [hybrid.py](../../../agents/document_repl/src/hybrid.py), [judge.py](../../../agents/document_repl/src/judge.py), [obliq_adapter.py](../../../agents/document_repl/src/obliq_adapter.py), [prepare_data.py](../../../agents/document_repl/src/prepare_data.py), [prompts.py](../../../agents/document_repl/src/prompts.py) …
- Original code that produced the result (verbatim, SHA-256 manifest): [agents/document_repl/original/](../../../agents/document_repl/original/MANIFEST.json)
- Configuration: [agents/document_repl/configs/FR-13.env](../../../agents/document_repl/configs/FR-13.env) · Launcher: [agents/document_repl/experiments/frames_qwen.sh](../../../agents/document_repl/experiments/frames_qwen.sh)
- Data loaders: [agents/shared/rlm_shared/](../../../agents/shared/rlm_shared/__init__.py) (`browsecomp_data.py`, `obliq_data.py`)
- Grading: [agents/shared/rlm_shared/grading/](../../../agents/shared/rlm_shared/grading) · Metrics: [metrics.py](../../../agents/shared/rlm_shared/metrics.py)
- Tests: [agents/document_repl/tests/](../../../agents/document_repl/tests), [tests/](../../../tests)
- **Per-question outputs in this repository** (Git LFS): [trajectories/document_repl/FR-13/](../../../trajectories/document_repl/FR-13/MANIFEST.json). 1,645 files, 9.31 GB as written by the run; extract with `python scripts/provenance/unpack_trajectories.py FR-13 OUT_DIR` ([format notes](../../../trajectories/README.md))
- Original locations on the machine that ran the experiments: `logs/frames/sup_q36_bm25f/`

## 7. Existing experimental results

Reported in Experiments.xlsx (every non-empty column of the row):

| Metric | Value |
|---|---|
| Accuracy | 0.6104 |
| Recall | 0.5981 |
| NDCG | — |
| Avg. Search Calls Per Query | 7.6 |
| Avg. Iterations Per Query | 20.2 |
| Time Per Query | 394 s (6.6 min) |
| Cached Cost Per Query | local GPU (no API cost) |
| Uncached Cost Per Query | local GPU (no API cost) |
| Input Tokens Per Query | 225873 |
| Output Tokens Per Query | 19312 |


**Failure breakdown** (from [Failure_Analysis.xlsx](../../../results/original_baselines/Failure_Analysis.xlsx), QA Summary; first matching category wins):

| Category | Questions | Share of failures |
|---|---|---|
| F1 did not finish | 15 | 4.7% |
| F2 no / abstained | 3 | 0.9% |
| F3 retrieval miss | 44 | 13.7% |
| F4 partial evidence | 192 | 59.8% |
| F5 all gold, wrong | 67 | 20.9% |
| **Failed in total** | 321 |  |


**Accuracy given the retrieved evidence** (By Recall sheet):

| Evidence retrieved | Questions | Accuracy |
|---|---|---|
| no gold retrieved | 81 | 45.7% |
| some gold | 481 | 59.7% |
| all gold | 247 | 72.5% |


**Turns** (By Turns sheet): limit 30, mean 20.2 (correct 16.8, not correct 25.7); 216 questions used every turn (26.2%), accuracy when answered before the limit 74.9%.

## 8. Results interpretation

**Paired comparisons on the same questions** (Pairwise sheet; exact McNemar test on the discordant pairs):

| Comparison | Other row | Only FR-13 right | Only other right | p | Reading |
|---|---|---|---|---|---|
| Snippet REPL vs document REPL, Qwen3.6, BM25 (FRAMES) | FR-9 | 81 | 89 | 0.59 | not significant |
| GPT-5 vs Qwen3.6, document REPL (FRAMES) | FR-11 | 22 | 184 | 4.9e-33 | significant at 0.05 |


- **Largest failure mode:** partial evidence (192 of 321 failures, 59.8%). Retrieval-side failures (F3 + F4) are 73.5% of failures; reading-side (F5) 20.9%. *(Measured.)*
- **With all gold evidence retrieved** the run is right 72.5% of the time (247 questions); with none, 45.7%. The gap between the two bounds how much better retrieval alone could help. *(Measured; causal reading is a hypothesis.)*
- Statements about *why* a configuration wins or loses are hypotheses unless the comparison isolates one variable; see the confounds in section 10.

## 9. Verification status

| Check | Result |
|---|---|
| Experiment previously executed | yes |
| Original outputs found | yes (on the original machine; paths in section 6) |
| Configuration identified | yes |
| Implementation reviewed | yes ([agent review](../../verification/REVIEW_document_repl.md)) |
| Lightweight tests | unit tests with mocks ([TESTING_REPORT.md](../../verification/TESTING_REPORT.md)); the experiment itself was not re-run |
| Numbers independently recomputed | **matches the workbook** (accuracy, recall, searches): accuracy 0.6104 from 818 result records ([audit](../../verification/RESULT_INTEGRITY.md)) |
| Reproduced by re-running | **no**: no experiment was re-run; full experimental equivalence of the refactored code is not established |


## 10. Limitations

- No random seed; model sampling is stochastic, so a re-run gives different per-question results (two identical 830-question snippet REPL agent runs disagreed on 17.8% of questions in the earlier campaign).
- Reasoning effort actually sent: as requested.
- The judge (gpt-4.1) is an LLM; its verdicts were not audited by hand.
- Comparisons with other rows can differ in more than one variable (search cap, turn limit, time limit, effort, index document length); see [AGENT_COMPARISON.md](../AGENT_COMPARISON.md).
- The number of documents per search is chosen by the model (up to 50), not fixed at 5 like the snippet REPL agent; the `llm_query` sub-LLM functions remain installed in the REPL (only the prompt says not to use them).

## 11. Related files

- [Experiments.xlsx](../../../results/original_baselines/Experiments.xlsx) · [Failure_Analysis.xlsx](../../../results/original_baselines/Failure_Analysis.xlsx) · [registry.yaml](../../../experiments/registry.yaml)
- [Known issues](../../verification/KNOWN_ISSUES.md) · [Result integrity](../../verification/RESULT_INTEGRITY.md) · [Changes from the original code](../../../agents/document_repl/CHANGES_FROM_ORIGINAL.md)
