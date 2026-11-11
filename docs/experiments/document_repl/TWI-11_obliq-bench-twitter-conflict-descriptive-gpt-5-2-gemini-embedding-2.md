# TWI-11: Document REPL agent · OBLIQ-Bench Twitter conflict, descriptive · gpt-5.2 · Gemini-Embedding-2 (Vertex AI)

Status: complete. Row 11 of the `OBLIQ - Twitter` sheet of [Experiments.xlsx](../../../results/original_baselines/Experiments.xlsx).

## 1. Experiment overview

|  |  |
|---|---|
| Identifier | `TWI-11` (sheet `OBLIQ - Twitter`, row 11) |
| Agent | [Document REPL agent](../../../agents/document_repl/README.md) |
| Benchmark | OBLIQ-Bench Twitter conflict, descriptive (281 queries) |
| Model | gpt-5.2 (Azure) |
| Retriever | Gemini-Embedding-2 (Vertex AI), raw query |
| Research question | Document REPL agent on the Twitter ranking task: the counterpart of TWI-9. |
| Original execution | 2026-10-27 |
| Run name in the logs | `obliq_fd_twitter` |

## 2. Research motivation

Document REPL agent on the Twitter ranking task: the counterpart of TWI-9. This row is one cell of the comparison matrix in Experiments.xlsx: the same benchmark is run with each agent × model × retriever combination, so this row is read against its neighbours (section 8). See [AGENT_COMPARISON.md](../AGENT_COMPARISON.md) for the matrix and [research_overview.md](../../research_overview.md) for the research questions.

## 3. Experimental methodology

The document REPL agent answers each question by writing Python in a persistent REPL. Each search puts the **whole** retrieved documents into a `DOCS` variable; the model reads slices of them itself (prints are capped per code block). Searches are capped client-side; at the turn limit the harness forces a final answer.
 Agent details: [agents/document_repl/README.md](../../../agents/document_repl/README.md).

Per question: the agent runs on one benchmark question with the budgets in section 4; the run record (answer, retrieved document ids, tool counts, turns, time, tokens) is written to the run directory.
 Scoring: none (NDCG@10 / Recall@10 against the gold qrels, binary relevance; ranking parsed from the answer text). A question without a parsable ranking scores 0.

## 4. Configuration

| Setting | Value |
|---|---|
| Model | gpt-5.2 (Azure) |
| Reasoning effort requested | high |
| Reasoning effort actually sent | as requested (run5d hook loaded) |
| Retriever / index | Gemini-Embedding-2 (Vertex AI), raw query |
| Turns | 30 + forced ranking (cap_ranking hook) |
| Searches | 10 |
| Results per search | fixed k 20 |
| Documents | whole passages into DOCS |
| Timeout | 3,600 s |
| Random seed | none set (API / vLLM sampling defaults); runs are not bit-reproducible |


Exact environment and arguments ([agents/document_repl/configs/TWI-11.env](../../../agents/document_repl/configs/TWI-11.env)):

```bash
# TWI-11 | OBLIQ twitter (descriptive, 281 queries) | Document REPL agent | GPT-5.2 (high) | Gemini-Embedding-2, raw text
# model: gpt-5.2 via Azure deployment AZURE_DEPLOYMENT_GPT52 (original: a personal deployment); served_models: null
# effort requested: high; effort actually sent: high -- startup hook run5d (retry429 + azure_sampling + cap_ranking;
#   OBLIQ_CAP_RANKING=1 replaces the forced final-answer message with a request for a plain ranking)
# retriever: src/obliq_adapter.py on :8768 in front of the Gemini-Embedding-2 server (upstream URL / --index of the adapter: UNKNOWN)
# prompts: GEPA_PROMPTS=configs/obliq/prompts_twitter.json (original: $RLM_DATA_DIR/obliq_fd_gemini/prompts_twitter.json) (record prompt_id 299e86ce3e1e); no grounding gate
# caps: --max-searches 10 (client-side), --max-iterations 30 (+1 forced final answer), FIXED k 20 (--fixed-k),
#   --max-timeout 3600, --lm-timeout 1800; run_experiment per-query guard 4200 s
# original output: <FULLDOC_ROOT>/runs/obliq_fd_twitter (281 records; config read from one record; launch argv read from
#   _provenance.json: --workers 5, ids file below, installed rlm = RLMLIB_PATH)
ROW_ID=TWI-11
BENCHMARK=obliq_twitter
ENTRY=src/run_experiment.py
LAUNCHER="experiments/obliq_fd.sh fd twitter <ids.json> <out_dir named obliq_fd_twitter> 5"
WORKERS=5
OBLIQ_TASK=descriptive
OBLIQ_SUBSET=twitter
IDS_FILE="${RLM_DATA_DIR}/obliq_fd/all281.json"
QUERIES_FILE="${FULLDOC_ROOT}/data/obliq_twitter/queries.jsonl"
# original: GEPA_PROMPTS="${RLM_DATA_DIR}/obliq_fd_gemini/prompts_twitter.json" (byte-identical copy below; configs/obliq/MANIFEST.json)
GEPA_PROMPTS="agents/document_repl/configs/obliq/prompts_twitter.json"
OBLIQ_CAP_RANKING=1
ROOT_MODEL=gpt-5.2
SUB_MODEL=gpt-5.2
REASONING_EFFORT=high
SUB_REASONING_EFFORT=high
EFFORT_SENT=high
RETRIEVER="Gemini-Embedding-2 (obliq-des-twitter)"
SERVER=http://127.0.0.1:8768
MAX_SEARCHES=10
MAX_ITERATIONS=30
MAX_TIMEOUT=3600
RUN_ARGS='--name obliq_fd_twitter --ids-file ${IDS_FILE} --workers 5 --stagger 0.3 --design B --server http://127.0.0.1:8768 --backend azure_openai --root-model gpt-5.2 --sub-model gpt-5.2 --root-deployment ${AZURE_DEPLOYMENT_GPT52} --sub-deployment ${AZURE_DEPLOYMENT_GPT52} --reasoning-effort high --sub-reasoning-effort high --default-k 20 --fixed-k --max-searches 10 --max-iterations 30 --seed-context --compaction --lean-prologue --no-delegation --solo-prompt --no-orchestrator --compaction-threshold-pct 0.85 --max-concurrent-subcalls 4 --max-timeout 3600 --lm-timeout 1800 --save-trajectory'
ENV_SET='AZURE_OPENAI_API_VERSION=2024-12-01-preview PYTHONPATH=<hooks>/run5d:<RLMLIB_PATH> OBLIQ_CAP_RANKING=1 RLM_USAGE_LOG=<out>/usage.jsonl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false'
ENV_UNSET='SUP_FENCE_TOLERANT SUP_MAX_BLOCKS SUP_PRESENCE_PENALTY'
ORIGINAL_OUTPUT="<FULLDOC_ROOT>/runs/obliq_fd_twitter"
```

## 5. Execution instructions (reproduction, not executed for this documentation)

These commands launch the full experiment: hundreds of model calls and hours of GPU or API time. They were **not** run while preparing this repository.

```bash
set -a; . configs/paths.env; . agents/document_repl/configs/TWI-11.env; set +a
# then the launcher; its header documents its arguments:
agents/document_repl/experiments/obliq_fd.sh
```

Launcher: [agents/document_repl/experiments/obliq_fd.sh](../../../agents/document_repl/experiments/obliq_fd.sh). It reads the benchmark questions, talks to the retrieval server named in the config and writes one record per question (plus trajectories) under `$RUNS_DIR`. Grade afterwards with the matching script in [agents/shared/rlm_shared/grading/](../../../agents/shared/rlm_shared/grading).

## 6. Implementation references

- Agent source: [bcp_bm25_split.py](../../../agents/document_repl/src/bcp_bm25_split.py), [bcp_qwen_dense.py](../../../agents/document_repl/src/bcp_qwen_dense.py), [build_bm25.py](../../../agents/document_repl/src/build_bm25.py), [encoder.py](../../../agents/document_repl/src/encoder.py), [fence_tolerant.py](../../../agents/document_repl/src/fence_tolerant.py), [frames_cap25_report.py](../../../agents/document_repl/src/frames_cap25_report.py), [grade_local.py](../../../agents/document_repl/src/grade_local.py), [hybrid.py](../../../agents/document_repl/src/hybrid.py), [judge.py](../../../agents/document_repl/src/judge.py), [obliq_adapter.py](../../../agents/document_repl/src/obliq_adapter.py), [prepare_data.py](../../../agents/document_repl/src/prepare_data.py), [prompts.py](../../../agents/document_repl/src/prompts.py) …
- Original code that produced the result (verbatim, SHA-256 manifest): [agents/document_repl/original/](../../../agents/document_repl/original/MANIFEST.json)
- Configuration: [agents/document_repl/configs/TWI-11.env](../../../agents/document_repl/configs/TWI-11.env) · Launcher: [agents/document_repl/experiments/obliq_fd.sh](../../../agents/document_repl/experiments/obliq_fd.sh)
- Data loaders: [agents/shared/rlm_shared/](../../../agents/shared/rlm_shared/__init__.py) (`browsecomp_data.py`, `obliq_data.py`)
- Grading: [agents/shared/rlm_shared/grading/](../../../agents/shared/rlm_shared/grading) · Metrics: [metrics.py](../../../agents/shared/rlm_shared/metrics.py)
- Tests: [agents/document_repl/tests/](../../../agents/document_repl/tests), [tests/](../../../tests)
- **Per-question outputs in this repository** (Git LFS): [trajectories/document_repl/TWI-11/](../../../trajectories/document_repl/TWI-11/MANIFEST.json). 285 files, 0.61 GB as written by the run; extract with `python scripts/provenance/unpack_trajectories.py TWI-11 OUT_DIR` ([format notes](../../../trajectories/README.md))
- Original locations on the machine that ran the experiments: `<FULLDOC_ROOT>/runs/obliq_fd_twitter/q*.json`

## 7. Existing experimental results

Reported in Experiments.xlsx (every non-empty column of the row):

| Metric | Value |
|---|---|
| Accuracy | — |
| Recall@10 (G) | 0.0951 |
| NDCG@10 (G / P) | 0.0950 / 0.1800 |
| NDCG@50 (G / P) | 0.1028 / 0.1814 |
| Avg. Search Calls Per Query | 10 |
| Avg. Iterations Per Query | 8.5 |
| Time Per Query | 179 s (3.0 min) |
| Cached Cost Per Query | 0.2298 |
| Uncached Cost Per Query | 0.271 |
| Input Tokens Per Query | 74073 |
| Output Tokens Per Query | 10096 |


**Per-question outcome** (Failure_Analysis.xlsx, OBLIQ Summary):

| Outcome | Queries | Share |
|---|---|---|
| Hit (≥ 1 gold in top 10) | 96 | 34.2% |
| O1 no usable ranking | 1 | 0.4% |
| O2 gold never retrieved | 85 | 30.2% |
| O3 retrieved, not ranked in top 10 | 99 | 35.2% |


Gold documents retrieved at any point: 397 of 2773; placed in the top 10: 134.

## 8. Results interpretation


- Statements about *why* a configuration wins or loses are hypotheses unless the comparison isolates one variable; see the confounds in section 10.

## 9. Verification status

| Check | Result |
|---|---|
| Experiment previously executed | yes |
| Original outputs found | yes (on the original machine; paths in section 6) |
| Configuration identified | yes |
| Implementation reviewed | yes ([agent review](../../verification/REVIEW_document_repl.md)) |
| Lightweight tests | unit tests with mocks ([TESTING_REPORT.md](../../verification/TESTING_REPORT.md)); the experiment itself was not re-run |
| Numbers independently recomputed | **matches the workbook**: NDCG@10 0.0950, Recall@10 0.0951 recomputed from the answer text ([audit](../../verification/RESULT_INTEGRITY.md)) |
| Reproduced by re-running | **no**: no experiment was re-run; full experimental equivalence of the refactored code is not established |


## 10. Limitations

- No random seed; model sampling is stochastic, so a re-run gives different per-question results (two identical 830-question snippet REPL agent runs disagreed on 17.8% of questions in the earlier campaign).
- Reasoning effort actually sent: as requested (run5d hook loaded).
- Binary relevance on gold qrels only (graded relevance ignored).
- Comparisons with other rows can differ in more than one variable (search cap, turn limit, time limit, effort, index document length); see [AGENT_COMPARISON.md](../AGENT_COMPARISON.md).
- The number of documents per search is chosen by the model (up to 50), not fixed at 5 like the snippet REPL agent; the `llm_query` sub-LLM functions remain installed in the REPL (only the prompt says not to use them).

## 11. Related files

- [Experiments.xlsx](../../../results/original_baselines/Experiments.xlsx) · [Failure_Analysis.xlsx](../../../results/original_baselines/Failure_Analysis.xlsx) · [registry.yaml](../../../experiments/registry.yaml)
- [Known issues](../../verification/KNOWN_ISSUES.md) · [Result integrity](../../verification/RESULT_INTEGRITY.md) · [Changes from the original code](../../../agents/document_repl/CHANGES_FROM_ORIGINAL.md)
