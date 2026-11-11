# BCP-31: Document REPL agent · BrowseComp-Plus · Qwen3.6-35B-A3B-FP8 · BM25 (bm25s

Status: complete. Row 31 of the `BrowseComp+` sheet of [Experiments.xlsx](../../../results/original_baselines/Experiments.xlsx).

## 1. Experiment overview

|  |  |
|---|---|
| Identifier | `BCP-31` (sheet `BrowseComp+`, row 31) |
| Agent | [Document REPL agent](../../../agents/document_repl/README.md) |
| Benchmark | BrowseComp-Plus (830 questions, 100,195 documents) |
| Model | Qwen3.6-35B-A3B-FP8 (vLLM, local) |
| Retriever | BM25 (bm25s, k1 0.9, b 0.4 per Experiments.xlsx) |
| Research question | Document REPL agent with an open-weight model and a lexical retriever. |
| Original execution | 2026-10-16 to 2026-10-18 |
| Run name in the logs | `cache key 11d0c5bc6b over bcp_bm25_test200, bcp_bm25_rest_gpu0, bcp_bm25_rest_gpu1` |

## 2. Research motivation

Document REPL agent with an open-weight model and a lexical retriever. This row is one cell of the comparison matrix in Experiments.xlsx: the same benchmark is run with each agent × model × retriever combination, so this row is read against its neighbours (section 8). See [AGENT_COMPARISON.md](../AGENT_COMPARISON.md) for the matrix and [research_overview.md](../../research_overview.md) for the research questions.

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
| Retriever / index | BM25 (bm25s, k1 0.9, b 0.4 per Experiments.xlsx) |
| Turns | 30 + forced final answer |
| Searches | 10 |
| Results per search | model-chosen k (≤ 50) |
| Documents | whole documents into DOCS |
| Timeout | 5,400 s |
| Random seed | none set (API / vLLM sampling defaults); runs are not bit-reproducible |


Exact environment and arguments ([agents/document_repl/configs/BCP-31.env](../../../agents/document_repl/configs/BCP-31.env)):

```bash
# BCP-31 | BrowseComp+ (830) | Document REPL agent | Qwen3.6-35B-A3B (medium) | BM25 (default Lucene k1 0.9, b 0.4) on :8043
# model: qwen36-35b-a3b served by vLLM (:8030 for test200 and rest_gpu0, :8031 for rest_gpu1), --backend openai
# effort requested: medium; sent as reasoning_effort=medium (OpenAI client forwards sampling_args); presence_penalty 1.5
# retriever: BM25 search server on :8043 (not part of this agent); prompt patched by src/run_bm25.py with
#   BM25_CORPUS=bcp (record prompt_id 758e43ed5b95)
# caps: --max-searches 10 (client-side), --max-iterations 30 (+1 forced final answer), k model-chosen 1..50 with
#   fallback 5, --max-timeout 5400, --lm-timeout 1800, subprocess kill at 6300 s; SUP_MAX_BLOCKS=5
# split runner: src/bcp_bm25_split.py -> split_runner; FINGERPRINT aa1d1029, cache key 11d0c5bc6b (reproduced by
#   tests/test_cache_keys.py). Three slices: test200 (8 workers, shuffled), rest_gpu0 = 230 ids from
#   bcp_gepa_splits/bm25_rest_gpu0.txt on :8030, rest_gpu1 = 400 ids from bm25_rest_gpu1.txt on :8031 (8 workers each).
# original output: <original repo>/logs/{bcp_bm25_test200, bcp_bm25_rest_gpu0, bcp_bm25_rest_gpu1}
#   (198 / 225 / 395 records; config read from one record each)
ROW_ID=BCP-31
BENCHMARK=browsecomp_plus
ENTRY=src/bcp_bm25_split.py
LAUNCHER="python src/bcp_bm25_split.py ; BCP_IDS=<ids.txt> BCP_OUT=<dir> BCP_SERVER=<vLLM URL> python src/bcp_bm25_split.py"
CACHE_KEY=11d0c5bc6b
FINGERPRINT=aa1d1029
ROOT_MODEL=qwen36-35b-a3b
SUB_MODEL=qwen36-35b-a3b
REASONING_EFFORT=medium
SUB_REASONING_EFFORT=medium
EFFORT_SENT=medium
RETRIEVER="BM25 k1 0.9 b 0.4 (BCP_BM25_INDEX_DIR)"
SERVER=http://127.0.0.1:8043
MAX_SEARCHES=10
MAX_ITERATIONS=30
MAX_TIMEOUT=5400
GEPA_BCP_RETRIEVAL=http://127.0.0.1:8043
GEPA_BCP_MODEL=qwen36-35b-a3b
LOCAL_JUDGE_MODEL=qwen36-35b-a3b
SUP_FENCE_TOLERANT=1
SUP_MAX_BLOCKS=5
SUP_PRESENCE_PENALTY=1.5
BM25_CORPUS=bcp
RUN_ARGS='--design B --server http://127.0.0.1:8043 --backend openai --api-key-env ALIBABA_TOKEN --root-model qwen36-35b-a3b --sub-model qwen36-35b-a3b --reasoning-effort medium --sub-reasoning-effort medium --max-iterations 30 --max-searches 10 --seed-context --compaction --grounding-gate --lean-prologue --no-delegation --solo-prompt --no-orchestrator --compaction-threshold-pct 0.85 --max-concurrent-subcalls 4 --max-timeout 5400 --lm-timeout 1800 --save-trajectory'
ENV_SET='OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false; rollouts: GEPA_PROMPTS=<seed blocks> PYTHONPATH=<RLMLIB_PATH> ALIBABA_TOKEN=EMPTY'
ENV_UNSET='QUERIES_FILE'
ORIGINAL_OUTPUT="<original repo>/logs/{bcp_bm25_test200,bcp_bm25_rest_gpu0,bcp_bm25_rest_gpu1}"
```

## 5. Execution instructions (reproduction, not executed for this documentation)

These commands launch the full experiment: hundreds of model calls and hours of GPU or API time. They were **not** run while preparing this repository.

```bash
set -a; . configs/paths.env; . agents/document_repl/configs/BCP-31.env; set +a
# then the launcher; its header documents its arguments:
agents/document_repl/src/bcp_bm25_split.py
```

Launcher: [agents/document_repl/src/bcp_bm25_split.py](../../../agents/document_repl/src/bcp_bm25_split.py). It reads the benchmark questions, talks to the retrieval server named in the config and writes one record per question (plus trajectories) under `$RUNS_DIR`. Grade afterwards with the matching script in [agents/shared/rlm_shared/grading/](../../../agents/shared/rlm_shared/grading).

## 6. Implementation references

- Agent source: [bcp_bm25_split.py](../../../agents/document_repl/src/bcp_bm25_split.py), [bcp_qwen_dense.py](../../../agents/document_repl/src/bcp_qwen_dense.py), [build_bm25.py](../../../agents/document_repl/src/build_bm25.py), [encoder.py](../../../agents/document_repl/src/encoder.py), [fence_tolerant.py](../../../agents/document_repl/src/fence_tolerant.py), [frames_cap25_report.py](../../../agents/document_repl/src/frames_cap25_report.py), [grade_local.py](../../../agents/document_repl/src/grade_local.py), [hybrid.py](../../../agents/document_repl/src/hybrid.py), [judge.py](../../../agents/document_repl/src/judge.py), [obliq_adapter.py](../../../agents/document_repl/src/obliq_adapter.py), [prepare_data.py](../../../agents/document_repl/src/prepare_data.py), [prompts.py](../../../agents/document_repl/src/prompts.py) …
- Original code that produced the result (verbatim, SHA-256 manifest): [agents/document_repl/original/](../../../agents/document_repl/original/MANIFEST.json)
- Configuration: [agents/document_repl/configs/BCP-31.env](../../../agents/document_repl/configs/BCP-31.env) · Launcher: [agents/document_repl/src/bcp_bm25_split.py](../../../agents/document_repl/src/bcp_bm25_split.py)
- Data loaders: [agents/shared/rlm_shared/](../../../agents/shared/rlm_shared/__init__.py) (`browsecomp_data.py`, `obliq_data.py`)
- Grading: [agents/shared/rlm_shared/grading/](../../../agents/shared/rlm_shared/grading) · Metrics: [metrics.py](../../../agents/shared/rlm_shared/metrics.py)
- Tests: [agents/document_repl/tests/](../../../agents/document_repl/tests), [tests/](../../../tests)
- Per-question outputs: **not in this repository** (see [trajectories/README.md](../../../trajectories/README.md) for why)
- Original locations on the machine that ran the experiments: `logs/{bcp_bm25_test200,bcp_bm25_rest_gpu0,bcp_bm25_rest_gpu1}/cache.json`; `logs/bcp_rejudge_gpt41/18.json`

## 7. Existing experimental results

Reported in Experiments.xlsx (every non-empty column of the row):

| Metric | Value |
|---|---|
| Accuracy | 0.2241 |
| Recall | 0.1806 |
| NDCG | — |
| Avg. Search Calls Per Query | 9.7 |
| Avg. Iterations Per Query | 27.3 |
| Time Per Query | 1388 s (23.1 min) |
| Cached Cost Per Query | local GPU (no API cost) |
| Uncached Cost Per Query | local GPU (no API cost) |
| Input Tokens Per Query | 617614 |
| Output Tokens Per Query | 58153 |


Workbook footnotes that apply to this row:

> Input / Output Tokens Per Query: all model calls of a question (root + sub-LLM), reasoning tokens included in output; these do not depend on how many runs shared a server, unlike Time Per Query. Exact from logged per-call usage, except values marked ≈ (Sept. 2026 snippet runs, whose logs overwrote the root model's usage with the sub-LLM's: rebuilt from the trajectories, about 10% low when the same method is checked on a run with exact usage). Uncached cost = input x list input price + output x list output price. Avg. Iterations Per Query: root-model turns per question, counted from the trajectory logs (snippet RLM) or the run records (document REPL agent); sub-LLM calls are not counted. A run that reaches its turn cap adds one forced final-answer turn.


**Failure breakdown** (from [Failure_Analysis.xlsx](../../../results/original_baselines/Failure_Analysis.xlsx), QA Summary; first matching category wins):

| Category | Questions | Share of failures |
|---|---|---|
| F1 did not finish | 18 | 2.8% |
| F2 no / abstained | 37 | 5.7% |
| F3 retrieval miss | 543 | 84.3% |
| F4 partial evidence | 35 | 5.4% |
| F5 all gold, wrong | 11 | 1.7% |
| **Failed in total** | 644 |  |


**Accuracy given the retrieved evidence** (By Recall sheet):

| Evidence retrieved | Questions | Accuracy |
|---|---|---|
| no gold retrieved | 596 | 2.9% |
| some gold | 118 | 69.5% |
| all gold | 98 | 88.8% |


**Turns** (By Turns sheet): limit 30, mean 27.3 (correct 18.7, not correct 29.8); 583 questions used every turn (70.2%), accuracy when answered before the limit 66.8%.

## 8. Results interpretation

**Paired comparisons on the same questions** (Pairwise sheet; exact McNemar test on the discordant pairs):

| Comparison | Other row | Only BCP-31 right | Only other right | p | Reading |
|---|---|---|---|---|---|
| GPT-5 vs Qwen3.6, document REPL, BM25 | BCP-25 | 20 | 332 | 4.7e-74 | significant at 0.05 |


- **Largest failure mode:** never retrieving a gold document (543 of 644 failures, 84.3%). Retrieval-side failures (F3 + F4) are 89.8% of failures; reading-side (F5) 1.7%. *(Measured.)*
- **With all gold evidence retrieved** the run is right 88.8% of the time (98 questions); with none, 2.9%. The gap between the two bounds how much better retrieval alone could help. *(Measured; causal reading is a hypothesis.)*
- Statements about *why* a configuration wins or loses are hypotheses unless the comparison isolates one variable; see the confounds in section 10.

## 9. Verification status

| Check | Result |
|---|---|
| Experiment previously executed | yes |
| Original outputs found | yes (on the original machine; paths in section 6) |
| Configuration identified | yes |
| Implementation reviewed | yes ([agent review](../../verification/REVIEW_document_repl.md)) |
| Lightweight tests | unit tests with mocks ([TESTING_REPORT.md](../../verification/TESTING_REPORT.md)); the experiment itself was not re-run |
| Numbers independently recomputed | **matches the workbook** (accuracy, recall, searches): accuracy 0.2241 from 812 result records ([audit](../../verification/RESULT_INTEGRITY.md)) |
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
