# FR-11: Full-Doc RLM · FRAMES · gpt-5 · BM25 (bm25s

Status: complete. Row 11 of the `FRAMES` sheet of [Experiments.xlsx](../../../results/original_baselines/Experiments.xlsx).

## 1. Experiment overview

|  |  |
|---|---|
| Identifier | `FR-11` (sheet `FRAMES`, row 11) |
| Algorithm | [Full-Doc RLM](../../../algorithms/full_doc_rlm/README.md) |
| Benchmark | FRAMES (824 questions, Wikipedia 20230601.en) |
| Model | gpt-5 (Azure) |
| Retriever | BM25 (bm25s, k1 0.9, b 0.4) over Wikipedia |
| Research question | Full-Doc RLM on FRAMES: the counterpart of FR-7. |
| Original execution | 2026-10-27 to 2026-10-28 |
| Run name in the logs | `frames_fd_gpt5` |

## 2. Research motivation

Full-Doc RLM on FRAMES: the counterpart of FR-7. This row is one cell of the comparison matrix in Experiments.xlsx: the same benchmark is run with each algorithm × model × retriever combination, so this row is read against its neighbours (section 8). See [ALGORITHM_COMPARISON.md](../ALGORITHM_COMPARISON.md) for the matrix and [research_overview.md](../../research_overview.md) for the research questions.

## 3. Experimental methodology

The Full-Doc RLM answers each question by writing Python in a persistent REPL. Each search puts the **whole** retrieved documents into a `DOCS` variable; the model reads slices of them itself (prints are capped per code block). Searches are capped client-side; at the turn limit the harness forces a final answer.
 Algorithm details: [algorithms/full_doc_rlm/README.md](../../../algorithms/full_doc_rlm/README.md).

Per question: the agent runs on one benchmark question with the budgets in section 4; the run record (answer, retrieved document ids, tool counts, turns, time, tokens) is written to the run directory.
 Correctness is judged afterwards by gpt-4.1 (Azure deployment gpt-4.1, served gpt-4.1-2025-04-14), official BrowseComp-Plus grader template, 3 votes, majority; a question without a completed answer is wrong. Recall is the share of the question's gold documents retrieved at any point of the run (definitions per row: [RESULT_INTEGRITY.md](../../verification/RESULT_INTEGRITY.md) §R1).

## 4. Configuration

| Setting | Value |
|---|---|
| Model | gpt-5 (Azure) |
| Reasoning effort requested | high |
| Reasoning effort actually sent | as requested (run5d hook loaded) |
| Retriever / index | BM25 (bm25s, k1 0.9, b 0.4) over Wikipedia |
| Turns | 30 + forced final answer |
| Searches | 10 |
| Results per search | model-chosen k (≤ 50) |
| Documents | whole documents into DOCS |
| Timeout | 2,700 s |
| Random seed | none set (API / vLLM sampling defaults); runs are not bit-reproducible |


Exact environment and arguments ([algorithms/full_doc_rlm/configs/FR-11.env](../../../algorithms/full_doc_rlm/configs/FR-11.env)):

```bash
# FR-11 | FRAMES (824) | Full-Doc RLM | GPT-5 (high) | BM25 over Wikipedia 20230601.en on :8041
# model: gpt-5 via Azure deployment AZURE_DEPLOYMENT_GPT5 (original: a personal deployment serving gpt-5-2025-08-07);
#   served_models in the records: null
# effort requested: high; effort actually sent: high -- startup hook run5d (retry429 + azure_sampling + cap_ranking,
#   the last inactive without OBLIQ_CAP_RANKING=1) makes rlm's Azure client send reasoning_effort
# retriever: FRAMES BM25 search server (FRAMES_DATA_DIR index, not part of this algorithm); prompt patched by
#   src/run_bm25.py for Wikipedia (BM25_CORPUS unset; record prompt_id 758e43ed5b95)
# caps: --max-searches 10 (client-side), --max-iterations 30 (+1 forced final answer: 48 of 824 records have
#   n_iterations 31), k model-chosen 1..50 with fallback 5, --max-timeout 2700, xargs timeout 3300
# original output: <original repo>/logs/run5d/full/frames_fd_gpt5 (824 records; config read from 0.json);
#   per-call usage in usage.jsonl (RLM_USAGE_LOG)
ROW_ID=FR-11
BENCHMARK=frames
ENTRY=src/run_bm25.py
LAUNCHER="experiments/frames_gpt5.sh <out_dir> <workers> 0 824"
WORKERS=UNKNOWN
ROOT_MODEL=gpt-5
SUB_MODEL=gpt-5
REASONING_EFFORT=high
SUB_REASONING_EFFORT=high
EFFORT_SENT=high
RETRIEVER="BM25 over Wikipedia 20230601.en (FRAMES_DATA_DIR)"
SERVER=http://127.0.0.1:8041
MAX_SEARCHES=10
MAX_ITERATIONS=30
MAX_TIMEOUT=2700
RUN_ARGS='--design B --server http://127.0.0.1:8041 --root-model gpt-5 --sub-model gpt-5 --root-deployment ${AZURE_DEPLOYMENT_GPT5} --sub-deployment ${AZURE_DEPLOYMENT_GPT5} --reasoning-effort high --sub-reasoning-effort high --default-k 5 --max-iterations 30 --max-searches 10 --seed-context --compaction --grounding-gate --lean-prologue --no-delegation --solo-prompt --no-orchestrator --compaction-threshold-pct 0.85 --max-concurrent-subcalls 4 --max-timeout 2700'
ENV_SET='AZURE_OPENAI_API_VERSION=2024-12-01-preview PYTHONPATH=<hooks>/run5d:<RLMLIB_PATH> QUERIES_FILE=<FRAMES_DATA_DIR>/frames_queries.jsonl RLM_USAGE_LOG=<out>/usage.jsonl'
ENV_UNSET='BM25_CORPUS SUP_FENCE_TOLERANT SUP_MAX_BLOCKS SUP_PRESENCE_PENALTY GEPA_PROMPTS'
ORIGINAL_OUTPUT="<original repo>/logs/run5d/full/frames_fd_gpt5"
```

## 5. Execution instructions (reproduction, not executed for this documentation)

These commands launch the full experiment: hundreds of model calls and hours of GPU or API time. They were **not** run while preparing this repository.

```bash
set -a; . configs/paths.env; . algorithms/full_doc_rlm/configs/FR-11.env; set +a
# then the launcher; its header documents its arguments:
algorithms/full_doc_rlm/experiments/frames_gpt5.sh
```

Launcher: [algorithms/full_doc_rlm/experiments/frames_gpt5.sh](../../../algorithms/full_doc_rlm/experiments/frames_gpt5.sh). It reads the benchmark questions, talks to the retrieval server named in the config and writes one record per question (plus trajectories) under `$RUNS_DIR`. Grade afterwards with the matching script in [shared/rlm_shared/grading/](../../../shared/rlm_shared/grading).

## 6. Implementation references

- Algorithm source: [bcp_bm25_split.py](../../../algorithms/full_doc_rlm/src/bcp_bm25_split.py), [bcp_qwen_dense.py](../../../algorithms/full_doc_rlm/src/bcp_qwen_dense.py), [build_bm25.py](../../../algorithms/full_doc_rlm/src/build_bm25.py), [encoder.py](../../../algorithms/full_doc_rlm/src/encoder.py), [fence_tolerant.py](../../../algorithms/full_doc_rlm/src/fence_tolerant.py), [grade_local.py](../../../algorithms/full_doc_rlm/src/grade_local.py), [hybrid.py](../../../algorithms/full_doc_rlm/src/hybrid.py), [judge.py](../../../algorithms/full_doc_rlm/src/judge.py), [obliq_adapter.py](../../../algorithms/full_doc_rlm/src/obliq_adapter.py), [prepare_data.py](../../../algorithms/full_doc_rlm/src/prepare_data.py), [prompts.py](../../../algorithms/full_doc_rlm/src/prompts.py), [provenance.py](../../../algorithms/full_doc_rlm/src/provenance.py) …
- Original code that produced the result (verbatim, SHA-256 manifest): [algorithms/full_doc_rlm/original/](../../../algorithms/full_doc_rlm/original/MANIFEST.json)
- Configuration: [algorithms/full_doc_rlm/configs/FR-11.env](../../../algorithms/full_doc_rlm/configs/FR-11.env) · Launcher: [algorithms/full_doc_rlm/experiments/frames_gpt5.sh](../../../algorithms/full_doc_rlm/experiments/frames_gpt5.sh)
- Data loaders: [shared/rlm_shared/](../../../shared/rlm_shared/__init__.py) (`browsecomp_data.py`, `obliq_data.py`)
- Grading: [shared/rlm_shared/grading/](../../../shared/rlm_shared/grading) · Metrics: [metrics.py](../../../shared/rlm_shared/metrics.py)
- Tests: [algorithms/full_doc_rlm/tests/](../../../algorithms/full_doc_rlm/tests), [tests/](../../../tests)
- Original outputs (not in this repository; on the machine that ran the experiments): `logs/run5d/full/frames_fd_gpt5/`

## 7. Existing experimental results

Reported in Experiments.xlsx (every non-empty column of the row):

| Metric | Value |
|---|---|
| Accuracy | 0.807 |
| Recall | 0.7039 |
| NDCG | — |
| Avg. Search Calls Per Query | 6.17 |
| Avg. Iterations Per Query | 10.4 |
| Time Per Query | 333 s (5.6 min) |
| Cached Cost Per Query | 0.3537 |
| Uncached Cost Per Query | 0.4161 |
| Input Tokens Per Query | 107126 |
| Output Tokens Per Query | 28214 |


**Failure breakdown** (from [Failure_Analysis.xlsx](../../../results/original_baselines/Failure_Analysis.xlsx), QA Summary; first matching category wins):

| Category | Questions | Share of failures |
|---|---|---|
| F1 did not finish | 0 | 0.0% |
| F2 no / abstained | 4 | 2.5% |
| F3 retrieval miss | 15 | 9.4% |
| F4 partial evidence | 85 | 53.5% |
| F5 all gold, wrong | 55 | 34.6% |
| **Failed in total** | 159 |  |


**Accuracy given the retrieved evidence** (By Recall sheet):

| Evidence retrieved | Questions | Accuracy |
|---|---|---|
| no gold retrieved | 51 | 70.6% |
| some gold | 423 | 79.2% |
| all gold | 350 | 84.0% |


**Turns** (By Turns sheet): limit 30, mean 10.4 (correct 9.5, not correct 14.2); 58 questions used every turn (7.0%), accuracy when answered before the limit 80.5%.

## 8. Results interpretation

**Paired comparisons on the same questions** (Pairwise sheet; exact McNemar test on the discordant pairs):

| Comparison | Other row | Only FR-11 right | Only other right | p | Reading |
|---|---|---|---|---|---|
| Snippet vs full-doc, GPT-5, BM25 (FRAMES) | FR-7 | 74 | 38 | 0.00086 | significant at 0.05 |
| GPT-5 vs Qwen3.6, full-doc (FRAMES) | FR-13 | 184 | 22 | 4.9e-33 | significant at 0.05 |


- **Largest failure mode:** partial evidence (85 of 159 failures, 53.5%). Retrieval-side failures (F3 + F4) are 62.9% of failures; reading-side (F5) 34.6%. *(Measured.)*
- **With all gold evidence retrieved** the run is right 84.0% of the time (350 questions); with none, 70.6%. The gap between the two bounds how much better retrieval alone could help. *(Measured; causal reading is a hypothesis.)*
- Statements about *why* a configuration wins or loses are hypotheses unless the comparison isolates one variable; see the confounds in section 10.

## 9. Verification status

| Check | Result |
|---|---|
| Experiment previously executed | yes |
| Original outputs found | yes (on the original machine; paths in section 6) |
| Configuration identified | yes |
| Implementation reviewed | yes ([algorithm review](../../verification/REVIEW_full_doc_rlm.md)) |
| Lightweight tests | unit tests with mocks ([TESTING_REPORT.md](../../verification/TESTING_REPORT.md)); the experiment itself was not re-run |
| Numbers independently recomputed | **matches the workbook** (accuracy, recall, searches): accuracy 0.8070 from 824 result records ([audit](../../verification/RESULT_INTEGRITY.md)) |
| Reproduced by re-running | **no**: no experiment was re-run; full experimental equivalence of the refactored code is not established |


## 10. Limitations

- No random seed; model sampling is stochastic, so a re-run gives different per-question results (two identical 830-question Snippet RLM runs disagreed on 17.8% of questions in the earlier campaign).
- Reasoning effort actually sent: as requested (run5d hook loaded).
- The judge (gpt-4.1) is an LLM; its verdicts were not audited by hand.
- Comparisons with other rows can differ in more than one variable (search cap, turn limit, time limit, effort, index document length); see [ALGORITHM_COMPARISON.md](../ALGORITHM_COMPARISON.md).
- The number of documents per search is chosen by the model (up to 50), not fixed at 5 like the Snippet RLM; the `llm_query` sub-LLM functions remain installed in the REPL (only the prompt says not to use them).

## 11. Related files

- [Experiments.xlsx](../../../results/original_baselines/Experiments.xlsx) · [Failure_Analysis.xlsx](../../../results/original_baselines/Failure_Analysis.xlsx) · [registry.yaml](../../../experiments/registry.yaml)
- [Known issues](../../verification/KNOWN_ISSUES.md) · [Result integrity](../../verification/RESULT_INTEGRITY.md) · [Changes from the original code](../../../algorithms/full_doc_rlm/CHANGES_FROM_ORIGINAL.md)
