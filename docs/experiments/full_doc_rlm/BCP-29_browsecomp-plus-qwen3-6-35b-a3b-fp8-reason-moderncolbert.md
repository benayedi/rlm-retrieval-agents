# BCP-29: Full-Doc RLM · BrowseComp-Plus · Qwen3.6-35B-A3B-FP8 · Reason-ModernColBERT

Status: complete. Row 29 of the `BrowseComp+` sheet of [Experiments.xlsx](../../../results/original_baselines/Experiments.xlsx).

## 1. Experiment overview

|  |  |
|---|---|
| Identifier | `BCP-29` (sheet `BrowseComp+`, row 29) |
| Algorithm | [Full-Doc RLM](../../../algorithms/full_doc_rlm/README.md) |
| Benchmark | BrowseComp-Plus (830 questions, 100,195 documents) |
| Model | Qwen3.6-35B-A3B-FP8 (vLLM, local) |
| Retriever | Reason-ModernColBERT |
| Research question | Full-Doc RLM with an open-weight model and ColBERT; the unoptimised seed prompt of the GEPA study, evaluated on all 830 questions. |
| Original execution | 2026-10-07 to 2026-10-12 |
| Run name in the logs | `baseline prompt (cache key ee5ea32a1c) collected from gepa_bcp2, bcp_testrest_ctrl and gepa_bcp` |

## 2. Research motivation

Full-Doc RLM with an open-weight model and ColBERT; the unoptimised seed prompt of the GEPA study, evaluated on all 830 questions. This row is one cell of the comparison matrix in Experiments.xlsx: the same benchmark is run with each algorithm × model × retriever combination, so this row is read against its neighbours (section 8). See [ALGORITHM_COMPARISON.md](../ALGORITHM_COMPARISON.md) for the matrix and [research_overview.md](../../research_overview.md) for the research questions.

## 3. Experimental methodology

The Full-Doc RLM answers each question by writing Python in a persistent REPL. Each search puts the **whole** retrieved documents into a `DOCS` variable; the model reads slices of them itself (prints are capped per code block). Searches are capped client-side; at the turn limit the harness forces a final answer.
 Algorithm details: [algorithms/full_doc_rlm/README.md](../../../algorithms/full_doc_rlm/README.md).

Per question: the agent runs on one benchmark question with the budgets in section 4; the run record (answer, retrieved document ids, tool counts, turns, time, tokens) is written to the run directory.
 Correctness is judged afterwards by gpt-4.1 (Azure deployment gpt-4.1, served gpt-4.1-2025-04-14), official BrowseComp-Plus grader template, 3 votes, majority; a question without a completed answer is wrong. Recall is the share of the question's gold documents retrieved at any point of the run (definitions per row: [RESULT_INTEGRITY.md](../../verification/RESULT_INTEGRITY.md) §R1).

## 4. Configuration

| Setting | Value |
|---|---|
| Model | Qwen3.6-35B-A3B-FP8 (vLLM, local) |
| Reasoning effort requested | medium |
| Reasoning effort actually sent | as requested |
| Retriever / index | Reason-ModernColBERT |
| Turns | 30 + forced final answer |
| Searches | 10 |
| Results per search | model-chosen k (≤ 50) |
| Documents | whole documents into DOCS |
| Timeout | 5,400 s |
| Random seed | none set (API / vLLM sampling defaults); runs are not bit-reproducible |


Exact environment and arguments ([algorithms/full_doc_rlm/configs/BCP-29.env](../../../algorithms/full_doc_rlm/configs/BCP-29.env)):

```bash
# BCP-29 | BrowseComp+ (830) | Full-Doc RLM | Qwen3.6-35B-A3B (medium) | Reason-ModernColBERT (PLAID 8192) on :8766
# model: qwen36-35b-a3b served by vLLM on :8030, --backend openai
# effort requested: medium; sent as reasoning_effort=medium (OpenAI client forwards sampling_args); presence_penalty 1.5
# retriever: src/retrieval_server_colbert.py on :8766 (record prompt_id 94f31e91f6b7)
# caps: --max-searches 10 (client-side), --max-iterations 30 (+1 forced final answer), k model-chosen 1..50 with
#   fallback 5, --max-timeout 5400, --lm-timeout 1800, subprocess kill at 6300 s; SUP_MAX_BLOCKS=5
# split runner: src/split_runner.py with the seed prompt; FINGERPRINT 29587b4e, cache key ee5ea32a1c (reproduced by
#   tests/test_cache_keys.py). Records under the same key in three directories:
#     gepa_bcp/        (GEPA baseline on pareto + --seed-split feedback; 300 records)
#     gepa_bcp2/       (test200; 199 records)          bcp_testrest_ctrl/ (--seed-split test_rest; 322 records)
#   Worker counts per invocation: 20 for gepa_bcp; UNKNOWN for the other two.
# original output: <original repo>/logs/{gepa_bcp2, bcp_testrest_ctrl, gepa_bcp} (config read from one record each)
ROW_ID=BCP-29
BENCHMARK=browsecomp_plus
ENTRY=src/split_runner.py
LAUNCHER="GEPA_OUT=<dir> GEPA_SPLIT=<split.json> python src/split_runner.py --seed-split <feedback|test_rest|...> --workers 20"
CACHE_KEY=ee5ea32a1c
FINGERPRINT=29587b4e
ROOT_MODEL=qwen36-35b-a3b
SUB_MODEL=qwen36-35b-a3b
REASONING_EFFORT=medium
SUB_REASONING_EFFORT=medium
EFFORT_SENT=medium
RETRIEVER="Reason-ModernColBERT PLAID 8192 (src/retrieval_server_colbert.py)"
SERVER=http://127.0.0.1:8766
MAX_SEARCHES=10
MAX_ITERATIONS=30
MAX_TIMEOUT=5400
GEPA_BCP_SERVERS=http://localhost:8030/v1
GEPA_BCP_MODEL=qwen36-35b-a3b
SUP_FENCE_TOLERANT=1
SUP_MAX_BLOCKS=5
SUP_PRESENCE_PENALTY=1.5
RUN_ARGS='--design B --server http://127.0.0.1:8766 --backend openai --api-key-env ALIBABA_TOKEN --root-model qwen36-35b-a3b --sub-model qwen36-35b-a3b --reasoning-effort medium --sub-reasoning-effort medium --max-iterations 30 --max-searches 10 --seed-context --compaction --grounding-gate --lean-prologue --no-delegation --solo-prompt --no-orchestrator --compaction-threshold-pct 0.85 --max-concurrent-subcalls 4 --max-timeout 5400 --lm-timeout 1800 --save-trajectory'
ENV_UNSET='GEPA_PROMPTS'
ORIGINAL_OUTPUT="<original repo>/logs/{gepa_bcp2,bcp_testrest_ctrl,gepa_bcp}"
```

## 5. Execution instructions (reproduction, not executed for this documentation)

These commands launch the full experiment: hundreds of model calls and hours of GPU or API time. They were **not** run while preparing this repository.

```bash
set -a; . configs/paths.env; . algorithms/full_doc_rlm/configs/BCP-29.env; set +a
# then the launcher; its header documents its arguments:
algorithms/full_doc_rlm/src/split_runner.py
```

Launcher: [algorithms/full_doc_rlm/src/split_runner.py](../../../algorithms/full_doc_rlm/src/split_runner.py). It reads the benchmark questions, talks to the retrieval server named in the config and writes one record per question (plus trajectories) under `$RUNS_DIR`. Grade afterwards with the matching script in [shared/rlm_shared/grading/](../../../shared/rlm_shared/grading).

## 6. Implementation references

- Algorithm source: [bcp_bm25_split.py](../../../algorithms/full_doc_rlm/src/bcp_bm25_split.py), [bcp_qwen_dense.py](../../../algorithms/full_doc_rlm/src/bcp_qwen_dense.py), [build_bm25.py](../../../algorithms/full_doc_rlm/src/build_bm25.py), [encoder.py](../../../algorithms/full_doc_rlm/src/encoder.py), [fence_tolerant.py](../../../algorithms/full_doc_rlm/src/fence_tolerant.py), [grade_local.py](../../../algorithms/full_doc_rlm/src/grade_local.py), [hybrid.py](../../../algorithms/full_doc_rlm/src/hybrid.py), [judge.py](../../../algorithms/full_doc_rlm/src/judge.py), [obliq_adapter.py](../../../algorithms/full_doc_rlm/src/obliq_adapter.py), [prepare_data.py](../../../algorithms/full_doc_rlm/src/prepare_data.py), [prompts.py](../../../algorithms/full_doc_rlm/src/prompts.py), [provenance.py](../../../algorithms/full_doc_rlm/src/provenance.py) …
- Original code that produced the result (verbatim, SHA-256 manifest): [algorithms/full_doc_rlm/original/](../../../algorithms/full_doc_rlm/original/MANIFEST.json)
- Configuration: [algorithms/full_doc_rlm/configs/BCP-29.env](../../../algorithms/full_doc_rlm/configs/BCP-29.env) · Launcher: [algorithms/full_doc_rlm/src/split_runner.py](../../../algorithms/full_doc_rlm/src/split_runner.py)
- Data loaders: [shared/rlm_shared/](../../../shared/rlm_shared/__init__.py) (`browsecomp_data.py`, `obliq_data.py`)
- Grading: [shared/rlm_shared/grading/](../../../shared/rlm_shared/grading) · Metrics: [metrics.py](../../../shared/rlm_shared/metrics.py)
- Tests: [algorithms/full_doc_rlm/tests/](../../../algorithms/full_doc_rlm/tests), [tests/](../../../tests)
- Original outputs (not in this repository; on the machine that ran the experiments): `logs/{gepa_bcp2,bcp_testrest_ctrl,gepa_bcp}/cache.json and rollouts/ee5ea32a1c/`; `logs/bcp_rejudge_gpt41/16.json (gpt-4.1 verdicts)`

## 7. Existing experimental results

Reported in Experiments.xlsx (every non-empty column of the row):

| Metric | Value |
|---|---|
| Accuracy | 0.5795 |
| Recall | 0.5123 |
| NDCG | — |
| Avg. Search Calls Per Query | 8.6 |
| Avg. Iterations Per Query | 20.9 |
| Time Per Query | 1829 s (30.5 min) |
| Cached Cost Per Query | local GPU (no API cost) |
| Uncached Cost Per Query | local GPU (no API cost) |
| Input Tokens Per Query | 335994 |
| Output Tokens Per Query | 30906 |


Workbook footnotes that apply to this row:

> Input / Output Tokens Per Query: all model calls of a question (root + sub-LLM), reasoning tokens included in output; these do not depend on how many runs shared a server, unlike Time Per Query. Exact from logged per-call usage, except values marked ≈ (Sept. 2026 snippet runs, whose logs overwrote the root model's usage with the sub-LLM's: rebuilt from the trajectories, about 10% low when the same method is checked on a run with exact usage). Uncached cost = input x list input price + output x list output price. Avg. Iterations Per Query: root-model turns per question, counted from the trajectory logs (snippet RLM) or the run records (full-doc RLM); sub-LLM calls are not counted. A run that reaches its turn cap adds one forced final-answer turn.


**Failure breakdown** (from [Failure_Analysis.xlsx](../../../results/original_baselines/Failure_Analysis.xlsx), QA Summary; first matching category wins):

| Category | Questions | Share of failures |
|---|---|---|
| F1 did not finish | 53 | 15.2% |
| F2 no / abstained | 10 | 2.9% |
| F3 retrieval miss | 203 | 58.2% |
| F4 partial evidence | 51 | 14.6% |
| F5 all gold, wrong | 32 | 9.2% |
| **Failed in total** | 349 |  |


**Accuracy given the retrieved evidence** (By Recall sheet):

| Evidence retrieved | Questions | Accuracy |
|---|---|---|
| no gold retrieved | 229 | 7.4% |
| some gold | 258 | 79.8% |
| all gold | 290 | 89.0% |


**Turns** (By Turns sheet): limit 30, mean 20.9 (correct 16.4, not correct 27.3); 272 questions used every turn (32.8%), accuracy when answered before the limit 84.6%.

## 8. Results interpretation

**Paired comparisons on the same questions** (Pairwise sheet; exact McNemar test on the discordant pairs):

| Comparison | Other row | Only BCP-29 right | Only other right | p | Reading |
|---|---|---|---|---|---|
| GPT-5 vs Qwen3.6, full-doc, Reason-ModernColBERT | BCP-23 | 17 | 262 | 1.4e-57 | significant at 0.05 |


- **Largest failure mode:** never retrieving a gold document (203 of 349 failures, 58.2%). Retrieval-side failures (F3 + F4) are 72.8% of failures; reading-side (F5) 9.2%. *(Measured.)*
- **With all gold evidence retrieved** the run is right 89.0% of the time (290 questions); with none, 7.4%. The gap between the two bounds how much better retrieval alone could help. *(Measured; causal reading is a hypothesis.)*
- Statements about *why* a configuration wins or loses are hypotheses unless the comparison isolates one variable; see the confounds in section 10.

## 9. Verification status

| Check | Result |
|---|---|
| Experiment previously executed | yes |
| Original outputs found | yes (on the original machine; paths in section 6) |
| Configuration identified | yes |
| Implementation reviewed | yes ([algorithm review](../../verification/REVIEW_full_doc_rlm.md)) |
| Lightweight tests | unit tests with mocks ([TESTING_REPORT.md](../../verification/TESTING_REPORT.md)); the experiment itself was not re-run |
| Numbers independently recomputed | **matches the workbook** (accuracy, recall, searches): accuracy 0.5795 from 777 result records ([audit](../../verification/RESULT_INTEGRITY.md)) |
| Reproduced by re-running | **no**: no experiment was re-run; full experimental equivalence of the refactored code is not established |


## 10. Limitations

- No random seed; model sampling is stochastic, so a re-run gives different per-question results (two identical 830-question Snippet RLM runs disagreed on 17.8% of questions in the earlier campaign).
- Reasoning effort actually sent: as requested.
- The judge (gpt-4.1) is an LLM; its verdicts were not audited by hand.
- Comparisons with other rows can differ in more than one variable (search cap, turn limit, time limit, effort, index document length); see [ALGORITHM_COMPARISON.md](../ALGORITHM_COMPARISON.md).
- The number of documents per search is chosen by the model (up to 50), not fixed at 5 like the Snippet RLM; the `llm_query` sub-LLM functions remain installed in the REPL (only the prompt says not to use them).

## 11. Related files

- [Experiments.xlsx](../../../results/original_baselines/Experiments.xlsx) · [Failure_Analysis.xlsx](../../../results/original_baselines/Failure_Analysis.xlsx) · [registry.yaml](../../../experiments/registry.yaml)
- [Known issues](../../verification/KNOWN_ISSUES.md) · [Result integrity](../../verification/RESULT_INTEGRITY.md) · [Changes from the original code](../../../algorithms/full_doc_rlm/CHANGES_FROM_ORIGINAL.md)
