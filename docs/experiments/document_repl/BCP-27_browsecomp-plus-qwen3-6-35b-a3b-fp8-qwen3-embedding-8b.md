# BCP-27: Document REPL agent · BrowseComp-Plus · Qwen3.6-35B-A3B-FP8 · Qwen3-Embedding-8B

Status: complete. Row 27 of the `BrowseComp+` sheet of [Experiments.xlsx](../../../results/original_baselines/Experiments.xlsx).

## 1. Experiment overview

|  |  |
|---|---|
| Identifier | `BCP-27` (sheet `BrowseComp+`, row 27) |
| Agent | [Document REPL agent](../../../agents/document_repl/README.md) |
| Benchmark | BrowseComp-Plus (830 questions, 100,195 documents) |
| Model | Qwen3.6-35B-A3B-FP8 (vLLM, local) |
| Retriever | Qwen3-Embedding-8B, exact dense (document REPL retrieval server, port 8765) |
| Research question | Document REPL agent with an open-weight model and a dense retriever. |
| Original execution | 2026-10-27 to 2026-10-29 |
| Run name in the logs | `bcp_fd_qwen_dense (cache key e925f0f6c7)` |

## 2. Research motivation

Document REPL agent with an open-weight model and a dense retriever. This row is one cell of the comparison matrix in Experiments.xlsx: the same benchmark is run with each agent × model × retriever combination, so this row is read against its neighbours (section 8). See [AGENT_COMPARISON.md](../AGENT_COMPARISON.md) for the matrix and [research_overview.md](../../research_overview.md) for the research questions.

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
| Retriever / index | Qwen3-Embedding-8B, exact dense (document REPL retrieval server, port 8765) |
| Turns | 30 + forced final answer |
| Searches | 10 |
| Results per search | model-chosen k (≤ 50) |
| Documents | whole documents into DOCS |
| Timeout | 5,400 s harness limit; 6,300 s runaway kill |
| Random seed | none set (API / vLLM sampling defaults); runs are not bit-reproducible |


Exact environment and arguments ([agents/document_repl/configs/BCP-27.env](../../../agents/document_repl/configs/BCP-27.env)):

```bash
# BCP-27 | BrowseComp+ (830) | Document REPL agent | Qwen3.6-35B-A3B (medium) | Qwen3-Embedding-8B dense on :8765
# model: qwen36-35b-a3b served by vLLM on :8030 and :8031 (alternating), --backend openai
# effort requested: medium; sent as the request field reasoning_effort=medium (rlm's OpenAI client forwards
#   sampling_args); presence_penalty 1.5 (SUP_PRESENCE_PENALTY). Whether vLLM's Qwen3.6 template honours
#   reasoning_effort is not verified.
# retriever: src/retrieval_server.py on :8765 (record prompt_id 94f31e91f6b7 = unpatched dense prompt)
# caps: --max-searches 10 (client-side), --max-iterations 30 (+1 forced final answer), k model-chosen 1..50 with
#   fallback 5, --max-timeout 5400, --lm-timeout 1800, subprocess kill at 6300 s; SUP_MAX_BLOCKS=5
# split runner: src/bcp_qwen_dense.py -> split_runner.run_batch(seed); FINGERPRINT 9e6ce99e, cache key e925f0f6c7
#   (reproduced by tests/test_cache_keys.py); interim grade grade_local on Qwen3.6, final grade gpt-4.1
# original output: <original repo>/logs/run5d/bcp_fd_qwen_dense (cache.json; rollouts/e925f0f6c7/, 817 records;
#   config read from rollouts/e925f0f6c7/1.json)
ROW_ID=BCP-27
BENCHMARK=browsecomp_plus
ENTRY=src/bcp_qwen_dense.py
LAUNCHER="python src/bcp_qwen_dense.py <workers>"
WORKERS=UNKNOWN
CACHE_KEY=e925f0f6c7
FINGERPRINT=9e6ce99e
ROOT_MODEL=qwen36-35b-a3b
SUB_MODEL=qwen36-35b-a3b
REASONING_EFFORT=medium
SUB_REASONING_EFFORT=medium
EFFORT_SENT=medium
RETRIEVER="Qwen3-Embedding-8B dense (src/retrieval_server.py)"
SERVER=http://127.0.0.1:8765
MAX_SEARCHES=10
MAX_ITERATIONS=30
MAX_TIMEOUT=5400
GEPA_BCP_SERVERS=http://localhost:8030/v1,http://localhost:8031/v1
GEPA_BCP_RETRIEVAL=http://127.0.0.1:8765
GEPA_BCP_MODEL=qwen36-35b-a3b
LOCAL_JUDGE_MODEL=qwen36-35b-a3b
SUP_FENCE_TOLERANT=1
SUP_MAX_BLOCKS=5
SUP_PRESENCE_PENALTY=1.5
RUN_ARGS='--design B --server http://127.0.0.1:8765 --backend openai --api-key-env ALIBABA_TOKEN --root-model qwen36-35b-a3b --sub-model qwen36-35b-a3b --reasoning-effort medium --sub-reasoning-effort medium --max-iterations 30 --max-searches 10 --seed-context --compaction --grounding-gate --lean-prologue --no-delegation --solo-prompt --no-orchestrator --compaction-threshold-pct 0.85 --max-concurrent-subcalls 4 --max-timeout 5400 --lm-timeout 1800 --save-trajectory'
ENV_SET='OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false; rollouts: GEPA_PROMPTS=<seed blocks> PYTHONPATH=<RLMLIB_PATH> ALIBABA_TOKEN=EMPTY'
ENV_UNSET='QUERIES_FILE GEPA_PROMPTS BM25_CORPUS'
ORIGINAL_OUTPUT="<original repo>/logs/run5d/bcp_fd_qwen_dense"
```

## 5. Execution instructions (reproduction, not executed for this documentation)

These commands launch the full experiment: hundreds of model calls and hours of GPU or API time. They were **not** run while preparing this repository.

```bash
set -a; . configs/paths.env; . agents/document_repl/configs/BCP-27.env; set +a
# then the launcher; its header documents its arguments:
agents/document_repl/src/bcp_qwen_dense.py
```

Launcher: [agents/document_repl/src/bcp_qwen_dense.py](../../../agents/document_repl/src/bcp_qwen_dense.py). It reads the benchmark questions, talks to the retrieval server named in the config and writes one record per question (plus trajectories) under `$RUNS_DIR`. Grade afterwards with the matching script in [agents/shared/rlm_shared/grading/](../../../agents/shared/rlm_shared/grading).

## 6. Implementation references

- Agent source: [bcp_bm25_split.py](../../../agents/document_repl/src/bcp_bm25_split.py), [bcp_qwen_dense.py](../../../agents/document_repl/src/bcp_qwen_dense.py), [build_bm25.py](../../../agents/document_repl/src/build_bm25.py), [encoder.py](../../../agents/document_repl/src/encoder.py), [fence_tolerant.py](../../../agents/document_repl/src/fence_tolerant.py), [frames_cap25_report.py](../../../agents/document_repl/src/frames_cap25_report.py), [grade_local.py](../../../agents/document_repl/src/grade_local.py), [hybrid.py](../../../agents/document_repl/src/hybrid.py), [judge.py](../../../agents/document_repl/src/judge.py), [obliq_adapter.py](../../../agents/document_repl/src/obliq_adapter.py), [prepare_data.py](../../../agents/document_repl/src/prepare_data.py), [prompts.py](../../../agents/document_repl/src/prompts.py) …
- Original code that produced the result (verbatim, SHA-256 manifest): [agents/document_repl/original/](../../../agents/document_repl/original/MANIFEST.json)
- Configuration: [agents/document_repl/configs/BCP-27.env](../../../agents/document_repl/configs/BCP-27.env) · Launcher: [agents/document_repl/src/bcp_qwen_dense.py](../../../agents/document_repl/src/bcp_qwen_dense.py)
- Data loaders: [agents/shared/rlm_shared/](../../../agents/shared/rlm_shared/__init__.py) (`browsecomp_data.py`, `obliq_data.py`)
- Grading: [agents/shared/rlm_shared/grading/](../../../agents/shared/rlm_shared/grading) · Metrics: [metrics.py](../../../agents/shared/rlm_shared/metrics.py)
- Tests: [agents/document_repl/tests/](../../../agents/document_repl/tests), [tests/](../../../tests)
- **Per-question outputs in this repository** (Git LFS): [trajectories/document_repl/BCP-27/](../../../trajectories/document_repl/BCP-27/MANIFEST.json). 820 files, 25.72 GB as written by the run, encrypted with the BrowseComp canary key; extract with `python scripts/provenance/unpack_trajectories.py BCP-27 OUT_DIR` ([format notes](../../../trajectories/README.md))
- Original locations on the machine that ran the experiments: `logs/run5d/bcp_fd_qwen_dense/ (cache.json, rollouts/e925f0f6c7/, graded_gpt41.json)`

## 7. Existing experimental results

Reported in Experiments.xlsx (every non-empty column of the row):

| Metric | Value |
|---|---|
| Accuracy | 0.4422 |
| Recall | 0.3973 |
| NDCG | — |
| Avg. Search Calls Per Query | 8.66 |
| Avg. Iterations Per Query | 22.4 |
| Time Per Query | 2439 s (40.6 min) |
| Cached Cost Per Query | local GPU (no API cost) |
| Uncached Cost Per Query | local GPU (no API cost) |
| Input Tokens Per Query | 392252 |
| Output Tokens Per Query | 36558 |


Workbook footnotes that apply to this row:

> Input / Output Tokens Per Query: all model calls of a question (root + sub-LLM), reasoning tokens included in output; these do not depend on how many runs shared a server, unlike Time Per Query. Exact from logged per-call usage, except values marked ≈ (Sept. 2026 snippet runs, whose logs overwrote the root model's usage with the sub-LLM's: rebuilt from the trajectories, about 10% low when the same method is checked on a run with exact usage). Uncached cost = input x list input price + output x list output price. Avg. Iterations Per Query: root-model turns per question, counted from the trajectory logs (snippet RLM) or the run records (document REPL agent); sub-LLM calls are not counted. A run that reaches its turn cap adds one forced final-answer turn.


**Failure breakdown** (from [Failure_Analysis.xlsx](../../../results/original_baselines/Failure_Analysis.xlsx), QA Summary; first matching category wins):

| Category | Questions | Share of failures |
|---|---|---|
| F1 did not finish | 121 | 26.1% |
| F2 no / abstained | 15 | 3.2% |
| F3 retrieval miss | 260 | 56.2% |
| F4 partial evidence | 44 | 9.5% |
| F5 all gold, wrong | 23 | 5.0% |
| **Failed in total** | 463 |  |


**Accuracy given the retrieved evidence** (By Recall sheet):

| Evidence retrieved | Questions | Accuracy |
|---|---|---|
| no gold retrieved | 291 | 6.2% |
| some gold | 194 | 76.8% |
| all gold | 224 | 89.3% |


**Turns** (By Turns sheet): limit 30, mean 22.4 (correct 16.6, not correct 27.2); 321 questions used every turn (38.7%), accuracy when answered before the limit 82.5%.

## 8. Results interpretation

**Paired comparisons on the same questions** (Pairwise sheet; exact McNemar test on the discordant pairs):

| Comparison | Other row | Only BCP-27 right | Only other right | p | Reading |
|---|---|---|---|---|---|
| GPT-5 vs Qwen3.6, document REPL, Qwen3-Embed-8B | BCP-21 | 9 | 332 | 7.1e-86 | significant at 0.05 |


- **Largest failure mode:** never retrieving a gold document (260 of 463 failures, 56.2%). Retrieval-side failures (F3 + F4) are 65.7% of failures; reading-side (F5) 5.0%. *(Measured.)*
- **With all gold evidence retrieved** the run is right 89.3% of the time (224 questions); with none, 6.2%. The gap between the two bounds how much better retrieval alone could help. *(Measured; causal reading is a hypothesis.)*
- Statements about *why* a configuration wins or loses are hypotheses unless the comparison isolates one variable; see the confounds in section 10.

## 9. Verification status

| Check | Result |
|---|---|
| Experiment previously executed | yes |
| Original outputs found | yes (on the original machine; paths in section 6) |
| Configuration identified | yes |
| Implementation reviewed | yes ([agent review](../../verification/REVIEW_document_repl.md)) |
| Lightweight tests | unit tests with mocks ([TESTING_REPORT.md](../../verification/TESTING_REPORT.md)); the experiment itself was not re-run |
| Numbers independently recomputed | **matches the workbook** (accuracy, recall, searches): accuracy 0.4422 from 830 result records ([audit](../../verification/RESULT_INTEGRITY.md)) |
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
