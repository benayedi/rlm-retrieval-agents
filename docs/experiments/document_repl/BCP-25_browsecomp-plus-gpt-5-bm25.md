# BCP-25: Document REPL agent · BrowseComp-Plus · gpt-5 · BM25 (bm25s

Status: complete. Row 25 of the `BrowseComp+` sheet of [Experiments.xlsx](../../../results/original_baselines/Experiments.xlsx).

## 1. Experiment overview

|  |  |
|---|---|
| Identifier | `BCP-25` (sheet `BrowseComp+`, row 25) |
| Agent | [Document REPL agent](../../../agents/document_repl/README.md) |
| Benchmark | BrowseComp-Plus (830 questions, 100,195 documents) |
| Model | gpt-5 (Azure) |
| Retriever | BM25 (bm25s, k1 0.9, b 0.4) |
| Research question | Document REPL agent with GPT-5 and a lexical retriever: the counterpart of BCP-13. |
| Original execution | 2026-10-26 to 2026-10-27 |
| Run name in the logs | `bcp_gpt5_bm25` |

## 2. Research motivation

Document REPL agent with GPT-5 and a lexical retriever: the counterpart of BCP-13. This row is one cell of the comparison matrix in Experiments.xlsx: the same benchmark is run with each agent × model × retriever combination, so this row is read against its neighbours (section 8). See [AGENT_COMPARISON.md](../AGENT_COMPARISON.md) for the matrix and [research_overview.md](../../research_overview.md) for the research questions.

## 3. Experimental methodology

The document REPL agent answers each question by writing Python in a persistent REPL. Each search puts the **whole** retrieved documents into a `DOCS` variable; the model reads slices of them itself (prints are capped per code block). Searches are capped client-side; at the turn limit the harness forces a final answer.
 Agent details: [agents/document_repl/README.md](../../../agents/document_repl/README.md).

Per question: the agent runs on one benchmark question with the budgets in section 4; the run record (answer, retrieved document ids, tool counts, turns, time, tokens) is written to the run directory.
 Correctness is judged afterwards by gpt-4.1 (Azure deployment gpt-4.1, served gpt-4.1-2025-04-14), official BrowseComp-Plus grader template, 3 votes, majority; a question without a completed answer is wrong. Recall is the share of the question's gold documents retrieved at any point of the run (definitions per row: [RESULT_INTEGRITY.md](../../verification/RESULT_INTEGRITY.md) §R1).

## 4. Configuration

| Setting | Value |
|---|---|
| Model | gpt-5 (Azure) |
| Reasoning effort requested | high |
| Reasoning effort actually sent | none sent: deployment default, measured as medium (only the retry429 hook was loaded); stated in the workbook footnote |
| Retriever / index | BM25 (bm25s, k1 0.9, b 0.4) |
| Turns | 30 + forced final answer |
| Searches | 10 |
| Results per search | model-chosen k (≤ 50) |
| Documents | whole documents into DOCS |
| Timeout | 2,700 s harness limit inside a 3,300 s process limit |
| Random seed | none set (API / vLLM sampling defaults); runs are not bit-reproducible |


Exact environment and arguments ([agents/document_repl/configs/BCP-25.env](../../../agents/document_repl/configs/BCP-25.env)):

```bash
# BCP-25 | BrowseComp+ (830) | Document REPL agent | GPT-5 | BM25 (default Lucene k1 0.9, b 0.4) on :8043
# model: gpt-5 via Azure deployment AZURE_DEPLOYMENT_GPT5 (original: a personal deployment serving gpt-5-2025-08-07);
#   served_models in the records: null (stock rlm client records nothing)
# effort requested: high (EFFORT default; record config reasoning_effort "high"). Effort actually sent: NONE --
#   only the retry429 startup hook was loaded, and rlm's Azure client drops sampling_args, so gpt-5 ran at the
#   deployment default (medium on the original deployment). Experiments.xlsx lists the row as "medium".
# retriever: BM25 search server over BCP_BM25_INDEX_DIR on :8043 (not part of this agent); prompt patched by
#   src/run_bm25.py with BM25_CORPUS=bcp (record prompt_id 758e43ed5b95)
# caps: --max-searches 10 (client-side), --max-iterations 30 (+1 forced final answer: 154 of 830 records have
#   n_iterations 31), k model-chosen 1..50 with fallback 5, --max-timeout 2700, xargs timeout 3300
# original output: <original repo>/logs/bcp_gpt5_bm25/runs (830 records; config read from 1.json)
ROW_ID=BCP-25
BENCHMARK=browsecomp_plus
ENTRY=src/run_bm25.py
LAUNCHER=experiments/bcp_gpt5_bm25.sh
ROOT_MODEL=gpt-5
SUB_MODEL=gpt-5
REASONING_EFFORT=high
SUB_REASONING_EFFORT=high
EFFORT_SENT=none
RETRIEVER="BM25 k1 0.9 b 0.4 (BCP_BM25_INDEX_DIR)"
SERVER=http://127.0.0.1:8043
MAX_SEARCHES=10
MAX_ITERATIONS=30
MAX_TIMEOUT=2700
W=16
EFFORT=high
RUN_ARGS='--design B --server http://127.0.0.1:8043 --root-model gpt-5 --sub-model gpt-5 --root-deployment ${DEP} --sub-deployment ${DEP} --reasoning-effort high --sub-reasoning-effort high --default-k 5 --max-iterations 30 --max-searches 10 --seed-context --compaction --grounding-gate --lean-prologue --no-delegation --solo-prompt --no-orchestrator --compaction-threshold-pct 0.85 --max-concurrent-subcalls 4 --max-timeout 2700'
ENV_SET='AZURE_OPENAI_API_VERSION=2024-12-01-preview BM25_CORPUS=bcp PYTHONPATH=<hooks>/retry429:<RLMLIB_PATH>'
ENV_UNSET='QUERIES_FILE SUP_FENCE_TOLERANT SUP_MAX_BLOCKS SUP_PRESENCE_PENALTY'
ORIGINAL_OUTPUT="<original repo>/logs/bcp_gpt5_bm25/runs"
```

## 5. Execution instructions (reproduction, not executed for this documentation)

These commands launch the full experiment: hundreds of model calls and hours of GPU or API time. They were **not** run while preparing this repository.

```bash
set -a; . configs/paths.env; . agents/document_repl/configs/BCP-25.env; set +a
# then the launcher; its header documents its arguments:
agents/document_repl/experiments/bcp_gpt5_bm25.sh
```

Launcher: [agents/document_repl/experiments/bcp_gpt5_bm25.sh](../../../agents/document_repl/experiments/bcp_gpt5_bm25.sh). It reads the benchmark questions, talks to the retrieval server named in the config and writes one record per question (plus trajectories) under `$RUNS_DIR`. Grade afterwards with the matching script in [agents/shared/rlm_shared/grading/](../../../agents/shared/rlm_shared/grading).

## 6. Implementation references

- Agent source: [bcp_bm25_split.py](../../../agents/document_repl/src/bcp_bm25_split.py), [bcp_qwen_dense.py](../../../agents/document_repl/src/bcp_qwen_dense.py), [build_bm25.py](../../../agents/document_repl/src/build_bm25.py), [encoder.py](../../../agents/document_repl/src/encoder.py), [fence_tolerant.py](../../../agents/document_repl/src/fence_tolerant.py), [grade_local.py](../../../agents/document_repl/src/grade_local.py), [hybrid.py](../../../agents/document_repl/src/hybrid.py), [judge.py](../../../agents/document_repl/src/judge.py), [obliq_adapter.py](../../../agents/document_repl/src/obliq_adapter.py), [prepare_data.py](../../../agents/document_repl/src/prepare_data.py), [prompts.py](../../../agents/document_repl/src/prompts.py), [provenance.py](../../../agents/document_repl/src/provenance.py) …
- Original code that produced the result (verbatim, SHA-256 manifest): [agents/document_repl/original/](../../../agents/document_repl/original/MANIFEST.json)
- Configuration: [agents/document_repl/configs/BCP-25.env](../../../agents/document_repl/configs/BCP-25.env) · Launcher: [agents/document_repl/experiments/bcp_gpt5_bm25.sh](../../../agents/document_repl/experiments/bcp_gpt5_bm25.sh)
- Data loaders: [agents/shared/rlm_shared/](../../../agents/shared/rlm_shared/__init__.py) (`browsecomp_data.py`, `obliq_data.py`)
- Grading: [agents/shared/rlm_shared/grading/](../../../agents/shared/rlm_shared/grading) · Metrics: [metrics.py](../../../agents/shared/rlm_shared/metrics.py)
- Tests: [agents/document_repl/tests/](../../../agents/document_repl/tests), [tests/](../../../tests)
- **Per-question outputs in this repository** (Git LFS): [trajectories/document_repl/BCP-25/](../../../trajectories/document_repl/BCP-25/MANIFEST.json). 1,712 files, 0.01 GB as written by the run, encrypted with the BrowseComp canary key; extract with `python scripts/provenance/unpack_trajectories.py BCP-25 OUT_DIR` ([format notes](../../../trajectories/README.md))
- Original locations on the machine that ran the experiments: `logs/bcp_gpt5_bm25/runs/<qid>.json`; `logs/bcp_rejudge_gpt41/bm25_gpt5.json (gpt-4.1 verdicts)`

## 7. Existing experimental results

Reported in Experiments.xlsx (every non-empty column of the row):

| Metric | Value |
|---|---|
| Accuracy | 0.6 |
| Recall | 0.4973 |
| NDCG | — |
| Avg. Search Calls Per Query | 9.2 |
| Avg. Iterations Per Query | 18.2 |
| Time Per Query | 567 s (9.5 min) |
| Cached Cost Per Query | not logged |
| Uncached Cost Per Query | 0.9266 |
| Input Tokens Per Query | 390854 |
| Output Tokens Per Query | 43804 |


Workbook footnotes that apply to this row:

> * GPT 5 (medium)*: Document REPL agent with Qwen3-Embedding-8B and document REPL agent with BM25 were run at reasoning effort medium, not high (no high-effort full run exists for these cells). The BM25 run requested high, but the Azure client dropped the effort setting, so GPT-5 ran at its default, medium.

> Input / Output Tokens Per Query: all model calls of a question (root + sub-LLM), reasoning tokens included in output; these do not depend on how many runs shared a server, unlike Time Per Query. Exact from logged per-call usage, except values marked ≈ (Sept. 2026 snippet runs, whose logs overwrote the root model's usage with the sub-LLM's: rebuilt from the trajectories, about 10% low when the same method is checked on a run with exact usage). Uncached cost = input x list input price + output x list output price. Avg. Iterations Per Query: root-model turns per question, counted from the trajectory logs (snippet RLM) or the run records (document REPL agent); sub-LLM calls are not counted. A run that reaches its turn cap adds one forced final-answer turn.


**Failure breakdown** (from [Failure_Analysis.xlsx](../../../results/original_baselines/Failure_Analysis.xlsx), QA Summary; first matching category wins):

| Category | Questions | Share of failures |
|---|---|---|
| F1 did not finish | 4 | 1.2% |
| F2 no / abstained | 118 | 35.5% |
| F3 retrieval miss | 112 | 33.7% |
| F4 partial evidence | 91 | 27.4% |
| F5 all gold, wrong | 7 | 2.1% |
| **Failed in total** | 332 |  |


**Accuracy given the retrieved evidence** (By Recall sheet):

| Evidence retrieved | Questions | Accuracy |
|---|---|---|
| no gold retrieved | 219 | 11.4% |
| some gold | 432 | 71.1% |
| all gold | 175 | 94.9% |


**Turns** (By Turns sheet): limit 30, mean 18.2 (correct 14.5, not correct 23.6); 169 questions used every turn (20.4%), accuracy when answered before the limit 66.7%.

## 8. Results interpretation

**Paired comparisons on the same questions** (Pairwise sheet; exact McNemar test on the discordant pairs):

| Comparison | Other row | Only BCP-25 right | Only other right | p | Reading |
|---|---|---|---|---|---|
| Snippet REPL vs document REPL, GPT-5, BM25 | BCP-13 | 74 | 136 | 2.3e-05 | significant at 0.05 |
| GPT-5 vs Qwen3.6, document REPL, BM25 | BCP-31 | 332 | 20 | 4.7e-74 | significant at 0.05 |
| ColBERT vs BM25, document REPL, GPT-5 | BCP-23 | 27 | 255 | 1e-47 | significant at 0.05 |


- **Largest failure mode:** abstentions (118 of 332 failures, 35.5%). Retrieval-side failures (F3 + F4) are 61.1% of failures; reading-side (F5) 2.1%. *(Measured.)*
- **With all gold evidence retrieved** the run is right 94.9% of the time (175 questions); with none, 11.4%. The gap between the two bounds how much better retrieval alone could help. *(Measured; causal reading is a hypothesis.)*
- Statements about *why* a configuration wins or loses are hypotheses unless the comparison isolates one variable; see the confounds in section 10.

## 9. Verification status

| Check | Result |
|---|---|
| Experiment previously executed | yes |
| Original outputs found | yes (on the original machine; paths in section 6) |
| Configuration identified | yes |
| Implementation reviewed | yes ([agent review](../../verification/REVIEW_document_repl.md)) |
| Lightweight tests | unit tests with mocks ([TESTING_REPORT.md](../../verification/TESTING_REPORT.md)); the experiment itself was not re-run |
| Numbers independently recomputed | **matches the workbook** (accuracy, recall, searches): accuracy 0.6000 from 826 result records ([audit](../../verification/RESULT_INTEGRITY.md)) |
| Reproduced by re-running | **no**: no experiment was re-run; full experimental equivalence of the refactored code is not established |


## 10. Limitations

- No random seed; model sampling is stochastic, so a re-run gives different per-question results (two identical 830-question snippet REPL agent runs disagreed on 17.8% of questions in the earlier campaign).
- Reasoning effort actually sent: none sent: deployment default, measured as medium (only the retry429 hook was loaded); stated in the workbook footnote.
- The judge (gpt-4.1) is an LLM; its verdicts were not audited by hand.
- Comparisons with other rows can differ in more than one variable (search cap, turn limit, time limit, effort, index document length); see [AGENT_COMPARISON.md](../AGENT_COMPARISON.md).
- The number of documents per search is chosen by the model (up to 50), not fixed at 5 like the snippet REPL agent; the `llm_query` sub-LLM functions remain installed in the REPL (only the prompt says not to use them).

## 11. Related files

- [Experiments.xlsx](../../../results/original_baselines/Experiments.xlsx) · [Failure_Analysis.xlsx](../../../results/original_baselines/Failure_Analysis.xlsx) · [registry.yaml](../../../experiments/registry.yaml)
- [Known issues](../../verification/KNOWN_ISSUES.md) · [Result integrity](../../verification/RESULT_INTEGRITY.md) · [Changes from the original code](../../../agents/document_repl/CHANGES_FROM_ORIGINAL.md)
