# BCP-21: Document REPL agent · BrowseComp-Plus · gpt-5 · Qwen3-Embedding-8B

Status: complete. Row 21 of the `BrowseComp+` sheet of [Experiments.xlsx](../../../results/original_baselines/Experiments.xlsx).

## 1. Experiment overview

|  |  |
|---|---|
| Identifier | `BCP-21` (sheet `BrowseComp+`, row 21) |
| Agent | [Document REPL agent](../../../agents/document_repl/README.md) |
| Benchmark | BrowseComp-Plus (830 questions, 100,195 documents) |
| Model | gpt-5 (Azure, served gpt-5-2025-08-07) |
| Retriever | Qwen3-Embedding-8B, exact dense |
| Research question | Document REPL agent with GPT-5 and a dense retriever: the counterpart of BCP-9. |
| Original execution | before 2026-10-26; exact dates not recorded in the repository |
| Run name in the logs | `final100_medium + final200_medium + final100_e + final_rest` |

## 2. Research motivation

Document REPL agent with GPT-5 and a dense retriever: the counterpart of BCP-9. This row is one cell of the comparison matrix in Experiments.xlsx: the same benchmark is run with each agent × model × retriever combination, so this row is read against its neighbours (section 8). See [AGENT_COMPARISON.md](../AGENT_COMPARISON.md) for the matrix and [research_overview.md](../../research_overview.md) for the research questions.

## 3. Experimental methodology

The document REPL agent answers each question by writing Python in a persistent REPL. Each search puts the **whole** retrieved documents into a `DOCS` variable; the model reads slices of them itself (prints are capped per code block). Searches are capped client-side; at the turn limit the harness forces a final answer.
 Agent details: [agents/document_repl/README.md](../../../agents/document_repl/README.md).

Per question: the agent runs on one benchmark question with the budgets in section 4; the run record (answer, retrieved document ids, tool counts, turns, time, tokens) is written to the run directory.
 Correctness is judged afterwards by gpt-4.1 (Azure deployment gpt-4.1, served gpt-4.1-2025-04-14), official BrowseComp-Plus grader template, 3 votes, majority; a question without a completed answer is wrong. Recall is the share of the question's gold documents retrieved at any point of the run (definitions per row: [RESULT_INTEGRITY.md](../../verification/RESULT_INTEGRITY.md) §R1).

## 4. Configuration

| Setting | Value |
|---|---|
| Model | gpt-5 (Azure, served gpt-5-2025-08-07) |
| Reasoning effort requested | medium |
| Reasoning effort actually sent | unverified: the run used a patched build of the rlm library that was not preserved |
| Retriever / index | Qwen3-Embedding-8B, exact dense |
| Turns | 30 + forced final answer |
| Searches | not recorded (flag absent in the run summaries; max observed 10) |
| Results per search | model-chosen k (≤ 50; median 30) |
| Documents | whole documents into DOCS |
| Timeout | not recorded |
| Random seed | none set (API / vLLM sampling defaults); runs are not bit-reproducible |


Exact environment and arguments ([agents/document_repl/configs/BCP-21.env](../../../agents/document_repl/configs/BCP-21.env)):

```bash
# BCP-21 | BrowseComp+ (830) | Document REPL agent | GPT-5 (medium) | Qwen3-Embedding-8B (dense, exact cosine)
# model: gpt-5 (root); served build gpt-5-2025-08-07 per the per-question run summaries
# effort requested: medium (root and sub-call). Effort actually sent: UNKNOWN from code -- these runs used a patched
#   rlm build of the original working directory (rlm_upstream, not preserved), not RLMLIB_PATH @ 854e688.
# retriever: src/retrieval_server.py (Qwen3-Embedding-8B); port UNKNOWN (server default :8765)
# caps: 30 turns + one forced final-answer call; search-call cap NOT RECORDED (the flag did not exist when the
#   summaries were built; max observed 10); k model-chosen 1..50 with fallback 5 (median k used 30, range 5-50)
# original output: four run directories of the original working directory, runs/{final100_medium, final200_medium,
#   final100_e, final_rest} -- not on this host; only per-question summaries survive (an HTML run-summary page in docs/ of the
#   original repository). Run records could not be read, so the flags below come from those summaries.
# launcher: NOT PRESERVED (experiments/bcp_gpt5_final830/final830_*.sh is a different run, see its header).
ROW_ID=BCP-21
BENCHMARK=browsecomp_plus
ENTRY=src/run_experiment.py
LAUNCHER=UNKNOWN
ROOT_MODEL=gpt-5
SUB_MODEL=UNKNOWN
REASONING_EFFORT=medium
SUB_REASONING_EFFORT=medium
EFFORT_SENT=UNKNOWN
RETRIEVER="Qwen3-Embedding-8B dense (src/retrieval_server.py)"
SERVER=UNKNOWN
MAX_SEARCHES=UNKNOWN
MAX_ITERATIONS=30
MAX_TIMEOUT=UNKNOWN
RUN_ARGS="--design B --root-model gpt-5 --reasoning-effort medium --sub-reasoning-effort medium --default-k 5 --max-iterations 30 --seed-context --compaction --grounding-gate --lean-prologue --no-delegation --solo-prompt --no-orchestrator --compaction-threshold-pct 0.85 --max-concurrent-subcalls 4"
ENV_SET=UNKNOWN
ORIGINAL_OUTPUT="<original working directory>/runs/{final100_medium,final200_medium,final100_e,final_rest}"
```

## 5. Execution instructions (reproduction, not executed for this documentation)

These commands launch the full experiment: hundreds of model calls and hours of GPU or API time. They were **not** run while preparing this repository.

```bash
set -a; . configs/paths.env; . agents/document_repl/configs/BCP-21.env; set +a
# then the launcher; its header documents its arguments:
agents/document_repl/src/run_experiment.py
```

Launcher: [agents/document_repl/src/run_experiment.py](../../../agents/document_repl/src/run_experiment.py). It reads the benchmark questions, talks to the retrieval server named in the config and writes one record per question (plus trajectories) under `$RUNS_DIR`. Grade afterwards with the matching script in [agents/shared/rlm_shared/grading/](../../../agents/shared/rlm_shared/grading).

## 6. Implementation references

- Agent source: [bcp_bm25_split.py](../../../agents/document_repl/src/bcp_bm25_split.py), [bcp_qwen_dense.py](../../../agents/document_repl/src/bcp_qwen_dense.py), [build_bm25.py](../../../agents/document_repl/src/build_bm25.py), [encoder.py](../../../agents/document_repl/src/encoder.py), [fence_tolerant.py](../../../agents/document_repl/src/fence_tolerant.py), [frames_cap25_report.py](../../../agents/document_repl/src/frames_cap25_report.py), [grade_local.py](../../../agents/document_repl/src/grade_local.py), [hybrid.py](../../../agents/document_repl/src/hybrid.py), [judge.py](../../../agents/document_repl/src/judge.py), [obliq_adapter.py](../../../agents/document_repl/src/obliq_adapter.py), [prepare_data.py](../../../agents/document_repl/src/prepare_data.py), [prompts.py](../../../agents/document_repl/src/prompts.py) …
- Original code that produced the result (verbatim, SHA-256 manifest): [agents/document_repl/original/](../../../agents/document_repl/original/MANIFEST.json)
- Configuration: [agents/document_repl/configs/BCP-21.env](../../../agents/document_repl/configs/BCP-21.env) · Launcher: [agents/document_repl/src/run_experiment.py](../../../agents/document_repl/src/run_experiment.py)
- Data loaders: [agents/shared/rlm_shared/](../../../agents/shared/rlm_shared/__init__.py) (`browsecomp_data.py`, `obliq_data.py`)
- Grading: [agents/shared/rlm_shared/grading/](../../../agents/shared/rlm_shared/grading) · Metrics: [metrics.py](../../../agents/shared/rlm_shared/metrics.py)
- Tests: [agents/document_repl/tests/](../../../agents/document_repl/tests), [tests/](../../../tests)
- Per-question outputs: **not in this repository** (see [trajectories/README.md](../../../trajectories/README.md) for why)
- Original locations on the machine that ran the experiments: `docs/supervisor_artifact.html (embedded per-question trace: verdict, evidence recall, searches, turns, time, answer cut at 1,500 chars); raw run records not available`

## 7. Existing experimental results

Reported in Experiments.xlsx (every non-empty column of the row):

| Metric | Value |
|---|---|
| Accuracy | 0.8313 |
| Recall | 0.7619 |
| NDCG | — |
| Avg. Search Calls Per Query | 7.9 |
| Avg. Iterations Per Query | 11.2 |
| Time Per Query | 300.2 s (5.0 min) |
| Cached Cost Per Query | not logged |
| Uncached Cost Per Query | 0.4395 |
| Input Tokens Per Query | 165685 |
| Output Tokens Per Query | 23236 |


Workbook footnotes that apply to this row:

> * GPT 5 (medium)*: Document REPL agent with Qwen3-Embedding-8B and document REPL agent with BM25 were run at reasoning effort medium, not high (no high-effort full run exists for these cells). The BM25 run requested high, but the Azure client dropped the effort setting, so GPT-5 ran at its default, medium.

> Input / Output Tokens Per Query: all model calls of a question (root + sub-LLM), reasoning tokens included in output; these do not depend on how many runs shared a server, unlike Time Per Query. Exact from logged per-call usage, except values marked ≈ (Sept. 2026 snippet runs, whose logs overwrote the root model's usage with the sub-LLM's: rebuilt from the trajectories, about 10% low when the same method is checked on a run with exact usage). Uncached cost = input x list input price + output x list output price. Avg. Iterations Per Query: root-model turns per question, counted from the trajectory logs (snippet RLM) or the run records (document REPL agent); sub-LLM calls are not counted. A run that reaches its turn cap adds one forced final-answer turn.


**Failure breakdown** (from [Failure_Analysis.xlsx](../../../results/original_baselines/Failure_Analysis.xlsx), QA Summary; first matching category wins):

| Category | Questions | Share of failures |
|---|---|---|
| F1 did not finish | 0 | 0.0% |
| F2 no / abstained | 20 | 14.3% |
| F3 retrieval miss | 27 | 19.3% |
| F4 partial evidence | 77 | 55.0% |
| F5 all gold, wrong | 16 | 11.4% |
| **Failed in total** | 140 |  |


**Accuracy given the retrieved evidence** (By Recall sheet):

| Evidence retrieved | Questions | Accuracy |
|---|---|---|
| no gold retrieved | 37 | 0.0% |
| some gold | 399 | 78.2% |
| all gold | 394 | 95.9% |


**Turns** (By Turns sheet): limit 30, mean 11.2 (correct 10.1, not correct 16.9); 4 questions used every turn (0.5%), accuracy when answered before the limit 83.5%.

## 8. Results interpretation

**Paired comparisons on the same questions** (Pairwise sheet; exact McNemar test on the discordant pairs):

| Comparison | Other row | Only BCP-21 right | Only other right | p | Reading |
|---|---|---|---|---|---|
| Snippet REPL vs document REPL, GPT-5, Qwen3-Embed-8B | BCP-9 | 88 | 57 | 0.012 | significant at 0.05 |
| GPT-5 vs Qwen3.6, document REPL, Qwen3-Embed-8B | BCP-27 | 332 | 9 | 7.1e-86 | significant at 0.05 |


- **Largest failure mode:** partial evidence (77 of 140 failures, 55.0%). Retrieval-side failures (F3 + F4) are 74.3% of failures; reading-side (F5) 11.4%. *(Measured.)*
- **With all gold evidence retrieved** the run is right 95.9% of the time (394 questions); with none, 0.0%. The gap between the two bounds how much better retrieval alone could help. *(Measured; causal reading is a hypothesis.)*
- Statements about *why* a configuration wins or loses are hypotheses unless the comparison isolates one variable; see the confounds in section 10.

## 9. Verification status

| Check | Result |
|---|---|
| Experiment previously executed | yes |
| Original outputs found | yes (on the original machine; paths in section 6) |
| Configuration identified | yes |
| Implementation reviewed | yes ([agent review](../../verification/REVIEW_document_repl.md)) |
| Lightweight tests | unit tests with mocks ([TESTING_REPORT.md](../../verification/TESTING_REPORT.md)); the experiment itself was not re-run |
| Numbers independently recomputed | **matches the workbook** (accuracy, recall, searches): accuracy 0.8313 from 830 result records ([audit](../../verification/RESULT_INTEGRITY.md)) |
| Reproduced by re-running | **no**: no experiment was re-run; full experimental equivalence of the refactored code is not established |


## 10. Limitations

- No random seed; model sampling is stochastic, so a re-run gives different per-question results (two identical 830-question snippet REPL agent runs disagreed on 17.8% of questions in the earlier campaign).
- Reasoning effort actually sent: unverified: the run used a patched build of the rlm library that was not preserved.
- The judge (gpt-4.1) is an LLM; its verdicts were not audited by hand.
- Comparisons with other rows can differ in more than one variable (search cap, turn limit, time limit, effort, index document length); see [AGENT_COMPARISON.md](../AGENT_COMPARISON.md).
- Per-question data exist only as the trace embedded in the results page; raw run records were not available for audit.
- The number of documents per search is chosen by the model (up to 50), not fixed at 5 like the snippet REPL agent; the `llm_query` sub-LLM functions remain installed in the REPL (only the prompt says not to use them).

## 11. Related files

- [Experiments.xlsx](../../../results/original_baselines/Experiments.xlsx) · [Failure_Analysis.xlsx](../../../results/original_baselines/Failure_Analysis.xlsx) · [registry.yaml](../../../experiments/registry.yaml)
- [Known issues](../../verification/KNOWN_ISSUES.md) · [Result integrity](../../verification/RESULT_INTEGRITY.md) · [Changes from the original code](../../../agents/document_repl/CHANGES_FROM_ORIGINAL.md)
