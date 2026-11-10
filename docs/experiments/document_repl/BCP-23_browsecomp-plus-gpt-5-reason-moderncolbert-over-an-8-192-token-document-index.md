# BCP-23: Document REPL agent · BrowseComp-Plus · gpt-5 · Reason-ModernColBERT over an 8,192-token-document index

Status: complete. Row 23 of the `BrowseComp+` sheet of [Experiments.xlsx](../../../results/original_baselines/Experiments.xlsx).

## 1. Experiment overview

|  |  |
|---|---|
| Identifier | `BCP-23` (sheet `BrowseComp+`, row 23) |
| Agent | [Document REPL agent](../../../agents/document_repl/README.md) |
| Benchmark | BrowseComp-Plus (830 questions, 100,195 documents) |
| Model | gpt-5 (Azure, served gpt-5-2025-08-07) |
| Retriever | Reason-ModernColBERT over an 8,192-token-document index |
| Research question | Document REPL agent with GPT-5 and ColBERT: the counterpart of BCP-11 and the best configuration in the table. |
| Original execution | before 2026-10-26; exact dates not recorded in the repository |
| Run name in the logs | `colbert8192_high400 + high100b + high100c + high_rest230` |

## 2. Research motivation

Document REPL agent with GPT-5 and ColBERT: the counterpart of BCP-11 and the best configuration in the table. This row is one cell of the comparison matrix in Experiments.xlsx: the same benchmark is run with each agent × model × retriever combination, so this row is read against its neighbours (section 8). See [AGENT_COMPARISON.md](../AGENT_COMPARISON.md) for the matrix and [research_overview.md](../../research_overview.md) for the research questions.

## 3. Experimental methodology

The document REPL agent answers each question by writing Python in a persistent REPL. Each search puts the **whole** retrieved documents into a `DOCS` variable; the model reads slices of them itself (prints are capped per code block). Searches are capped client-side; at the turn limit the harness forces a final answer.
 Agent details: [agents/document_repl/README.md](../../../agents/document_repl/README.md).

Per question: the agent runs on one benchmark question with the budgets in section 4; the run record (answer, retrieved document ids, tool counts, turns, time, tokens) is written to the run directory.
 Correctness is judged afterwards by gpt-4.1 (Azure deployment gpt-4.1, served gpt-4.1-2025-04-14), official BrowseComp-Plus grader template, 3 votes, majority; a question without a completed answer is wrong. Recall is the share of the question's gold documents retrieved at any point of the run (definitions per row: [RESULT_INTEGRITY.md](../../verification/RESULT_INTEGRITY.md) §R1).

## 4. Configuration

| Setting | Value |
|---|---|
| Model | gpt-5 (Azure, served gpt-5-2025-08-07) |
| Reasoning effort requested | high |
| Reasoning effort actually sent | unverified: the run used a patched build of the rlm library that was not preserved |
| Retriever / index | Reason-ModernColBERT over an 8,192-token-document index |
| Turns | 30 + forced final answer |
| Searches | 10 (client-side cap; the trace logs up to 19, produced by the unpreserved patched build) |
| Results per search | model-chosen k (≤ 50) |
| Documents | whole documents into DOCS |
| Timeout | 5,400 s (harness max_timeout) |
| Random seed | none set (API / vLLM sampling defaults); runs are not bit-reproducible |


Exact environment and arguments ([agents/document_repl/configs/BCP-23.env](../../../agents/document_repl/configs/BCP-23.env)):

```bash
# BCP-23 | BrowseComp+ (830) | Document REPL agent | GPT-5 (high) | Reason-ModernColBERT (PLAID, document_length 8192)
# model: gpt-5, deployment pinned to the model (--root-deployment/--sub-deployment, original value "gpt-5"),
#   --expect-served-model gpt-5-2025-08-07; served build gpt-5-2025-08-07 per the per-question run summaries
# effort requested: high (root and sub-call). Effort actually sent: UNKNOWN from code -- these runs used a patched rlm
#   build of the original working directory (rlm_upstream, not preserved). The run summaries report +33% cost over the
#   medium arm on the same questions, consistent with the setting reaching the model.
# retriever: src/retrieval_server_colbert.py on :8766 (8192-token index; server environment not recorded)
# caps: --max-iterations 30 (+1 forced final-answer call), --max-searches 10 (client-side), k model-chosen 1..50,
#   fallback 5. DISCREPANCY: the run summaries list "search CALL cap: not recorded" and up to 19 searches in
#   colbert8192_high400; whether the cap of the launcher bound is UNKNOWN.
# original output: <original working directory>/runs/{colbert8192_high400, colbert8192_high100b,
#   colbert8192_high100c, colbert8192_high_rest230} -- not on this host; run records not readable here.
# launchers: experiments/bcp_gpt5_final830/launch_high{400,100b,100c,_rest230}.sh (one per batch)
ROW_ID=BCP-23
BENCHMARK=browsecomp_plus
ENTRY=src/run_experiment.py
LAUNCHER="experiments/bcp_gpt5_final830/launch_high400.sh experiments/bcp_gpt5_final830/launch_high100b.sh experiments/bcp_gpt5_final830/launch_high100c.sh experiments/bcp_gpt5_final830/launch_high_rest230.sh"
ROOT_MODEL=gpt-5
SUB_MODEL=gpt-5
REASONING_EFFORT=high
SUB_REASONING_EFFORT=high
EFFORT_SENT=UNKNOWN
RETRIEVER="Reason-ModernColBERT PLAID, document_length 8192 (src/retrieval_server_colbert.py)"
SERVER=http://127.0.0.1:8766
MAX_SEARCHES=10
MAX_ITERATIONS=30
MAX_TIMEOUT=5400
RUN_ARGS='--design B --workers 12 --stagger 0.4 --server http://127.0.0.1:8766 --root-model gpt-5 --sub-model gpt-5 --root-deployment ${AZURE_DEPLOYMENT_GPT5} --sub-deployment ${AZURE_DEPLOYMENT_GPT5} --expect-served-model gpt-5-2025-08-07 --reasoning-effort high --sub-reasoning-effort high --max-iterations 30 --max-searches 10 --seed-context --compaction --grounding-gate --lean-prologue --no-delegation --solo-prompt --no-orchestrator --compaction-threshold-pct 0.85 --max-concurrent-subcalls 4 --max-timeout 5400 --save-trajectory'
RUN_NAMES="colbert8192_high400 colbert8192_high100b colbert8192_high100c colbert8192_high_rest230"
IDS_FILES="data/prepared/colbert8192_high400.json data/prepared/colbert8192_high100b.json data/prepared/colbert8192_high100c.json data/prepared/colbert8192_high_rest230.json"
ENV_SET="Azure credentials from <working directory>/.env (AZURE2_*); no startup hooks"
ORIGINAL_OUTPUT="<original working directory>/runs/{colbert8192_high400,colbert8192_high100b,colbert8192_high100c,colbert8192_high_rest230}"
```

## 5. Execution instructions (reproduction, not executed for this documentation)

These commands launch the full experiment: hundreds of model calls and hours of GPU or API time. They were **not** run while preparing this repository.

```bash
set -a; . configs/paths.env; . agents/document_repl/configs/BCP-23.env; set +a
# then the launcher; its header documents its arguments:
agents/document_repl/experiments/bcp_gpt5_final830/launch_high400.sh
```

Launcher: [agents/document_repl/experiments/bcp_gpt5_final830/launch_high400.sh](../../../agents/document_repl/experiments/bcp_gpt5_final830/launch_high400.sh). It reads the benchmark questions, talks to the retrieval server named in the config and writes one record per question (plus trajectories) under `$RUNS_DIR`. Grade afterwards with the matching script in [agents/shared/rlm_shared/grading/](../../../agents/shared/rlm_shared/grading).

## 6. Implementation references

- Agent source: [bcp_bm25_split.py](../../../agents/document_repl/src/bcp_bm25_split.py), [bcp_qwen_dense.py](../../../agents/document_repl/src/bcp_qwen_dense.py), [build_bm25.py](../../../agents/document_repl/src/build_bm25.py), [encoder.py](../../../agents/document_repl/src/encoder.py), [fence_tolerant.py](../../../agents/document_repl/src/fence_tolerant.py), [grade_local.py](../../../agents/document_repl/src/grade_local.py), [hybrid.py](../../../agents/document_repl/src/hybrid.py), [judge.py](../../../agents/document_repl/src/judge.py), [obliq_adapter.py](../../../agents/document_repl/src/obliq_adapter.py), [prepare_data.py](../../../agents/document_repl/src/prepare_data.py), [prompts.py](../../../agents/document_repl/src/prompts.py), [provenance.py](../../../agents/document_repl/src/provenance.py) …
- Original code that produced the result (verbatim, SHA-256 manifest): [agents/document_repl/original/](../../../agents/document_repl/original/MANIFEST.json)
- Configuration: [agents/document_repl/configs/BCP-23.env](../../../agents/document_repl/configs/BCP-23.env) · Launcher: [agents/document_repl/experiments/bcp_gpt5_final830/launch_high400.sh](../../../agents/document_repl/experiments/bcp_gpt5_final830/launch_high400.sh)
- Data loaders: [agents/shared/rlm_shared/](../../../agents/shared/rlm_shared/__init__.py) (`browsecomp_data.py`, `obliq_data.py`)
- Grading: [agents/shared/rlm_shared/grading/](../../../agents/shared/rlm_shared/grading) · Metrics: [metrics.py](../../../agents/shared/rlm_shared/metrics.py)
- Tests: [agents/document_repl/tests/](../../../agents/document_repl/tests), [tests/](../../../tests)
- Per-question outputs: **not in this repository** (see [trajectories/README.md](../../../trajectories/README.md) for why)
- Original locations on the machine that ran the experiments: `docs/supervisor_artifact.html (embedded trace); raw run records not available`

## 7. Existing experimental results

Reported in Experiments.xlsx (every non-empty column of the row):

| Metric | Value |
|---|---|
| Accuracy | 0.8747 |
| Recall | 0.7875 |
| NDCG | — |
| Avg. Search Calls Per Query | 8.12 |
| Avg. Iterations Per Query | 10.8 |
| Time Per Query | 482.5 s (8.0 min) |
| Cached Cost Per Query | not logged |
| Uncached Cost Per Query | 0.5847 |
| Input Tokens Per Query | 166432 |
| Output Tokens Per Query | 37666 |


Workbook footnotes that apply to this row:

> Input / Output Tokens Per Query: all model calls of a question (root + sub-LLM), reasoning tokens included in output; these do not depend on how many runs shared a server, unlike Time Per Query. Exact from logged per-call usage, except values marked ≈ (Sept. 2026 snippet runs, whose logs overwrote the root model's usage with the sub-LLM's: rebuilt from the trajectories, about 10% low when the same method is checked on a run with exact usage). Uncached cost = input x list input price + output x list output price. Avg. Iterations Per Query: root-model turns per question, counted from the trajectory logs (snippet RLM) or the run records (document REPL agent); sub-LLM calls are not counted. A run that reaches its turn cap adds one forced final-answer turn.


**Failure breakdown** (from [Failure_Analysis.xlsx](../../../results/original_baselines/Failure_Analysis.xlsx), QA Summary; first matching category wins):

| Category | Questions | Share of failures |
|---|---|---|
| F1 did not finish | 0 | 0.0% |
| F2 no / abstained | 11 | 10.6% |
| F3 retrieval miss | 24 | 23.1% |
| F4 partial evidence | 54 | 51.9% |
| F5 all gold, wrong | 15 | 14.4% |
| **Failed in total** | 104 |  |


**Accuracy given the retrieved evidence** (By Recall sheet):

| Evidence retrieved | Questions | Accuracy |
|---|---|---|
| no gold retrieved | 31 | 3.2% |
| some gold | 384 | 84.6% |
| all gold | 415 | 96.4% |


**Turns** (By Turns sheet): limit 30, mean 10.8 (correct 9.9, not correct 17.2); 2 questions used every turn (0.2%), accuracy when answered before the limit 87.7%.

## 8. Results interpretation

**Paired comparisons on the same questions** (Pairwise sheet; exact McNemar test on the discordant pairs):

| Comparison | Other row | Only BCP-23 right | Only other right | p | Reading |
|---|---|---|---|---|---|
| Snippet REPL vs document REPL, GPT-5, Reason-ModernColBERT | BCP-11 | 65 | 51 | 0.23 | not significant |
| GPT-5 vs Qwen3.6, document REPL, Reason-ModernColBERT | BCP-29 | 262 | 17 | 1.4e-57 | significant at 0.05 |
| ColBERT vs BM25, document REPL, GPT-5 | BCP-25 | 255 | 27 | 1e-47 | significant at 0.05 |


- **Largest failure mode:** partial evidence (54 of 104 failures, 51.9%). Retrieval-side failures (F3 + F4) are 75.0% of failures; reading-side (F5) 14.4%. *(Measured.)*
- **With all gold evidence retrieved** the run is right 96.4% of the time (415 questions); with none, 3.2%. The gap between the two bounds how much better retrieval alone could help. *(Measured; causal reading is a hypothesis.)*
- Statements about *why* a configuration wins or loses are hypotheses unless the comparison isolates one variable; see the confounds in section 10.

## 9. Verification status

| Check | Result |
|---|---|
| Experiment previously executed | yes |
| Original outputs found | yes (on the original machine; paths in section 6) |
| Configuration identified | yes |
| Implementation reviewed | yes ([agent review](../../verification/REVIEW_document_repl.md)) |
| Lightweight tests | unit tests with mocks ([TESTING_REPORT.md](../../verification/TESTING_REPORT.md)); the experiment itself was not re-run |
| Numbers independently recomputed | **matches the workbook** (accuracy, recall, searches): accuracy 0.8747 from 830 result records ([audit](../../verification/RESULT_INTEGRITY.md)) |
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
