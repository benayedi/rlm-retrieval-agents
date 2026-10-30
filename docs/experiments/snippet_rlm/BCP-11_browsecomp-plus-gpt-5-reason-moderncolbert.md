# BCP-11: Snippet RLM · BrowseComp-Plus · gpt-5 · Reason-ModernColBERT (149M

Status: complete. Row 11 of the `BrowseComp+` sheet of [Experiments.xlsx](../../../results/original_baselines/Experiments.xlsx).

## 1. Experiment overview

|  |  |
|---|---|
| Identifier | `BCP-11` (sheet `BrowseComp+`, row 11) |
| Algorithm | [Snippet RLM](../../../algorithms/snippet_rlm/README.md) |
| Benchmark | BrowseComp-Plus (830 questions, 100,195 documents) |
| Model | gpt-5 (Azure) |
| Retriever | Reason-ModernColBERT (149M, multi-vector late interaction, 512-token documents) |
| Research question | Does a small late-interaction retriever (149M) beat an 8B dense retriever under the same Snippet RLM agent? |
| Original execution | 2026-09-11 (trajectory timestamps) |
| Run name in the logs | `rlmfix24 + rlmfix25/29/30/31 (five 100-to-230-question blocks of one frozen configuration)` |

## 2. Research motivation

Does a small late-interaction retriever (149M) beat an 8B dense retriever under the same Snippet RLM agent? This row is one cell of the comparison matrix in Experiments.xlsx: the same benchmark is run with each algorithm × model × retriever combination, so this row is read against its neighbours (section 8). See [ALGORITHM_COMPARISON.md](../ALGORITHM_COMPARISON.md) for the matrix and [research_overview.md](../../research_overview.md) for the research questions.

## 3. Experimental methodology

The Snippet RLM answers each question by writing Python in a persistent REPL, one turn at a time. Its tools return search results as short snippets (the first tokens of each document) into a variable; it can open a whole document and pass text to a sub-LLM (`llm_query`) to read it. A run ends when the model sets a final answer or the turn limit is reached.
 Algorithm details: [algorithms/snippet_rlm/README.md](../../../algorithms/snippet_rlm/README.md).

Per question: the agent runs on one benchmark question with the budgets in section 4; the run record (answer, retrieved document ids, tool counts, turns, time, tokens) is written to the run directory.
 Correctness is judged afterwards by gpt-4.1 (Azure deployment gpt-4.1, served gpt-4.1-2025-04-14), official BrowseComp-Plus grader template, 3 votes, majority; a question without a completed answer is wrong. Recall is the share of the question's gold documents retrieved at any point of the run (definitions per row: [RESULT_INTEGRITY.md](../../verification/RESULT_INTEGRITY.md) §R1).

## 4. Configuration

| Setting | Value |
|---|---|
| Model | gpt-5 (Azure) |
| Reasoning effort requested | root high, sub-LLM medium |
| Reasoning effort actually sent | deployment default for both (no hook loaded; gpt-5 default measured as medium) |
| Retriever / index | Reason-ModernColBERT (149M, multi-vector late interaction, 512-token documents) |
| Turns | 100 |
| Searches | no cap |
| Results per search | 5 (default; the model may request more) |
| Snippet | min(2,000 chars, 512 tokens) |
| Timeout | none |
| Compaction | off |
| Random seed | none set (API / vLLM sampling defaults); runs are not bit-reproducible |
| Harness version | September state; closest preserved copy: algorithms/snippet_rlm/original/RLM_git_7685538. Launchers of rlmfix25/29/30/31 are not in the repository |


Exact environment and arguments ([algorithms/snippet_rlm/configs/BCP-11.env](../../../algorithms/snippet_rlm/configs/BCP-11.env)):

```bash
# BCP-11  Snippet RLM | GPT-5 (requested root high, llm_query medium; see the effort note) | Reason-ModernColBERT
#         (multi-vector, index bc_reason_mcb) on :8024 | BrowseComp+ 830 q, assembled from five runs (September 2026):
#           rlmfix24  q0-99     launcher for the 17 "run 19 failures": experiments/september/bcp_snip_no_orchestrator.sh
#                               (was scripts/run_no_orchestrator.sh; question list /tmp/noorch.txt, contents UNKNOWN);
#                               launcher of the other 83 questions: UNKNOWN (not in the repository)
#           rlmfix25  q100-199  launcher UNKNOWN (not in the repository)
#           rlmfix29  q200-299  launcher UNKNOWN (not in the repository)
#           rlmfix30  q300-399  launcher UNKNOWN (not in the repository)
#           rlmfix31  q400-829  launcher UNKNOWN; the second header block of scripts/run_qwen_full.sh ("Completes
#                               BrowseComp-Plus end to end ... Reason-ModernColBERT on 8024") is evidently copied from it
# Budgets: 100 turns, depth 1, NO compaction, NO search cap, top_k default 5, 512-token snippets, 60K REPL ceiling,
#          no per-question timeout. That header states the configuration of all five runs is identical ("q0-399 are already
#          done under this exact configuration (runs 24, 25, 29, 30)"; "CONFIG, unchanged and frozen").
# Original outputs: <original repo>/logs/rlmfix{24,25,29,30,31}_q<N>.json (100+100+100+100+430 = 830 files); re-graded
#          with gpt-4.1: <original repo>/logs/bcp_rejudge_gpt41/9.json.
# EFFORT NOTE: no startup hook on PYTHONPATH, so reasoning_effort was dropped by rlm's Azure client (deployment default).
# CODE NOTE: the records lack the child_* fields, so the harness that ran predates commit 7685538 (see CHANGES).
# Use: NOORCH_QIDS_FILE=<file> experiments/run_config.sh configs/BCP-11.env   (reproduces the rlmfix24 launcher only)
ROW=BCP-11
LAUNCHER=experiments/september/bcp_snip_no_orchestrator.sh
LAUNCHER_ARGS=()
export W=5   # rlmfix24 worker loops; launcher default. Actual value UNKNOWN
LAUNCHER_RLMFIX25=UNKNOWN
LAUNCHER_RLMFIX29=UNKNOWN
LAUNCHER_RLMFIX30=UNKNOWN
LAUNCHER_RLMFIX31=UNKNOWN
NOORCH_QUESTION_LIST=UNKNOWN   # /tmp/noorch.txt: run 19's 17 failures, indices not preserved
# --- effective environment of rlmfix24 (set by the launcher; per the header above, the same for 25/29/30/31) ---
export EMBEDDING_SERVER_URL=http://localhost:8024 AZURE_OPENAI_REASONING_EFFORT=high SUB_LM_REASONING_EFFORT=medium
export AZURE_OPENAI_API_VERSION=2025-04-01-preview LIGHTON_TOOL_DOCS=1 LIGHTON_TASK_TEXT=1
export SEARCH_SNIPPET_TOKENS=512 OFFICIAL_TOP_K=5 ORCHESTRATOR=0
# RLM_TRAJ_DIR=logs/rlm_traj24; unset: MAX_TOP_K SEARCH_SNIPPET_LEN (rlmfix24's launcher does NOT unset SUB_LM_MODEL).
# CLI per question: harness.py --start Q --end Q+1 --max-iterations 100 --max-depth 1 --out logs/rlmfix24_qQ
# --- from <original machine>/master/.env (file gone) ---
AZURE_OPENAI_MODEL_ORIGINAL=UNKNOWN
AZURE_OPENAI_JUDGE_MODEL_ORIGINAL=UNKNOWN
OTHER_DOTENV_VARIABLES=UNKNOWN
```

## 5. Execution instructions (reproduction, not executed for this documentation)

These commands launch the full experiment: hundreds of model calls and hours of GPU or API time. They were **not** run while preparing this repository.

```bash
set -a; . configs/paths.env; . algorithms/snippet_rlm/configs/BCP-11.env; set +a
# then the launcher; its header documents its arguments:
algorithms/snippet_rlm/experiments/september/bcp_snip_no_orchestrator.sh
```

Launcher: [algorithms/snippet_rlm/experiments/september/bcp_snip_no_orchestrator.sh](../../../algorithms/snippet_rlm/experiments/september/bcp_snip_no_orchestrator.sh). It reads the benchmark questions, talks to the retrieval server named in the config and writes one record per question (plus trajectories) under `$RUNS_DIR`. Grade afterwards with the matching script in [shared/rlm_shared/grading/](../../../shared/rlm_shared/grading).

## 6. Implementation references

- Algorithm source: [harness.py](../../../algorithms/snippet_rlm/src/harness.py), [sitecustomize.py](../../../algorithms/snippet_rlm/src/hooks/azure_sampling/sitecustomize.py), [sitecustomize.py](../../../algorithms/snippet_rlm/src/hooks/cap_ranking/sitecustomize.py), [sitecustomize.py](../../../algorithms/snippet_rlm/src/hooks/retry429/sitecustomize.py), [sitecustomize.py](../../../algorithms/snippet_rlm/src/hooks/retry429_cap_ranking/sitecustomize.py), [sitecustomize.py](../../../algorithms/snippet_rlm/src/hooks/run5d/sitecustomize.py), [prompt_modules.py](../../../algorithms/snippet_rlm/src/prompt_modules.py), [run_browsecomp.py](../../../algorithms/snippet_rlm/src/run_browsecomp.py), [run_frames.py](../../../algorithms/snippet_rlm/src/run_frames.py), [run_obliq.py](../../../algorithms/snippet_rlm/src/run_obliq.py)
- Original code that produced the result (verbatim, SHA-256 manifest): [algorithms/snippet_rlm/original/](../../../algorithms/snippet_rlm/original/MANIFEST.json)
- Configuration: [algorithms/snippet_rlm/configs/BCP-11.env](../../../algorithms/snippet_rlm/configs/BCP-11.env) · Launcher: [algorithms/snippet_rlm/experiments/september/bcp_snip_no_orchestrator.sh](../../../algorithms/snippet_rlm/experiments/september/bcp_snip_no_orchestrator.sh)
- Data loaders: [shared/rlm_shared/](../../../shared/rlm_shared/__init__.py) (`browsecomp_data.py`, `obliq_data.py`)
- Grading: [shared/rlm_shared/grading/](../../../shared/rlm_shared/grading) · Metrics: [metrics.py](../../../shared/rlm_shared/metrics.py)
- Tests: [algorithms/snippet_rlm/tests/](../../../algorithms/snippet_rlm/tests), [tests/](../../../tests)
- Original outputs (not in this repository; on the machine that ran the experiments): `logs/rlmfix{24,25,29,30,31}_q*.json`; `logs/bcp_rejudge_gpt41/9.json (gpt-4.1 verdicts)`

## 7. Existing experimental results

Reported in Experiments.xlsx (every non-empty column of the row):

| Metric | Value |
|---|---|
| Accuracy | 0.8578 |
| Recall | 0.824 |
| NDCG | — |
| Avg. Search Calls Per Query | 22.3 |
| Avg. Iterations Per Query | 4.7 |
| Time Per Query | 166 s (2.8 min) |
| Cached Cost Per Query | not logged |
| Uncached Cost Per Query | ≈ 0.36 |
| Input Tokens Per Query | ≈ 180,623 |
| Output Tokens Per Query | ≈ 13,144 |


Workbook footnotes that apply to this row:

> * GPT 5 (high)* (Snippet RLM rows): the root model runs at high effort, but its sub-LLM reading calls (llm_query) run at medium.

> Input / Output Tokens Per Query: all model calls of a question (root + sub-LLM), reasoning tokens included in output; these do not depend on how many runs shared a server, unlike Time Per Query. Exact from logged per-call usage, except values marked ≈ (Sept. 2026 snippet runs, whose logs overwrote the root model's usage with the sub-LLM's: rebuilt from the trajectories, about 10% low when the same method is checked on a run with exact usage). Uncached cost = input x list input price + output x list output price. Avg. Iterations Per Query: root-model turns per question, counted from the trajectory logs (snippet RLM) or the run records (full-doc RLM); sub-LLM calls are not counted. A run that reaches its turn cap adds one forced final-answer turn.

> **Correction (this audit):** the launcher of this run loaded no hook that forwards `reasoning_effort` to Azure, so both the root model and the sub-LLM ran at the deployment's default effort (measured as medium for gpt-5), not high/medium as the footnote says. See [KNOWN_ISSUES.md](../../verification/KNOWN_ISSUES.md) K3.


**Failure breakdown** (from [Failure_Analysis.xlsx](../../../results/original_baselines/Failure_Analysis.xlsx), QA Summary; first matching category wins):

| Category | Questions | Share of failures |
|---|---|---|
| F1 did not finish | 0 | 0.0% |
| F2 no / abstained | 29 | 24.6% |
| F3 retrieval miss | 33 | 28.0% |
| F4 partial evidence | 17 | 14.4% |
| F5 all gold, wrong | 39 | 33.1% |
| **Failed in total** | 118 |  |


**Accuracy given the retrieved evidence** (By Recall sheet):

| Evidence retrieved | Questions | Accuracy |
|---|---|---|
| no gold retrieved | 62 | 12.9% |
| some gold | 227 | 91.6% |
| all gold | 541 | 91.7% |


**Turns** (By Turns sheet): limit 100, mean 4.7 (correct 3.9, not correct 9.6); 0 questions used every turn (0.0%), accuracy when answered before the limit 85.7%.

## 8. Results interpretation

**Paired comparisons on the same questions** (Pairwise sheet; exact McNemar test on the discordant pairs):

| Comparison | Other row | Only BCP-11 right | Only other right | p | Reading |
|---|---|---|---|---|---|
| Snippet vs full-doc, GPT-5, Reason-ModernColBERT | BCP-23 | 51 | 65 | 0.23 | not significant |
| ColBERT vs BM25, snippet, GPT-5 | BCP-13 | 196 | 44 | 4.4e-24 | significant at 0.05 |
| ColBERT vs dense, snippet, GPT-5 | BCP-9 | 100 | 47 | 1.5e-05 | significant at 0.05 |


- **Largest failure mode:** wrong answers despite all gold evidence (39 of 118 failures, 33.1%). Retrieval-side failures (F3 + F4) are 42.4% of failures; reading-side (F5) 33.1%. *(Measured.)*
- **With all gold evidence retrieved** the run is right 91.7% of the time (541 questions); with none, 12.9%. The gap between the two bounds how much better retrieval alone could help. *(Measured; causal reading is a hypothesis.)*
- Statements about *why* a configuration wins or loses are hypotheses unless the comparison isolates one variable; see the confounds in section 10.

## 9. Verification status

| Check | Result |
|---|---|
| Experiment previously executed | yes |
| Original outputs found | yes (on the original machine; paths in section 6) |
| Configuration identified | yes |
| Implementation reviewed | yes ([algorithm review](../../verification/REVIEW_snippet_rlm.md)) |
| Lightweight tests | unit tests with mocks ([TESTING_REPORT.md](../../verification/TESTING_REPORT.md)); the experiment itself was not re-run |
| Numbers independently recomputed | **matches the workbook** (accuracy, recall, searches): accuracy 0.8578 from 830 result records ([audit](../../verification/RESULT_INTEGRITY.md)) |
| Reproduced by re-running | **no**: no experiment was re-run; full experimental equivalence of the refactored code is not established |


## 10. Limitations

- No random seed; model sampling is stochastic, so a re-run gives different per-question results (two identical 830-question Snippet RLM runs disagreed on 17.8% of questions in the earlier campaign).
- Reasoning effort actually sent: deployment default for both (no hook loaded; gpt-5 default measured as medium).
- The judge (gpt-4.1) is an LLM; its verdicts were not audited by hand.
- Comparisons with other rows can differ in more than one variable (search cap, turn limit, time limit, effort, index document length); see [ALGORITHM_COMPARISON.md](../ALGORITHM_COMPARISON.md).

## 11. Related files

- [Experiments.xlsx](../../../results/original_baselines/Experiments.xlsx) · [Failure_Analysis.xlsx](../../../results/original_baselines/Failure_Analysis.xlsx) · [registry.yaml](../../../experiments/registry.yaml)
- [Known issues](../../verification/KNOWN_ISSUES.md) · [Result integrity](../../verification/RESULT_INTEGRITY.md) · [Changes from the original code](../../../algorithms/snippet_rlm/CHANGES_FROM_ORIGINAL.md)
