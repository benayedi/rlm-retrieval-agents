# BCP-9: Snippet RLM · BrowseComp-Plus · gpt-5 · Qwen3-Embedding-8B

Status: complete. Row 9 of the `BrowseComp+` sheet of [Experiments.xlsx](../../../results/original_baselines/Experiments.xlsx).

## 1. Experiment overview

|  |  |
|---|---|
| Identifier | `BCP-9` (sheet `BrowseComp+`, row 9) |
| Algorithm | [Snippet RLM](../../../algorithms/snippet_rlm/README.md) |
| Benchmark | BrowseComp-Plus (830 questions, 100,195 documents) |
| Model | gpt-5 (Azure) |
| Retriever | Qwen3-Embedding-8B, single-vector dense, FAISS |
| Research question | How accurate is the Snippet RLM with GPT-5 when the retriever is a strong dense embedding model? |
| Original execution | 2026-09-11 to 2026-09-12 (trajectory timestamps) |
| Run name in the logs | `rlmfix32` |

## 2. Research motivation

How accurate is the Snippet RLM with GPT-5 when the retriever is a strong dense embedding model? This row is one cell of the comparison matrix in Experiments.xlsx: the same benchmark is run with each algorithm × model × retriever combination, so this row is read against its neighbours (section 8). See [ALGORITHM_COMPARISON.md](../ALGORITHM_COMPARISON.md) for the matrix and [research_overview.md](../../research_overview.md) for the research questions.

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
| Retriever / index | Qwen3-Embedding-8B, single-vector dense, FAISS |
| Turns | 100 |
| Searches | no cap |
| Results per search | 5 (default; the model may request more) |
| Snippet | min(2,000 chars, 512 tokens) |
| Timeout | none |
| Compaction | off |
| Random seed | none set (API / vLLM sampling defaults); runs are not bit-reproducible |
| Harness version | September state of the harness; closest preserved copy: algorithms/snippet_rlm/original/RLM_git_7685538 (commit of 2026-09-22, after the run) |


Exact environment and arguments ([algorithms/snippet_rlm/configs/BCP-9.env](../../../algorithms/snippet_rlm/configs/BCP-9.env)):

```bash
# BCP-9  Snippet RLM | GPT-5 (requested root high, llm_query medium; see the effort note) | Qwen3-Embedding-8B (FAISS
#        IVF-SQ8, nprobe 64) on :8001 | BrowseComp+ 830 q | run name rlmfix32 (September 2026)
# Budgets: 100 turns, depth 1, NO compaction, NO search cap, top_k default 5, 512-token snippets, 60K REPL ceiling
#          (default), no per-question timeout, W=8 worker loops (W default; actual W UNKNOWN).
# Launcher: experiments/september/bcp_snip_qwen_embed.sh (was scripts/run_qwen_full.sh, run from <original machine>/master,
#          which no longer exists). Re-graded with gpt-4.1: <original repo>/logs/bcp_rejudge_gpt41/10.json.
# Original outputs: <original repo>/logs/rlmfix32_q<0..829>.json (830 files; their .log files and the trajectories
#          logs/rlm_traj32 were written under master/ and are not preserved there).
# EFFORT NOTE: no startup hook was on PYTHONPATH, so rlm's Azure client dropped reasoning_effort and GPT-5 ran at the
#          deployment default (measured: medium); the workbook marks this row "GPT 5 (high)*".
# CODE NOTE: the records lack the child_* fields, so the harness that ran predates commit 7685538 (see CHANGES).
# Use: experiments/run_config.sh configs/BCP-9.env
ROW=BCP-9
LAUNCHER=experiments/september/bcp_snip_qwen_embed.sh
LAUNCHER_ARGS=()
export W=8   # worker loops; launcher default. Actual value UNKNOWN (not logged)
# --- effective environment (set by the launcher; listed for the record) ---
export EMBEDDING_SERVER_URL=http://localhost:8001 AZURE_OPENAI_REASONING_EFFORT=high SUB_LM_REASONING_EFFORT=medium
export AZURE_OPENAI_API_VERSION=2025-04-01-preview LIGHTON_TOOL_DOCS=1 LIGHTON_TASK_TEXT=1
export SEARCH_SNIPPET_TOKENS=512 OFFICIAL_TOP_K=5 ORCHESTRATOR=0
# RLM_TRAJ_DIR=logs/rlm_traj32 (relative to the working directory); unset: MAX_TOP_K SEARCH_SNIPPET_LEN SUB_LM_MODEL.
# CLI per question: harness.py --start Q --end Q+1 --max-iterations 100 --max-depth 1 --out logs/rlmfix32_qQ
# --- from <original machine>/master/.env (file gone) ---
AZURE_OPENAI_MODEL_ORIGINAL=UNKNOWN        # deployment name of gpt-5 then; the launcher now uses $AZURE_DEPLOYMENT_GPT5
AZURE_OPENAI_JUDGE_MODEL_ORIGINAL=UNKNOWN  # in-run judge of the `correct` field (the inventory's Sept. judge: gpt-5.4-mini)
OTHER_DOTENV_VARIABLES=UNKNOWN             # anything else master/.env exported
```

## 5. Execution instructions (reproduction, not executed for this documentation)

These commands launch the full experiment: hundreds of model calls and hours of GPU or API time. They were **not** run while preparing this repository.

```bash
set -a; . configs/paths.env; . algorithms/snippet_rlm/configs/BCP-9.env; set +a
# then the launcher; its header documents its arguments:
algorithms/snippet_rlm/experiments/september/bcp_snip_qwen_embed.sh
```

Launcher: [algorithms/snippet_rlm/experiments/september/bcp_snip_qwen_embed.sh](../../../algorithms/snippet_rlm/experiments/september/bcp_snip_qwen_embed.sh). It reads the benchmark questions, talks to the retrieval server named in the config and writes one record per question (plus trajectories) under `$RUNS_DIR`. Grade afterwards with the matching script in [shared/rlm_shared/grading/](../../../shared/rlm_shared/grading).

## 6. Implementation references

- Algorithm source: [harness.py](../../../algorithms/snippet_rlm/src/harness.py), [sitecustomize.py](../../../algorithms/snippet_rlm/src/hooks/azure_sampling/sitecustomize.py), [sitecustomize.py](../../../algorithms/snippet_rlm/src/hooks/cap_ranking/sitecustomize.py), [sitecustomize.py](../../../algorithms/snippet_rlm/src/hooks/retry429/sitecustomize.py), [sitecustomize.py](../../../algorithms/snippet_rlm/src/hooks/retry429_cap_ranking/sitecustomize.py), [sitecustomize.py](../../../algorithms/snippet_rlm/src/hooks/run5d/sitecustomize.py), [prompt_modules.py](../../../algorithms/snippet_rlm/src/prompt_modules.py), [run_browsecomp.py](../../../algorithms/snippet_rlm/src/run_browsecomp.py), [run_frames.py](../../../algorithms/snippet_rlm/src/run_frames.py), [run_obliq.py](../../../algorithms/snippet_rlm/src/run_obliq.py)
- Original code that produced the result (verbatim, SHA-256 manifest): [algorithms/snippet_rlm/original/](../../../algorithms/snippet_rlm/original/MANIFEST.json)
- Configuration: [algorithms/snippet_rlm/configs/BCP-9.env](../../../algorithms/snippet_rlm/configs/BCP-9.env) · Launcher: [algorithms/snippet_rlm/experiments/september/bcp_snip_qwen_embed.sh](../../../algorithms/snippet_rlm/experiments/september/bcp_snip_qwen_embed.sh)
- Data loaders: [shared/rlm_shared/](../../../shared/rlm_shared/__init__.py) (`browsecomp_data.py`, `obliq_data.py`)
- Grading: [shared/rlm_shared/grading/](../../../shared/rlm_shared/grading) · Metrics: [metrics.py](../../../shared/rlm_shared/metrics.py)
- Tests: [algorithms/snippet_rlm/tests/](../../../algorithms/snippet_rlm/tests), [tests/](../../../tests)
- Original outputs (not in this repository; on the machine that ran the experiments): `logs/rlmfix32_q*.json`; `logs/rlm_traj32/ (deleted; turn counts kept in scripts/workbook/tokmaps)`; `logs/bcp_rejudge_gpt41/10.json (gpt-4.1 verdicts)`

## 7. Existing experimental results

Reported in Experiments.xlsx (every non-empty column of the row):

| Metric | Value |
|---|---|
| Accuracy | 0.794 |
| Recall | 0.804 |
| NDCG | — |
| Avg. Search Calls Per Query | 29.5 |
| Avg. Iterations Per Query | 5.6 |
| Time Per Query | 191 s (3.2 min) |
| Cached Cost Per Query | not logged |
| Uncached Cost Per Query | ≈ 0.50 |
| Input Tokens Per Query | ≈ 267,564 |
| Output Tokens Per Query | ≈ 16,474 |


Workbook footnotes that apply to this row:

> * GPT 5 (high)* (Snippet RLM rows): the root model runs at high effort, but its sub-LLM reading calls (llm_query) run at medium.

> Input / Output Tokens Per Query: all model calls of a question (root + sub-LLM), reasoning tokens included in output; these do not depend on how many runs shared a server, unlike Time Per Query. Exact from logged per-call usage, except values marked ≈ (Sept. 2026 snippet runs, whose logs overwrote the root model's usage with the sub-LLM's: rebuilt from the trajectories, about 10% low when the same method is checked on a run with exact usage). Uncached cost = input x list input price + output x list output price. Avg. Iterations Per Query: root-model turns per question, counted from the trajectory logs (snippet RLM) or the run records (full-doc RLM); sub-LLM calls are not counted. A run that reaches its turn cap adds one forced final-answer turn.

> **Correction (this audit):** the launcher of this run loaded no hook that forwards `reasoning_effort` to Azure, so both the root model and the sub-LLM ran at the deployment's default effort (measured as medium for gpt-5), not high/medium as the footnote says. See [KNOWN_ISSUES.md](../../verification/KNOWN_ISSUES.md) K3.


**Failure breakdown** (from [Failure_Analysis.xlsx](../../../results/original_baselines/Failure_Analysis.xlsx), QA Summary; first matching category wins):

| Category | Questions | Share of failures |
|---|---|---|
| F1 did not finish | 0 | 0.0% |
| F2 no / abstained | 45 | 26.3% |
| F3 retrieval miss | 58 | 33.9% |
| F4 partial evidence | 25 | 14.6% |
| F5 all gold, wrong | 43 | 25.1% |
| **Failed in total** | 171 |  |


**Accuracy given the retrieved evidence** (By Recall sheet):

| Evidence retrieved | Questions | Accuracy |
|---|---|---|
| no gold retrieved | 97 | 7.2% |
| some gold | 177 | 82.5% |
| all gold | 556 | 91.0% |


**Turns** (By Turns sheet): limit 100, mean 5.6 (correct 4.2, not correct 11.2); 0 questions used every turn (0.0%), accuracy when answered before the limit 79.4%.

## 8. Results interpretation

**Paired comparisons on the same questions** (Pairwise sheet; exact McNemar test on the discordant pairs):

| Comparison | Other row | Only BCP-9 right | Only other right | p | Reading |
|---|---|---|---|---|---|
| Snippet vs full-doc, GPT-5, Qwen3-Embed-8B | BCP-21 | 57 | 88 | 0.012 | significant at 0.05 |
| ColBERT vs dense, snippet, GPT-5 | BCP-11 | 47 | 100 | 1.5e-05 | significant at 0.05 |


- **Largest failure mode:** never retrieving a gold document (58 of 171 failures, 33.9%). Retrieval-side failures (F3 + F4) are 48.5% of failures; reading-side (F5) 25.1%. *(Measured.)*
- **With all gold evidence retrieved** the run is right 91.0% of the time (556 questions); with none, 7.2%. The gap between the two bounds how much better retrieval alone could help. *(Measured; causal reading is a hypothesis.)*
- Statements about *why* a configuration wins or loses are hypotheses unless the comparison isolates one variable; see the confounds in section 10.

## 9. Verification status

| Check | Result |
|---|---|
| Experiment previously executed | yes |
| Original outputs found | yes (on the original machine; paths in section 6) |
| Configuration identified | yes |
| Implementation reviewed | yes ([algorithm review](../../verification/REVIEW_snippet_rlm.md)) |
| Lightweight tests | unit tests with mocks ([TESTING_REPORT.md](../../verification/TESTING_REPORT.md)); the experiment itself was not re-run |
| Numbers independently recomputed | **matches the workbook** (accuracy, recall, searches): accuracy 0.7940 from 830 result records ([audit](../../verification/RESULT_INTEGRITY.md)) |
| Reproduced by re-running | **no**: no experiment was re-run; full experimental equivalence of the refactored code is not established |


## 10. Limitations

- No random seed; model sampling is stochastic, so a re-run gives different per-question results (two identical 830-question Snippet RLM runs disagreed on 17.8% of questions in the earlier campaign).
- Reasoning effort actually sent: deployment default for both (no hook loaded; gpt-5 default measured as medium).
- The judge (gpt-4.1) is an LLM; its verdicts were not audited by hand.
- Comparisons with other rows can differ in more than one variable (search cap, turn limit, time limit, effort, index document length); see [ALGORITHM_COMPARISON.md](../ALGORITHM_COMPARISON.md).

## 11. Related files

- [Experiments.xlsx](../../../results/original_baselines/Experiments.xlsx) · [Failure_Analysis.xlsx](../../../results/original_baselines/Failure_Analysis.xlsx) · [registry.yaml](../../../experiments/registry.yaml)
- [Known issues](../../verification/KNOWN_ISSUES.md) · [Result integrity](../../verification/RESULT_INTEGRITY.md) · [Changes from the original code](../../../algorithms/snippet_rlm/CHANGES_FROM_ORIGINAL.md)
