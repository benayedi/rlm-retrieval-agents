# BCP-13: Snippet RLM · BrowseComp-Plus · gpt-5 · BM25 (bm25s

Status: complete. Row 13 of the `BrowseComp+` sheet of [Experiments.xlsx](../../../results/original_baselines/Experiments.xlsx).

## 1. Experiment overview

|  |  |
|---|---|
| Identifier | `BCP-13` (sheet `BrowseComp+`, row 13) |
| Algorithm | [Snippet RLM](../../../algorithms/snippet_rlm/README.md) |
| Benchmark | BrowseComp-Plus (830 questions, 100,195 documents) |
| Model | gpt-5 (Azure) |
| Retriever | BM25 (bm25s, k1 0.9, b 0.4) |
| Research question | How much does a lexical retriever cost the Snippet RLM, and can unlimited searching compensate? |
| Original execution | 2026-10-27 to 2026-10-28 |
| Run name in the logs | `bcp_snip_gpt5_bm25` |

## 2. Research motivation

How much does a lexical retriever cost the Snippet RLM, and can unlimited searching compensate? This row is one cell of the comparison matrix in Experiments.xlsx: the same benchmark is run with each algorithm × model × retriever combination, so this row is read against its neighbours (section 8). See [ALGORITHM_COMPARISON.md](../ALGORITHM_COMPARISON.md) for the matrix and [research_overview.md](../../research_overview.md) for the research questions.

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
| Reasoning effort actually sent | as requested (run5d hook loaded) |
| Retriever / index | BM25 (bm25s, k1 0.9, b 0.4) |
| Turns | 100 |
| Searches | no cap |
| Results per search | 5 (default; the model may request more) |
| Snippet | min(2,000 chars, 512 tokens) |
| Timeout | 7,200 s per question; $10 cost watchdog |
| Compaction | on |
| Random seed | none set (API / vLLM sampling defaults); runs are not bit-reproducible |


Exact environment and arguments ([algorithms/snippet_rlm/configs/BCP-13.env](../../../algorithms/snippet_rlm/configs/BCP-13.env)):

```bash
# BCP-13  Snippet RLM | GPT-5 (root high, llm_query medium) | BM25 (k1 0.9, b 0.4) on :8044 (rlm wire) | BrowseComp+ 830 q
# Budgets: 100 turns, depth 1, --compaction, NO search cap, top_k default 5, 512-token snippets, 7200 s per question
#          (timeout), 6 workers. Ops: experiments/question_watchdog.sh <out> 5 10 (WARN at $5, kill at $10 per question;
#          it killed q212) and experiments/budget_guard.sh <out>/usage.jsonl <budget> <pid> with budget UNKNOWN (only
#          its exit line "logged spend $806.14" survives); graded by experiments/ops/bcp_snip_gpt5_bm25_finish.sh.
# Launch line (logs/run5d/full/bcp_snip_gpt5_bm25.out): 2026-10-27 18:27:04 model=gpt5 retriever=http://localhost:8044
#          q=[0,830) workers=6; done 2026-10-28 12:31:43 with 828 result files.
# Original outputs: <original repo>/logs/run5d/full/bcp_snip_gpt5_bm25/ (q<N>.json, q<N>.log, traj/, usage.jsonl),
#          grades logs/run5d/full/bcp_snip_gpt5_bm25.grade.txt (gpt-4.1).
# Use: experiments/run_config.sh configs/BCP-13.env   (sources configs/paths.env first)
ROW=BCP-13
LAUNCHER=experiments/bcp_snip.sh
LAUNCHER_ARGS=(gpt5 http://localhost:8044 "$RUNS_DIR/run5d/full/bcp_snip_gpt5_bm25" 6)
# --- effective environment of every question process (the launcher sets all of these itself; listed for the record) ---
export EMBEDDING_SERVER_URL=http://localhost:8044 ORCHESTRATOR=0 LIGHTON_TOOL_DOCS=1 LIGHTON_TASK_TEXT=1
export SEARCH_SNIPPET_TOKENS=512 OFFICIAL_TOP_K=5 AZURE_OPENAI_API_VERSION=2025-04-01-preview
export AZURE_OPENAI_REASONING_EFFORT=high SUB_LM_REASONING_EFFORT=medium   # sent to Azure: hook src/hooks/run5d loaded
# AZURE_OPENAI_MODEL = $AZURE_DEPLOYMENT_GPT5 (gpt-5); PYTHONPATH hook = src/hooks/run5d; CLI per question:
#   run_browsecomp.py --start Q --end Q+1 --max-iterations 100 --max-depth 1 --compaction --out <out>/qQ
# Unset by the launcher: MAX_TOP_K SEARCH_SNIPPET_LEN SUB_LM_MODEL AZURE_OPENAI_DEPLOYMENT LOCAL_LLM_BASE_URL LOCAL_LLM_MODEL
#   QWEN_PRESENCE_PENALTY QWEN_FENCE_TOLERANT BCP_SEARCH_CAP
# --- implicit environment (not set by the launcher) ---
# The original rag/agent.py ran load_dotenv() on its first import (first search_index call), which filled variables
# missing from the environment from <original repo>/.env (last modified 2026-10-27 14:11, before this run). The only
# one the snippet path reads afterwards is the IN-RUN judge model of the `correct` field (not the gpt-4.1 grade):
export AZURE_OPENAI_JUDGE_MODEL=gpt-5.4-mini
```

## 5. Execution instructions (reproduction, not executed for this documentation)

These commands launch the full experiment: hundreds of model calls and hours of GPU or API time. They were **not** run while preparing this repository.

```bash
set -a; . configs/paths.env; . algorithms/snippet_rlm/configs/BCP-13.env; set +a
# then the launcher; its header documents its arguments:
algorithms/snippet_rlm/experiments/bcp_snip.sh
```

Launcher: [algorithms/snippet_rlm/experiments/bcp_snip.sh](../../../algorithms/snippet_rlm/experiments/bcp_snip.sh). It reads the benchmark questions, talks to the retrieval server named in the config and writes one record per question (plus trajectories) under `$RUNS_DIR`. Grade afterwards with the matching script in [shared/rlm_shared/grading/](../../../shared/rlm_shared/grading).

## 6. Implementation references

- Algorithm source: [harness.py](../../../algorithms/snippet_rlm/src/harness.py), [sitecustomize.py](../../../algorithms/snippet_rlm/src/hooks/azure_sampling/sitecustomize.py), [sitecustomize.py](../../../algorithms/snippet_rlm/src/hooks/cap_ranking/sitecustomize.py), [sitecustomize.py](../../../algorithms/snippet_rlm/src/hooks/retry429/sitecustomize.py), [sitecustomize.py](../../../algorithms/snippet_rlm/src/hooks/retry429_cap_ranking/sitecustomize.py), [sitecustomize.py](../../../algorithms/snippet_rlm/src/hooks/run5d/sitecustomize.py), [prompt_modules.py](../../../algorithms/snippet_rlm/src/prompt_modules.py), [run_browsecomp.py](../../../algorithms/snippet_rlm/src/run_browsecomp.py), [run_frames.py](../../../algorithms/snippet_rlm/src/run_frames.py), [run_obliq.py](../../../algorithms/snippet_rlm/src/run_obliq.py)
- Original code that produced the result (verbatim, SHA-256 manifest): [algorithms/snippet_rlm/original/](../../../algorithms/snippet_rlm/original/MANIFEST.json)
- Configuration: [algorithms/snippet_rlm/configs/BCP-13.env](../../../algorithms/snippet_rlm/configs/BCP-13.env) · Launcher: [algorithms/snippet_rlm/experiments/bcp_snip.sh](../../../algorithms/snippet_rlm/experiments/bcp_snip.sh)
- Data loaders: [shared/rlm_shared/](../../../shared/rlm_shared/__init__.py) (`browsecomp_data.py`, `obliq_data.py`)
- Grading: [shared/rlm_shared/grading/](../../../shared/rlm_shared/grading) · Metrics: [metrics.py](../../../shared/rlm_shared/metrics.py)
- Tests: [algorithms/snippet_rlm/tests/](../../../algorithms/snippet_rlm/tests), [tests/](../../../tests)
- **Per-question outputs in this repository** (Git LFS): [trajectories/snippet_rlm/BCP-13/](../../../trajectories/snippet_rlm/BCP-13/MANIFEST.json). 2,496 files, 11.88 GB as written by the run, encrypted with the BrowseComp canary key; extract with `python scripts/provenance/unpack_trajectories.py BCP-13 OUT_DIR` ([format notes](../../../trajectories/README.md))
- Original locations on the machine that ran the experiments: `logs/run5d/full/bcp_snip_gpt5_bm25/ (q*.json, traj/, usage.jsonl, graded_gpt41.json)`

## 7. Existing experimental results

Reported in Experiments.xlsx (every non-empty column of the row):

| Metric | Value |
|---|---|
| Accuracy | 0.6747 |
| Recall | 0.6815 |
| NDCG | — |
| Avg. Search Calls Per Query | 60.87 |
| Avg. Iterations Per Query | 8.3 |
| Time Per Query | 452 s (7.5 min) |
| Cached Cost Per Query | 0.9737 |
| Uncached Cost Per Query | 1.1999 |
| Input Tokens Per Query | 638846 |
| Output Tokens Per Query | 40138 |


Workbook footnotes that apply to this row:

> * GPT 5 (high)* (Snippet RLM rows): the root model runs at high effort, but its sub-LLM reading calls (llm_query) run at medium.

> Input / Output Tokens Per Query: all model calls of a question (root + sub-LLM), reasoning tokens included in output; these do not depend on how many runs shared a server, unlike Time Per Query. Exact from logged per-call usage, except values marked ≈ (Sept. 2026 snippet runs, whose logs overwrote the root model's usage with the sub-LLM's: rebuilt from the trajectories, about 10% low when the same method is checked on a run with exact usage). Uncached cost = input x list input price + output x list output price. Avg. Iterations Per Query: root-model turns per question, counted from the trajectory logs (snippet RLM) or the run records (full-doc RLM); sub-LLM calls are not counted. A run that reaches its turn cap adds one forced final-answer turn.


**Failure breakdown** (from [Failure_Analysis.xlsx](../../../results/original_baselines/Failure_Analysis.xlsx), QA Summary; first matching category wins):

| Category | Questions | Share of failures |
|---|---|---|
| F1 did not finish | 9 | 3.3% |
| F2 no / abstained | 57 | 21.1% |
| F3 retrieval miss | 123 | 45.6% |
| F4 partial evidence | 43 | 15.9% |
| F5 all gold, wrong | 38 | 14.1% |
| **Failed in total** | 270 |  |


**Accuracy given the retrieved evidence** (By Recall sheet):

| Evidence retrieved | Questions | Accuracy |
|---|---|---|
| no gold retrieved | 169 | 4.1% |
| some gold | 199 | 73.9% |
| all gold | 453 | 89.6% |


**Turns** (By Turns sheet): limit 100, mean 8.4 (correct 5.8, not correct 13.8); 0 questions used every turn (0.0%), accuracy when answered before the limit 68.2%.

## 8. Results interpretation

**Paired comparisons on the same questions** (Pairwise sheet; exact McNemar test on the discordant pairs):

| Comparison | Other row | Only BCP-13 right | Only other right | p | Reading |
|---|---|---|---|---|---|
| Snippet vs full-doc, GPT-5, BM25 | BCP-25 | 136 | 74 | 2.3e-05 | significant at 0.05 |
| ColBERT vs BM25, snippet, GPT-5 | BCP-11 | 44 | 196 | 4.4e-24 | significant at 0.05 |


- **Largest failure mode:** never retrieving a gold document (123 of 270 failures, 45.6%). Retrieval-side failures (F3 + F4) are 61.5% of failures; reading-side (F5) 14.1%. *(Measured.)*
- **With all gold evidence retrieved** the run is right 89.6% of the time (453 questions); with none, 4.1%. The gap between the two bounds how much better retrieval alone could help. *(Measured; causal reading is a hypothesis.)*
- Statements about *why* a configuration wins or loses are hypotheses unless the comparison isolates one variable; see the confounds in section 10.

## 9. Verification status

| Check | Result |
|---|---|
| Experiment previously executed | yes |
| Original outputs found | yes (on the original machine; paths in section 6) |
| Configuration identified | yes |
| Implementation reviewed | yes ([algorithm review](../../verification/REVIEW_snippet_rlm.md)) |
| Lightweight tests | unit tests with mocks ([TESTING_REPORT.md](../../verification/TESTING_REPORT.md)); the experiment itself was not re-run |
| Numbers independently recomputed | **matches the workbook** (accuracy, recall, searches): accuracy 0.6747 from 828 result records ([audit](../../verification/RESULT_INTEGRITY.md)) |
| Reproduced by re-running | **no**: no experiment was re-run; full experimental equivalence of the refactored code is not established |


## 10. Limitations

- No random seed; model sampling is stochastic, so a re-run gives different per-question results (two identical 830-question Snippet RLM runs disagreed on 17.8% of questions in the earlier campaign).
- Reasoning effort actually sent: as requested (run5d hook loaded).
- The judge (gpt-4.1) is an LLM; its verdicts were not audited by hand.
- Comparisons with other rows can differ in more than one variable (search cap, turn limit, time limit, effort, index document length); see [ALGORITHM_COMPARISON.md](../ALGORITHM_COMPARISON.md).

## 11. Related files

- [Experiments.xlsx](../../../results/original_baselines/Experiments.xlsx) · [Failure_Analysis.xlsx](../../../results/original_baselines/Failure_Analysis.xlsx) · [registry.yaml](../../../experiments/registry.yaml)
- [Known issues](../../verification/KNOWN_ISSUES.md) · [Result integrity](../../verification/RESULT_INTEGRITY.md) · [Changes from the original code](../../../algorithms/snippet_rlm/CHANGES_FROM_ORIGINAL.md)
