# FR-7: Snippet RLM · FRAMES · gpt-5 · BM25 (bm25s

Status: complete. Row 7 of the `FRAMES` sheet of [Experiments.xlsx](../../../results/original_baselines/Experiments.xlsx).

## 1. Experiment overview

|  |  |
|---|---|
| Identifier | `FR-7` (sheet `FRAMES`, row 7) |
| Algorithm | [Snippet RLM](../../../algorithms/snippet_rlm/README.md) |
| Benchmark | FRAMES (824 questions, Wikipedia 20230601.en) |
| Model | gpt-5 (Azure) |
| Retriever | BM25 (bm25s, k1 0.9, b 0.4) over Wikipedia |
| Research question | Snippet RLM on multi-hop Wikipedia questions where answers sit deep inside long articles. |
| Original execution | 2026-10-27 |
| Run name in the logs | `frames_snip_gpt5` |

## 2. Research motivation

Snippet RLM on multi-hop Wikipedia questions where answers sit deep inside long articles. This row is one cell of the comparison matrix in Experiments.xlsx: the same benchmark is run with each algorithm × model × retriever combination, so this row is read against its neighbours (section 8). See [ALGORITHM_COMPARISON.md](../ALGORITHM_COMPARISON.md) for the matrix and [research_overview.md](../../research_overview.md) for the research questions.

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
| Retriever / index | BM25 (bm25s, k1 0.9, b 0.4) over Wikipedia |
| Turns | 50 |
| Searches | 25 |
| Results per search | 5 (default; the model may request more) |
| Snippet | min(2,000 chars, 512 tokens) |
| Timeout | 7,200 s per question |
| Compaction | on |
| Random seed | none set (API / vLLM sampling defaults); runs are not bit-reproducible |


Exact environment and arguments ([algorithms/snippet_rlm/configs/FR-7.env](../../../algorithms/snippet_rlm/configs/FR-7.env)):

```bash
# FR-7  Snippet RLM | GPT-5 (root high, llm_query medium) | BM25 over Wikipedia 2023-06-01 on :8040 (rlm wire) | FRAMES 824 q
# Budgets: 50 turns, depth 1 (harness default), --compaction, search cap 25 (FRAMES_SEARCH_CAP), top_k default 5,
#          512-token snippets, 7200 s per question, 6 workers; RETRIEVER_LINE=0 + FRAMES_BM25_LINE=1 (BM25 sentence instead
#          of the dense-retriever sentence). In-run judge: local Qwen3.6 on :8030 (interim only; graded later with gpt-4.1).
# Launch lines (logs/run5d/full/frames_snip_gpt5.out, frames_snip_gpt5_retry.out): 2026-10-27 18:27:04 q=[0,824) workers=6,
#          done 22:47:39 with 819 results; then the same command with workers=3 (22:48:27) filled the 5 missing questions,
#          done 22:59:19 with 824 results.
# Original outputs: <original repo>/logs/run5d/full/frames_snip_gpt5/ (q<NNN>.json, .log, traj/, usage.jsonl)
# Use: experiments/run_config.sh configs/FR-7.env
ROW=FR-7
LAUNCHER=experiments/frames_snip.sh
LAUNCHER_ARGS=(gpt5 "$RUNS_DIR/run5d/full/frames_snip_gpt5" 6)
# --- effective environment of every question process (set by the launcher; listed for the record) ---
export EMBEDDING_SERVER_URL=http://localhost:8040 FRAMES_SEARCH_CAP=25 RETRIEVER_LINE=0 FRAMES_BM25_LINE=1
export ORCHESTRATOR=0 LIGHTON_TOOL_DOCS=1 LIGHTON_TASK_TEXT=1 SEARCH_SNIPPET_TOKENS=512 OFFICIAL_TOP_K=5
export AZURE_OPENAI_API_VERSION=2025-04-01-preview LOCAL_JUDGE_BASE_URL=http://localhost:8030/v1 LOCAL_JUDGE_MODEL=qwen36-35b-a3b
export AZURE_OPENAI_REASONING_EFFORT=high SUB_LM_REASONING_EFFORT=medium   # sent to Azure: hook src/hooks/run5d loaded
# AZURE_OPENAI_MODEL = $AZURE_DEPLOYMENT_GPT5 (gpt-5). CLI per question:
#   run_frames.py --start Q --end Q+1 --max-iterations 50 --compaction --out <out>/qNNN   (timeout 7200)
# Unset: MAX_TOP_K SEARCH_SNIPPET_LEN SUB_LM_MODEL AZURE_OPENAI_DEPLOYMENT LOCAL_LLM_BASE_URL LOCAL_LLM_MODEL FRAMES_REPL_CUE
#   FRAMES_FENCE_TOLERANT. Implicit .env (see BCP-13.env): nothing the FRAMES path reads.
```

## 5. Execution instructions (reproduction, not executed for this documentation)

These commands launch the full experiment: hundreds of model calls and hours of GPU or API time. They were **not** run while preparing this repository.

```bash
set -a; . configs/paths.env; . algorithms/snippet_rlm/configs/FR-7.env; set +a
# then the launcher; its header documents its arguments:
algorithms/snippet_rlm/experiments/frames_snip.sh
```

Launcher: [algorithms/snippet_rlm/experiments/frames_snip.sh](../../../algorithms/snippet_rlm/experiments/frames_snip.sh). It reads the benchmark questions, talks to the retrieval server named in the config and writes one record per question (plus trajectories) under `$RUNS_DIR`. Grade afterwards with the matching script in [shared/rlm_shared/grading/](../../../shared/rlm_shared/grading).

## 6. Implementation references

- Algorithm source: [harness.py](../../../algorithms/snippet_rlm/src/harness.py), [sitecustomize.py](../../../algorithms/snippet_rlm/src/hooks/azure_sampling/sitecustomize.py), [sitecustomize.py](../../../algorithms/snippet_rlm/src/hooks/cap_ranking/sitecustomize.py), [sitecustomize.py](../../../algorithms/snippet_rlm/src/hooks/retry429/sitecustomize.py), [sitecustomize.py](../../../algorithms/snippet_rlm/src/hooks/retry429_cap_ranking/sitecustomize.py), [sitecustomize.py](../../../algorithms/snippet_rlm/src/hooks/run5d/sitecustomize.py), [prompt_modules.py](../../../algorithms/snippet_rlm/src/prompt_modules.py), [run_browsecomp.py](../../../algorithms/snippet_rlm/src/run_browsecomp.py), [run_frames.py](../../../algorithms/snippet_rlm/src/run_frames.py), [run_obliq.py](../../../algorithms/snippet_rlm/src/run_obliq.py)
- Original code that produced the result (verbatim, SHA-256 manifest): [algorithms/snippet_rlm/original/](../../../algorithms/snippet_rlm/original/MANIFEST.json)
- Configuration: [algorithms/snippet_rlm/configs/FR-7.env](../../../algorithms/snippet_rlm/configs/FR-7.env) · Launcher: [algorithms/snippet_rlm/experiments/frames_snip.sh](../../../algorithms/snippet_rlm/experiments/frames_snip.sh)
- Data loaders: [shared/rlm_shared/](../../../shared/rlm_shared/__init__.py) (`browsecomp_data.py`, `obliq_data.py`)
- Grading: [shared/rlm_shared/grading/](../../../shared/rlm_shared/grading) · Metrics: [metrics.py](../../../shared/rlm_shared/metrics.py)
- Tests: [algorithms/snippet_rlm/tests/](../../../algorithms/snippet_rlm/tests), [tests/](../../../tests)
- **Per-question outputs in this repository** (Git LFS): [trajectories/snippet_rlm/FR-7/](../../../trajectories/snippet_rlm/FR-7/MANIFEST.json). 2,500 files, 0.50 GB as written by the run; extract with `python scripts/provenance/unpack_trajectories.py FR-7 OUT_DIR` ([format notes](../../../trajectories/README.md))
- Original locations on the machine that ran the experiments: `logs/run5d/full/frames_snip_gpt5/`

## 7. Existing experimental results

Reported in Experiments.xlsx (every non-empty column of the row):

| Metric | Value |
|---|---|
| Accuracy | 0.7633 |
| Recall | 0.6716 |
| NDCG | — |
| Avg. Search Calls Per Query | 6.59 |
| Avg. Iterations Per Query | 2.6 |
| Time Per Query | 107 s (1.8 min) |
| Cached Cost Per Query | 0.1424 |
| Uncached Cost Per Query | 0.146 |
| Input Tokens Per Query | 35135 |
| Output Tokens Per Query | 10206 |


Workbook footnotes that apply to this row:

> * GPT 5 (high)* (Snippet RLM row): the root model runs at high effort, but its sub-LLM reading calls (llm_query) run at medium. Snippet RLM: 25-search cap, 50 turns, compaction; Full-Doc RLM: max 10 searches, 30 turns. A question without a usable answer counts wrong. Qwen3.6 Full-Doc row re-judged with gpt-4.1 on 2026-10-27 (0.6104; the Qwen3.6 self-judge had given 0.6274). Costs from the logged Azure usage: gpt-5 $1.25 / $0.125 (cached) / $10 per 1M input / cached input / output tokens. Input / Output Tokens Per Query: all model calls of a question (root + sub-LLM), reasoning tokens included in output; these do not depend on how many runs shared a server, unlike Time Per Query. Exact from logged per-call usage, except values marked ≈ (Sept. 2026 snippet runs, whose logs overwrote the root model's usage with the sub-LLM's: rebuilt from the trajectories, about 10% low when the same method is checked on a run with exact usage). Uncached cost = input x list input price + output x list output price. Avg. Iterations Per Query: root-model turns per question, counted from the trajectory logs (snippet RLM) or the run records (full-doc RLM); sub-LLM calls are not counted. A run that reaches its turn cap adds one forced final-answer turn.


**Failure breakdown** (from [Failure_Analysis.xlsx](../../../results/original_baselines/Failure_Analysis.xlsx), QA Summary; first matching category wins):

| Category | Questions | Share of failures |
|---|---|---|
| F1 did not finish | 0 | 0.0% |
| F2 no / abstained | 13 | 6.7% |
| F3 retrieval miss | 23 | 11.8% |
| F4 partial evidence | 99 | 50.8% |
| F5 all gold, wrong | 60 | 30.8% |
| **Failed in total** | 195 |  |


**Accuracy given the retrieved evidence** (By Recall sheet):

| Evidence retrieved | Questions | Accuracy |
|---|---|---|
| no gold retrieved | 69 | 65.2% |
| some gold | 451 | 76.5% |
| all gold | 304 | 78.6% |


**Turns** (By Turns sheet): limit 50, mean 2.6 (correct 2.3, not correct 3.6); 0 questions used every turn (0.0%), accuracy when answered before the limit 76.3%.

## 8. Results interpretation

**Paired comparisons on the same questions** (Pairwise sheet; exact McNemar test on the discordant pairs):

| Comparison | Other row | Only FR-7 right | Only other right | p | Reading |
|---|---|---|---|---|---|
| Snippet vs full-doc, GPT-5, BM25 (FRAMES) | FR-11 | 38 | 74 | 0.00086 | significant at 0.05 |
| GPT-5 vs Qwen3.6, snippet (FRAMES) | FR-9 | 168 | 50 | 4.1e-16 | significant at 0.05 |


- **Largest failure mode:** partial evidence (99 of 195 failures, 50.8%). Retrieval-side failures (F3 + F4) are 62.6% of failures; reading-side (F5) 30.8%. *(Measured.)*
- **With all gold evidence retrieved** the run is right 78.6% of the time (304 questions); with none, 65.2%. The gap between the two bounds how much better retrieval alone could help. *(Measured; causal reading is a hypothesis.)*
- Statements about *why* a configuration wins or loses are hypotheses unless the comparison isolates one variable; see the confounds in section 10.

## 9. Verification status

| Check | Result |
|---|---|
| Experiment previously executed | yes |
| Original outputs found | yes (on the original machine; paths in section 6) |
| Configuration identified | yes |
| Implementation reviewed | yes ([algorithm review](../../verification/REVIEW_snippet_rlm.md)) |
| Lightweight tests | unit tests with mocks ([TESTING_REPORT.md](../../verification/TESTING_REPORT.md)); the experiment itself was not re-run |
| Numbers independently recomputed | **matches the workbook** (accuracy, recall, searches): accuracy 0.7633 from 824 result records ([audit](../../verification/RESULT_INTEGRITY.md)) |
| Reproduced by re-running | **no**: no experiment was re-run; full experimental equivalence of the refactored code is not established |


## 10. Limitations

- No random seed; model sampling is stochastic, so a re-run gives different per-question results (two identical 830-question Snippet RLM runs disagreed on 17.8% of questions in the earlier campaign).
- Reasoning effort actually sent: as requested (run5d hook loaded).
- The judge (gpt-4.1) is an LLM; its verdicts were not audited by hand.
- Comparisons with other rows can differ in more than one variable (search cap, turn limit, time limit, effort, index document length); see [ALGORITHM_COMPARISON.md](../ALGORITHM_COMPARISON.md).

## 11. Related files

- [Experiments.xlsx](../../../results/original_baselines/Experiments.xlsx) · [Failure_Analysis.xlsx](../../../results/original_baselines/Failure_Analysis.xlsx) · [registry.yaml](../../../experiments/registry.yaml)
- [Known issues](../../verification/KNOWN_ISSUES.md) · [Result integrity](../../verification/RESULT_INTEGRITY.md) · [Changes from the original code](../../../algorithms/snippet_rlm/CHANGES_FROM_ORIGINAL.md)
