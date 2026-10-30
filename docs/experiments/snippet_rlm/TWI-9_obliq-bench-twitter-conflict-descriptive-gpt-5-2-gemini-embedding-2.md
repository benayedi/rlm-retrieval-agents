# TWI-9: Snippet RLM · OBLIQ-Bench Twitter conflict, descriptive · gpt-5.2 · Gemini-Embedding-2 (Vertex AI)

Status: complete. Row 9 of the `OBLIQ - Twitter` sheet of [Experiments.xlsx](../../../results/original_baselines/Experiments.xlsx).

## 1. Experiment overview

|  |  |
|---|---|
| Identifier | `TWI-9` (sheet `OBLIQ - Twitter`, row 9) |
| Algorithm | [Snippet RLM](../../../algorithms/snippet_rlm/README.md) |
| Benchmark | OBLIQ-Bench Twitter conflict, descriptive (281 queries) |
| Model | gpt-5.2 (Azure) |
| Retriever | Gemini-Embedding-2 (Vertex AI), raw query |
| Research question | Snippet RLM on a ranking task where relevance is an implied stance. |
| Original execution | 2026-10-27 |
| Run name in the logs | `obliq_snip_twitter` |

## 2. Research motivation

Snippet RLM on a ranking task where relevance is an implied stance. This row is one cell of the comparison matrix in Experiments.xlsx: the same benchmark is run with each algorithm × model × retriever combination, so this row is read against its neighbours (section 8). See [ALGORITHM_COMPARISON.md](../ALGORITHM_COMPARISON.md) for the matrix and [research_overview.md](../../research_overview.md) for the research questions.

## 3. Experimental methodology

The Snippet RLM answers each question by writing Python in a persistent REPL, one turn at a time. Its tools return search results as short snippets (the first tokens of each document) into a variable; it can open a whole document and pass text to a sub-LLM (`llm_query`) to read it. A run ends when the model sets a final answer or the turn limit is reached.
 Algorithm details: [algorithms/snippet_rlm/README.md](../../../algorithms/snippet_rlm/README.md).

Per question: the agent runs on one benchmark question with the budgets in section 4; the run record (answer, retrieved document ids, tool counts, turns, time, tokens) is written to the run directory.
 Scoring: none (NDCG@10 / Recall@10 against the gold qrels, binary relevance; ranking parsed from the answer text). A question without a parsable ranking scores 0.

## 4. Configuration

| Setting | Value |
|---|---|
| Model | gpt-5.2 (Azure) |
| Reasoning effort requested | root high, sub-LLM medium |
| Reasoning effort actually sent | as requested (run5d hook loaded) |
| Retriever / index | Gemini-Embedding-2 (Vertex AI), raw query |
| Turns | 12 |
| Searches | no cap |
| Results per search | 20 |
| Snippet | server text ≤ 2,000 chars |
| Timeout | none |
| Compaction | off |
| Random seed | none set (API / vLLM sampling defaults); runs are not bit-reproducible |


Exact environment and arguments ([algorithms/snippet_rlm/configs/TWI-9.env](../../../algorithms/snippet_rlm/configs/TWI-9.env)):

```bash
# TWI-9  Snippet RLM | GPT-5.2 (Azure; root effort high, llm_query medium) | Gemini-Embedding-2 raw-text server on 8021 | OBLIQ twitter (281 q)
# Budgets: 12 turns, depth 1, no compaction, NO search cap, top_k default 20 (OFFICIAL_TOP_K), 512-token snippets, no
#          per-question timeout, 5 worker shards (experiments/obliq_shard.sh), per-LM-call timeout 1800 s (run_obliq.py
#          default). Format gate: ON (OBLIQ_NO_FORMAT_GATE not set; see CHANGES_FROM_ORIGINAL.md issues).
# Launch line (logs/run5d/full/obliq_snip_twitter.out): 2026-10-27 18:22:19 arm=snip task=twitter ids=all281.json workers=5;
#          merged NDCG@10 = 0.1019 (the merged all.json holds every question twice, see issues; the mean is unaffected).
# Original outputs: <original repo>/logs/run5d/full/obliq_snip_twitter/ (all.json, all_s<k>.json, all_s<k>.log, traj/, usage.jsonl)
# Question ids: configs/obliq/twitter/all281.json (byte-identical copy of the ids file passed to the original launcher)
# Use: experiments/run_config.sh configs/TWI-9.env
ROW=TWI-9
LAUNCHER=experiments/obliq_snip.sh
LAUNCHER_ARGS=(snip twitter "$ALG/configs/obliq/twitter/all281.json" "$RUNS_DIR/run5d/full/obliq_snip_twitter" 5)
# --- effective environment of every shard process (set by the launchers; listed for the record) ---
export AZURE_OPENAI_API_VERSION=2025-04-01-preview LOCAL_LLM_BASE_URL="" OBLIQ_RELATIVE_SCORES=1
export OBLIQ_TASK=descriptive OBLIQ_SUBSET=twitter OBLIQ_INDEX=obliq-des-twitter EMBEDDING_SERVER_URL=http://localhost:8021
export OBLIQ_REPL_CUE=1 OBLIQ_TAG_SEARCH=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export AZURE_OPENAI_REASONING_EFFORT=high SUB_LM_REASONING_EFFORT=medium   # ROOT_EFFORT / SUB_EFFORT; sent: hook run5d loaded
export ORCHESTRATOR=0 LIGHTON_TOOL_DOCS=1 LIGHTON_TASK_TEXT=1 SEARCH_SNIPPET_TOKENS=512 OFFICIAL_TOP_K=20
# No OBLIQ_TASK_FILE / OBLIQ_STRATEGY_FILE: the built-in OBLIQ task text and OBLIQ_STRATEGY of src/run_obliq.py, with
#   OBLIQ_RELATIVE_SCORES=1 rewriting their score cutoffs.
# AZURE_OPENAI_MODEL = $AZURE_DEPLOYMENT_GPT52 (gpt-5.2), used for root and llm_query. Unset: AZURE_OPENAI_DEPLOYMENT
#   OBLIQ_FENCE_TOLERANT OBLIQ_PRESENCE_PENALTY. LOCAL_LLM_MODEL defaults to qwen3.8-27b-bf16 in obliq_shard.sh but is
#   unused (LOCAL_LLM_BASE_URL is empty, so the Azure backend is used).
#   CLI per shard: run_obliq.py --ids-file <shard>.json --max-iterations 12 --max-depth 1 --out <out>/all_s<k>
```

## 5. Execution instructions (reproduction, not executed for this documentation)

These commands launch the full experiment: hundreds of model calls and hours of GPU or API time. They were **not** run while preparing this repository.

```bash
set -a; . configs/paths.env; . algorithms/snippet_rlm/configs/TWI-9.env; set +a
# then the launcher; its header documents its arguments:
algorithms/snippet_rlm/experiments/obliq_snip.sh
```

Launcher: [algorithms/snippet_rlm/experiments/obliq_snip.sh](../../../algorithms/snippet_rlm/experiments/obliq_snip.sh). It reads the benchmark questions, talks to the retrieval server named in the config and writes one record per question (plus trajectories) under `$RUNS_DIR`. Grade afterwards with the matching script in [shared/rlm_shared/grading/](../../../shared/rlm_shared/grading).

## 6. Implementation references

- Algorithm source: [harness.py](../../../algorithms/snippet_rlm/src/harness.py), [sitecustomize.py](../../../algorithms/snippet_rlm/src/hooks/azure_sampling/sitecustomize.py), [sitecustomize.py](../../../algorithms/snippet_rlm/src/hooks/cap_ranking/sitecustomize.py), [sitecustomize.py](../../../algorithms/snippet_rlm/src/hooks/retry429/sitecustomize.py), [sitecustomize.py](../../../algorithms/snippet_rlm/src/hooks/retry429_cap_ranking/sitecustomize.py), [sitecustomize.py](../../../algorithms/snippet_rlm/src/hooks/run5d/sitecustomize.py), [prompt_modules.py](../../../algorithms/snippet_rlm/src/prompt_modules.py), [run_browsecomp.py](../../../algorithms/snippet_rlm/src/run_browsecomp.py), [run_frames.py](../../../algorithms/snippet_rlm/src/run_frames.py), [run_obliq.py](../../../algorithms/snippet_rlm/src/run_obliq.py)
- Original code that produced the result (verbatim, SHA-256 manifest): [algorithms/snippet_rlm/original/](../../../algorithms/snippet_rlm/original/MANIFEST.json)
- Configuration: [algorithms/snippet_rlm/configs/TWI-9.env](../../../algorithms/snippet_rlm/configs/TWI-9.env) · Launcher: [algorithms/snippet_rlm/experiments/obliq_snip.sh](../../../algorithms/snippet_rlm/experiments/obliq_snip.sh)
- Data loaders: [shared/rlm_shared/](../../../shared/rlm_shared/__init__.py) (`browsecomp_data.py`, `obliq_data.py`)
- Grading: [shared/rlm_shared/grading/](../../../shared/rlm_shared/grading) · Metrics: [metrics.py](../../../shared/rlm_shared/metrics.py)
- Tests: [algorithms/snippet_rlm/tests/](../../../algorithms/snippet_rlm/tests), [tests/](../../../tests)
- Original outputs (not in this repository; on the machine that ran the experiments): `logs/run5d/full/obliq_snip_twitter/all.json`

## 7. Existing experimental results

Reported in Experiments.xlsx (every non-empty column of the row):

| Metric | Value |
|---|---|
| Accuracy | — |
| Recall@10 (G) | 0.1058 |
| NDCG@10 (G / P) | 0.1019 / 0.1607 |
| NDCG@50 (G / P) | 0.1168 / 0.1765 |
| Avg. Search Calls Per Query | 34.36 |
| Avg. Iterations Per Query | 12 |
| Time Per Query | 225 s (3.7 min) |
| Cached Cost Per Query | 0.5161 |
| Uncached Cost Per Query | 0.6018 |
| Input Tokens Per Query | 196280 |
| Output Tokens Per Query | 18451 |


Workbook footnotes that apply to this row:

> * GPT 5.2 (high)* (Snippet RLM row): the root model runs at high effort, its sub-LLM reading calls (llm_query) at medium; top-k 20, 12 turns, no search cap. Full-Doc RLM: effort high, fixed k 20, max 10 searches, 30 turns, no sub-LLM. Both on Gemini-Embedding-2 over raw text (the paper's setup), run 2026-10-27. Costs from the logged Azure usage: gpt-5.2 $1.75 / $0.175 (cached) / $14 per 1M input / cached input / output tokens. Input / Output Tokens Per Query: all model calls of a question (root + sub-LLM), reasoning tokens included in output; these do not depend on how many runs shared a server, unlike Time Per Query. Exact from logged per-call usage, except values marked ≈ (Sept. 2026 snippet runs, whose logs overwrote the root model's usage with the sub-LLM's: rebuilt from the trajectories, about 10% low when the same method is checked on a run with exact usage). Uncached cost = input x list input price + output x list output price. Avg. Iterations Per Query: root-model turns per question, counted from the trajectory logs (snippet RLM) or the run records (full-doc RLM); sub-LLM calls are not counted. A run that reaches its turn cap adds one forced final-answer turn.


**Per-question outcome** (Failure_Analysis.xlsx, OBLIQ Summary):

| Outcome | Queries | Share |
|---|---|---|
| Hit (≥ 1 gold in top 10) | 103 | 36.7% |
| O1 no usable ranking | 37 | 13.2% |
| O2 gold never retrieved | 57 | 20.3% |
| O3 retrieved, not ranked in top 10 | 84 | 29.9% |


Gold documents retrieved at any point: 485 of 2773; placed in the top 10: 141.

## 8. Results interpretation


- Statements about *why* a configuration wins or loses are hypotheses unless the comparison isolates one variable; see the confounds in section 10.

## 9. Verification status

| Check | Result |
|---|---|
| Experiment previously executed | yes |
| Original outputs found | yes (on the original machine; paths in section 6) |
| Configuration identified | yes |
| Implementation reviewed | yes ([algorithm review](../../verification/REVIEW_snippet_rlm.md)) |
| Lightweight tests | unit tests with mocks ([TESTING_REPORT.md](../../verification/TESTING_REPORT.md)); the experiment itself was not re-run |
| Numbers independently recomputed | **matches the workbook**: NDCG@10 0.1019, Recall@10 0.1058 recomputed from the answer text ([audit](../../verification/RESULT_INTEGRITY.md)) |
| Reproduced by re-running | **no**: no experiment was re-run; full experimental equivalence of the refactored code is not established |


## 10. Limitations

- No random seed; model sampling is stochastic, so a re-run gives different per-question results (two identical 830-question Snippet RLM runs disagreed on 17.8% of questions in the earlier campaign).
- Reasoning effort actually sent: as requested (run5d hook loaded).
- Binary relevance on gold qrels only (graded relevance ignored).
- Comparisons with other rows can differ in more than one variable (search cap, turn limit, time limit, effort, index document length); see [ALGORITHM_COMPARISON.md](../ALGORITHM_COMPARISON.md).

## 11. Related files

- [Experiments.xlsx](../../../results/original_baselines/Experiments.xlsx) · [Failure_Analysis.xlsx](../../../results/original_baselines/Failure_Analysis.xlsx) · [registry.yaml](../../../experiments/registry.yaml)
- [Known issues](../../verification/KNOWN_ISSUES.md) · [Result integrity](../../verification/RESULT_INTEGRITY.md) · [Changes from the original code](../../../algorithms/snippet_rlm/CHANGES_FROM_ORIGINAL.md)
