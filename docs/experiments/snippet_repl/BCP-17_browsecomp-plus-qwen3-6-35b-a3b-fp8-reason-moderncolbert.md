# BCP-17: Snippet REPL agent · BrowseComp-Plus · Qwen3.6-35B-A3B-FP8 · Reason-ModernColBERT

Status: **not run yet** (queued). Row 17 of the `BrowseComp+` sheet of [Experiments.xlsx](../../../results/original_baselines/Experiments.xlsx).

## 1. Experiment overview

|  |  |
|---|---|
| Identifier | `BCP-17` (sheet `BrowseComp+`, row 17) |
| Agent | [Snippet REPL agent](../../../agents/snippet_repl/README.md) |
| Benchmark | BrowseComp-Plus (830 questions, 100,195 documents) |
| Model | Qwen3.6-35B-A3B-FP8 (vLLM, local) |
| Retriever | Reason-ModernColBERT |
| Research question | Snippet REPL agent with an open-weight model: ColBERT retrieval. |
| Original execution | queued after BCP-15 |
| Run name in the logs | `bcp_snip_qwen_colbert` |

## 2. Research motivation

Snippet REPL agent with an open-weight model: ColBERT retrieval. This row is one cell of the comparison matrix in Experiments.xlsx: the same benchmark is run with each agent × model × retriever combination, so this row is read against its neighbours (section 8). See [AGENT_COMPARISON.md](../AGENT_COMPARISON.md) for the matrix and [research_overview.md](../../research_overview.md) for the research questions.

## 3. Experimental methodology

The snippet REPL agent answers each question by writing Python in a persistent REPL, one turn at a time. Its tools return search results as short snippets (the first tokens of each document) into a variable; it can open a whole document and pass text to a sub-LLM (`llm_query`) to read it. A run ends when the model sets a final answer or the turn limit is reached.
 Agent details: [agents/snippet_repl/README.md](../../../agents/snippet_repl/README.md).

Per question: the agent runs on one benchmark question with the budgets in section 4; the run record (answer, retrieved document ids, tool counts, turns, time, tokens) is written to the run directory.
 Correctness is judged afterwards by gpt-4.1 (Azure deployment gpt-4.1, served gpt-4.1-2025-04-14), official BrowseComp-Plus grader template, 3 votes, majority; a question without a completed answer is wrong. Recall is the share of the question's gold documents retrieved at any point of the run (definitions per row: [RESULT_INTEGRITY.md](../../verification/RESULT_INTEGRITY.md) §R1).

## 4. Configuration

| Setting | Value |
|---|---|
| Model | Qwen3.6-35B-A3B-FP8 (vLLM, local) |
| Reasoning effort requested | medium |
| Reasoning effort actually sent | as requested |
| Retriever / index | Reason-ModernColBERT |
| Turns | 50 |
| Searches | 25 |
| Results per search | 5 (default; the model may request more) |
| Snippet | min(2,000 chars, 512 tokens) |
| Timeout | 3,600 s per question |
| Max output tokens | 8192 |
| Compaction | on |
| Random seed | none set (API / vLLM sampling defaults); runs are not bit-reproducible |


Exact environment and arguments ([agents/snippet_repl/configs/BCP-17.env](../../../agents/snippet_repl/configs/BCP-17.env)):

```bash
# BCP-17  snippet REPL agent | Qwen3.6-35B-A3B-FP8 (vLLM :8030/:8031, effort medium) | Reason-ModernColBERT on :8024 | BrowseComp+ 830 q
# Budgets: 50 turns (MAX_ITERS_QWEN default), depth 1, --compaction, search cap 25 (BCP_SEARCH_CAP), QWEN_MAX_TOKENS 8192
#          per call, 3600 s per question (QTIMEOUT_QWEN default), per-LM-call timeout 1800 s, 8 workers.
# Status: QUEUED at refactor time (third run of qwen_bcp_snip_queue.sh, after BCP-19 and BCP-15); no output yet.
#          night_finisher.sh, if still alive, raises it to 20 workers.
# Original outputs: <original repo>/logs/run5d/full/bcp_snip_qwen_colbert/
# Use: experiments/run_config.sh configs/BCP-17.env   (or the whole queue: experiments/qwen_bcp_snip_queue.sh)
ROW=BCP-17
LAUNCHER=experiments/bcp_snip2.sh
LAUNCHER_ARGS=(qwen http://localhost:8024 "$RUNS_DIR/run5d/full/bcp_snip_qwen_colbert" 8)
# --- effective environment of every question process (set by the launcher; listed for the record) ---
export EMBEDDING_SERVER_URL=http://localhost:8024 ORCHESTRATOR=0 LIGHTON_TOOL_DOCS=1 LIGHTON_TASK_TEXT=1
export SEARCH_SNIPPET_TOKENS=512 OFFICIAL_TOP_K=5 AZURE_OPENAI_API_VERSION=2025-04-01-preview
export LOCAL_LLM_MODEL=qwen36-35b-a3b LOCAL_LLM_API_KEY=EMPTY LOCAL_LM_TIMEOUT=1800
export AZURE_OPENAI_REASONING_EFFORT=medium SUB_LM_REASONING_EFFORT=medium QWEN_PRESENCE_PENALTY=1.5 QWEN_FENCE_TOLERANT=1
export QWEN_REPL_CUE=1 QWEN_PROSE_FINAL=1 QWEN_MAX_TOKENS=8192 BCP_SEARCH_CAP=25
# LOCAL_LLM_BASE_URL=http://localhost:$((8030 + Q % 2))/v1 per question, re-pointed at start to the less loaded server
#   by the balance_local part of src/hooks/run5d. CLI per question:
#   run_browsecomp.py --start Q --end Q+1 --max-iterations 50 --max-depth 1 --compaction --out <out>/qQ   (timeout 3600)
# --- implicit environment (see BCP-13.env): in-run judge of the `correct` field, from <original repo>/.env ---
export AZURE_OPENAI_JUDGE_MODEL=gpt-5.4-mini
```

## 5. Execution instructions (reproduction, not executed for this documentation)

These commands launch the full experiment: hundreds of model calls and hours of GPU or API time. They were **not** run while preparing this repository.

```bash
set -a; . configs/paths.env; . agents/snippet_repl/configs/BCP-17.env; set +a
# then the launcher; its header documents its arguments:
agents/snippet_repl/experiments/bcp_snip2.sh
```

Launcher: [agents/snippet_repl/experiments/bcp_snip2.sh](../../../agents/snippet_repl/experiments/bcp_snip2.sh). It reads the benchmark questions, talks to the retrieval server named in the config and writes one record per question (plus trajectories) under `$RUNS_DIR`. Grade afterwards with the matching script in [agents/shared/rlm_shared/grading/](../../../agents/shared/rlm_shared/grading).

## 6. Implementation references

- Agent source: [harness.py](../../../agents/snippet_repl/src/harness.py), [sitecustomize.py](../../../agents/snippet_repl/src/hooks/azure_sampling/sitecustomize.py), [sitecustomize.py](../../../agents/snippet_repl/src/hooks/cap_ranking/sitecustomize.py), [sitecustomize.py](../../../agents/snippet_repl/src/hooks/retry429/sitecustomize.py), [sitecustomize.py](../../../agents/snippet_repl/src/hooks/retry429_cap_ranking/sitecustomize.py), [sitecustomize.py](../../../agents/snippet_repl/src/hooks/run5d/sitecustomize.py), [obliq_pick_best.py](../../../agents/snippet_repl/src/obliq_pick_best.py), [prompt_modules.py](../../../agents/snippet_repl/src/prompt_modules.py), [run_browsecomp.py](../../../agents/snippet_repl/src/run_browsecomp.py), [run_frames.py](../../../agents/snippet_repl/src/run_frames.py), [run_obliq.py](../../../agents/snippet_repl/src/run_obliq.py)
- Original code that produced the result (verbatim, SHA-256 manifest): [agents/snippet_repl/original/](../../../agents/snippet_repl/original/MANIFEST.json)
- Configuration: [agents/snippet_repl/configs/BCP-17.env](../../../agents/snippet_repl/configs/BCP-17.env) · Launcher: [agents/snippet_repl/experiments/bcp_snip2.sh](../../../agents/snippet_repl/experiments/bcp_snip2.sh)
- Data loaders: [agents/shared/rlm_shared/](../../../agents/shared/rlm_shared/__init__.py) (`browsecomp_data.py`, `obliq_data.py`)
- Grading: [agents/shared/rlm_shared/grading/](../../../agents/shared/rlm_shared/grading) · Metrics: [metrics.py](../../../agents/shared/rlm_shared/metrics.py)
- Tests: [agents/snippet_repl/tests/](../../../agents/snippet_repl/tests), [tests/](../../../tests)
- Per-question outputs: **not in this repository** (see [trajectories/README.md](../../../trajectories/README.md) for why)
- Original locations on the machine that ran the experiments: `logs/run5d/full/bcp_snip_qwen_colbert/ (not yet created)`

## 7. Existing experimental results

Reported in Experiments.xlsx (every non-empty column of the row):

| Metric | Value |
|---|---|


Workbook footnotes that apply to this row:

> Input / Output Tokens Per Query: all model calls of a question (root + sub-LLM), reasoning tokens included in output; these do not depend on how many runs shared a server, unlike Time Per Query. Exact from logged per-call usage, except values marked ≈ (Sept. 2026 snippet runs, whose logs overwrote the root model's usage with the sub-LLM's: rebuilt from the trajectories, about 10% low when the same method is checked on a run with exact usage). Uncached cost = input x list input price + output x list output price. Avg. Iterations Per Query: root-model turns per question, counted from the trajectory logs (snippet RLM) or the run records (document REPL agent); sub-LLM calls are not counted. A run that reaches its turn cap adds one forced final-answer turn.

## 8. Results interpretation

Not interpreted: the run is not complete.

## 9. Verification status

| Check | Result |
|---|---|
| Experiment previously executed | queued |
| Original outputs found | no (not run) |
| Configuration identified | yes |
| Implementation reviewed | yes ([agent review](../../verification/REVIEW_snippet_repl.md)) |
| Lightweight tests | unit tests with mocks ([TESTING_REPORT.md](../../verification/TESTING_REPORT.md)); the experiment itself was not re-run |
| Numbers independently recomputed | no (run not complete) |
| Reproduced by re-running | **no**: no experiment was re-run; full experimental equivalence of the refactored code is not established |


## 10. Limitations

- No random seed; model sampling is stochastic, so a re-run gives different per-question results (two identical 830-question snippet REPL agent runs disagreed on 17.8% of questions in the earlier campaign).
- Reasoning effort actually sent: as requested.
- The judge (gpt-4.1) is an LLM; its verdicts were not audited by hand.
- Comparisons with other rows can differ in more than one variable (search cap, turn limit, time limit, effort, index document length); see [AGENT_COMPARISON.md](../AGENT_COMPARISON.md).

## 11. Related files

- [Experiments.xlsx](../../../results/original_baselines/Experiments.xlsx) · [Failure_Analysis.xlsx](../../../results/original_baselines/Failure_Analysis.xlsx) · [registry.yaml](../../../experiments/registry.yaml)
- [Known issues](../../verification/KNOWN_ISSUES.md) · [Result integrity](../../verification/RESULT_INTEGRITY.md) · [Changes from the original code](../../../agents/snippet_repl/CHANGES_FROM_ORIGINAL.md)
