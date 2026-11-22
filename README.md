# Recursive-language-model retrieval agents: DSPy.RLM, snippet REPL agent and document REPL agent

Code, exact configurations, results and audit for a study of **recursive language model (RLM) agents** on
retrieval-heavy benchmarks. An RLM answers by writing Python, one turn at a time, in a persistent REPL that holds its
search tools and data. The study compares three ways of building such an agent, several retrievers, and a frontier vs
an open-weight model, on **BrowseComp-Plus**, **FRAMES** and **OBLIQ-Bench**.

| Agent | Idea | In the final results |
|---|---|---|
| [**DSPy.RLM**](agents/dspy_rlm/README.md) | `dspy.RLM` with typed signatures and task-specific strategy text | no (earlier campaign only) |
| [**Snippet REPL agent**](agents/snippet_repl/README.md) | the RLM authors' library; search returns short snippets; documents are read through `llm_query` sub-LLM calls | yes, 11 rows |
| [**Document REPL agent**](agents/document_repl/README.md) | the same library; search puts whole documents into a `DOCS` variable that the model reads itself | yes, 11 rows |

**Start here:**
- [docs/research_overview.md](docs/research_overview.md): the questions, methods and findings.
- [docs/experiments/AGENT_COMPARISON.md](docs/experiments/AGENT_COMPARISON.md): the results side by side, with
  what they do and do not show.
- [docs/FINAL_RESEARCH_AUDIT.md](docs/FINAL_RESEARCH_AUDIT.md): what was reviewed, tested and verified, and what was not.
- [docs/experiments/FOLLOW_UP_STUDIES.md](docs/experiments/FOLLOW_UP_STUDIES.md): the work after the results table
  (search policy, GEPA, the FRAMES search cap, the DSPy.RLM ablation, run-to-run variation, FRAMES error analysis).

## Results at a glance

Accuracy, judged by gpt-4.1 (BrowseComp-Plus 830 questions, FRAMES 824):

| Benchmark | Model | Retriever | Snippet REPL agent | Document REPL agent |
|---|---|---|---|---|
| BrowseComp-Plus | GPT-5 | Qwen3-Embedding-8B | 0.794 | 0.831 |
| BrowseComp-Plus | GPT-5 | Reason-ModernColBERT | 0.858 | **0.875** |
| BrowseComp-Plus | GPT-5 | BM25 | 0.675 | 0.600 |
| BrowseComp-Plus | Qwen3.6-35B-A3B | Qwen3-Embedding-8B | 0.3675 | 0.442 |
| BrowseComp-Plus | Qwen3.6-35B-A3B | Reason-ModernColBERT | 0.6012 | 0.580 |
| BrowseComp-Plus | Qwen3.6-35B-A3B | BM25 | 0.249 | 0.224 |
| FRAMES | GPT-5 | BM25 | 0.763 | **0.807** |
| FRAMES | Qwen3.6-35B-A3B | BM25 | 0.620 | 0.610 |

OBLIQ-Bench NDCG@10, GPT-5.2 with Gemini-Embedding-2: Math 0.149 / 0.159, Twitter 0.102 / 0.095, Congress 0.188 / 0.133
(Snippet REPL / document REPL).

Read these with [AGENT_COMPARISON.md](docs/experiments/AGENT_COMPARISON.md):
- Paired rows differ in more than the reading strategy (search and turn budgets).
- Each cell is one unseeded run.
- Some effort labels and two document REPL rows carry caveats listed in
  [docs/verification/KNOWN_ISSUES.md](docs/verification/KNOWN_ISSUES.md).
- Every number above was recomputed from the per-question records
  ([RESULT_INTEGRITY.md](docs/verification/RESULT_INTEGRITY.md)).

## Follow-up studies

After the results table, two ways of improving the agent's search without changing the agent were tested on the
document REPL agent ([FOLLOW_UP_STUDIES.md](docs/experiments/FOLLOW_UP_STUDIES.md); code in [`studies/`](studies/README.md)).
Accuracies judged by gpt-4.1; every number with its source in [results/follow_up/studies.md](results/follow_up/studies.md):

| Study | Setting | Result |
|---|---|---|
| Search policy (novelty filter + Qwen3-Reranker-4B), SP-1 | BrowseComp-Plus, Qwen3.6, all 830 | 67.5 % vs 58.0 % (control BCP-29) |
| Reranker only, SP-2 | BrowseComp-Plus test200 | 70.1 % vs 57.9 % (37/13, p 0.00094) |
| Search policy with GPT-5, SP-3 | 100-question pilot | no better than a plain re-run |
| Search policy on FRAMES, SP-4 | Qwen3.6, search cap 25, all 824 | 67.8 % vs 67.8 % |
| Search cap 25 instead of 10, FRC-1 | FRAMES, Qwen3.6, all 824 | 67.8 % vs 61.0 % (105/49, p 7.5e-6) |
| GEPA prompt optimisation, GEPA-2 / GEPA-3 | BrowseComp-Plus test200 / FRAMES unused questions | 59.3 % vs 57.3 % (p 0.64) / 63.2 % vs 61.6 % (p 0.59) |
| GEPA pipeline check on HoVer, GEPA-5 | the GEPA paper's task | 38.3 % -> 67.3 % (exact metric) |
| DSPy.RLM with models and search output matched, DSA-5 | BrowseComp-Plus, 100 stratified questions | 0.791 vs 0.783 for the snippet REPL agent |

## Repository structure

```text
agents/
  dspy_rlm/  snippet_repl/  document_repl/
    src/            readable, runnable code (behaviour identical to original/)
    original/       the exact files that produced the results, with SHA-256 manifest (never edited)
    experiments/    launchers
    configs/        one <ID>.env per results row: the exact settings it ran with
    tests/          lightweight tests (mocks only)
    README.md  CHANGES_FROM_ORIGINAL.md
  shared/rlm_shared/  paths, benchmark loaders, reference metrics, grading (gpt-4.1 judge, NDCG)
benchmarks/         retrieval servers and index builders; benchmark sources and licences (README)
experiments/        registry.yaml: one entry per experiment
results/
  original_baselines/   Experiments.xlsx (the results table), Failure_Analysis.xlsx (BrowseComp-Plus text removed)
  comparisons/          figures generated from the workbooks
trajectories/       per-question records and full trajectories of every available row and follow-up study (Git LFS;
                    BrowseComp-Plus encrypted)
scripts/            provenance/ (snapshots, leak scan), audit/ (recompute the results), docs/ (generators), reporting/
docs/               research_overview, experiments/ (one document per experiment), architecture/, verification/
studies/            work around an agent (search_policy, prompt_optimisation, run_variation, error_analysis,
                    frames_baselines): original/, src/, experiments/, tests/ as for the agents (studies/README.md)
results/follow_up/  verdicts/ (gpt-4.1), numbers.yaml (every follow-up number with its source), studies.md
benchmarks/models/  serve_qwen36.sh: the vLLM server of every Qwen3.6 run (262,144 context, 16,384-token output cap)
tests/              cross-cutting tests (metrics, grading, retrieval servers, follow-up registry)
```

## Agent and run names

The agents and folders use the names of the paper and its replication package
([neulandAI/program-state-agents](https://github.com/neulandAI/program-state-agents)). Run names, the workbooks in
`results/original_baselines/`, the run records and the `original/` snapshots keep the names used during the research.
They map as follows:

| In run names, workbooks and records | In the paper and in this repository |
|---|---|
| Full-Doc RLM, full-doc, `fd_*` runs, `FULLDOC_ROOT`, `original/FULLDOC/` | document REPL agent, [`agents/document_repl/`](agents/document_repl/README.md) |
| Snippet RLM, `snip_*` runs | snippet REPL agent, [`agents/snippet_repl/`](agents/snippet_repl/README.md) |
| DSPy.RLM | DSPy.RLM, [`agents/dspy_rlm/`](agents/dspy_rlm/README.md) (not in the paper) |
| `algorithms/<name>/`, `shared/` | `agents/<name>/`, `agents/shared/` |

The documentation generators in `scripts/docs/` write the paper's names (`scripts/docs/naming.py`); the reporting scripts
in `scripts/reporting/` keep the research-time labels, so they still reproduce the workbooks.

## Installation

The experiments used three Python environments and several model and retrieval servers. To read the code and run the
tests you need only the first:

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt            # pins the versions that ran, incl. the rlm library at commit 854e688
python -m pytest                           # 838 tests, no network, no GPU; see docs/verification/TESTING_REPORT.md
```

Running an experiment additionally needs:
- **Settings.** Copy `configs/paths.example.env` to `configs/paths.env` and fill in the data locations and the Azure
  credentials file.
- **Retrieval server.** One per retriever: see [benchmarks/README.md](benchmarks/README.md). The ColBERT and Gemini
  servers use their own environments, listed at the end of `requirements.txt`.
- **Models.** Either Azure OpenAI deployments of GPT-5 / GPT-5.2 / gpt-4.1, or a vLLM server for Qwen3.6-35B-A3B-FP8
  (`benchmarks/models/serve_qwen36.sh`; the runs used a 262,144-token context and `GEN_OVERRIDE='{"max_new_tokens": 16384}'`).
- **Deno,** for DSPy.RLM's default interpreter.

## Running

```bash
set -a; . configs/paths.env; set +a
bash agents/snippet_repl/experiments/run_config.sh agents/snippet_repl/configs/FR-7.env   # example: one full row
```

**This launches the full experiment:** hundreds of model calls and hours of compute. Each experiment document gives its
exact command, configuration, original output location and grading step: start at
[docs/experiments/README.md](docs/experiments/README.md). No experiment was re-run while preparing this repository.

## Data and licences

Per-question records and trajectories are in [`trajectories/`](trajectories/README.md), stored with Git LFS. Pull only
the rows you need: `git lfs pull --include "trajectories/<agent>/<ID>/*"`. Benchmark data, corpora and indexes are
not included. [benchmarks/README.md](benchmarks/README.md) explains how to
obtain and build them.

- **BrowseComp-Plus.** Its questions and answers are distributed encrypted, with a canary asking that they never
  appear online in plain text. This repository contains none; `scripts/provenance/bcp_leak_scan.py` checks that.
- **Third-party code.** The RLM library (MIT, Alex Zhang) is installed from its repository, not vendored.
- **This repository's licence.** Not chosen yet; it needs to be decided before any public release.

## Known limitations

See [docs/verification/KNOWN_ISSUES.md](docs/verification/KNOWN_ISSUES.md). The main ones:
- one run per cell, without seeds;
- Snippet REPL vs document REPL pairs differ in budgets;
- reasoning effort did not always reach the model as labelled;
- two document REPL rows (BCP-21, BCP-23) ran on an unpreserved library build;
- DSPy.RLM was not evaluated under the final protocol.
