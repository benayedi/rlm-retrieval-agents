# Recursive-language-model retrieval agents: DSPy.RLM, Snippet RLM and Full-Doc RLM

Code, exact configurations, results and audit for a study of **recursive language model (RLM) agents** on
retrieval-heavy benchmarks. An RLM answers by writing Python, one turn at a time, in a persistent REPL that holds its
search tools and data. The study compares three ways of building such an agent, several retrievers, and a frontier vs
an open-weight model, on **BrowseComp-Plus**, **FRAMES** and **OBLIQ-Bench**.

| Algorithm | Idea | In the final results |
|---|---|---|
| [**DSPy.RLM**](algorithms/dspy_rlm/README.md) | `dspy.RLM` with typed signatures and task-specific strategy text | no (earlier campaign only) |
| [**Snippet RLM**](algorithms/snippet_rlm/README.md) | the RLM authors' library; search returns short snippets; documents are read through `llm_query` sub-LLM calls | yes, 11 rows (2 still running) |
| [**Full-Doc RLM**](algorithms/full_doc_rlm/README.md) | the same library; search puts whole documents into a `DOCS` variable that the model reads itself | yes, 11 rows |

**Start here:**
- [docs/research_overview.md](docs/research_overview.md): the questions, methods and findings.
- [docs/experiments/ALGORITHM_COMPARISON.md](docs/experiments/ALGORITHM_COMPARISON.md): the results side by side, with
  what they do and do not show.
- [docs/FINAL_RESEARCH_AUDIT.md](docs/FINAL_RESEARCH_AUDIT.md): what was reviewed, tested and verified, and what was not.

## Results at a glance

Accuracy, judged by gpt-4.1 (BrowseComp-Plus 830 questions, FRAMES 824):

| Benchmark | Model | Retriever | Snippet RLM | Full-Doc RLM |
|---|---|---|---|---|
| BrowseComp-Plus | GPT-5 | Qwen3-Embedding-8B | 0.794 | 0.831 |
| BrowseComp-Plus | GPT-5 | Reason-ModernColBERT | 0.858 | **0.875** |
| BrowseComp-Plus | GPT-5 | BM25 | 0.675 | 0.600 |
| BrowseComp-Plus | Qwen3.6-35B-A3B | Qwen3-Embedding-8B | running | 0.442 |
| BrowseComp-Plus | Qwen3.6-35B-A3B | Reason-ModernColBERT | queued | 0.580 |
| BrowseComp-Plus | Qwen3.6-35B-A3B | BM25 | 0.249 | 0.224 |
| FRAMES | GPT-5 | BM25 | 0.763 | **0.807** |
| FRAMES | Qwen3.6-35B-A3B | BM25 | 0.620 | 0.610 |

OBLIQ-Bench NDCG@10, GPT-5.2 with Gemini-Embedding-2: Math 0.149 / 0.159, Twitter 0.102 / 0.095, Congress 0.188 / 0.133
(Snippet / Full-Doc).

Read these with [ALGORITHM_COMPARISON.md](docs/experiments/ALGORITHM_COMPARISON.md):
- Paired rows differ in more than the reading strategy (search and turn budgets).
- Each cell is one unseeded run.
- Some effort labels and two Full-Doc rows carry caveats listed in
  [docs/verification/KNOWN_ISSUES.md](docs/verification/KNOWN_ISSUES.md).
- Every number above was recomputed from the per-question records
  ([RESULT_INTEGRITY.md](docs/verification/RESULT_INTEGRITY.md)).

## Repository structure

```text
algorithms/
  dspy_rlm/  snippet_rlm/  full_doc_rlm/
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
trajectories/       per-question records and full trajectories of every available row (Git LFS; BrowseComp-Plus encrypted)
scripts/            provenance/ (snapshots, leak scan), audit/ (recompute the results), docs/ (generators), reporting/
docs/               research_overview, experiments/ (one document per experiment), architecture/, verification/
tests/              cross-cutting tests (metrics, grading, retrieval servers)
```

## Installation

The experiments used three Python environments and several model and retrieval servers. To read the code and run the
tests you need only the first:

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt            # pins the versions that ran, incl. the rlm library at commit 854e688
python -m pytest                           # 446 tests, no network, no GPU; see docs/verification/TESTING_REPORT.md
```

Running an experiment additionally needs:
- **Settings.** Copy `configs/paths.example.env` to `configs/paths.env` and fill in the data locations and the Azure
  credentials file.
- **Retrieval server.** One per retriever: see [benchmarks/README.md](benchmarks/README.md). The ColBERT and Gemini
  servers use their own environments, listed at the end of `requirements.txt`.
- **Models.** Either Azure OpenAI deployments of GPT-5 / GPT-5.2 / gpt-4.1, or a vLLM server for Qwen3.6-35B-A3B-FP8.
- **Deno,** for DSPy.RLM's default interpreter.

## Running

```bash
set -a; . configs/paths.env; set +a
bash algorithms/snippet_rlm/experiments/run_config.sh algorithms/snippet_rlm/configs/FR-7.env   # example: one full row
```

**This launches the full experiment:** hundreds of model calls and hours of compute. Each experiment document gives its
exact command, configuration, original output location and grading step: start at
[docs/experiments/README.md](docs/experiments/README.md). No experiment was re-run while preparing this repository.

## Data and licences

Per-question records and trajectories are in [`trajectories/`](trajectories/README.md), stored with Git LFS. Pull only
the rows you need: `git lfs pull --include "trajectories/<algorithm>/<ID>/*"`. Benchmark data, corpora and indexes are
not included. [benchmarks/README.md](benchmarks/README.md) explains how to
obtain and build them.

- **BrowseComp-Plus.** Its questions and answers are distributed encrypted, with a canary asking that they never
  appear online in plain text. This repository contains none; `scripts/provenance/bcp_leak_scan.py` checks that.
- **Third-party code.** The RLM library (MIT, Alex Zhang) is installed from its repository, not vendored.
- **This repository's licence.** Not chosen yet; it needs to be decided before any public release.

## Known limitations

See [docs/verification/KNOWN_ISSUES.md](docs/verification/KNOWN_ISSUES.md). The main ones:
- one run per cell, without seeds;
- Snippet vs Full-Doc pairs differ in budgets;
- reasoning effort did not always reach the model as labelled;
- two Full-Doc rows (BCP-21, BCP-23) ran on an unpreserved library build;
- DSPy.RLM was not evaluated under the final protocol.
