# Migration plan (approved 2026-10-30 with decisions D1–D3; executed)

Companion to [INITIAL_CODEBASE_INVENTORY.md](INITIAL_CODEBASE_INVENTORY.md). Nothing below has been executed.

## Principles

1. **The original stays untouched.** All work happens in `/home/extstu_00/rlm-research-artifact`, a new directory
   with a new git history. The original repo, `rlmrepo`, `suprlm` and every running job are read-only to this work.
2. **The code that produced the results is preserved verbatim.** Each agent keeps an `original/` snapshot of the
   exact files that ran (copied, with a manifest of source path + SHA-256). Refactored code sits next to it in `src/`.
   A refactor never changes behaviour silently: any intended behaviour change is a separate, documented commit.
3. **No new experiments.** Verification is static review, source-level comparison against `original/`, mocked unit
   tests and checks on existing result files.

## Target structure

```text
rlm-research-artifact/
├── agents/
│   ├── dspy_rlm/      src/ (from rag/ + the two drivers), original/, experiments/ (launch scripts), configs/, tests/, README.md
│   ├── snippet_repl/   src/ (harness, runners, hooks, prompt modules), original/, experiments/, configs/, tests/, README.md
│   └── document_repl/  src/ (harness and wrappers), original/, experiments/, configs/, tests/, README.md
├── agents/shared/            retrieval wire-format client, data loaders (BrowseComp+, FRAMES, OBLIQ), judges, metrics (recall, NDCG), turn counting
├── benchmarks/        README per benchmark: source, licence, how to obtain, how indexes were built; retrieval servers in benchmarks/retrieval/
├── experiments/
│   └── registry.yaml  one entry per experiment: id, agent, benchmark, model, retriever, budgets, runner command, env, result paths, workbook cell, judge, verification status
├── results/
│   ├── original_baselines/{dspy_rlm,snippet_repl,document_repl}/   per-question summaries (no trajectories), graded verdicts, manifests with SHA-256 of the source files
│   └── comparisons/   the workbooks (gold answers stripped where needed), comparison tables and figures from existing data
├── scripts/           report generators (iteration counts, tokens, failure analysis, workbook builders)
├── tests/             cross-cutting tests (registry ↔ results ↔ workbook consistency)
├── docs/              inventory, agents/, experiments/{dspy_rlm,snippet_repl,document_repl}/ (one .md per experiment), architecture/, verification/, research_overview.md, FINAL_RESEARCH_AUDIT.md
├── README.md  CONTRIBUTING.md  CITATION.cff (only with real metadata)  pyproject.toml  requirements.txt  .gitignore
```

Experiment documents are generated from `registry.yaml` for the factual sections (configuration, command, paths,
metrics read from the result files) and hand-written for motivation and interpretation, so numbers cannot drift from
the evidence. Configurations become YAML files that reproduce the env vars and CLI args the wrappers set.

## What does not go into the repository

- `logs/` trajectories (431 GB), `data/` corpora (89 GB), indexes, model weights: documented with retrieval
  instructions instead.
- Decrypted BrowseComp+ gold answers (the benchmark ships them encrypted): stripped from published result files and
  workbooks; a script re-derives them locally from the HF dataset.
- `.env`, credential paths, personal deployment names (replaced by placeholders), `/tmp` scratch paths.
- One-off probes, `night*.sh` chains, `.bak` files: listed in an appendix of the inventory, not migrated.

## Phases

| Phase | Output | Needs approval |
|---|---|---|
| 3 Baseline preservation | `results/original_baselines/` with manifests; checksums of every source result file | — |
| 4–5 Refactor | agents/*, agents/shared/, original/ snapshots, configs, registry | D1, D2 |
| 6 Review | `docs/verification/review_<agent>.md`; issues C1–C12 confirmed or rejected with evidence | — |
| 7 Tests | pytest suite (mocks only, no network), `docs/verification/TESTING_REPORT.md` | — |
| 8 Result audit | `docs/verification/RESULT_INTEGRITY.md`: recompute every workbook number from per-question files | — |
| 9–10 Docs | one .md per experiment, agent READMEs, AGENT_COMPARISON.md, architecture, research overview | — |
| 11 Audit | `docs/FINAL_RESEARCH_AUDIT.md` | — |
| 12 Publish | private GitHub repo, invite `Torky24` | **yes, explicit** |

## Deviations from this plan (as executed)

- **Results:** published as the Excel workbooks (decision D3). Later, at the owner's request, the per-question records
  and trajectories of every available row were added as Git LFS archives in `trajectories/`: byte-exact, with
  BrowseComp-Plus archives encrypted with the benchmark's canary key. BCP-31 is left out for now.
- **Experiment documents:** one per row of Experiments.xlsx (22 rows). DSPy.RLM has none, because it has no row
  (decision D2).
- **Document REPL code:** included as `agents/document_repl/` with no separate origin note (decision D1).
- **Shared code:** lives in `agents/shared/rlm_shared/` (loaders, metrics, grading) and `benchmarks/retrieval/` (servers,
  index builders).
- **Not migrated:** the historical in-place patchers of the document REPL harness (`apply_*_patch.py`). They are kept only
  in `original/`, and `src/PATCH_HISTORY.md` describes them.
