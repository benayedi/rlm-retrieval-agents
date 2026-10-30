# Results

## original_baselines/

The results as reported by the original experiments. Nothing here was produced or changed by re-running anything.

| File | What it is |
|---|---|
| [Experiments.xlsx](original_baselines/Experiments.xlsx) | The results table: one sheet per benchmark, one row per algorithm × model × retriever. Per-row documentation: [docs/experiments/](../docs/experiments/README.md) |
| [Failure_Analysis.xlsx](original_baselines/Failure_Analysis.xlsx) | Per-question failure analysis of every row: failure categories, accuracy given the retrieved evidence, turn outcomes, paired McNemar tests, formulas over a per-question detail sheet. **BrowseComp-Plus questions, gold answers and model answers are removed** (the benchmark is distributed encrypted); FRAMES and OBLIQ text is kept. Produced by `scripts/reporting/` and redacted by `scripts/provenance/sanitize_failure_analysis.py` |

The per-question records and full trajectories behind these numbers are in [`../trajectories/`](../trajectories/README.md):
one Git LFS archive per row, byte-exact, with BrowseComp-Plus archives encrypted. The exceptions (BCP-31 for now,
BCP-21/23, the deleted BCP-9/11 trajectories, and the unfinished BCP-15/17) are listed there. [docs/verification/RESULT_INTEGRITY.md](../docs/verification/RESULT_INTEGRITY.md)
recomputes every headline number from them and finds a match for every complete row.

There are no per-algorithm result folders here: the two workbooks cover all algorithms, and the per-question files
are organised by algorithm and row under `trajectories/`. DSPy.RLM's results come from an earlier campaign and are
summarised in its README.

## comparisons/

Figures generated from the workbooks by `scripts/docs/make_comparison.py` and used in
[ALGORITHM_COMPARISON.md](../docs/experiments/ALGORITHM_COMPARISON.md).
