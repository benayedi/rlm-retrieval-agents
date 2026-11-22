# Error analysis (ERR-1): changes from the original code

`original/` holds the exact files (`original/MANIFEST.json`); `src/` is the same code made runnable from this
repository under [REFACTOR_RULES.md](../../docs/architecture/REFACTOR_RULES.md). No logic was changed.

**Judge.** Every bucket rests on the `correct` field of each run's `graded.json` and of the single-shot baselines,
i.e. the Qwen3.6 judge (FRAMES used it until 2026-10-27). The taxonomy was not rebuilt from the gpt-4.1 verdicts
(owner's decision, 2026-11-11), so its counts are **Qwen3.6 judge only**.

| Original | New |
|---|---|
| `original/RLM/scripts/frames_failure_taxonomy.py` | `src/failure_taxonomy.py` |
| `original/RLM/scripts/frames_s25_failure_drill.py` | `src/failure_drill_s25.py` |
| (new) | `tests/test_failure_taxonomy.py` |

## Configuration
`LOGS_ROOT` (run directories `frames/sup_q36_bm25f*`, the single-shot files `frames/llm_{oracle,closed,bm25_4}.jsonl`,
and the output `frames/failure_taxonomy[_s25].json`, as in the original), `FRAMES_DATA_DIR` (`frames_rows.json`,
`test.tsv`, `corpus_meta.npz`).

## Per-file diff summary
Both files: module docstring (rule 4); `R = "<original repository>"` replaced by `L = LOGS_ROOT`,
`FRD = FRAMES_DATA_DIR` and their import (rules 2–3); every `f"{R}/logs/..."` / `f"{R}/data/frames/..."` rewritten on
them (rule 3); `failure_drill_s25.py` imports `os, sys` for that. No other line differs.

## Issues found, not fixed
- **EA-I1. Inputs from FB-1 beyond the oracle.** `src/failure_taxonomy.py:55-56,117` (original 37-38, 99) read the
  closed-book and BM25 top-4 single-shot files besides the oracle: the closed-book flag is reported per bucket and
  the BM25-4 overlap is printed. They are migrated as ERR-1 inputs (studies/frames_baselines/results/).
- **EA-I2. Re-running overwrites the original result.** `src/failure_taxonomy.py:42,123` (original 24, 105) write to
  `LOGS_ROOT/frames/failure_taxonomy[_s25].json`, the file the original run produced (the copies in `results/` are
  unaffected). Kept, as in the original.
- **EA-I3. Bucket C uses one oracle run.** "Beyond the model" means the single oracle run with all gold articles was
  also wrong; with an 18 % run-to-run flip rate on FRAMES (the original repository's DATASET_BENCHMARKS.md §9.7),
  part of C (and of D, which needs the oracle right) is sampling noise. Affects how the bucket shares are read.

## How verified
`py_compile`; `tests/test_failure_taxonomy.py` (synthetic 824-question tree: both scripts print identical output and
write a byte-identical taxonomy file, with the original's `R` pointed at the tree); `preserve_originals.py --check`.
