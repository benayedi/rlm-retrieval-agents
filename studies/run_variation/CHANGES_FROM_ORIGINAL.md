# Run variation (VAR-2): changes from the original code

`original/` holds the code that produced VAR-2 (`original/MANIFEST.json`, SHA-256 checked by
`scripts/provenance/preserve_originals.py --check`). `src/` is the same code made runnable from this repository under
[REFACTOR_RULES.md](../../docs/architecture/REFACTOR_RULES.md). No logic was changed.

**Source.** VAR-2's computation was never saved as a file in the original repository: it was run as a
`python - <<'EOF'` heredoc in a Claude Code session on 2026-10-27. `original/TRANSCRIPT/` holds the two heredoc bodies,
extracted byte for byte from that session's transcript (`preserve_originals.py`, function `extract_heredoc`); the
manifest gives for each the transcript file, the 1-based line, the tool_use id, the timestamp and the SHA-256 of the
whole command. Its output (eight pairs, n 50–350, break 3–14 %, fix 25–64 %) is the source of the sentence in the
original repository's `docs/EXPERIMENTS_STATUS.md` §1 ("eight pairs of the supervisor's gpt-5 runs ... fix ~45%
(25–64%) ... break ~8% (3–14%)").

| Original | New |
|---|---|
| `original/TRANSCRIPT/var2_flip_rates.py` (02:05:25 UTC, the final version) | `src/rerun_flip_rates.py` |
| `original/TRANSCRIPT/var2_flip_rates_draft.py` (02:05:18 UTC) | not copied: the previous attempt in the same session. It required the six configuration fields to be equal and effort `high`, and found no pair (it printed nothing). The final version relaxed the match to effort + served model |
| (new) | `tests/test_rerun_flip_rates.py` |

## Configuration
`ORIGINAL_REPO_ROOT`: the artifact is read from `$ORIGINAL_REPO_ROOT/docs/supervisor_artifact.html` (the original ran
with the original repository as cwd). The artifact holds BrowseComp-Plus answers (cut at 1,500 characters); its
encrypted copy is in `trajectories/run_variation/VAR-2/`.

## Per-file diff summary
`src/rerun_flip_rates.py`: module docstring (rule 4); `import os,sys` and the `rlm_shared.paths` import (rule 2); the
`open('docs/supervisor_artifact.html')` path (rule 3). Every other line is the heredoc unchanged.

## Issues found, not fixed
- **RV-I1. The eight pairs are not identical configurations.** `src/rerun_flip_rates.py:44-47` (original 18-21) pairs
  two runs when only `reasoning effort` and `model actually served` agree. Pairs therefore differ in other settings,
  e.g. `gpt5_colbert_pair50` (the published 512-token ColBERT index) vs `gpt5_colbert8192_pair50` (the 8,192-token
  index); `final_rest`, `final100_e`, `final200_medium` and `final100_medium` are later runs of the document REPL
  harness. All eight pairs ran at effort medium; the pilot they are compared with (SP-3) ran at high. The draft
  (`original/TRANSCRIPT/var2_flip_rates_draft.py`) shows that no pair with identical configuration and effort high
  exists in the artifact. So the 3–14 % / 25–64 % ranges are re-run-plus-configuration variation, not pure re-run
  variation.
- **RV-I2. Fix and break rates are not symmetric samples.** For a pair (A, B) the rates are conditioned on A's
  outcome only; `itertools.combinations` gives each pair once, so swapping A and B (a different break/fix split) is
  not reported. Affects only how the ranges are read.

## How verified
`py_compile`; `tests/test_rerun_flip_rates.py` (a synthetic artifact: the readable copy and the original heredoc,
run with the original's cwd convention, print identical output; manifest records the transcript source);
`preserve_originals.py --check`.
