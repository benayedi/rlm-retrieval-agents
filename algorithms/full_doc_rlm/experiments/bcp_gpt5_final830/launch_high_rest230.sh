#!/usr/bin/env bash
# From original/FULLDOC/src/launch_high_rest230.sh. Row BCP-23 (one of its four batches). Arguments unchanged; locations, deployment
# and credentials from configs/paths.env. NOTE: --expect-served-model needs assert_served_model.py, which was not
# preserved (see CHANGES_FROM_ORIGINAL.md); as shipped, run_experiment.py aborts at that gate.
# FINAL batch: the remaining 230 of the 830 not yet run at high effort. This is the
# exact complement of high400 + high100b + high100c, not a sample. Flags byte-identical to the medium 8192 arm except
# --reasoning-effort/--sub-reasoning-effort, same as high400.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
. "$(dirname "${BASH_SOURCE[0]}")/_azure.sh"
cd "$FULLDOC_ROOT" || exit 1   # original: cd <working directory>
# original: exec .venv/bin/python src/run_experiment.py \
exec "$PYTHON" "$SRC/run_experiment.py" \
  --name colbert8192_high_rest230 \
  --design B \
  --ids-file data/prepared/colbert8192_high_rest230.json \
  --workers 12 --stagger 0.4 \
  --server http://127.0.0.1:8766 \
  --root-model gpt-5 --sub-model gpt-5 \
  --root-deployment "$DEP" --sub-deployment "$DEP" \
  --expect-served-model gpt-5-2025-08-07 \
  --reasoning-effort high --sub-reasoning-effort high \
  --max-iterations 30 --max-searches 10 \
  --seed-context --compaction --grounding-gate --lean-prologue \
  --no-delegation --solo-prompt --no-orchestrator \
  --compaction-threshold-pct 0.85 --max-concurrent-subcalls 4 \
  --max-timeout 5400 --save-trajectory
