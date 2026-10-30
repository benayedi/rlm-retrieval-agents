#!/usr/bin/env bash
# From original/FULLDOC/src/launch_high100c.sh. Row BCP-23 (one of its four batches). Arguments unchanged; locations, deployment
# and credentials from configs/paths.env. NOTE: --expect-served-model needs assert_served_model.py, which was not
# preserved (see CHANGES_FROM_ORIGINAL.md); as shipped, run_experiment.py aborts at that gate.
# Extend the high-effort ColBERT-8192 arm by 100 more of the 830 (seed 100200, drawn
# from the 430 not in high400). Flags byte-identical to the medium 8192 arm except
# --reasoning-effort/--sub-reasoning-effort, same as high400.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
. "$(dirname "${BASH_SOURCE[0]}")/_azure.sh"
cd "$FULLDOC_ROOT" || exit 1   # original: cd <working directory>
# one run at a time: wait for the in-flight high100b orchestrator to exit first
WAIT_PID=${WAIT_PID:-}   # original: 1207148 (the high100b orchestrator on the original machine)
[ -n "$WAIT_PID" ] && while [ -e /proc/$WAIT_PID ]; do sleep 20; done
echo "$(date '+%F %T') predecessor ${WAIT_PID:-none} exited; starting high100c"
# original: exec .venv/bin/python src/run_experiment.py \
exec "$PYTHON" "$SRC/run_experiment.py" \
  --name colbert8192_high100c \
  --design B \
  --ids-file data/prepared/colbert8192_high100c.json \
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
