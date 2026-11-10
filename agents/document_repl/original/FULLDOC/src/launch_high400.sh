#!/usr/bin/env bash
# Wait for the in-flight cap-15 retry to finish, then run ColBERT-8192 at HIGH effort
# on a seeded random 400 of the 830. Flags are byte-identical to the medium 8192 arm
# (gpt5_colbert8192_rest630) except --reasoning-effort/--sub-reasoning-effort.
# The --expect-served-model gate is mandatory here: final830_G_high silently ran on
# gpt-5.4 because it lacked this flag.
set -u
cd /home/extstu_00/suprlm || exit 1
WAIT_PID=3534882
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
echo "$(date '+%F %T') predecessor $WAIT_PID exited; starting high400"
exec .venv/bin/python src/run_experiment.py \
  --name colbert8192_high400 \
  --design B \
  --ids-file data/prepared/colbert8192_high400.json \
  --workers 12 --stagger 0.4 \
  --server http://127.0.0.1:8766 \
  --root-model gpt-5 --sub-model gpt-5 \
  --root-deployment gpt-5 --sub-deployment gpt-5 \
  --expect-served-model gpt-5-2025-08-07 \
  --reasoning-effort high --sub-reasoning-effort high \
  --max-iterations 30 --max-searches 10 \
  --seed-context --compaction --grounding-gate --lean-prologue \
  --no-delegation --solo-prompt --no-orchestrator \
  --compaction-threshold-pct 0.85 --max-concurrent-subcalls 4 \
  --max-timeout 5400 --save-trajectory
