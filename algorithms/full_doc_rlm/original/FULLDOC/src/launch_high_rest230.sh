#!/usr/bin/env bash
# FINAL batch: the remaining 230 of the 830 not yet run at high effort. This is the
# exact complement of high400 + high100b + high100c, not a sample. Flags byte-identical to the medium 8192 arm except
# --reasoning-effort/--sub-reasoning-effort, same as high400.
set -u
cd /home/extstu_00/suprlm || exit 1
exec .venv/bin/python src/run_experiment.py \
  --name colbert8192_high_rest230 \
  --design B \
  --ids-file data/prepared/colbert8192_high_rest230.json \
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
