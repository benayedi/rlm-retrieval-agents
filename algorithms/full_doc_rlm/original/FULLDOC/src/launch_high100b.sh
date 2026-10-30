#!/usr/bin/env bash
# Extend the high-effort ColBERT-8192 arm by 100 more of the 830 (seed 100100, drawn
# from the 430 not in high400). Flags byte-identical to the medium 8192 arm except
# --reasoning-effort/--sub-reasoning-effort, same as high400.
set -u
cd /home/extstu_00/suprlm || exit 1
exec .venv/bin/python src/run_experiment.py \
  --name colbert8192_high100b \
  --design B \
  --ids-file data/prepared/colbert8192_high100b.json \
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
