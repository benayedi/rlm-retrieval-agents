#!/usr/bin/env bash
# GEPA-4 run 3, paper-shaped at paper-like size (2026-10-04, logs/gepa_paper): D_feedback 100, D_pareto 100,
# minibatch 3, budget 1500, Qwen3.8-27B on :8030 and :8031, fresh run directory.
# From original/RLM/scripts/overnight2.sh lines 34-38 (PHASE 4). overnight2.sh exports no environment of its own.
#   bash studies/prompt_optimisation/obliq_twitter/experiments/run3_gepa_paper.sh
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../../experiments/_common.sh"
cd "$RUNS_DIR"
L=$RUNS_DIR/night2; mkdir -p $L                      # original: L=logs/night2
# Run 3 resumed a stale gepa_state.bin and mixed a 50-question valset into our 48, which
# corrupted its own selection. A fresh directory is mandatory.
rm -rf $RUNS_DIR/gepa_paper && mkdir -p $RUNS_DIR/gepa_paper
$PYV -u $OBLIQ_SRC/gepa_paper.py --workers 16 --valset 100 --minibatch 3 \
  --max-metric-calls 1500 2>&1 | tee $L/gepa_paper.log
