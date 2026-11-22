#!/usr/bin/env bash
# GEPA-4 run 4 verdict: pick GEPA's best candidate by its D_pareto score and compare it with the seed (2026-10-05
# 13:12, logs/night8/validate.log: no candidate beat the seed). From original/RLM/scripts/night8_gepa27_finish.sh
# line 61. A winner would then be tested on the 81 held-out questions (lines 63-87, not reproduced: never reached).
#   bash studies/prompt_optimisation/obliq_twitter/experiments/run4_validate.sh
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../../experiments/_common.sh"
cd "$RUNS_DIR"
L=$RUNS_DIR/night8; mkdir -p $L
export PYTHONPATH=$RLMLIB_PATH HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
OUT=$RUNS_DIR/gepa_wide                              # original: OUT=logs/gepa_wide
GEPA_OUT=$OUT $PYV -u $OBLIQ_SRC/validate_wide.py 2>&1 | tee $L/validate.log
