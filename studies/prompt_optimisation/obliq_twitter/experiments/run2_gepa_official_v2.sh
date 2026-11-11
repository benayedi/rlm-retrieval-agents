#!/usr/bin/env bash
# GEPA-4 run 2 (2026-09-29, logs/gepa_official_v2: progress_w12_aborted.txt, then progress_leaked.txt /
# candidates_leaked.json): fixed 50-question paired train set (minibatch = |dataset|), 50 held out.
# RECONSTRUCTED: no launcher survives; the driver's defaults (train 50, budget 420, workers 12, seed 0) are used. The
# aborted attempt used 12 workers (file name); the arguments of the second attempt are not recorded.
#   bash studies/prompt_optimisation/obliq_twitter/experiments/run2_gepa_official_v2.sh [args]
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../../experiments/_common.sh"
cd "$RUNS_DIR"
export PYTHONPATH=$RLMLIB_PATH HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
$PYV -u $OBLIQ_SRC/gepa_official_obliq_v2.py "$@"
