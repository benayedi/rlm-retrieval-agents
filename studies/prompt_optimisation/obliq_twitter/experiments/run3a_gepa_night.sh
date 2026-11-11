#!/usr/bin/env bash
# GEPA-4, the first paper-shaped run (D_feedback 100, D_pareto 50, minibatch 3, budget 400; Qwen3.8-27B on :8030 and
# :8031). Its output directory is logs/gepa_official_v2 (progress.txt from 2026-10-03 05:55; restarted 06:28 with the
# final gepa_night.py). The chain's first attempt (10-02 23:47) ran original/RLM/scripts/gepa_night.py.bak-leak, whose
# candidates leaked the objective (progress_leaked.txt).
# From original/RLM/scripts/night_chain.sh lines 10-11 (environment) and 79-80 (command; BEST_W chosen by the
# throughput probe = 16, logs/night/chain.log). The 05:55 / 06:28 relaunches were made by hand; their arguments are not
# recorded and are assumed equal.
#   bash studies/prompt_optimisation/obliq_twitter/experiments/run3a_gepa_night.sh
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../../experiments/_common.sh"
cd "$RUNS_DIR"
L=$RUNS_DIR/night; mkdir -p $L                       # original: L=logs/night
export PYTHONPATH=$RLMLIB_PATH
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
BEST_W=${BEST_W:-16}
$PYV -u $OBLIQ_SRC/gepa_night.py --workers "$BEST_W" --valset 50 --minibatch 3 \
  --max-metric-calls 400 2>&1 | tee $L/gepa.log
