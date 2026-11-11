#!/usr/bin/env bash
# GEPA-4 run 4 (logs/gepa_wide): STRATEGY + TASK_SUFFIX mutable, round-robin, D_pareto 100, minibatch 3, budget 800,
# Qwen3.8-27B on :8030 and :8031. One command, started and resumed by four chains; STAGE selects which one's
# environment is reproduced (the command line is identical in all of them):
#   overnight3  original/RLM/scripts/overnight3.sh lines 10, 27-30 (2026-10-04 10:07, fresh run directory)
#   night5      original/RLM/scripts/night5.sh lines 7-10 (13:16, resumed; GEPA_SHARD_WALL_CAP=5400)
#   night6      original/RLM/scripts/night6.sh lines 7-10 (20:19, resumed; GEPA_PER_QUESTION_CAP=3600)
#   night7      original/RLM/scripts/night7_newmodel.sh lines 17, 25-28 (restart_gepa_27b; 23:33, resumed)
#   night8      original/RLM/scripts/night8_gepa27_finish.sh lines 18, 37-38 (starts it only if none runs)
# The final gepa_wide.py (2026-10-04 20:38, the one in original/) ran from the night7 restart on; the earlier stages
# ran earlier, unpreserved versions (CHANGES_FROM_ORIGINAL.md PO-I6). night8's 12-hour deadline watch, trace checks and
# the held-out test (skipped: no winner) are not reproduced here; the verdict is run4_validate.sh.
#   STAGE=night8 bash studies/prompt_optimisation/obliq_twitter/experiments/run4_gepa_wide.sh
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../../experiments/_common.sh"
cd "$RUNS_DIR"
STAGE=${STAGE:?set STAGE to overnight3, night5, night6, night7 or night8}
case "$STAGE" in overnight3) L=$RUNS_DIR/night3 ;; *) L=$RUNS_DIR/$STAGE ;; esac   # original: L=logs/night3, logs/night5, ...
mkdir -p $L
export PYTHONPATH=$RLMLIB_PATH HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
case "$STAGE" in
  overnight3) rm -rf $RUNS_DIR/gepa_wide && mkdir -p $RUNS_DIR/gepa_wide
              $PYV -u $OBLIQ_SRC/gepa_wide.py --workers 16 --valset 100 --minibatch 3 \
                --max-metric-calls 800 2>&1 | tee $L/gepa_wide.log ;;
  night5)     export GEPA_SHARD_WALL_CAP=5400
              $PYV -u $OBLIQ_SRC/gepa_wide.py --workers 16 --valset 100 --minibatch 3 --max-metric-calls 800 2>&1 | tee $L/gepa_wide.log ;;
  night6)     export GEPA_PER_QUESTION_CAP=3600
              $PYV -u $OBLIQ_SRC/gepa_wide.py --workers 16 --valset 100 --minibatch 3 --max-metric-calls 800 2>&1 | tee $L/gepa_wide.log ;;
  night7)     setsid nohup $PYV -u $OBLIQ_SRC/gepa_wide.py --workers 16 --valset 100 --minibatch 3 \
                --max-metric-calls 800 > $L/gepa27_restart.log 2>&1 < /dev/null & ;;
  night8)     $PYV -u $OBLIQ_SRC/gepa_wide.py --workers 16 --valset 100 --minibatch 3 \
                --max-metric-calls 800 > $L/gepa27.log 2>&1 ;;
  *) echo "unknown STAGE $STAGE" >&2; exit 2 ;;
esac
