#!/usr/bin/env bash
# GEPA-3 confirmation: GEPA's best FRAMES prompt (598a758e3a with the leaked "counts toward your score" phrase removed
# = 8ba345b481) vs the seed on the 324 questions no GEPA split used (2026-10-17 07:18 / 07:45).
# RECONSTRUCTED: no wrapper script existed; the command is the usage line of original/RLM/scripts/gepa_frames_confirm.py
# (`python scripts/gepa_frames_confirm.py`, run.log line 1), with PYTHONPATH as in the GEPA-3 chain. The script pins
# GEPA_BCP_SERVERS to :8030 itself and runs 8 workers. Add --report to score the cache only.
#   bash studies/prompt_optimisation/experiments/gepa3_frames_confirm.sh [--report]
set -u
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$RUNS_DIR"
export PYTHONPATH=$RLMLIB_PATH
mkdir -p $RUNS_DIR/gepa_frames_confirm
$PYV -u $SRC/gepa_frames_confirm.py "$@" 2>&1 | tee -a $RUNS_DIR/gepa_frames_confirm/run.log
