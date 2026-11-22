#!/usr/bin/env bash
# GEPA-3: GEPA on FRAMES after the 2026-10-15 14:33 kernel crash/reboot, ONE GPU (:8030, GPU 0) -- the chain that
# produced logs/gepa_frames (from 15:25):
#   1. finish grading the seed run   2. splits   3. GEPA (8 workers, budget 800)   4. test split
# From original/RLM/scripts/frames_gepa_chain_1gpu.sh (arguments and environment unchanged; locations from
# configs/paths.env). The seed run is FR-13 (RUNS_DIR/frames/sup_q36_bm25f). Needs FRAMES_GRADE_SUP (see gepa3_frames.sh).
#   setsid nohup bash studies/prompt_optimisation/experiments/gepa3_frames_1gpu.sh > $RUNS_DIR/frames/gepa_chain_1gpu.log 2>&1 &
set -u
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$RUNS_DIR"                                       # original: cd <original repo>
PY=$PYV
export PYTHONPATH=$RLMLIB_PATH GEPA_BCP_SERVERS=http://localhost:8030/v1
say() { printf '%s  [gepa-1gpu] %s\n' "$(date '+%F %T')" "$*"; }

say "1. final grade of the seed run"
need_file FRAMES_GRADE_SUP "the original repository's scripts/frames_grade_sup.py"
$PY $FRAMES_GRADE_SUP $RUNS_DIR/frames/sup_q36_bm25f 2>&1 | grep -v '^\['
say "2. splits"
$PY $SRC/gepa_frames.py --make-splits 2>&1 | grep splits || { say "ABORT: splits failed"; exit 1; }
curl -s -m10 localhost:8030/v1/models | grep -q qwen36 && curl -s -m10 127.0.0.1:8041/health | grep -q ok \
  || { say "ABORT: model or search server down"; exit 1; }
say "3. GEPA (preloaded seed, 8 workers on :8030, budget 800)"
$PY $SRC/gepa_frames.py --preload-seed --workers 8 --max-metric-calls 800 > $RUNS_DIR/gepa_frames.out 2>&1
say "GEPA exited ($?): $(tail -1 $RUNS_DIR/gepa_frames/progress.txt)"
if [ -s $RUNS_DIR/gepa_frames/best_candidate.json ]; then
  say "4. test split: seed vs best candidate"
  $PY $SRC/gepa_frames.py --test $RUNS_DIR/gepa_frames/best_candidate.json 2>&1 | grep '\[test\]'
fi
say "chain done"
