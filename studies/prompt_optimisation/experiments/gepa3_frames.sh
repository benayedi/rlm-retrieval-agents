#!/usr/bin/env bash
# GEPA-3, two-GPU variant (2026-10-15 09:51; superseded by gepa3_frames_1gpu.sh after the 14:33 kernel crash, which is
# the chain that produced the run). After the 824-question seed run (RUNS_DIR/frames/sup_q36_bm25f, the FR-13 run):
#   1. final grade of the seed run     2. GEPA splits (200 feedback / 100 pareto / 200 test)
#   3. GEPA, both GPUs, budget 800     4. best candidate vs seed on the test split
# From original/RLM/scripts/frames_gepa_chain.sh (arguments and environment unchanged; locations from configs/paths.env).
# Needs FRAMES_GRADE_SUP = the original repository's scripts/frames_grade_sup.py (interim Qwen3.6 judge; not part of
# this repository). The wait for the seed run matches this repository's launcher (frames_qwen.sh) instead of the
# original run_frames_sup.sh / frames_chain[56].sh.
#   setsid nohup bash studies/prompt_optimisation/experiments/gepa3_frames.sh > $RUNS_DIR/frames/gepa_chain.log 2>&1 &
set -u
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$RUNS_DIR"                                       # original: cd <original repo>
PY=$PYV
export PYTHONPATH=$RLMLIB_PATH
say() { printf '%s  [gepa-chain] %s\n' "$(date '+%F %T')" "$*"; }

say "waiting for the seed run (frames_qwen.sh; original: run_frames_sup.sh / frames_chain5 / frames_chain6)"
while pgrep -f "^bash [^ ]*frames_qwen[.]sh|^/bin/bash [^ ]*frames_qwen[.]sh" >/dev/null; do sleep 60; done
say "seed run finished: $(ls $RUNS_DIR/frames/sup_q36_bm25f/[0-9]*.json | wc -l) result files"

say "1. final grade"
need_file FRAMES_GRADE_SUP "the original repository's scripts/frames_grade_sup.py"
$PY $FRAMES_GRADE_SUP $RUNS_DIR/frames/sup_q36_bm25f 2>&1 | grep -v '^\['

say "2. splits"
$PY $SRC/gepa_frames.py --make-splits 2>&1 | grep splits || { say "ABORT: splits failed"; exit 1; }

curl -s -m10 localhost:8030/v1/models | grep -q qwen36 && curl -s -m10 localhost:8031/v1/models | grep -q qwen36 \
  || { say "ABORT: a model server is down"; exit 1; }
say "3. GEPA (preloaded seed, 16 workers over :8030 + :8031, budget 800)"
$PY $SRC/gepa_frames.py --preload-seed --workers 16 --max-metric-calls 800 > $RUNS_DIR/gepa_frames.out 2>&1
say "GEPA exited ($?): $(tail -1 $RUNS_DIR/gepa_frames/progress.txt)"

if [ -s $RUNS_DIR/gepa_frames/best_candidate.json ]; then
  say "4. test split: seed vs best candidate"
  $PY $SRC/gepa_frames.py --test $RUNS_DIR/gepa_frames/best_candidate.json 2>&1 | grep '\[test\]'
fi
say "chain done"
