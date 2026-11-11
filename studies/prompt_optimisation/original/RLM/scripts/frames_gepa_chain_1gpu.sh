#!/usr/bin/env bash
# GEPA on FRAMES after the 2026-10-15 14:33 kernel crash/reboot, ONE GPU (:8030, GPU 0):
#   1. finish grading the seed run   2. splits   3. GEPA (8 workers, budget 800)   4. test split
#   setsid nohup scripts/frames_gepa_chain_1gpu.sh > logs/frames/gepa_chain_1gpu.log 2>&1 &
set -u
cd /home/extstu_00/RLM
PY=/home/extstu_00/miniconda3/envs/vllm/bin/python
export PYTHONPATH=/home/extstu_00/rlmrepo GEPA_BCP_SERVERS=http://localhost:8030/v1
say() { printf '%s  [gepa-1gpu] %s\n' "$(date '+%F %T')" "$*"; }

say "1. final grade of the seed run"
$PY scripts/frames_grade_sup.py logs/frames/sup_q36_bm25f 2>&1 | grep -v '^\['
say "2. splits"
$PY scripts/gepa_frames.py --make-splits 2>&1 | grep splits || { say "ABORT: splits failed"; exit 1; }
curl -s -m10 localhost:8030/v1/models | grep -q qwen36 && curl -s -m10 127.0.0.1:8041/health | grep -q ok \
  || { say "ABORT: model or search server down"; exit 1; }
say "3. GEPA (preloaded seed, 8 workers on :8030, budget 800)"
$PY scripts/gepa_frames.py --preload-seed --workers 8 --max-metric-calls 800 > logs/gepa_frames.out 2>&1
say "GEPA exited ($?): $(tail -1 logs/gepa_frames/progress.txt)"
if [ -s logs/gepa_frames/best_candidate.json ]; then
  say "4. test split: seed vs best candidate"
  $PY scripts/gepa_frames.py --test logs/gepa_frames/best_candidate.json 2>&1 | grep '\[test\]'
fi
say "chain done"
