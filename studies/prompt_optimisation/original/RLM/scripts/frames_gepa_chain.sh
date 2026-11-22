#!/usr/bin/env bash
# After the 824-question seed run (logs/frames/sup_q36_bm25f) finishes on both GPUs:
#   1. final grade of the seed run     2. GEPA splits (200 feedback / 100 pareto / 200 test)
#   3. GEPA, both GPUs, budget 800     4. best candidate vs seed on the test split
#   setsid nohup scripts/frames_gepa_chain.sh > logs/frames/gepa_chain.log 2>&1 &
set -u
cd /home/extstu_00/RLM
PY=/home/extstu_00/miniconda3/envs/vllm/bin/python
export PYTHONPATH=/home/extstu_00/rlmrepo
say() { printf '%s  [gepa-chain] %s\n' "$(date '+%F %T')" "$*"; }

say "waiting for the seed run (all run_frames_sup.sh / frames_chain5 / frames_chain6)"
while pgrep -f "^bash scripts/run_frames_sup.sh|^/bin/bash scripts/run_frames_sup.sh|^bash scripts/frames_chain[56].sh" >/dev/null; do sleep 60; done
say "seed run finished: $(ls logs/frames/sup_q36_bm25f/[0-9]*.json | wc -l) result files"

say "1. final grade"
$PY scripts/frames_grade_sup.py logs/frames/sup_q36_bm25f 2>&1 | grep -v '^\['

say "2. splits"
$PY scripts/gepa_frames.py --make-splits 2>&1 | grep splits || { say "ABORT: splits failed"; exit 1; }

curl -s -m10 localhost:8030/v1/models | grep -q qwen36 && curl -s -m10 localhost:8031/v1/models | grep -q qwen36 \
  || { say "ABORT: a model server is down"; exit 1; }
say "3. GEPA (preloaded seed, 16 workers over :8030 + :8031, budget 800)"
$PY scripts/gepa_frames.py --preload-seed --workers 16 --max-metric-calls 800 > logs/gepa_frames.out 2>&1
say "GEPA exited ($?): $(tail -1 logs/gepa_frames/progress.txt)"

if [ -s logs/gepa_frames/best_candidate.json ]; then
  say "4. test split: seed vs best candidate"
  $PY scripts/gepa_frames.py --test logs/gepa_frames/best_candidate.json 2>&1 | grep '\[test\]'
fi
say "chain done"
