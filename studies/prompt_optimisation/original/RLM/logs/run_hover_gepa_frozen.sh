#!/usr/bin/env bash
# HoVer GEPA sanity check: baseline (val+test) -> GEPA (paper budget 7051) -> best on test.
set -uo pipefail
cd /home/extstu_00/RLM
PYV=/home/extstu_00/miniconda3/envs/vllm/bin/python
export HOVER_OUT=logs/hover_gepa HF_HOME=/home/extstu_00/.cache/huggingface
L=$HOVER_OUT; mkdir -p $L
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L/chain.log; }
curl -s -m 5 -X POST localhost:8770/search -d '{"query":"x","k":1}' | grep -q results || { say "ABORT: retriever :8770 down"; exit 1; }
say "baseline: seed instructions on val 300 + test 300"
# full output kept (an earlier grep filter hid a traceback); results line copied to chain.log
env -u GEPA_PROMPTS $PYV -u scripts/hover_gepa.py --baseline --test --workers 32 > $L/baseline.log 2>&1
grep -E '\[baseline\]' $L/baseline.log | tee -a $L/chain.log
grep -q '\[baseline\] test' $L/baseline.log || { say "ABORT: baseline incomplete -- see $L/baseline.log"; tail -20 $L/baseline.log; exit 1; }
say "GEPA: budget 7051 (paper), minibatch 3, critic claude"
env -u GEPA_PROMPTS $PYV -u scripts/hover_gepa.py --max-metric-calls 7051 --workers 32 > $L/gepa.log 2>&1
rc=$?; say "GEPA exited ($rc) -- scoring best candidate on test"
[ "$rc" = 0 ] || { say "ABORT: GEPA failed"; tail -20 $L/gepa.log; exit 1; }
[ -s $L/best_candidate.json ] || { say "ABORT: no best_candidate.json -- see $L/gepa.log"; exit 1; }
env -u GEPA_PROMPTS $PYV -u scripts/hover_gepa.py --eval-best --test --workers 32 > $L/best_eval.log 2>&1
grep -E '\[best\]' $L/best_eval.log | tee -a $L/chain.log
