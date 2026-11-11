#!/usr/bin/env bash
# GEPA-5, HoVer GEPA sanity check: baseline (val+test) -> GEPA (paper budget 7051) -> best on test.
# From original/RLM/scripts/run_hover_gepa.sh (identical to the frozen copy that ran, original/RLM/logs/
# run_hover_gepa_frozen.sh; arguments and environment unchanged; locations from configs/paths.env). The run was stopped
# at about 3,800 of 7,051 evaluations (GEPA_BROWSECOMP_PLAN.md 7c). Needs the HoVer BM25 server on :8770
# (hover/src/hover_retriever.py --serve, HOVER_DATA_DIR) and Qwen3.6 on :8030.
#   bash studies/prompt_optimisation/hover/experiments/run_hover_gepa.sh
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../../experiments/_common.sh"
cd "$RUNS_DIR"                                       # original: cd <original repo>
export HOVER_OUT=$RUNS_DIR/hover_gepa HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}   # original: HOVER_OUT=logs/hover_gepa
L=$HOVER_OUT; mkdir -p $L
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L/chain.log; }
curl -s -m 5 -X POST localhost:8770/search -d '{"query":"x","k":1}' | grep -q results || { say "ABORT: retriever :8770 down"; exit 1; }
say "baseline: seed instructions on val 300 + test 300"
# full output kept (an earlier grep filter hid a traceback); results line copied to chain.log
env -u GEPA_PROMPTS $PYV -u $HOVER_SRC/hover_gepa.py --baseline --test --workers 32 > $L/baseline.log 2>&1
grep -E '\[baseline\]' $L/baseline.log | tee -a $L/chain.log
grep -q '\[baseline\] test' $L/baseline.log || { say "ABORT: baseline incomplete -- see $L/baseline.log"; tail -20 $L/baseline.log; exit 1; }
say "GEPA: budget 7051 (paper), minibatch 3, critic claude"
env -u GEPA_PROMPTS $PYV -u $HOVER_SRC/hover_gepa.py --max-metric-calls 7051 --workers 32 > $L/gepa.log 2>&1
rc=$?; say "GEPA exited ($rc) -- scoring best candidate on test"
[ "$rc" = 0 ] || { say "ABORT: GEPA failed"; tail -20 $L/gepa.log; exit 1; }
[ -s $L/best_candidate.json ] || { say "ABORT: no best_candidate.json -- see $L/gepa.log"; exit 1; }
env -u GEPA_PROMPTS $PYV -u $HOVER_SRC/hover_gepa.py --eval-best --test --workers 32 > $L/best_eval.log 2>&1
grep -E '\[best\]' $L/best_eval.log | tee -a $L/chain.log
