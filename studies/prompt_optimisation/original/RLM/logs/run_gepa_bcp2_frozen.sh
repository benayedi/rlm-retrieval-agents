#!/usr/bin/env bash
# BrowseComp+ GEPA run 2 (plan A+B+D, 2026-10-08), chained after run 1. Same model/harness/cache.
#   A: minibatch gate 8 instead of 3  (run 1: lucky 3-q passes cost ~100 rollouts each)
#   B: D_pareto enriched with the seed's own failures (scripts/bcp_enrich_split.py)
#   D: critic rule against cutting search effort (in gepa_bcp.py CRITIC_SYSTEM)
# Run from a frozen copy:  cp scripts/run_gepa_bcp2.sh logs/run_gepa_bcp2_frozen.sh
set -uo pipefail
cd /home/extstu_00/RLM
PYV=/home/extstu_00/miniconda3/envs/vllm/bin/python
L=logs/gepa_bcp2; mkdir -p $L
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L/chain.log; }
export GEPA_BCP_MODEL=qwen36-35b-a3b SUP_FENCE_TOLERANT=1 SUP_MAX_BLOCKS=5 SUP_PRESENCE_PENALTY=1.5
export GEPA_BCP_SERVERS=http://localhost:8030/v1 HF_HOME=/home/extstu_00/.cache/huggingface
BUDGET=${BUDGET:-1200}

say "waiting for GEPA run 1 to finish"
while pgrep -u "$(whoami)" -f 'scripts/gepa_bcp[.]py' >/dev/null; do sleep 60; done
say "run 1 finished -- final validation"
env -u GEPA_PROMPTS $PYV scripts/validate_bcp.py 2>&1 | grep -v Warning | tee $L/run1_validate.log

say "seed on the 200-question feedback pool (cache-aware) to find Qwen3.6's own failures"
env -u GEPA_PROMPTS GEPA_OUT=logs/gepa_bcp $PYV -u scripts/gepa_bcp.py --seed-split feedback --workers 20 \
  2>&1 | grep -E 'seed-split|Traceback' | tee -a $L/chain.log
$PYV scripts/bcp_enrich_split.py 2>&1 | tee -a $L/chain.log
[ -s data/bcp_gepa_splits/split_enriched_q36.json ] || { say "ABORT: enriched split not written"; exit 1; }

cp logs/gepa_bcp/cache.json $L/cache.json
say "GEPA run 2: enriched D_pareto, minibatch 8, budget $BUDGET evaluations, critic claude"
env -u GEPA_PROMPTS GEPA_OUT=$L GEPA_SPLIT=data/bcp_gepa_splits/split_enriched_q36.json \
  $PYV -u scripts/gepa_bcp.py --max-metric-calls $BUDGET --minibatch 8 --workers 20 > $L/gepa.log 2>&1
say "GEPA run 2 exited ($?)"
