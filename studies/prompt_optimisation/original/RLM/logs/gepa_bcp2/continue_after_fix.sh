#!/usr/bin/env bash
# After the relative-GEPA_OUT fix: let orphaned workers finish, recover their misfiled rollouts,
# finish the seed on the feedback pool, enrich, launch GEPA run 2.
cd /home/extstu_00/RLM
PYV=/home/extstu_00/miniconda3/envs/vllm/bin/python; L=logs/gepa_bcp2
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $L/chain.log; }
export GEPA_BCP_MODEL=qwen36-35b-a3b SUP_FENCE_TOLERANT=1 SUP_MAX_BLOCKS=5 SUP_PRESENCE_PENALTY=1.5
export GEPA_BCP_SERVERS=http://localhost:8030/v1 HF_HOME=/home/extstu_00/.cache/huggingface
say "fix: GEPA_OUT made absolute; recovering misfiled rollouts (suprlm/logs/gepa_bcp)"
$PYV scripts/bcp_recover_misfiled.py 2>&1 | grep -v Warning >> $L/chain.log
while pgrep -u "$(whoami)" -f 'run_one[.]py --query-id' >/dev/null; do sleep 60; done
$PYV scripts/bcp_recover_misfiled.py 2>&1 | grep -v Warning >> $L/chain.log
say "seed on the remaining feedback questions (missing/killed only)"
env -u GEPA_PROMPTS GEPA_OUT=logs/gepa_bcp $PYV -u scripts/gepa_bcp.py --seed-split feedback --workers 20 2>&1 | grep -E 'seed-split|Traceback' >> $L/chain.log
$PYV scripts/bcp_enrich_split.py >> $L/chain.log 2>&1 || { say "ABORT: enrich failing"; exit 1; }
cp logs/gepa_bcp/cache.json $L/cache.json
say "GEPA run 2: enriched D_pareto, minibatch 8, budget 1200 evaluations, critic claude"
env -u GEPA_PROMPTS GEPA_OUT=$L GEPA_SPLIT=data/bcp_gepa_splits/split_enriched_q36.json \
  $PYV -u scripts/gepa_bcp.py --max-metric-calls 1200 --minibatch 8 --workers 20 > $L/gepa.log 2>&1
say "GEPA run 2 exited ($?)"
