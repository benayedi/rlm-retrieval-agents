#!/usr/bin/env bash
# Restart the GEPA driver (resuming gepa_state.bin) right after c36c56c1c5's 100-q evaluation,
# so the next proposal is the first to get per-search nDCG feedback. GEPA saves state at the
# START of each iteration, i.e. just after this evaluation -- restarting before would lose it.
cd /home/extstu_00/RLM
L=logs/gepa_bcp; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $L/chain.log; }
until grep -q 'cand c36c56c1c5 | n=100 ' $L/progress.txt; do sleep 30; done
sleep 90   # let GEPA finish the iteration and write gepa_state.bin
cp $L/gepa_state.bin $L/gepa_state.before_feedback_restart.bin
for pat in 'scripts/gepa_bcp[.]py' 'src/run_one[.]py --query-id'; do
  for p in $(pgrep -u "$(whoami)" -f "$pat"); do [ "$p" != "$$" ] && kill "$p" 2>/dev/null; done; done
sleep 10
say "restart (resume) with per-search nDCG@10 in critic feedback; c36c56c1c5 evaluation complete and kept"
env -u GEPA_PROMPTS GEPA_BCP_MODEL=qwen36-35b-a3b SUP_FENCE_TOLERANT=1 SUP_MAX_BLOCKS=5 SUP_PRESENCE_PENALTY=1.5 \
  GEPA_BCP_SERVERS=http://localhost:8030/v1 \
  setsid nohup /home/extstu_00/miniconda3/envs/vllm/bin/python -u scripts/gepa_bcp.py \
  --max-metric-calls 800 --workers 20 >> $L/gepa.log 2>&1 < /dev/null &
say "resumed driver pid $!"
