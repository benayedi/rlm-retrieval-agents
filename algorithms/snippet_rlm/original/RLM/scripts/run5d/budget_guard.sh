#!/usr/bin/env bash
# Stop a run when its logged Azure spend passes a budget (2026-10-27). Reads <usage.jsonl> (RLM_USAGE_LOG lines) every
# 60 s, prices them (gpt-5 $1.25 in / $0.125 cached in / $10 out; gpt-5.2 $1.75 / $0.175 / $14 per 1M tokens) and kills
# the process tree under <pid> once spend > <budget_usd>. Prints the spend when it acts.
#   scripts/run5d/budget_guard.sh <usage.jsonl> <budget_usd> <pid>
U=$1; B=$2; P=$3
kill_tree(){ for c in $(pgrep -P $1); do kill_tree $c; done; kill $1 2>/dev/null; }
while kill -0 $P 2>/dev/null; do
  s=$(/home/extstu_00/miniconda3/envs/vllm/bin/python -c "
import json,sys
pr={'5.2':(1.75,0.175,14)}; t=0
for l in open('$U'):
    x=json.loads(l); a,c,o=pr['5.2'] if '5.2' in x['model'] else (1.25,0.125,10)
    t+=((x['prompt']-x['cached'])*a+x['cached']*c+x['completion']*o)/1e6
print(f'{t:.2f}')" 2>/dev/null || echo 0)
  if /home/extstu_00/miniconda3/envs/vllm/bin/python -c "import sys; sys.exit(0 if float('$s')>$B else 1)"; then
    echo "$(date '+%F %T') BUDGET GUARD: spend \$$s > \$$B, stopping run under pid $P"; kill_tree $P; exit 0
  fi
  sleep 60
done
echo "$(date '+%F %T') run under pid $P ended; logged spend \$$s"
