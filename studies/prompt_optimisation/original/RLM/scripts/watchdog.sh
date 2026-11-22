#!/usr/bin/env bash
# Health monitor for long GEPA / eval runs. Samples every 5 min and logs anomalies.
#
# WHY IT EXISTS. On 10-04 a single runaway question froze GEPA for 2h41m -- it was
# generating at 22 tok/s, so nothing looked "crashed", but 15 of 16 workers sat idle and
# the ETA went 18h -> 46h. Nothing surfaced that except a manual check. The signature was:
# batch timestamps frozen + worker count collapsed + one GPU at 0%.
#
# AUTO-KILL IS OFF BY DEFAULT (CAP=0), deliberately. The old default killed any worker
# older than 6480s. That is far shorter than a HEALTHY shard: gepa_wide.py budgets 3600s
# PER QUESTION and a 16-worker shard of D_pareto=100 holds 6-7 questions (~25,000s), while
# a PHASE 4 baseline shard holds 12-13. The old cap would have killed healthy work -- the
# same mistake, from outside, that cost this project 48 kills and three fake 0.0000 scores.
# gepa_wide.py now enforces its own correctly-scaled cap, so this one is redundant there.
# Set CAP=<seconds-per-question> to re-enable, and it will scale by the shard's size.
#
# This script's own command line is "bash scripts/watchdog.sh", which does NOT contain the
# worker pattern, so pgrep cannot match the watchdog itself. That self-match trap has cost
# this project four shells.
set -uo pipefail
cd /home/extstu_00/RLM
PYV=/home/extstu_00/miniconda3/envs/vllm/bin/python
W=logs/watchdog.log
CAP=${CAP:-0}                 # 0 = never auto-kill
STALL_MIN=${STALL_MIN:-45}
CHECK_EVERY=${CHECK_EVERY:-12}   # run a trace check every N samples (12 x 5min = 1h)
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $W; }
say "watchdog started (cap=${CAP} 0=off, stall=${STALL_MIN}min)"
last_n=-1; last_change=$(date +%s); i=0
while true; do
  now=$(date +%s); i=$((i+1))
  # follow whichever GEPA run is currently newest, instead of a dir fixed at launch
  PROG=${PROG_OVERRIDE:-$(ls -t logs/*/progress.txt 2>/dev/null | head -1)}
  n=0; [ -n "${PROG:-}" ] && { n=$(grep -c ' batch ' "$PROG" 2>/dev/null | head -1); n=${n:-0}; }
  roll=$(grep ' batch ' "${PROG:-/dev/null}" 2>/dev/null | tail -1 | grep -oE 'rollouts [0-9]+' | awk '{print $2}')
  wk=$(ps -eo comm,args | awk '$1 ~ /^python/' | grep -c 'run_obliq_mine' || true)
  g0=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0 2>/dev/null)
  g1=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 1 2>/dev/null)
  up8030=$(curl -s -m 4 http://localhost:8030/v1/models >/dev/null 2>&1 && echo 1 || echo 0)
  up8031=$(curl -s -m 4 http://localhost:8031/v1/models >/dev/null 2>&1 && echo 1 || echo 0)
  up8010=$(curl -s -m 4 http://localhost:8010/health  >/dev/null 2>&1 && echo 1 || echo 0)
  # the chain runs from a frozen copy (logs/night7/chain_frozen.sh) so that editing the
  # source cannot corrupt a live run; match either name.
  chain=$(pgrep -c -f 'chain_frozen[.]sh|night8_frozen[.]sh|night[789]_' 2>/dev/null | head -1); chain=${chain:-0}
  [ "$n" != "$last_n" ] && { last_n=$n; last_change=$now; }
  stalled=$(( (now - last_change) / 60 ))
  say "prog=$(basename $(dirname ${PROG:-none/none})) batches=$n rollouts=${roll:-?} workers=$wk gpu=$g0/$g1 srv=$up8010$up8030$up8031 chain=$chain stalled=${stalled}min"
  # --- anomaly flags -------------------------------------------------------------
  [ "$up8010$up8030$up8031" != "111" ] && say "  !! A SERVER IS DOWN (8010=$up8010 8030=$up8030 8031=$up8031)"
  [ "$chain" = "0" ] && say "  !! the night7 chain is NOT running"
  [ "$stalled" -ge "$STALL_MIN" ] && [ "$wk" -lt 4 ] && \
     say "  !! STALL: no new batch in ${stalled}min and only $wk workers -- likely a runaway shard"
  [ "${g0:-0}" -lt 5 ] && [ "${g1:-0}" -lt 5 ] && [ "$wk" -gt 0 ] && \
     say "  !! BOTH GPUs IDLE with $wk workers alive -- workers may be blocked"
  # --- hourly trace check on whatever arm is producing results right now ----------
  if [ $((i % CHECK_EVERY)) -eq 0 ]; then
    newest=$(ls -t logs/q36_*_s*.json logs/gepa_q36/shards/*/out*.json 2>/dev/null | head -1)
    if [ -n "$newest" ]; then
      case "$newest" in
        *t81_best*) pat='logs/t81_best*_s*.json';            tdir=logs/traj_t81_best ;;
        *t81_seed*) pat='logs/t81_seed*_s*.json';            tdir=logs/traj_t81_seed ;;
        *)          pat='logs/gepa_wide/shards/*/out*.json'; tdir=logs/rlm_traj_gepa2 ;;
      esac
      out=$($PYV scripts/trace_check.py "$pat" --traj "$tdir" --label "watchdog hourly" 2>&1)
      rc=$?
      echo "$out" >> logs/watchdog_checks.log
      [ $rc -eq 2 ] && { say "  !! TRACE CHECK SAYS BROKEN:"; echo "$out" | grep '!!' | sed 's/^/      /' >> $W; }
      [ $rc -eq 1 ] && say "  ~~ trace check warnings: $(echo "$out" | grep -c '~~') (see logs/watchdog_checks.log)"
    fi
  fi
  # --- optional auto-fix: kill a worker past a PER-QUESTION cap, scaled by shard ---
  if [ "$CAP" -gt 0 ]; then
    for p in $(pgrep -f 'python.*run_obliq_mine[.]py --ids-file' 2>/dev/null); do
      idf=$(tr '\0' '\n' < /proc/$p/cmdline 2>/dev/null | grep -A0 '\.json$' | head -1)
      nq=1; [ -n "$idf" ] && [ -f "$idf" ] && nq=$($PYV -c "import json;print(max(1,len(json.load(open('$idf')))))" 2>/dev/null || echo 1)
      grace=$(( CAP * nq + CAP / 5 ))
      es=$(ps -o etimes= -p "$p" 2>/dev/null | tr -d ' ')
      if [ -n "$es" ] && [ "$es" -gt "$grace" ]; then
        kill -9 "$p" 2>/dev/null && say "  >> AUTO-FIX: killed worker $p after ${es}s (${nq}q x ${CAP}s = ${grace}s)"
      fi
    done
  fi
  sleep 300
done
