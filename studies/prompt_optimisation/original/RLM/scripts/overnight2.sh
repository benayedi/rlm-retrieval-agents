#!/usr/bin/env bash
# Unattended: cap test -> trace health check -> paper-scale GEPA on a fresh run_dir.
# Every wait matches the python worker by a flag no shell of mine will contain.
set -uo pipefail
cd /home/extstu_00/RLM
PYV=/home/extstu_00/miniconda3/envs/vllm/bin/python
L=logs/night2; mkdir -p $L
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L/chain.log; }

say "PHASE 1: waiting for the 133 test to finish"
while pgrep -f 'python.*--out logs/test133' >/dev/null 2>&1; do sleep 120; done
say "133 test done"
$PYV -u scripts/score_test133.py 2>&1 | tee $L/test133_final.log

# RLM_TRAJ_DIR is hardcoded in run_obliq_shard.sh the same way LOCAL_LLM_BASE_URL was, so
# every arm's trajectories land in one directory and cannot be told apart. Safe to patch now
# that no run is using the script.
say "PHASE 2: un-hardcoding RLM_TRAJ_DIR"
$PYV - <<'PY'
p='scripts/run_obliq_shard.sh'; s=open(p).read()
old="RLM_TRAJ_DIR=logs/rlm_traj_obliq_sweep"
new="RLM_TRAJ_DIR=${RLM_TRAJ_DIR:-logs/rlm_traj_obliq_sweep}"
if old in s and "${RLM_TRAJ_DIR" not in s:
    open(p,'w').write(s.replace(old,new,1)); print("  patched")
else: print("  already patched or anchor moved")
PY

say "PHASE 3: supervisor cap test (--max-searches 8 -> 20)"
./scripts/sup_cap_test.sh > $L/sup_cap20.log 2>&1
say "cap test finished"
$PYV -u scripts/score_cap_test.py 2>&1 | tee $L/cap_result.log

say "PHASE 4: paper-scale GEPA, FRESH run_dir"
# Run 3 resumed a stale gepa_state.bin and mixed a 50-question valset into our 48, which
# corrupted its own selection. A fresh directory is mandatory.
rm -rf logs/gepa_paper && mkdir -p logs/gepa_paper
$PYV -u scripts/gepa_paper.py --workers 16 --valset 100 --minibatch 3 \
  --max-metric-calls 1500 2>&1 | tee $L/gepa_paper.log
say "OVERNIGHT2 COMPLETE"
