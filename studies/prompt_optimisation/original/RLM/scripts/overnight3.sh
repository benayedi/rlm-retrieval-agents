#!/usr/bin/env bash
# EXP 1: hand-fixed answer contract (tests the diagnosis directly, cheap)
# EXP 2: GEPA with a 44% mutable surface (STRATEGY + TASK_SUFFIX, round-robin)
# EXP 3: held-out validation of whichever wins
set -uo pipefail
cd /home/extstu_00/RLM
PYV=/home/extstu_00/miniconda3/envs/vllm/bin/python
L=logs/night3; mkdir -p $L
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L/chain.log; }
export PYTHONPATH=/home/extstu_00/rlmrepo HF_HOME=/home/extstu_00/.cache/huggingface

say "EXP1: hand-fixed TASK_SUFFIX vs control, dev100, both arms simultaneous"
COMMON="PYTHONUNBUFFERED=1 EMBEDDING_SERVER_URL=http://localhost:8010 OBLIQ_INDEX=obliq-des-twitter OBLIQ_TASK=descriptive OBLIQ_SUBSET=twitter LOCAL_LM_TIMEOUT=1800 AZURE_OPENAI_REASONING_EFFORT=medium SUB_LM_REASONING_EFFORT=medium ORCHESTRATOR=0 LIGHTON_TOOL_DOCS=1 LIGHTON_TASK_TEXT=1 SEARCH_SNIPPET_TOKENS=512 OFFICIAL_TOP_K=20 LOCAL_LLM_MODEL=qwen3.8-27b-bf16"
env $COMMON OBLIQ_TASK_FILE=$PWD/experiments/rank_ab/A_control.txt \
    LOCAL_LLM_BASE_URL=http://localhost:8030/v1 RLM_TRAJ_DIR=logs/traj_fix_ctrl \
    ./scripts/run_obliq_shard.sh data/obliq_splits/twitter_dev.json logs/fix_control 20 20 8 \
    > $L/fix_control.log 2>&1 &
sleep 5
env $COMMON OBLIQ_TASK_FILE=$PWD/experiments/rank_ab/D_fixed_contract.txt \
    LOCAL_LLM_BASE_URL=http://localhost:8031/v1 RLM_TRAJ_DIR=logs/traj_fix_trt \
    ./scripts/run_obliq_shard.sh data/obliq_splits/twitter_dev.json logs/fix_treat 20 20 8 \
    > $L/fix_treat.log 2>&1 &
wait
say "EXP1 done"
$PYV -u scripts/score_pair.py logs/fix_control logs/fix_treat "control contract" "fixed contract" 2>&1 | tee $L/exp1_result.log

say "EXP2: GEPA, STRATEGY + TASK_SUFFIX mutable (44% of the prompt), fresh run_dir"
rm -rf logs/gepa_wide && mkdir -p logs/gepa_wide
$PYV -u scripts/gepa_wide.py --workers 16 --valset 100 --minibatch 3 \
  --max-metric-calls 800 2>&1 | tee $L/gepa_wide.log
say "EXP2 done"
say "OVERNIGHT3 COMPLETE"
