#!/usr/bin/env bash
# Fires when GEPA ends: pick the best candidate, then run it AND the seed on the
# 81 held-out questions, simultaneously at matched concurrency. Writes one verdict file.
set -uo pipefail
cd /home/extstu_00/RLM
PYV=/home/extstu_00/miniconda3/envs/vllm/bin/python
L=logs/night5; mkdir -p $L
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L/finish.log; }
export PYTHONPATH=/home/extstu_00/rlmrepo HF_HOME=/home/extstu_00/.cache/huggingface

while pgrep -f 'python.*scripts/gepa_wide[.]py' >/dev/null 2>&1; do sleep 120; done
say "GEPA finished -- selecting best candidate"
$PYV -u scripts/validate_wide.py 2>&1 | tee $L/candidates.log
[ -f /tmp/best_strategy.txt ] || { say "no candidate fully evaluated; nothing to validate"; exit 0; }

$PYV -c "import json; json.dump(json.load(open('logs/gepa_wide/test_split.json'))['test'], open('/tmp/test81.json','w'))"
say "HELD-OUT TEST: seed vs best on 81 questions never used in optimisation"
C="PYTHONUNBUFFERED=1 EMBEDDING_SERVER_URL=http://localhost:8010 OBLIQ_INDEX=obliq-des-twitter OBLIQ_TASK=descriptive OBLIQ_SUBSET=twitter LOCAL_LM_TIMEOUT=1800 AZURE_OPENAI_REASONING_EFFORT=medium SUB_LM_REASONING_EFFORT=medium ORCHESTRATOR=0 LIGHTON_TOOL_DOCS=1 LIGHTON_TASK_TEXT=1 SEARCH_SNIPPET_TOKENS=512 OFFICIAL_TOP_K=20 LOCAL_LLM_MODEL=qwen3.8-27b-bf16"
env $C OBLIQ_STRATEGY_FILE=/tmp/seed_strategy.txt OBLIQ_TASK_FILE=/tmp/seed_task.txt \
    LOCAL_LLM_BASE_URL=http://localhost:8030/v1 RLM_TRAJ_DIR=logs/traj_t81_seed \
    ./scripts/run_obliq_shard.sh /tmp/test81.json logs/test81_seed 20 20 8 > $L/t81_seed.log 2>&1 &
sleep 5
env $C OBLIQ_STRATEGY_FILE=/tmp/best_strategy.txt OBLIQ_TASK_FILE=/tmp/best_task.txt \
    LOCAL_LLM_BASE_URL=http://localhost:8031/v1 RLM_TRAJ_DIR=logs/traj_t81_best \
    ./scripts/run_obliq_shard.sh /tmp/test81.json logs/test81_best 20 20 8 > $L/t81_best.log 2>&1 &
wait
say "test done -- scoring"
$PYV -u scripts/score_pair.py logs/test81_seed logs/test81_best "SEED" "GEPA best" 2>&1 | tee $L/VERDICT.log
say "VERDICT written to $L/VERDICT.log"
