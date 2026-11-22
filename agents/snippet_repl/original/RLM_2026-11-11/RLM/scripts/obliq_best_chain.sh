#!/usr/bin/env bash
# After arm B: if B (full-doc, 20 turns) beats A (full-doc, 12 turns) on dev100, first give the snippet RLM 20 turns
# on dev100 too (SNIP20). Then pick the best version on dev100 (A / B / SNIP / [SNIP20]) and run it on the remaining
# 181 Twitter questions over both GPUs, with that version's exact dev100 settings. Report all 281.
cd /home/extstu_00/RLM
PYV=/home/extstu_00/miniconda3/envs/vllm/bin/python
say() { printf '%s  [obliq-best] %s\n' "$(date '+%F %T')" "$*"; }
snip_run() {  # iters ids-prefix out-prefix  -> starts one shard runner per GPU, sets P0 P1
  for g in 0 1; do
    ( OBLIQ_REPL_CUE=1 OBLIQ_FENCE_TOLERANT=1 OBLIQ_PRESENCE_PENALTY=1.5 OBLIQ_TAG_SEARCH=1 LOCAL_LM_TIMEOUT=1800 \
      LOCAL_LLM_MODEL=qwen36-35b-a3b LOCAL_LLM_BASE_URL=http://localhost:803$g/v1 RLM_TRAJ_DIR=logs/obliq_mine_rr/traj_$(basename $3) \
      OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
      scripts/run_obliq_shard.sh $2_g$g.json $3_g$g 20 $1 8 > logs/$(basename $3)_g$g.log 2>&1 ) & eval P$g=\$!
  done
}
say "waiting for arm B"
while ps -eo args | grep -qE "^(/bin/)?bash scripts/obliq_fd_chain.sh"; do sleep 60; done
AB=$($PYV scripts/obliq_best_full.py pick A B 2>>logs/obliq_best_chain.log)
say "A vs B on dev100: better = $AB"
CANDS="A B SNIP"
if [ "$AB" = B ]; then
  say "B beat A -> snippet RLM with 20 turns on dev100 first"
  snip_run 20 data/obliq_fd/dev100 logs/obliq_mine_rr/snip20_dev100
  wait $P0 $P1
  CANDS="A B SNIP SNIP20"
fi
BEST=$($PYV scripts/obliq_best_full.py pick $CANDS 2>>logs/obliq_best_chain.log)
say "best on dev100 among [$CANDS]: $BEST"
case "$BEST" in
  SNIP)   snip_run 12 data/obliq_fd/rest181 logs/obliq_mine_rr/snip_rest181 ;;
  SNIP20) snip_run 20 data/obliq_fd/rest181 logs/obliq_mine_rr/snip20_rest181 ;;
  *)
    IT=12; [ "$BEST" = B ] && IT=20
    cd /home/extstu_00/suprlm
    export PYTHONPATH=/home/extstu_00/rlmrepo HF_HOME=/home/extstu_00/.cache/huggingface ALIBABA_TOKEN=EMPTY
    export QUERIES_FILE=/home/extstu_00/suprlm/data/obliq_twitter/queries.jsonl GEPA_PROMPTS=/home/extstu_00/RLM/data/obliq_fd/prompts_obliq_cue.json
    export SUP_FENCE_TOLERANT=1 SUP_MAX_BLOCKS=5 SUP_PRESENCE_PENALTY=1.5 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
    for g in 0 1; do
      $PYV -u src/run_experiment.py --name obliq_q36_fd${BEST}_rest_g$g --ids-file /home/extstu_00/RLM/data/obliq_fd/rest181_g$g.json \
        --workers 8 --stagger 0.3 --design B --server http://127.0.0.1:8767 --backend openai --base-url http://localhost:803$g/v1 \
        --api-key-env ALIBABA_TOKEN --root-model qwen36-35b-a3b --sub-model qwen36-35b-a3b --reasoning-effort medium \
        --sub-reasoning-effort medium --default-k 20 --fixed-k --max-searches 20 --max-iterations $IT --seed-context --compaction \
        --lean-prologue --no-delegation --solo-prompt --no-orchestrator --compaction-threshold-pct 0.85 --max-concurrent-subcalls 4 \
        --max-timeout 3600 --lm-timeout 1800 --save-trajectory > /home/extstu_00/RLM/logs/obliq_best_g$g.log 2>&1 & eval P$g=\$!
    done
    cd /home/extstu_00/RLM ;;
esac
say "running $BEST on the remaining 181 (pids $P0 $P1)"
wait $P0 $P1
say "rest181 done"
$PYV scripts/obliq_best_full.py report $BEST 2>&1 | tail -2
say "chain done"
