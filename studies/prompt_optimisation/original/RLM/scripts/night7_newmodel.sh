#!/usr/bin/env bash
# UNATTENDED MODEL SWAP + GEPA: Qwen3.8-27B (dense) -> Qwen3.6-35B-A3B (3B active MoE).
#
# Gates, in order. Any failure restores the 27B AND restarts the 27B GEPA (which resumes
# from its cache), rather than leaving the box idle or producing numbers that look like
# results but are not.
#   A  both vLLM servers answer           (FP8/Marlin on Ampere is unproven)
#   B  smoke test: the harness works      (the gate gpt-oss-20b failed, 8/8 content=None)
#   C  trace check on the baseline        (scripts/trace_check.py)
#   D  baseline quality: NDCG >= $GATE_MIN_NDCG -- below this GEPA has nothing to optimise
# then PHASE 5 GEPA on the new model, PHASE 6 held-out test.
set -uo pipefail
cd /home/extstu_00/RLM
PYV=/home/extstu_00/miniconda3/envs/vllm/bin/python
L=logs/night7; mkdir -p $L
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L/chain.log; }
export PYTHONPATH=/home/extstu_00/rlmrepo HF_HOME=/home/extstu_00/.cache/huggingface
COMMON="PYTHONUNBUFFERED=1 EMBEDDING_SERVER_URL=http://localhost:8010 OBLIQ_INDEX=obliq-des-twitter OBLIQ_TASK=descriptive OBLIQ_SUBSET=twitter LOCAL_LM_TIMEOUT=1800 AZURE_OPENAI_REASONING_EFFORT=medium SUB_LM_REASONING_EFFORT=medium ORCHESTRATOR=0 LIGHTON_TOOL_DOCS=1 LIGHTON_TASK_TEXT=1 SEARCH_SNIPPET_TOKENS=512 OFFICIAL_TOP_K=20 OBLIQ_FENCE_TOLERANT=1 OBLIQ_REPL_CUE=1"
NEW=qwen36-35b-a3b
GATE_MIN_NDCG=${GATE_MIN_NDCG:-0.080}   # 27B scores .1302; below ~60% of that, GEPA can't catch up

# every arm gets checked the moment it finishes; 2 = BROKEN
check(){ $PYV scripts/trace_check.py "$1" --traj "$2" --label "$3" 2>&1 | tee -a $L/checks.log; return ${PIPESTATUS[0]}; }

restart_gepa_27b(){
  say "restarting the 27B GEPA (resumes from logs/gepa_wide/cache.json)"
  setsid nohup $PYV -u scripts/gepa_wide.py --workers 16 --valset 100 --minibatch 3 \
    --max-metric-calls 800 > $L/gepa27_restart.log 2>&1 < /dev/null &
}
restore_old(){
  say "RESTORING Qwen3.8-27B"
  for q in $(pgrep -f 'vllm serve' 2>/dev/null); do kill $q 2>/dev/null; done; sleep 40
  GPU=0 PORT=8030 MAX_NUM_SEQS=64 USE_MTP=0 setsid nohup ./scripts/serve_qwen27b_bf16_tuned.sh > $L/restore0.log 2>&1 < /dev/null &
  sleep 20
  GPU=1 PORT=8031 MAX_NUM_SEQS=64 USE_MTP=0 setsid nohup ./scripts/serve_qwen27b_bf16_tuned.sh > $L/restore1.log 2>&1 < /dev/null &
  for i in $(seq 1 90); do curl -s -m 4 http://localhost:8030/v1/models >/dev/null 2>&1 && break; sleep 10; done
  restart_gepa_27b
  say "old model restored, 27B GEPA resumed; stopping here"
}

# SKIP_TO=<n> resumes at phase n, for when the servers are already loaded and gated and
# only a later phase changed. Re-running PHASE 1-2 would kill and reload both models for
# nothing.
SKIP_TO=${SKIP_TO:-0}
if [ "$SKIP_TO" -ge 4 ]; then say "SKIP_TO=$SKIP_TO -- phases 0-3 skipped, servers assumed live and smoked"; fi
if [ "$SKIP_TO" -lt 4 ]; then
say "PHASE 0: waiting for the Qwen3.6 download"
while pgrep -f 'snapshot_download' >/dev/null 2>&1; do sleep 60; done
grep -q "^OK " logs/qwen36_download.log || { say "ABORT: download failed"; tail -3 logs/qwen36_download.log; exit 1; }
MODEL=$(grep "^OK " logs/qwen36_download.log | head -1 | awk '{print $2}')
say "downloaded $MODEL"

say "PHASE 1: stopping the 27B GEPA and the 27B servers"
for q in $(pgrep -f 'python.*scripts/gepa_wide[.]py' 2>/dev/null); do kill $q 2>/dev/null; done
for q in $(pgrep -f 'python.*run_obliq_mine[.]py --ids-file' 2>/dev/null); do kill $q 2>/dev/null; done
sleep 10
for q in $(pgrep -f 'vllm serve' 2>/dev/null); do kill $q 2>/dev/null; done
sleep 45

say "PHASE 2: serving $MODEL on both GPUs"
MODEL="$MODEL" GPU=0 PORT=8030 setsid nohup ./scripts/serve_qwen36_moe.sh > $L/vllm0.log 2>&1 < /dev/null &
sleep 25
MODEL="$MODEL" GPU=1 PORT=8031 setsid nohup ./scripts/serve_qwen36_moe.sh > $L/vllm1.log 2>&1 < /dev/null &
for i in $(seq 1 150); do
  curl -s -m 4 http://localhost:8030/v1/models >/dev/null 2>&1 \
    && curl -s -m 4 http://localhost:8031/v1/models >/dev/null 2>&1 && break
  sleep 10
done
curl -s -m 5 http://localhost:8030/v1/models >/dev/null 2>&1 || { say "GATE A FAILED: :8030 never came up"; tail -25 $L/vllm0.log | tee -a $L/chain.log; restore_old; exit 1; }
curl -s -m 5 http://localhost:8031/v1/models >/dev/null 2>&1 || { say "GATE A FAILED: :8031 never came up"; tail -25 $L/vllm1.log | tee -a $L/chain.log; restore_old; exit 1; }
for P in 8030 8031; do say "  :$P KV = $(curl -s -m 5 http://localhost:$P/metrics | grep -oE 'kv_cache_size_tokens=\"[0-9]*\"' | head -1)"; done
say "GATE A passed -- both servers live"

say "PHASE 3: smoke test (3 questions) -- the gate gpt-oss-20b failed"
$PYV -c "import json; json.dump(['q0000','q0034','q0048'],open('/tmp/smoke3.json','w'))"
env $COMMON LOCAL_LLM_BASE_URL=http://localhost:8030/v1 LOCAL_LLM_MODEL=$NEW \
  RLM_TRAJ_DIR=logs/traj_q36_smoke \
  ./scripts/run_obliq_shard.sh /tmp/smoke3.json logs/q36_smoke 20 20 3 > $L/smoke.log 2>&1
check 'logs/q36_smoke_s*.json' logs/traj_q36_smoke "GATE B smoke (Qwen3.6-35B-A3B)"
[ $? -eq 2 ] && { say "GATE B FAILED -- the harness does not work on this model"; tail -25 $L/smoke.log | tee -a $L/chain.log; restore_old; exit 1; }
say "GATE B passed"
fi   # end SKIP_TO<4

# Arm B runs the 100 D_pareto questions, NOT the full 181-question heldout set. The other
# 81 are the TEST split, which PHASE 6 runs fresh under both prompts -- running them here
# too would be ~81 rollouts of pure waste. These 100 are exactly what GEPA preloads, so
# the seed costs it zero.
say "PHASE 4a: seed baseline on dev100, split across BOTH GPUs (8 workers each)"
# run_obliq_shard.sh talks to one server, so dev100 is halved by hand; leaving all 16
# workers on :8030 would idle GPU1 and halve throughput.
$PYV -c "
import json
d=json.load(open('data/obliq_splits/twitter_dev.json'))
json.dump(d[0::2], open('/tmp/dev_h1.json','w')); json.dump(d[1::2], open('/tmp/dev_h2.json','w'))
print('dev100 split %d + %d'%(len(d[0::2]),len(d[1::2])))" | tee -a $L/chain.log
env $COMMON LOCAL_LLM_BASE_URL=http://localhost:8030/v1 LOCAL_LLM_MODEL=$NEW RLM_TRAJ_DIR=logs/traj_q36_base \
  ./scripts/run_obliq_shard.sh /tmp/dev_h1.json logs/q36_base_a 20 20 8 > $L/base_a.log 2>&1 &
sleep 5
env $COMMON LOCAL_LLM_BASE_URL=http://localhost:8031/v1 LOCAL_LLM_MODEL=$NEW RLM_TRAJ_DIR=logs/traj_q36_base \
  ./scripts/run_obliq_shard.sh /tmp/dev_h2.json logs/q36_base_a2 20 20 8 > $L/base_a2.log 2>&1 &
wait
$PYV -u scripts/score_q36_baseline.py 2>&1 | tee $L/baseline.log
check 'logs/q36_base_*_s*.json' logs/traj_q36_base "GATE C baseline (Qwen3.6-35B-A3B)"
[ $? -eq 2 ] && { say "GATE C FAILED -- baseline traces are broken"; restore_old; exit 1; }

B_NDCG=$($PYV -c "import json;print('%.4f'%json.load(open('$L/baseline.json'))['ndcg_dev'])" 2>/dev/null || echo 0)
say "GATE D: baseline dev100 NDCG@10 = $B_NDCG (need >= $GATE_MIN_NDCG)"
if [ "$($PYV -c "print(1 if $B_NDCG >= $GATE_MIN_NDCG else 0)")" != "1" ]; then
  say "GATE D FAILED -- $B_NDCG is too far below the 27B's 0.1302 for GEPA to close the gap."
  say "  The 27B stays the production model. Not spending GPU hours optimising a worse one."
  restore_old; exit 1
fi
say "GATE D passed -- this model is worth optimising"

say "PHASE 4b: seed on the 100 D_pareto questions (only now that GATE D has passed)"
$PYV -c "
import json
d=json.load(open('data/obliq_splits/twitter_pareto100.json'))
json.dump(d[0::2], open('/tmp/par_h1.json','w')); json.dump(d[1::2], open('/tmp/par_h2.json','w'))"
env $COMMON LOCAL_LLM_BASE_URL=http://localhost:8030/v1 LOCAL_LLM_MODEL=$NEW RLM_TRAJ_DIR=logs/traj_q36_base \
  ./scripts/run_obliq_shard.sh /tmp/par_h1.json logs/q36_base_b 20 20 8 > $L/base_b.log 2>&1 &
sleep 5
env $COMMON LOCAL_LLM_BASE_URL=http://localhost:8031/v1 LOCAL_LLM_MODEL=$NEW RLM_TRAJ_DIR=logs/traj_q36_base \
  ./scripts/run_obliq_shard.sh /tmp/par_h2.json logs/q36_base_b2 20 20 8 > $L/base_b2.log 2>&1 &
wait
check 'logs/q36_base_b*_s*.json' logs/traj_q36_base "PHASE 4b pareto seed"
[ $? -eq 2 ] && { say "PHASE 4b traces are broken"; restore_old; exit 1; }

# BUDGET = the largest search that fits the time available, not a guess from the speedup.
# Each accepted candidate costs |D_pareto| + b = 103 rollouts, spread over 16 workers, so
# the budget that fits T hours is 16*T*3600/latency. 800 calls buys only ~7 candidates
# against the GEPA paper's 1839-7051, so spend any speedup on SEARCH -- an under-powered
# budget (200) was the real defect in run 1 -- but never overrun the night.
SP=$($PYV -c "
import json
try: v=json.load(open('$L/baseline.json'))['speedup_vs_27b'] or 1.0
except Exception: v=1.0
print('%.2f'%v)" 2>/dev/null || echo 1.0)
BUDGET=$($PYV -c "
import json
T=float('${GEPA_TARGET_H:-11}')
try: lat=json.load(open('$L/baseline.json')).get('lat_dev') or 1500.0
except Exception: lat=1500.0
fits = 16*T*3600/max(lat,1.0)
print(int(max(800, min(2400, fits//103*103))))")
say "PHASE 5: GEPA on $NEW (speedup ${SP}x, budget $BUDGET metric calls ~= $((BUDGET/103)) candidates)"
# FRESH run dir. The cache key is (prompt_sha, qid) and carries NO model, so pointing this
# at logs/gepa_wide would mix 27B and 35B rollouts under the same keys and silently void
# every comparison. The seed preload comes from THIS model's own PHASE 4 run instead.
export GEPA_OUT=logs/gepa_q36
export GEPA_PRELOAD_GLOB='logs/q36_base_b*_s*.json'      # D_pareto seed rollouts, this model
export GEPA_PRELOAD_DEV_GLOB='logs/q36_base_a*_s*.json'  # dev100 seed rollouts, this model
export GEPA_BASELINE=$B_NDCG
export LOCAL_LLM_MODEL=$NEW
export GEPA_TRAJ_DIR=logs/traj_q36_gepa
export OBLIQ_FENCE_TOLERANT=1
export OBLIQ_REPL_CUE=1
# The per-question wall cap is a runaway catcher, so set it from what this model actually
# measured: 4x the mean is a decisive outlier, not a healthy question. A flat 3600s would be
# ~12x the mean if this model is as fast as hoped, letting a 2h41m stall pass unnoticed again.
export GEPA_PER_QUESTION_CAP=$($PYV -c "
import json
try: lat=json.load(open('$L/baseline.json')).get('lat_dev') or 900
except Exception: lat=900
print(int(min(3600,max(1200,4*lat))))")
mkdir -p $GEPA_OUT $GEPA_TRAJ_DIR
$PYV -u scripts/gepa_wide.py --workers 16 --valset 100 --minibatch 3 \
  --max-metric-calls $BUDGET > $L/gepa_q36.log 2>&1 < /dev/null &
GP=$!
_gstart=$(date +%s)
DEADLINE_H=${DEADLINE_H:-20}
say "  GEPA pid $GP -> $GEPA_OUT (budget $BUDGET, valset 100, minibatch 3, baseline $B_NDCG, per-q cap ${GEPA_PER_QUESTION_CAP}s, deadline ${DEADLINE_H}h)"
# periodic trace check while it runs, so a silent failure does not burn the whole night
while kill -0 $GP 2>/dev/null; do
  sleep 1800
  kill -0 $GP 2>/dev/null || break
  say "  [gepa] $(tail -1 $GEPA_OUT/progress.txt 2>/dev/null | cut -c1-150)"
  R=$(grep -c '\[runaway\]' $GEPA_OUT/progress.txt 2>/dev/null | head -1); R=${R:-0}
  I=$(grep -c 'scoring it as NEUTRAL' $GEPA_OUT/progress.txt 2>/dev/null | head -1); I=${I:-0}
  [ "$R" -gt 8 ] && say "  !! $R runaway kills -- the per-question cap may be too tight"
  [ "$I" -gt 2 ] && say "  !! $I candidates had every rollout die -- infrastructure problem"
  check "$GEPA_OUT/shards/*/out*.json" "$GEPA_TRAJ_DIR" "GEPA in-flight" >/dev/null 2>&1
  [ $? -eq 2 ] && say "  !! in-flight trace check says BROKEN -- see $L/checks.log"
  # Guarantee a verdict. A hung GEPA must not eat the night and leave nothing to report:
  # past the deadline we stop it and score the candidates it did produce.
  if [ $(( ($(date +%s) - _gstart) / 3600 )) -ge "$DEADLINE_H" ]; then
    say "  !! GEPA hit the ${DEADLINE_H}h deadline -- stopping it and scoring what it found"
    kill $GP 2>/dev/null; sleep 20
    for q in $(pgrep -f 'python.*run_obliq_mine[.]py --ids-file' 2>/dev/null); do kill $q 2>/dev/null; done
    break
  fi
done
say "PHASE 5 done"
$PYV -u scripts/validate_wide.py 2>&1 | tee $L/validate.log

say "PHASE 6: held-out test, GEPA winner vs seed on the untouched test split"
if [ -s $GEPA_OUT/best_strategy.txt ]; then
  $PYV -c "
import json; d=json.load(open('$GEPA_OUT/test_split.json'))['test']
json.dump(d, open('/tmp/q36_test.json','w')); print('test split:', len(d), 'questions')" | tee -a $L/chain.log
  env $COMMON LOCAL_LLM_BASE_URL=http://localhost:8030/v1 LOCAL_LLM_MODEL=$NEW \
    OBLIQ_STRATEGY_FILE=$GEPA_OUT/seed_strategy.txt OBLIQ_TASK_FILE=$GEPA_OUT/seed_task.txt \
    RLM_TRAJ_DIR=logs/traj_q36_test_seed \
    ./scripts/run_obliq_shard.sh /tmp/q36_test.json logs/q36_test_seed 20 20 8 > $L/test_seed.log 2>&1 &
  sleep 5
  env $COMMON LOCAL_LLM_BASE_URL=http://localhost:8031/v1 LOCAL_LLM_MODEL=$NEW \
    OBLIQ_STRATEGY_FILE=$GEPA_OUT/best_strategy.txt OBLIQ_TASK_FILE=$GEPA_OUT/best_task.txt \
    RLM_TRAJ_DIR=logs/traj_q36_test_best \
    ./scripts/run_obliq_shard.sh /tmp/q36_test.json logs/q36_test_best 20 20 8 > $L/test_best.log 2>&1 &
  wait
  check 'logs/q36_test_seed_s*.json' logs/traj_q36_test_seed "test arm: SEED prompt"
  check 'logs/q36_test_best_s*.json' logs/traj_q36_test_best "test arm: GEPA winner"
  $PYV -u scripts/score_pair.py logs/q36_test_seed logs/q36_test_best SEED GEPA 2>&1 | tee $L/verdict.log
else
  say "no GEPA winner was written -- nothing to test"
fi
say "NIGHT7 COMPLETE -- baseline $L/baseline.log | verdict $L/verdict.log | checks $L/checks.log"
