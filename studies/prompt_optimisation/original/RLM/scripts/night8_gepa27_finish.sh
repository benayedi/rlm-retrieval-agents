#!/usr/bin/env bash
# Take the restored Qwen3.8-27B GEPA run through to an actual VERDICT.
#
# restore_old() in the night7 chain restarts gepa_wide.py bare: no deadline, no validation,
# no held-out test. That leaves a half-finished search and nothing to report. This watches
# that run, checks its traces, stops it at a deadline if needed, then scores the winner on
# the 81-question TEST split that neither D_feedback nor D_pareto ever touched.
#
# NOTE: no OBLIQ_REPL_CUE / OBLIQ_FENCE_TOLERANT here, deliberately. Qwen3.8-27B emits
# ```repl on 100% of turns, and those flags change TURN_RULE -- which gepa_wide.py does NOT
# include in its candidate hash, so enabling them would silently invalidate the cache this
# run resumes from.
set -uo pipefail
cd /home/extstu_00/RLM
PYV=/home/extstu_00/miniconda3/envs/vllm/bin/python
L=logs/night8; mkdir -p $L
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L/chain.log; }
export PYTHONPATH=/home/extstu_00/rlmrepo HF_HOME=/home/extstu_00/.cache/huggingface
COMMON="PYTHONUNBUFFERED=1 EMBEDDING_SERVER_URL=http://localhost:8010 OBLIQ_INDEX=obliq-des-twitter OBLIQ_TASK=descriptive OBLIQ_SUBSET=twitter LOCAL_LM_TIMEOUT=1800 AZURE_OPENAI_REASONING_EFFORT=medium SUB_LM_REASONING_EFFORT=medium ORCHESTRATOR=0 LIGHTON_TOOL_DOCS=1 LIGHTON_TASK_TEXT=1 SEARCH_SNIPPET_TOKENS=512 OFFICIAL_TOP_K=20"
OUT=logs/gepa_wide
DEADLINE_H=${DEADLINE_H:-14}
check(){ $PYV scripts/trace_check.py "$1" --traj "$2" --label "$3" 2>&1 | tee -a $L/checks.log; return ${PIPESTATUS[0]}; }

say "waiting for a 27B server on :8030"
for i in $(seq 1 180); do curl -s -m 4 http://localhost:8030/v1/models >/dev/null 2>&1 && break; sleep 10; done
curl -s -m 5 http://localhost:8030/v1/models >/dev/null 2>&1 || { say "ABORT: :8030 never came up"; exit 1; }
M=$(curl -s -m 5 http://localhost:8030/v1/models | $PYV -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
say "server up, serving '$M'"
case "$M" in
  *27b*|*27B*) : ;;
  *) say "ABORT: :8030 is serving '$M', not the 27B -- refusing to score a mixed-model cache"; exit 1 ;;
esac

GP=$(pgrep -f 'python.*scripts/gepa_wide[.]py' 2>/dev/null | head -1)
if [ -z "${GP:-}" ]; then
  say "no GEPA running -- starting it (resumes from $OUT/cache.json)"
  $PYV -u scripts/gepa_wide.py --workers 16 --valset 100 --minibatch 3 \
    --max-metric-calls 800 > $L/gepa27.log 2>&1 &
  GP=$!
fi
say "watching GEPA pid $GP (deadline ${DEADLINE_H}h)"
_t0=$(date +%s)
while kill -0 $GP 2>/dev/null; do
  sleep 1800
  kill -0 $GP 2>/dev/null || break
  say "  [gepa] $(tail -1 $OUT/progress.txt 2>/dev/null | cut -c1-150)"
  R=$(grep -c '\[runaway\]' $OUT/progress.txt 2>/dev/null | head -1); R=${R:-0}
  I=$(grep -c 'scoring it as NEUTRAL' $OUT/progress.txt 2>/dev/null | head -1); I=${I:-0}
  [ "$R" -gt 8 ] && say "  !! $R runaway kills"
  [ "$I" -gt 2 ] && say "  !! $I candidates lost every rollout -- infrastructure problem"
  check "$OUT/shards/*/out*.json" logs/rlm_traj_gepa2 "27B GEPA in-flight" >/dev/null 2>&1
  [ $? -eq 2 ] && say "  !! in-flight trace check says BROKEN -- see $L/checks.log"
  if [ $(( ($(date +%s) - _t0) / 3600 )) -ge "$DEADLINE_H" ]; then
    say "  !! ${DEADLINE_H}h deadline -- stopping GEPA and scoring what it found"
    kill $GP 2>/dev/null; sleep 20
    for q in $(pgrep -f 'python.*run_obliq_mine[.]py --ids-file' 2>/dev/null); do kill $q 2>/dev/null; done
    break
  fi
done
say "GEPA finished -- validating"
GEPA_OUT=$OUT $PYV -u scripts/validate_wide.py 2>&1 | tee $L/validate.log

say "HELD-OUT TEST: GEPA winner vs seed on the 81 questions neither split touched"
if [ -s $OUT/best_strategy.txt ]; then
  $PYV -c "
import json; d=json.load(open('$OUT/test_split.json'))['test']
json.dump(d[0::2],open('/tmp/t81_h1.json','w')); json.dump(d[1::2],open('/tmp/t81_h2.json','w'))
print('test split:',len(d),'questions')" | tee -a $L/chain.log
  # each PROMPT arm is split over both GPUs, so the arms stay balanced
  for ARM in seed best; do
    env $COMMON LOCAL_LLM_BASE_URL=http://localhost:8030/v1 \
      OBLIQ_STRATEGY_FILE=$OUT/${ARM}_strategy.txt OBLIQ_TASK_FILE=$OUT/${ARM}_task.txt \
      RLM_TRAJ_DIR=logs/traj_t81_$ARM \
      ./scripts/run_obliq_shard.sh /tmp/t81_h1.json logs/t81_${ARM} 20 20 4 > $L/t81_${ARM}_a.log 2>&1 &
    sleep 3
    env $COMMON LOCAL_LLM_BASE_URL=http://localhost:8031/v1 \
      OBLIQ_STRATEGY_FILE=$OUT/${ARM}_strategy.txt OBLIQ_TASK_FILE=$OUT/${ARM}_task.txt \
      RLM_TRAJ_DIR=logs/traj_t81_$ARM \
      ./scripts/run_obliq_shard.sh /tmp/t81_h2.json logs/t81_${ARM}2 20 20 4 > $L/t81_${ARM}_b.log 2>&1 &
    sleep 3
  done
  wait
  check 'logs/t81_seed*_s*.json' logs/traj_t81_seed "test arm: SEED"
  check 'logs/t81_best*_s*.json' logs/traj_t81_best "test arm: GEPA winner"
  # one paired test over all 81 -- score_pair merges logs/t81_X and logs/t81_X2
  $PYV -u scripts/score_pair.py logs/t81_seed logs/t81_best SEED GEPA 2>&1 | tee $L/verdict.log
else
  say "no winner written -- GEPA produced nothing to test"
fi
say "NIGHT8 COMPLETE -- verdict $L/verdict.log | validate $L/validate.log | checks $L/checks.log"
