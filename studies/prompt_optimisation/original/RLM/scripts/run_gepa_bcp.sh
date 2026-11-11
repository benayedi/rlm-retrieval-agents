#!/usr/bin/env bash
# GEPA on BrowseComp-Plus, end to end, once the GPUs are free. Plan: docs/GEPA_BROWSECOMP_PLAN.md
#
#   GPU 0: Qwen3.8-27B bf16 on :8030 (agent + judge + fallback critic)
#   GPU 1: Reason-ModernColBERT-8192 PLAID on :8766 (retrieval) -- the 79.4% baseline's layout
#   critic: Claude via `claude -p` (no GPU)
#
# Phases: preflight (GPUs really free) -> servers -> smoke (2 real questions; abort on any
# crash) -> GEPA. The seed is RE-RUN on D_pareto first (inside gepa_bcp.py), which also
# measures Qwen's run-to-run noise on these 100 questions.
#
# RUN IT FROM A FROZEN COPY so editing this file mid-run cannot corrupt it (HANDOFF s7 trap 2):
#   cp scripts/run_gepa_bcp.sh logs/run_gepa_bcp_frozen.sh
#   setsid nohup bash logs/run_gepa_bcp_frozen.sh > logs/gepa_bcp_chain.out 2>&1 < /dev/null & disown
set -uo pipefail
cd /home/extstu_00/RLM
PYV=/home/extstu_00/miniconda3/envs/vllm/bin/python
L=logs/gepa_bcp; mkdir -p $L
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L/chain.log; }
BUDGET=${BUDGET:-800}
WORKERS=${WORKERS:-8}
export GEPA_BCP_MODEL=${GEPA_BCP_MODEL:-qwen3.8-27b-bf16}   # or qwen36-35b-a3b
case "$GEPA_BCP_MODEL" in
  qwen36-35b-a3b) SERVE=scripts/serve_qwen36_moe.sh
    # per-turn output cap: Qwen3.6 loops inside a turn until the 131k context overflows
    export GEN_OVERRIDE='{"max_new_tokens": 16384}'
    # and at most 5 code blocks per turn: its loops emit 30-169 blocks in one turn
    export SUP_FENCE_TOLERANT=1 SUP_MAX_BLOCKS=5
    # the model card's thinking-mode setting "to reduce endless repetitions"; vLLM cannot set it
    # server-side, so the harness sends it per request (suprlm/src/run_one.py _sampling)
    export SUP_PRESENCE_PENALTY=1.5 ;;
  qwen3.8-27b-bf16) SERVE=scripts/serve_qwen27b_bf16.sh ;;
  *) echo "unknown GEPA_BCP_MODEL $GEPA_BCP_MODEL"; exit 1 ;;
esac

say "preflight: GPU processes must be ours or none"
others=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | while read p; do
  [ -n "$p" ] && ps -o user= -p "$p" 2>/dev/null; done | grep -v "^$(whoami)$" | sort -u)
if [ -n "$others" ]; then say "ABORT: GPUs in use by: $others"; exit 1; fi

if ! curl -s -m 4 localhost:8030/v1/models >/dev/null; then
  say "starting $GEPA_BCP_MODEL on GPU 0 :8030 ($SERVE)"
  GPU=0 PORT=8030 setsid nohup bash $SERVE > $L/vllm8030.log 2>&1 < /dev/null &
fi
if ! curl -s -m 4 localhost:8766/ >/dev/null 2>&1 && ! grep -q READY $L/colbert8766.log 2>/dev/null; then
  say "starting ColBERT-8192 on GPU 1 :8766"
  CUDA_VISIBLE_DEVICES=1 HF_HOME=/home/extstu_00/.cache/huggingface \
    COLBERT_INDEX_NAME=bc_rmcb_dl8192 COLBERT_DOC_LEN=8192 COLBERT_SEARCH_BATCH=2048 COLBERT_LOW_MEMORY=1 \
    setsid nohup /home/extstu_00/miniconda3/envs/pylate/bin/python -u /home/extstu_00/suprlm/src/retrieval_server_colbert.py --port 8766 \
    > $L/colbert8766.log 2>&1 < /dev/null &
fi
for i in $(seq 1 120); do curl -s -m 4 localhost:8030/v1/models >/dev/null && break; sleep 15; done
curl -s -m 4 localhost:8030/v1/models >/dev/null || { say "ABORT: :8030 never came up"; tail -20 $L/vllm8030.log; exit 1; }
for i in $(seq 1 120); do grep -qE "READY|Traceback" $L/colbert8766.log 2>/dev/null && break; sleep 10; done
grep -q READY $L/colbert8766.log || { say "ABORT: ColBERT failed"; tail -20 $L/colbert8766.log; exit 1; }
M=$(curl -s -m 5 localhost:8030/v1/models | $PYV -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])")
[ "$M" = "$GEPA_BCP_MODEL" ] || { say "ABORT: :8030 serves '$M', expected '$GEPA_BCP_MODEL'"; exit 1; }
# NOT sharing GPU 1 with a second agent server: ColBERT's footprint grows while serving
# (measured 3 -> 13.6 -> 25.6 GB on 2026-10-07; PyTorch's caching allocator keeps it), which
# OOM-killed a co-located Qwen3.6 and blocked its restart. Opt in with SECOND_SERVER=1 only
# after capping ColBERT's memory.
if [ "${SECOND_SERVER:-0}" = "1" ] && [ "$GEPA_BCP_MODEL" = "qwen36-35b-a3b" ]; then
  if ! curl -s -m 4 localhost:8031/v1/models >/dev/null; then
    say "starting a second $GEPA_BCP_MODEL on GPU 1 :8031 (UTIL 0.55, beside ColBERT)"
    GPU=1 PORT=8031 UTIL=0.55 setsid nohup bash $SERVE > $L/vllm8031.log 2>&1 < /dev/null &
    for i in $(seq 1 80); do curl -s -m 4 localhost:8031/v1/models >/dev/null && break; sleep 15; done
  fi
  curl -s -m 4 localhost:8031/v1/models >/dev/null && export GEPA_BCP_SERVERS=http://localhost:8030/v1,http://localhost:8031/v1
fi
say "servers up ($M) | agent servers: ${GEPA_BCP_SERVERS:-http://localhost:8030/v1}"

smoke(){  # $1 = label; real rollouts on 2 D_pareto questions, then the turn-format check
  rm -rf logs/gepa_bcp_smoke
  env -u GEPA_PROMPTS $PYV -u scripts/gepa_bcp.py --smoke 2 2>&1 | tail -4 | tee -a $L/chain.log
  local rc=${PIPESTATUS[0]}
  $PYV scripts/bcp_format_check.py "$(ls -d logs/gepa_bcp_smoke/rollouts/*/ | head -1)" 2>&1 | tee -a $L/chain.log
  local fc=${PIPESTATUS[0]}
  say "smoke[$1]: rollouts rc=$rc format rc=$fc"; [ "$rc" = 0 ] && [ "$fc" = 0 ]
}
say "smoke: seed on 2 D_pareto questions, strict repl-fence parsing"
if ! smoke strict; then
  # Qwen3.6 lost 40% of its OBLIQ turns to the fence label. Retry once with the vetted
  # tolerant parser (frozen harness code, NOT a prompt change, so GEPA's search stays clean).
  export SUP_FENCE_TOLERANT=1
  say "retrying smoke with SUP_FENCE_TOLERANT=1"
  smoke tolerant || { say "ABORT: smoke failed with tolerant parsing too -- see logs/gepa_bcp_smoke/"; exit 1; }
fi
say "harness settled: SUP_FENCE_TOLERANT=${SUP_FENCE_TOLERANT:-0}"

say "GEPA: model $GEPA_BCP_MODEL, budget $BUDGET, workers $WORKERS, critic claude (baseline on D_pareto first)"
env -u GEPA_PROMPTS $PYV -u scripts/gepa_bcp.py --max-metric-calls $BUDGET --workers $WORKERS \
  > $L/gepa.log 2>&1
say "GEPA exited ($?). progress: $L/progress.txt"
