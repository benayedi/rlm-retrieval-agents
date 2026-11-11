#!/usr/bin/env bash
# GEPA-1: GEPA on BrowseComp-Plus, run 1 (proportional D_pareto of 100, minibatch 3, budget 800, Claude critic).
# From original/RLM/scripts/run_gepa_bcp.sh (arguments and environment unchanged; locations from configs/paths.env).
# The driver is the document REPL agent's split runner, agents/document_repl/src/split_runner.py (original
# scripts/gepa_bcp.py). Run 1 itself ran on the version of that file before its 2026-10-12 edit
# (agents/document_repl/original/RLM_2026-11-11/RLM/logs/gepa_bcp.py.bak_1012; CHANGES_FROM_ORIGINAL.md F-B).
#
# What actually ran differs from this file in two documented ways (both copies are in original/):
#   * original/RLM/logs/run_gepa_bcp_frozen.sh -- the frozen copy that was started 2026-10-07 -- did not yet export
#     SUP_FENCE_TOLERANT/SUP_MAX_BLOCKS/SUP_PRESENCE_PENALTY in the qwen36 branch and started a second Qwen3.6 server
#     on GPU 1 (UTIL 0.70) unconditionally; this file (the later state of run_gepa_bcp.sh) exports them and starts
#     the second server only with SECOND_SERVER=1.
#   * original/RLM/logs/gepa_bcp/restart_after_c36.sh resumed the driver after candidate c36c56c1c5 with
#     GEPA_BCP_MODEL=qwen36-35b-a3b SUP_FENCE_TOLERANT=1 SUP_MAX_BLOCKS=5 SUP_PRESENCE_PENALTY=1.5
#     GEPA_BCP_SERVERS=http://localhost:8030/v1 --max-metric-calls 800 --workers 20 (so most of run 1 used 20 workers).
# Run 1 was launched with GEPA_BCP_MODEL=qwen36-35b-a3b (the default below is the original's qwen3.8-27b-bf16).
#
# Needs: SERVE_QWEN36 (or SERVE_QWEN27B) = the vLLM launcher (original scripts/serve_qwen36_moe.sh /
# serve_qwen27b_bf16.sh), PYLATE_PYTHON (ColBERT server env), BCP_FORMAT_CHECK = the original repository's
# scripts/bcp_format_check.py (not part of this repository), the `claude` CLI (CLAUDE_BIN) for the critic.
#   GEPA_BCP_MODEL=qwen36-35b-a3b bash studies/prompt_optimisation/experiments/gepa1_bcp_run1.sh
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$RUNS_DIR"                                       # original: cd <original repo>
L=$RUNS_DIR/gepa_bcp; mkdir -p $L                    # original: L=logs/gepa_bcp
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L/chain.log; }
BUDGET=${BUDGET:-800}
WORKERS=${WORKERS:-8}
export GEPA_BCP_MODEL=${GEPA_BCP_MODEL:-qwen3.8-27b-bf16}   # or qwen36-35b-a3b
case "$GEPA_BCP_MODEL" in
  qwen36-35b-a3b) SERVE=${SERVE_QWEN36:-$REPO/benchmarks/models/serve_qwen36.sh}   # original: scripts/serve_qwen36_moe.sh (its copy)
    # per-turn output cap: Qwen3.6 loops inside a turn until the 131k context overflows
    export GEN_OVERRIDE='{"max_new_tokens": 16384}'
    # and at most 5 code blocks per turn: its loops emit 30-169 blocks in one turn
    export SUP_FENCE_TOLERANT=1 SUP_MAX_BLOCKS=5
    # the model card's thinking-mode setting "to reduce endless repetitions"; vLLM cannot set it
    # server-side, so the harness sends it per request (src/run_one.py _sampling)
    export SUP_PRESENCE_PENALTY=1.5 ;;
  qwen3.8-27b-bf16) SERVE=${SERVE_QWEN27B:-} ;;     # original: scripts/serve_qwen27b_bf16.sh
  *) echo "unknown GEPA_BCP_MODEL $GEPA_BCP_MODEL"; exit 1 ;;
esac

say "preflight: GPU processes must be ours or none"
others=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | while read p; do
  [ -n "$p" ] && ps -o user= -p "$p" 2>/dev/null; done | grep -v "^$(whoami)$" | sort -u)
if [ -n "$others" ]; then say "ABORT: GPUs in use by: $others"; exit 1; fi

if ! curl -s -m 4 localhost:8030/v1/models >/dev/null; then
  [ -f "$SERVE" ] || { say "ABORT: set SERVE_QWEN36 / SERVE_QWEN27B (the vLLM launcher of $GEPA_BCP_MODEL)"; exit 1; }
  say "starting $GEPA_BCP_MODEL on GPU 0 :8030 ($SERVE)"
  GPU=0 PORT=8030 setsid nohup bash $SERVE > $L/vllm8030.log 2>&1 < /dev/null &
fi
if ! curl -s -m 4 localhost:8766/ >/dev/null 2>&1 && ! grep -q READY $L/colbert8766.log 2>/dev/null; then
  : "${PYLATE_PYTHON:?set PYLATE_PYTHON in configs/paths.env}"
  say "starting ColBERT-8192 on GPU 1 :8766"
  CUDA_VISIBLE_DEVICES=1 HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} \
    COLBERT_INDEX_NAME=bc_rmcb_dl8192 COLBERT_DOC_LEN=8192 COLBERT_SEARCH_BATCH=2048 COLBERT_LOW_MEMORY=1 \
    setsid nohup $PYLATE_PYTHON -u $DOC_SRC/retrieval_server_colbert.py --port 8766 \
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
  need_file BCP_FORMAT_CHECK "the original repository's scripts/bcp_format_check.py (not part of this repository)"
  rm -rf $RUNS_DIR/gepa_bcp_smoke                    # original: logs/gepa_bcp_smoke
  env -u GEPA_PROMPTS $PYV -u $DOC_SRC/split_runner.py --smoke 2 2>&1 | tail -4 | tee -a $L/chain.log
  local rc=${PIPESTATUS[0]}
  $PYV $BCP_FORMAT_CHECK "$(ls -d $RUNS_DIR/gepa_bcp_smoke/rollouts/*/ | head -1)" 2>&1 | tee -a $L/chain.log
  local fc=${PIPESTATUS[0]}
  say "smoke[$1]: rollouts rc=$rc format rc=$fc"; [ "$rc" = 0 ] && [ "$fc" = 0 ]
}
say "smoke: seed on 2 D_pareto questions, strict repl-fence parsing"
if ! smoke strict; then
  # Qwen3.6 lost 40% of its OBLIQ turns to the fence label. Retry once with the vetted
  # tolerant parser (frozen harness code, NOT a prompt change, so GEPA's search stays clean).
  export SUP_FENCE_TOLERANT=1
  say "retrying smoke with SUP_FENCE_TOLERANT=1"
  smoke tolerant || { say "ABORT: smoke failed with tolerant parsing too -- see $RUNS_DIR/gepa_bcp_smoke/"; exit 1; }
fi
say "harness settled: SUP_FENCE_TOLERANT=${SUP_FENCE_TOLERANT:-0}"

say "GEPA: model $GEPA_BCP_MODEL, budget $BUDGET, workers $WORKERS, critic claude (baseline on D_pareto first)"
env -u GEPA_PROMPTS $PYV -u $DOC_SRC/split_runner.py --max-metric-calls $BUDGET --workers $WORKERS \
  > $L/gepa.log 2>&1
say "GEPA exited ($?). progress: $L/progress.txt"
