#!/usr/bin/env bash
# FRAMES (824 q) full-document RLM with GPT-5 high, 2026-10-27: the Full-Doc RLM harness with the same flags as the
# BrowseComp+ 0.8747 / BM25 0.600 runs (design B, max 10 searches, 30 turns, seeded first search, compaction,
# grounding gate, no sub-LLM), on the FRAMES BM25 server :8041 (this harness's wire protocol), through
# src/run_bm25.py, which tells the model the retriever is BM25 over the Wikipedia dump.
# From original/RLM/scripts/run5d/frames_fd_gpt5.sh (arguments and environment unchanged; locations from
# configs/paths.env). Row FR-11. Startup hooks: run5d = retry429 + azure_sampling (reasoning_effort IS sent) +
# cap_ranking (inactive: OBLIQ_CAP_RANKING unset).
#   experiments/frames_gpt5.sh <out_dir> [workers] [first_q] [last_q_exclusive]
set -u
OUT=$(realpath -m $1); W=${2:-10}; Q0=${3:-0}; Q1=${4:-824}
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
PY=$PYTHON; SUP=$FULLDOC_ROOT                # original: the vllm conda env's python; the Full-Doc RLM working directory
DEP=${AZURE_DEPLOYMENT_GPT5:?set AZURE_DEPLOYMENT_GPT5}
set -a; . "${AZ_ENV:?set AZ_ENV}"; set +a    # original: AZ_ENV defaulted to a file in a session scratch directory
mkdir -p "$OUT"
SERVER=http://127.0.0.1:8041
curl -s -m10 $SERVER/health | grep -q '"ok"' || { echo "ABORT: $SERVER down"; exit 1; }
export AZURE_OPENAI_ENDPOINT="$AZ_EP/" AZURE_OPENAI_API_KEY="$AZ_KEY" AZURE_OPENAI_API_VERSION=2024-12-01-preview AZURE_OPENAI_DEPLOYMENT=$DEP
need_hook run5d
export PYTHONUNBUFFERED=1 PYTHONPATH=$HOOKS_DIR/run5d:$RLMLIB_PATH HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export QUERIES_FILE=${FRAMES_DATA_DIR:?set FRAMES_DATA_DIR}/frames_queries.jsonl RLM_USAGE_LOG=$OUT/usage.jsonl
unset BM25_CORPUS SUP_FENCE_TOLERANT SUP_MAX_BLOCKS SUP_PRESENCE_PENALTY GEPA_PROMPTS
export OUT PY SUP SERVER SRC DEP
echo "$(date '+%F %T') FRAMES full-doc GPT-5 high, $W workers -> $OUT"
seq $Q0 $((Q1-1)) | xargs -P "$W" -I{} sh -c '
  f=$OUT/{}.json
  [ -s "$f" ] && exit 0
  cd $SUP && timeout 3300 $PY $SRC/run_bm25.py --query-id {} --out "$f" \
    --design B --server $SERVER \
    --root-model gpt-5 --sub-model gpt-5 --root-deployment $DEP --sub-deployment $DEP \
    --reasoning-effort high --sub-reasoning-effort high --default-k 5 \
    --max-iterations 30 --max-searches 10 --seed-context --compaction --grounding-gate \
    --lean-prologue --no-delegation --solo-prompt --no-orchestrator --compaction-threshold-pct 0.85 \
    --max-concurrent-subcalls 4 --max-timeout 2700 > "${f%.json}.log" 2>&1
'
echo "$(date '+%F %T') done: $(ls $OUT/*.json 2>/dev/null | wc -l) results"
