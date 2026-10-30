#!/usr/bin/env bash
# FRAMES (824 q) full-document RLM with GPT-5 high (Azure ismail-gpt-5), 2026-10-27: the supervisor's harness with the
# same flags as the BrowseComp+ 0.8747 / BM25 0.600 runs (design B, max 10 searches, 30 turns, seeded first search,
# compaction, grounding gate, no sub-LLM), on the FRAMES BM25 server :8041 (sup wire), through
# scripts/frames_sup_run_one.py, which tells the model the retriever is BM25 over the Wikipedia dump.
#   scripts/run5d/frames_fd_gpt5.sh <out_dir> [workers] [first_q] [last_q_exclusive]
set -u
OUT=$(realpath -m $1); W=${2:-10}; Q0=${3:-0}; Q1=${4:-824}
R=/home/extstu_00/RLM; SUP=/home/extstu_00/suprlm; PY=/home/extstu_00/miniconda3/envs/vllm/bin/python
AZ_ENV=${AZ_ENV:-/tmp/claude-1003/-home-extstu-00-RLM/2a1318b5-524a-4d81-9aba-3918a191383c/scratchpad/azure.env}
set -a; . "$AZ_ENV"; set +a
mkdir -p "$OUT"
SERVER=http://127.0.0.1:8041
curl -s -m10 $SERVER/health | grep -q '"ok"' || { echo "ABORT: $SERVER down"; exit 1; }
export AZURE_OPENAI_ENDPOINT="$AZ_EP/" AZURE_OPENAI_API_KEY="$AZ_KEY" AZURE_OPENAI_API_VERSION=2024-12-01-preview AZURE_OPENAI_DEPLOYMENT=ismail-gpt-5
export PYTHONUNBUFFERED=1 PYTHONPATH=$R/scripts/hooks/run5d:/home/extstu_00/rlmrepo HF_HOME=/home/extstu_00/.cache/huggingface
export QUERIES_FILE=$R/data/frames/frames_queries.jsonl RLM_USAGE_LOG=$OUT/usage.jsonl
unset BM25_CORPUS SUP_FENCE_TOLERANT SUP_MAX_BLOCKS SUP_PRESENCE_PENALTY GEPA_PROMPTS
export OUT PY SUP SERVER
echo "$(date '+%F %T') FRAMES full-doc GPT-5 high, $W workers -> $OUT"
seq $Q0 $((Q1-1)) | xargs -P "$W" -I{} sh -c '
  f=$OUT/{}.json
  [ -s "$f" ] && exit 0
  cd $SUP && timeout 3300 $PY /home/extstu_00/RLM/scripts/frames_sup_run_one.py --query-id {} --out "$f" \
    --design B --server $SERVER \
    --root-model gpt-5 --sub-model gpt-5 --root-deployment ismail-gpt-5 --sub-deployment ismail-gpt-5 \
    --reasoning-effort high --sub-reasoning-effort high --default-k 5 \
    --max-iterations 30 --max-searches 10 --seed-context --compaction --grounding-gate \
    --lean-prologue --no-delegation --solo-prompt --no-orchestrator --compaction-threshold-pct 0.85 \
    --max-concurrent-subcalls 4 --max-timeout 2700 > "${f%.json}.log" 2>&1
'
echo "$(date '+%F %T') done: $(ls $OUT/*.json 2>/dev/null | wc -l) results"
