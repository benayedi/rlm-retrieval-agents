#!/usr/bin/env bash
# BrowseComp-Plus, all 830, the supervisor's full-document RLM with gpt-5 (EFFORT, default high = the 0.8747 ColBERT control, row 11; medium = row 12)
# Experiments row 12) with ONE change: retrieval = BM25 (default Lucene k1 0.9 / b 0.4, :8043,
# scripts/frames_search_server.py --protocol sup --dataset bcp), through scripts/frames_sup_run_one.py with
# BM25_CORPUS=bcp so the retriever description is true for BM25 (as in the Qwen3.6 BM25 row 18).
# Agent flags = the control's recorded config (docs/supervisor_artifact.html): design B, 30 turns, 10 searches,
# model-chosen k (fallback 5), seeded first search, compaction at 0.85, grounding gate, solo/lean prompt, no
# delegation, no orchestrator, 4 concurrent sub-calls, root and sub gpt-5 medium.
# Model: Azure deployment ismail-gpt-5, verified to serve gpt-5-2025-08-07 (same build as the control).
# No --save-trajectory: with whole BM25 pages a trajectory is 0.3-2.3 GB per question (smoke 2026-10-26).
# Credentials come from AZ_ENV (a file with AZ_EP / AZ_KEY), never from this repo.
#   AZ_ENV=/path/azure.env W=16 [IDS_FILE=ids.txt] OUT=logs/bcp_gpt5_bm25/runs scripts/bcp_gpt5_bm25.sh
set -u
cd /home/extstu_00/RLM
PY=/home/extstu_00/miniconda3/envs/vllm/bin/python
SUP=/home/extstu_00/suprlm
W=${W:-16}
OUT=$(realpath -m "${OUT:-logs/bcp_gpt5_bm25/runs}"); mkdir -p "$OUT"
SERVER=${SERVER:-http://127.0.0.1:8043}
DEP=${DEP:-ismail-gpt-5}
say() { printf '%s  [bcp-gpt5-bm25] %s\n' "$(date '+%F %T')" "$*"; }
set -a; . "${AZ_ENV:?set AZ_ENV}"; set +a
export AZURE_OPENAI_ENDPOINT="$AZ_EP/" AZURE_OPENAI_API_KEY="$AZ_KEY" AZURE_OPENAI_API_VERSION=2024-12-01-preview \
       AZURE_OPENAI_DEPLOYMENT="$DEP"
curl -s -m10 $SERVER/health | grep -q '"ok"' || { say "ABORT: $SERVER down"; exit 1; }
export PYTHONUNBUFFERED=1 PYTHONPATH=/home/extstu_00/RLM/scripts/hooks/retry429:/home/extstu_00/rlmrepo HF_HOME=/home/extstu_00/.cache/huggingface BM25_CORPUS=bcp
unset QUERIES_FILE SUP_FENCE_TOLERANT SUP_MAX_BLOCKS SUP_PRESENCE_PENALTY
ENTRY=/home/extstu_00/RLM/scripts/frames_sup_run_one.py
EFFORT=${EFFORT:-high}
export EFFORT OUT PY SUP ENTRY SERVER DEP
IDS_FILE=${IDS_FILE:-}
if [ -z "$IDS_FILE" ]; then IDS_FILE=$OUT/../ids830.txt
  $PY -c "import json;[print(json.loads(l)['query_id']) for l in open('$SUP/data/prepared/queries.jsonl')]" > "$IDS_FILE"; fi
say "$(wc -l < $IDS_FILE) questions, $W workers, deployment $DEP, effort $EFFORT, server $SERVER -> $OUT"
cat "$IDS_FILE" | xargs -P "$W" -I{} sh -c '
  f=$OUT/{}.json
  [ -s "$f" ] && exit 0
  cd $SUP && timeout 3300 $PY $ENTRY --query-id {} --out "$f" \
    --design B --server $SERVER \
    --root-model gpt-5 --sub-model gpt-5 --root-deployment $DEP --sub-deployment $DEP \
    --reasoning-effort $EFFORT --sub-reasoning-effort $EFFORT --default-k 5 \
    --max-iterations 30 --max-searches 10 --seed-context --compaction --grounding-gate \
    --lean-prologue --no-delegation --solo-prompt --no-orchestrator --compaction-threshold-pct 0.85 \
    --max-concurrent-subcalls 4 --max-timeout 2700 \
    > "${f%.json}.log" 2>&1
'
say "rollouts done: $(ls "$OUT"/*.json 2>/dev/null | wc -l)"
