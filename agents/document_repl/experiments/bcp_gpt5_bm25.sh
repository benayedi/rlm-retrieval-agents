#!/usr/bin/env bash
# BrowseComp-Plus, all 830, the document REPL agent harness with gpt-5 (EFFORT, default high) with ONE change against the
# ColBERT control: retrieval = BM25 (default Lucene k1 0.9 / b 0.4, :8043, a BM25 search server over
# BCP_BM25_INDEX_DIR speaking this harness's wire protocol -- frames_search_server.py --protocol sup --dataset bcp in
# the original repository, not part of this agent), through src/run_bm25.py with BM25_CORPUS=bcp so the retriever
# description is true for BM25 (as in the Qwen3.6 BM25 row BCP-31).
# From original/RLM/scripts/bcp_gpt5_bm25.sh (arguments and environment unchanged; locations from configs/paths.env).
# Agent flags: design B, 30 turns, 10 searches, model-chosen k (fallback 5), seeded first search, compaction at 0.85,
# grounding gate, solo/lean prompt, no delegation, no orchestrator, 4 concurrent sub-calls, root and sub gpt-5.
# Row BCP-25. Only the retry429 hook is loaded, so EFFORT is NOT sent to Azure: gpt-5 ran at the deployment's default
# effort (medium on the original deployment). See CHANGES_FROM_ORIGINAL.md.
# Model: Azure deployment $DEP (default AZURE_DEPLOYMENT_GPT5), verified to serve gpt-5-2025-08-07.
# No --save-trajectory: with whole BM25 pages a trajectory is 0.3-2.3 GB per question (smoke 2026-10-26).
# Credentials come from AZ_ENV (a file with AZ_EP / AZ_KEY), never from this repo.
#   W=16 [IDS_FILE=ids.txt] [OUT=<dir>] experiments/bcp_gpt5_bm25.sh
set -u
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$REPO"                                   # original: cd <original repo>
PY=$PYTHON                                   # original: the vllm conda env's python
SUP=$FULLDOC_ROOT                            # original: the document REPL agent working directory
W=${W:-16}
OUT=$(realpath -m "${OUT:-$RUNS_DIR/bcp_gpt5_bm25/runs}"); mkdir -p "$OUT"   # original default: logs/bcp_gpt5_bm25/runs
SERVER=${SERVER:-http://127.0.0.1:8043}
DEP=${DEP:-${AZURE_DEPLOYMENT_GPT5:?set AZURE_DEPLOYMENT_GPT5}}
say() { printf '%s  [bcp-gpt5-bm25] %s\n' "$(date '+%F %T')" "$*"; }
set -a; . "${AZ_ENV:?set AZ_ENV}"; set +a
export AZURE_OPENAI_ENDPOINT="$AZ_EP/" AZURE_OPENAI_API_KEY="$AZ_KEY" AZURE_OPENAI_API_VERSION=2024-12-01-preview \
       AZURE_OPENAI_DEPLOYMENT="$DEP"
curl -s -m10 $SERVER/health | grep -q '"ok"' || { say "ABORT: $SERVER down"; exit 1; }
need_hook retry429
export PYTHONUNBUFFERED=1 PYTHONPATH=$HOOKS_DIR/retry429:$RLMLIB_PATH HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} BM25_CORPUS=bcp
unset QUERIES_FILE SUP_FENCE_TOLERANT SUP_MAX_BLOCKS SUP_PRESENCE_PENALTY
ENTRY=$SRC/run_bm25.py                       # original: scripts/frames_sup_run_one.py
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
