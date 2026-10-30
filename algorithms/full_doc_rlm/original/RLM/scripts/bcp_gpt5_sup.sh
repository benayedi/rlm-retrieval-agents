#!/usr/bin/env bash
# BrowseComp-Plus, the supervisor's full-document RLM verbatim (suprlm/src/run_one.py, no prompt patch) with gpt-5
# at medium: the config of the 0.8627 ColBERT-8192 control (Experiments row 12, docs/supervisor_artifact.html).
# Only SERVER varies: :8766 = ColBERT-8192 direct, :8772 = scripts/bcp_rerank_proxy.py (novelty + Qwen3-Reranker-4B).
# Same flags as scripts/bcp_gpt5_bm25.sh (which differs only by its BM25 server and BM25 prompt wrapper).
# Model: Azure deployment ismail-gpt-5 = gpt-5-2025-08-07. Credentials from AZ_ENV (AZ_EP / AZ_KEY), not this repo.
#   AZ_ENV=/path/azure.env SERVER=http://127.0.0.1:8772 IDS_FILE=ids.txt OUT=logs/x W=8 scripts/bcp_gpt5_sup.sh
set -u
cd /home/extstu_00/RLM
PY=/home/extstu_00/miniconda3/envs/vllm/bin/python
SUP=/home/extstu_00/suprlm
W=${W:-8}
OUT=$(realpath -m "${OUT:?set OUT}"); mkdir -p "$OUT"
SERVER=${SERVER:?set SERVER}
DEP=${DEP:-ismail-gpt-5}
IDS_FILE=${IDS_FILE:?set IDS_FILE}
say() { printf '%s  [bcp-gpt5-sup] %s\n' "$(date '+%F %T')" "$*"; }
set -a; . "${AZ_ENV:?set AZ_ENV}"; set +a
export AZURE_OPENAI_ENDPOINT="$AZ_EP/" AZURE_OPENAI_API_KEY="$AZ_KEY" AZURE_OPENAI_API_VERSION=2024-12-01-preview \
       AZURE_OPENAI_DEPLOYMENT="$DEP"
export PYTHONUNBUFFERED=1 PYTHONPATH=/home/extstu_00/RLM/scripts/hooks/retry429:/home/extstu_00/rlmrepo HF_HOME=/home/extstu_00/.cache/huggingface
unset QUERIES_FILE SUP_FENCE_TOLERANT SUP_MAX_BLOCKS SUP_PRESENCE_PENALTY BM25_CORPUS
EFFORT=${EFFORT:-high}
export EFFORT OUT PY SUP SERVER DEP
say "$(wc -l < $IDS_FILE) questions, $W workers, deployment $DEP, effort $EFFORT, server $SERVER -> $OUT"
cat "$IDS_FILE" | xargs -P "$W" -I{} sh -c '
  f=$OUT/{}.json
  [ -s "$f" ] && exit 0
  cd $SUP && timeout 3300 $PY src/run_one.py --query-id {} --out "$f" \
    --design B --server $SERVER \
    --root-model gpt-5 --sub-model gpt-5 --root-deployment $DEP --sub-deployment $DEP \
    --reasoning-effort $EFFORT --sub-reasoning-effort $EFFORT --default-k 5 \
    --max-iterations 30 --max-searches 10 --seed-context --compaction --grounding-gate \
    --lean-prologue --no-delegation --solo-prompt --no-orchestrator --compaction-threshold-pct 0.85 \
    --max-concurrent-subcalls 4 --max-timeout 2700 \
    > "${f%.json}.log" 2>&1
'
say "rollouts done: $(ls "$OUT"/*.json 2>/dev/null | wc -l)"
