#!/usr/bin/env bash
# BrowseComp-Plus, the document REPL agent harness verbatim (src/run_one.py, no prompt patch) with gpt-5.
# From original/RLM/scripts/bcp_gpt5_sup.sh (arguments and environment unchanged; locations from configs/paths.env).
# Only SERVER varies: :8766 = ColBERT-8192 direct, :8772 = a rerank proxy in front of it (not part of this agent).
# Same flags as experiments/bcp_gpt5_bm25.sh (which differs only by its BM25 server and BM25 prompt wrapper).
# Model: Azure deployment $DEP (default AZURE_DEPLOYMENT_GPT5) serving gpt-5-2025-08-07. Credentials from AZ_ENV
# (a file with AZ_EP / AZ_KEY). Startup hook: retry429 only, so reasoning_effort is NOT sent to Azure (see
# CHANGES_FROM_ORIGINAL.md); the effort actually used is the deployment default.
#   SERVER=http://127.0.0.1:8766 IDS_FILE=ids.txt OUT=<dir> W=8 [EFFORT=high] experiments/bcp_gpt5_colbert.sh
set -u
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$REPO"                                   # original: cd <original repo>
PY=$PYTHON                                   # original: the vllm conda env's python
SUP=$FULLDOC_ROOT                            # original: the document REPL agent working directory
W=${W:-8}
OUT=$(realpath -m "${OUT:?set OUT}"); mkdir -p "$OUT"
SERVER=${SERVER:?set SERVER}
DEP=${DEP:-${AZURE_DEPLOYMENT_GPT5:?set AZURE_DEPLOYMENT_GPT5}}
IDS_FILE=${IDS_FILE:?set IDS_FILE}
say() { printf '%s  [bcp-gpt5-sup] %s\n' "$(date '+%F %T')" "$*"; }
set -a; . "${AZ_ENV:?set AZ_ENV}"; set +a
export AZURE_OPENAI_ENDPOINT="$AZ_EP/" AZURE_OPENAI_API_KEY="$AZ_KEY" AZURE_OPENAI_API_VERSION=2024-12-01-preview \
       AZURE_OPENAI_DEPLOYMENT="$DEP"
need_hook retry429
export PYTHONUNBUFFERED=1 PYTHONPATH=$HOOKS_DIR/retry429:$RLMLIB_PATH HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
unset QUERIES_FILE SUP_FENCE_TOLERANT SUP_MAX_BLOCKS SUP_PRESENCE_PENALTY BM25_CORPUS
EFFORT=${EFFORT:-high}
export EFFORT OUT PY SUP SERVER DEP SRC
say "$(wc -l < $IDS_FILE) questions, $W workers, deployment $DEP, effort $EFFORT, server $SERVER -> $OUT"
cat "$IDS_FILE" | xargs -P "$W" -I{} sh -c '
  f=$OUT/{}.json
  [ -s "$f" ] && exit 0
  cd $SUP && timeout 3300 $PY $SRC/run_one.py --query-id {} --out "$f" \
    --design B --server $SERVER \
    --root-model gpt-5 --sub-model gpt-5 --root-deployment $DEP --sub-deployment $DEP \
    --reasoning-effort $EFFORT --sub-reasoning-effort $EFFORT --default-k 5 \
    --max-iterations 30 --max-searches 10 --seed-context --compaction --grounding-gate \
    --lean-prologue --no-delegation --solo-prompt --no-orchestrator --compaction-threshold-pct 0.85 \
    --max-concurrent-subcalls 4 --max-timeout 2700 \
    > "${f%.json}.log" 2>&1
'
say "rollouts done: $(ls "$OUT"/*.json 2>/dev/null | wc -l)"
