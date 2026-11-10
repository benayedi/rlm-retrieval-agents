#!/usr/bin/env bash
# The SUPERVISOR harness (suprlm/src/run_one.py) on FRAMES, over the BM25 index of the FRAMES paper's
# June 2023 Wikipedia (scripts/frames_search_server.py --protocol sup on :8041).
#
# Agent config = BASE_ARGS of scripts/gepa_bcp.py verbatim (the 79.4% BrowseComp+ baseline: design B,
# 30 iterations, 10 searches, compaction, grounding gate, solo prompt, ...), plus the Qwen3.6 switches
# the running GEPA runs use (SUP_FENCE_TOLERANT=1, SUP_MAX_BLOCKS=5, SUP_PRESENCE_PENALTY=1.5).
# Only the queries (QUERIES_FILE) and the retrieval server differ. The harness never sees the gold;
# scripts/frames_grade_sup.py grades afterwards. Resumable: finished questions are skipped.
#   W=8 START=0 END=100 OUT=logs/frames/sup_q36 scripts/run_frames_sup.sh
set -u
cd /home/extstu_00/RLM
PY=/home/extstu_00/miniconda3/envs/vllm/bin/python
SUP=/home/extstu_00/suprlm
W=${W:-8}; START=${START:-0}; END=${END:-100}
MODEL=${MODEL:-qwen36-35b-a3b}
BASE_URL=${BASE_URL:-http://localhost:8030/v1}   # :8031 = second server, GPU 1
OUT=$(realpath -m "${OUT:-logs/frames/sup_q36}")   # absolute: run_one.py runs with cwd=suprlm
mkdir -p "$OUT"
say() { printf '%s  [frames-sup] %s\n' "$(date '+%F %T')" "$*"; }

# SERVER: the retrieval endpoint (default the BM25 server; e.g. :8042 = scripts/bcp_rerank_proxy.py in front of it)
SERVER=${SERVER:-http://127.0.0.1:8041}
curl -s -m10 $SERVER/health | grep -q '"ok"' || { say "ABORT: $SERVER retrieval server down"; exit 1; }
curl -s -m10 $BASE_URL/models | grep -q "$MODEL" || { say "ABORT: $BASE_URL does not serve $MODEL"; exit 1; }

export PYTHONUNBUFFERED=1 PYTHONPATH=/home/extstu_00/rlmrepo HF_HOME=/home/extstu_00/.cache/huggingface
export ALIBABA_TOKEN=EMPTY QUERIES_FILE=/home/extstu_00/RLM/data/frames/frames_queries.jsonl
case "$MODEL" in qwen36*) export SUP_FENCE_TOLERANT=1 SUP_MAX_BLOCKS=5 SUP_PRESENCE_PENALTY=1.5 ;;
                 *) unset SUP_FENCE_TOLERANT SUP_MAX_BLOCKS SUP_PRESENCE_PENALTY ;; esac
# BM25_PROMPT=1: run_one.py through scripts/frames_sup_run_one.py, which makes the retriever
# description true for BM25 (his text describes BrowseComp+'s dense index); otherwise verbatim.
if [ "${BM25_PROMPT:-0}" = 1 ]; then ENTRY=/home/extstu_00/RLM/scripts/frames_sup_run_one.py; else ENTRY=$SUP/src/run_one.py; fi
# MAX_SEARCHES / MAX_ITERS: the supervisor baseline's 10 / 30 unless set (budget ablation, 2026-10-18)
MAX_SEARCHES=${MAX_SEARCHES:-10}; MAX_ITERS=${MAX_ITERS:-30}
export OUT MODEL PY SUP ENTRY BASE_URL SERVER MAX_SEARCHES MAX_ITERS

say "questions $START..$((END-1)), $W workers, model $MODEL @ $BASE_URL, entry $ENTRY, server $SERVER, searches $MAX_SEARCHES, iters $MAX_ITERS${IDS_FILE:+, ids $IDS_FILE} -> $OUT"
# REVERSE=1 walks the range backwards, so a second runner on the other GPU can share a range from
# the far end; the only overlap is the question in flight where the two meet.
# IDS_FILE: run exactly these question ids (one per line) instead of START..END
if [ -n "${IDS_FILE:-}" ]; then SEQ="cat $IDS_FILE"
elif [ "${REVERSE:-0}" = 1 ]; then SEQ="seq $((END-1)) -1 $START"; else SEQ="seq $START $((END-1))"; fi
$SEQ | xargs -P "$W" -I{} sh -c '
  f=$OUT/$(printf "%03d" {}).json
  [ -s "$f" ] && exit 0
  cd $SUP && timeout 6300 $PY $ENTRY --query-id {} --out "$f" \
    --base-url $BASE_URL \
    --design B --server $SERVER --backend openai --api-key-env ALIBABA_TOKEN \
    --root-model $MODEL --sub-model $MODEL --reasoning-effort medium --sub-reasoning-effort medium \
    --max-iterations $MAX_ITERS --max-searches $MAX_SEARCHES --seed-context --compaction --grounding-gate \
    --lean-prologue --no-delegation --solo-prompt --no-orchestrator --compaction-threshold-pct 0.85 \
    --max-concurrent-subcalls 4 --max-timeout 5400 --lm-timeout 1800 --save-trajectory \
    > "${f%.json}.log" 2>&1
'
say "rollouts done: $(ls "$OUT"/[0-9]*.json 2>/dev/null | wc -l)"
$PY scripts/frames_grade_sup.py "$OUT"
