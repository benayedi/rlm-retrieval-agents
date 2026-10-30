#!/usr/bin/env bash
# FRAMES (824 q) snippet RLM (examples/frames_eval_rlm.py), 2026-10-27, with FRAMES's existing snippet setup (the
# Qwen3.6 test200 run, 0.655): BM25 :8040 (rlm wire), the BM25 retriever line, 25-search cap, 50 turns, compaction,
# LightOn tool docs, 5 x 512-token snippets, root/llm_query efforts as in the BrowseComp+ snippet rows.
#   scripts/run5d/frames_snip.sh <gpt5|qwen> <out_dir> [workers] [first_q] [last_q_exclusive]
set -u
MODEL=$1; OUT=$(realpath -m $2); W=${3:-8}; Q0=${4:-0}; Q1=${5:-824}
R=/home/extstu_00/RLM; PY=/home/extstu_00/miniconda3/envs/vllm/bin/python
AZ_ENV=${AZ_ENV:-/tmp/claude-1003/-home-extstu-00-RLM/2a1318b5-524a-4d81-9aba-3918a191383c/scratchpad/azure.env}
set -a; . "$AZ_ENV"; set +a
mkdir -p "$OUT/traj"; cd $R
curl -s -m10 http://localhost:8040/health | grep -q '"status":"ok"' || { echo "ABORT: :8040 BM25 down"; exit 1; }
export PYTHONUNBUFFERED=1 PYTHONPATH=$R/scripts/hooks/run5d:/home/extstu_00/rlmrepo HF_HOME=/home/extstu_00/.cache/huggingface
export EMBEDDING_SERVER_URL=http://localhost:8040 FRAMES_SEARCH_CAP=25 RETRIEVER_LINE=0 FRAMES_BM25_LINE=1
export ORCHESTRATOR=0 LIGHTON_TOOL_DOCS=1 LIGHTON_TASK_TEXT=1 SEARCH_SNIPPET_TOKENS=512 OFFICIAL_TOP_K=5 RLM_TRAJ_DIR=$OUT/traj RLM_USAGE_LOG=$OUT/usage.jsonl
export AZURE_OPENAI_API_KEY="$AZ_KEY" AZURE_OPENAI_ENDPOINT="$AZ_EP/" AZURE_OPENAI_API_VERSION=2025-04-01-preview
export LOCAL_JUDGE_BASE_URL=http://localhost:8030/v1 LOCAL_JUDGE_MODEL=qwen36-35b-a3b   # interim judge only; graded later with gpt-4.1
unset MAX_TOP_K SEARCH_SNIPPET_LEN SUB_LM_MODEL AZURE_OPENAI_DEPLOYMENT
if [ "$MODEL" = gpt5 ]; then
  export AZURE_OPENAI_MODEL=ismail-gpt-5 AZURE_OPENAI_REASONING_EFFORT=high SUB_LM_REASONING_EFFORT=medium
  unset LOCAL_LLM_BASE_URL LOCAL_LLM_MODEL FRAMES_REPL_CUE FRAMES_FENCE_TOLERANT
elif [ "$MODEL" = qwen ]; then
  export LOCAL_LLM_MODEL=qwen36-35b-a3b LOCAL_LM_TIMEOUT=1800 FRAMES_REPL_CUE=1 FRAMES_FENCE_TOLERANT=1
  export AZURE_OPENAI_REASONING_EFFORT=medium SUB_LM_REASONING_EFFORT=medium
else echo "model must be gpt5 or qwen"; exit 2; fi
export OUT PY MODEL
echo "$(date '+%F %T') FRAMES snippet model=$MODEL q=[$Q0,$Q1) workers=$W -> $OUT"
seq $Q0 $((Q1-1)) | xargs -P "$W" -I{} sh -c '
  q=$(printf "%03d" {}); f=$OUT/q$q
  [ -s "$f.json" ] && exit 0
  if [ "$MODEL" = qwen ]; then export LOCAL_LLM_BASE_URL=http://localhost:$((8030 + {} % 2))/v1; fi
  timeout 7200 $PY examples/frames_eval_rlm.py --start {} --end $(({}+1)) --max-iterations 50 --compaction --out "$f" > "$f.log" 2>&1
'
echo "$(date '+%F %T') done: $(ls $OUT/q*.json 2>/dev/null | wc -l) results"
