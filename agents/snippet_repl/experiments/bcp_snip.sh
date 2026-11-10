#!/usr/bin/env bash
# PUBLICATION COPY of scripts/run5d/bcp_snip.sh (rows BCP-13 GPT-5 BM25; also the first, superseded Qwen3.6 BM25 attempt
# bcp_snip_qwen_bm25_nocue). Arguments, environment settings, caps and timeouts are the original's; only paths, the
# Azure deployment name and the credential file come from configs/paths.env, and the runner is src/run_browsecomp.py
# (was scripts/run_bcp_snippet.py). See ../CHANGES_FROM_ORIGINAL.md.
#
# Original header:
# BrowseComp+ snippet RLM (the official RLM harness via scripts/run_bcp_snippet.py), 2026-10-27. Same frozen setup as
# the GPT-5 snippet rows (rlmfix24..32: root high, llm_query medium, LightOn tool docs + task text, 5 x 512-token
# snippets, no orchestrator, 100 turns, depth 1); only the model and the retriever change. Trajectories are saved.
#   experiments/bcp_snip.sh <gpt5|qwen> <retriever_url> <out_dir> [workers] [first_q] [last_q_exclusive]
#   retrievers (rlm wire): :8001 Qwen3-Embedding-8B FAISS IVFSQ8 | :8044 BM25 k1 0.9 b 0.4 | :8024 Reason-ModernColBERT
# qwen: Qwen3.6-35B-A3B-FP8 on :8030/:8031 (alternating by question), effort medium, presence penalty 1.5,
#       tolerant code fences, 1800 s per-call timeout (src/run_browsecomp.py), history compaction (without it
#       Qwen3.6 overflowed its 262k context on both smoke questions; GPT runs stay without, as the GPT-5 snippet rows).
# NOTE (refactor): despite the sentence above, COMPACT=--compaction below is passed for BOTH models.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
MODEL=$1; RETR=$2; OUT=$(realpath -m $3); W=${4:-8}; Q0=${5:-0}; Q1=${6:-830}
PYV=${PYTHON:?}
set -a; . "${AZ_ENV:?set AZ_ENV in configs/paths.env (file with AZ_EP and AZ_KEY)}"; set +a
mkdir -p "$OUT/traj"
curl -s -m10 $RETR/health | grep -q ok || { echo "ABORT: retriever $RETR down"; exit 1; }
export PYTHONUNBUFFERED=1 PYTHONPATH=$(snip_pythonpath run5d) HF_HOME=${HF_HOME:?}
export EMBEDDING_SERVER_URL=$RETR ORCHESTRATOR=0 LIGHTON_TOOL_DOCS=1 LIGHTON_TASK_TEXT=1 SEARCH_SNIPPET_TOKENS=512 OFFICIAL_TOP_K=5
export RLM_USAGE_LOG=$OUT/usage.jsonl RLM_TRAJ_DIR=$OUT/traj AZURE_OPENAI_API_KEY="$AZ_KEY" AZURE_OPENAI_ENDPOINT="$AZ_EP/" AZURE_OPENAI_API_VERSION=2025-04-01-preview
unset MAX_TOP_K SEARCH_SNIPPET_LEN SUB_LM_MODEL AZURE_OPENAI_DEPLOYMENT
COMPACT=--compaction   # only fires near the context limit; without it GPT-5 on BM25 overflowed its 272k context on 1 of 2
                       # smoke questions (215 searches) and Qwen3.6 on both. The GPT-5 snippet rows (rlmfix) never reached
                       # the limit, so for them it would not have changed anything.
if [ "$MODEL" = gpt5 ]; then
  export AZURE_OPENAI_MODEL=${AZURE_DEPLOYMENT_GPT5:?} AZURE_OPENAI_REASONING_EFFORT=high SUB_LM_REASONING_EFFORT=medium
  unset LOCAL_LLM_BASE_URL LOCAL_LLM_MODEL QWEN_PRESENCE_PENALTY QWEN_FENCE_TOLERANT BCP_SEARCH_CAP
elif [ "$MODEL" = qwen ]; then
  export LOCAL_LLM_MODEL=qwen36-35b-a3b LOCAL_LLM_API_KEY=EMPTY LOCAL_LM_TIMEOUT=1800
  export AZURE_OPENAI_REASONING_EFFORT=medium SUB_LM_REASONING_EFFORT=medium QWEN_PRESENCE_PENALTY=1.5 QWEN_FENCE_TOLERANT=1
  export BCP_SEARCH_CAP=25   # 2026-10-27: uncapped, Qwen3.6 ran 100-300 searches per question (~1-2 weeks of GPU time
                             # for three runs); 25 as the FRAMES snippet setup. The GPT-5 snippet rows have no cap.
  for p in 8030 8031; do curl -s -m5 localhost:$p/v1/models | grep -q qwen36 || { echo "ABORT: :$p model server down"; exit 1; }; done
else
  echo "model must be gpt5 or qwen"; exit 2
fi
export OUT PYV MODEL COMPACT SRC
echo "$(date '+%F %T') model=$MODEL retriever=$RETR q=[$Q0,$Q1) workers=$W out=$OUT"
seq $Q0 $((Q1-1)) | xargs -P "$W" -I{} sh -c '
  f=$OUT/q{}
  [ -s "$f.json" ] && exit 0
  if [ "$MODEL" = qwen ]; then export LOCAL_LLM_BASE_URL=http://localhost:$((8030 + {} % 2))/v1; fi
  timeout 7200 $PYV -u $SRC/run_browsecomp.py --start {} --end $(({}+1)) \
    --max-iterations 100 --max-depth 1 $COMPACT --out "$f" > "$f.log" 2>&1
'
echo "$(date '+%F %T') done: $(ls $OUT/q*.json 2>/dev/null | wc -l) result files"
