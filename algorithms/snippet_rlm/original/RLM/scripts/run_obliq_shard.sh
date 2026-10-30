#!/usr/bin/env bash
# Shard run_obliq_mine.py across W processes. The harness is sequential by design -- the
# BrowseComp+ runs got their parallelism the same way, from W worker PROCESSES each handed a
# disjoint slice, not from threads inside one run.
set -u
# LOCAL_LLM_BASE_URL set but EMPTY = use Azure (AZURE_OPENAI_* from the environment); unset = local :8030.
IDS=$1; OUT=$2; TOPK=${3:-20}; ITERS=${4:-12}; W=${5:-6}
PYV=/home/extstu_00/miniconda3/envs/vllm/bin/python
cd /home/extstu_00/RLM
mkdir -p "$(dirname "$OUT")" /tmp/obliq_shards
# Shard lists are keyed on the full OUT path (2026-10-27): keyed on basename, two runs with OUT=.../all launched together
# overwrote each other's lists and one of them ran nothing.
SHK=$(echo "$OUT" | md5sum | cut -c1-12)
$PYV -c "
import json,sys
ids=json.load(open('$IDS')); W=$W
for w in range(W):
    json.dump(ids[w::W], open(f'/tmp/obliq_shards/${SHK}_{w}.json','w'))
print('split', len(ids), 'ids into', W, 'shards')"
for w in $(seq 0 $((W-1))); do
(
  PYTHONPATH=${EXTRA_PYTHONPATH:+$EXTRA_PYTHONPATH:}/home/extstu_00/rlmrepo HF_HOME=/home/extstu_00/.cache/huggingface \
  PYTHONUNBUFFERED=1 EMBEDDING_SERVER_URL=${EMBEDDING_SERVER_URL:-http://localhost:8010} OBLIQ_INDEX=${OBLIQ_INDEX:-obliq-des-twitter} \
  OBLIQ_TASK=${OBLIQ_TASK:-descriptive} OBLIQ_SUBSET=${OBLIQ_SUBSET:-twitter} \
  LOCAL_LLM_BASE_URL=${LOCAL_LLM_BASE_URL-http://localhost:8030/v1} \
  LOCAL_LLM_MODEL=${LOCAL_LLM_MODEL:-qwen3.8-27b-bf16} \
  AZURE_OPENAI_REASONING_EFFORT=${ROOT_EFFORT:-medium} SUB_LM_REASONING_EFFORT=${SUB_EFFORT:-medium} \
  ORCHESTRATOR=0 LIGHTON_TOOL_DOCS=1 LIGHTON_TASK_TEXT=1 \
  SEARCH_SNIPPET_TOKENS=512 OFFICIAL_TOP_K=$TOPK RLM_TRAJ_DIR=${RLM_TRAJ_DIR:-logs/rlm_traj_obliq_sweep} \
  $PYV -u scripts/run_obliq_mine.py --ids-file /tmp/obliq_shards/${SHK}_${w}.json \
    --max-iterations $ITERS --max-depth 1 --out ${OUT}_s${w} > ${OUT}_s${w}.log 2>&1
) &
done
wait
$PYV -c "
import json,glob,statistics as st
rows=[]
for f in glob.glob('${OUT}_s*.json'):
    rows += json.load(open(f))
json.dump(rows, open('${OUT}.json','w'), indent=2)
n=len(rows) or 1
print('MERGED n=%d  NDCG@10=%.4f  recall@10=%.4f'%(len(rows),
  sum(r.get('ndcg@10',0) for r in rows)/n, sum(r.get('recall@10',0) for r in rows)/n))"
