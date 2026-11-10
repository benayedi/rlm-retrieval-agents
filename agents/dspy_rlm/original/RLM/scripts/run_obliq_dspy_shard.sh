#!/usr/bin/env bash
# Shard the DSPY harness over the dev split. This is the arm that isolates the MODEL:
# same harness, same ObliqTwitterSig, same index as the recorded gpt-5.2 run (0.1284) --
# only the LLM changes. Our other two OBLIQ arms changed the harness too, so they cannot
# separate "different model" from "different scaffold".
#
# Sequential by design, like the official harness, so parallelism comes from W processes.
set -u
IDS=$1; OUT=$2; W=${3:-6}
PYV=/home/extstu_00/miniconda3/envs/vllm/bin/python
# OBLIQ_TASK / OBLIQ_SUBSET / OBLIQ_INDEX / EMBEDDING_SERVER_URL (2026-10-22, Math arm A): unset = Twitter as before.
TASK=${OBLIQ_TASK:-descriptive}; SUBSET=${OBLIQ_SUBSET:-twitter}; INDEX=${OBLIQ_INDEX:-obliq-des-twitter}
SH=/tmp/obliq_dspy_shards/$(basename $OUT)   # per run: two concurrent runs must not share shard files
cd /home/extstu_00/RLM
mkdir -p $SH "$(dirname "$OUT")"
# map query_ids -> dataset indices, then split
$PYV - "$IDS" "$W" "$TASK" "$SUBSET" "$SH" <<'PY'
import json,sys
sys.path.insert(0,'/home/extstu_00/RLM')
from examples import obliq_data as OD
want=set(json.load(open(sys.argv[1]))); W=int(sys.argv[2])
rows=OD.load_queries(sys.argv[3],sys.argv[4])
idx=[i for i,r in enumerate(rows) if r['_id'] in want]
for w in range(W):
    open(f'{sys.argv[5]}/{w}.txt','w').write(','.join(str(i) for i in idx[w::W]))
print('mapped %d ids -> indices, %d shards'%(len(idx),W))
PY
for w in $(seq 0 $((W-1))); do
(
  PATH=/home/extstu_00/.deno/bin:$PATH HF_HOME=/home/extstu_00/.cache/huggingface \
  EMBEDDING_SERVER_URL=${EMBEDDING_SERVER_URL:-http://localhost:8010} OBLIQ_INDEX=$INDEX \
  LOCAL_LLM_BASE_URL=${LOCAL_LLM_BASE_URL:-http://localhost:8030/v1} LOCAL_LLM_MODEL=${LOCAL_LLM_MODEL:-qwen3.8-27b-bf16} \
  LOCAL_LLM_EFFORT=medium LOCAL_LM_TIMEOUT=1800 PYTHONUNBUFFERED=1 \
  $PYV -u examples/obliq_eval_dspy_rlm.py --task $TASK --subset $SUBSET \
    --indices "$(cat $SH/${w}.txt)" \
    --max-iters 12 --max-search 8 --top-k 20 --timeout 0 \
    --out ${OUT}_s${w} > ${OUT}_s${w}.log 2>&1
) &
done
wait
echo "ALL SHARDS DONE"
