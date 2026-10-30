#!/usr/bin/env bash
# Run src/run_obliq.py (DSPy.RLM on OBLIQ-Bench) over a list of query ids, split into W parallel shards.
#
# Publication copy of scripts/run_obliq_dspy_shard.sh. Arguments, per-shard environment and driver flags are the
# original's; only machine-specific values come from configs/paths.env (PYTHON, HF_HOME, OBLIQ_BENCH_ROOT), the
# driver path points at src/, and PYTHONPATH is set to the new layout. DSPy.RLM has no row in the final results
# table (docs/Experiments.xlsx); this is the launcher of the earlier campaign's OBLIQ runs.
#
# Usage:   algorithms/dspy_rlm/experiments/obliq_shard.sh IDS_JSON OUT_PREFIX [W=6]
#   IDS_JSON    JSON list of OBLIQ query _ids to run (e.g. a dev split)
#   OUT_PREFIX  shard i writes OUT_PREFIX_s<i>.{json,txt,log}; relative paths are relative to the repository root
# Env:     OBLIQ_TASK (descriptive) / OBLIQ_SUBSET (twitter) / OBLIQ_INDEX (obliq-des-twitter),
#          EMBEDDING_SERVER_URL (http://localhost:8010), LOCAL_LLM_BASE_URL (http://localhost:8030/v1),
#          LOCAL_LLM_MODEL (qwen3.8-27b-bf16); Azure credentials (AZURE_OPENAI_*) for llm_query, see ../configs/example.env.
#
# Original header:
# Shard the DSPY harness over the dev split. This is the arm that isolates the MODEL:
# same harness, same ObliqTwitterSig, same index as the recorded gpt-5.2 run (0.1284) --
# only the LLM changes. Our other two OBLIQ arms changed the harness too, so they cannot
# separate "different model" from "different scaffold".
#
# Sequential by design, like the official harness, so parallelism comes from W processes.
set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ALGO_DIR="$REPO_ROOT/algorithms/dspy_rlm"
if [ ! -f "$REPO_ROOT/configs/paths.env" ]; then
  echo "missing $REPO_ROOT/configs/paths.env: copy configs/paths.example.env and fill it in" >&2; exit 1
fi
set -a; . "$REPO_ROOT/configs/paths.env"; set +a
export PYTHONPATH="$REPO_ROOT/shared:$ALGO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
: "${OBLIQ_BENCH_ROOT:?set OBLIQ_BENCH_ROOT in configs/paths.env (OBLIQ-Bench data checkout)}"
: "${HF_HOME:?set HF_HOME in configs/paths.env}"   # original: the user's ~/.cache/huggingface
IDS=$1; OUT=$2; W=${3:-6}
PYV=${PYTHON:?set PYTHON in configs/paths.env}   # original: the vllm conda env's python
# OBLIQ_TASK / OBLIQ_SUBSET / OBLIQ_INDEX / EMBEDDING_SERVER_URL (2026-10-22, Math arm A): unset = Twitter as before.
TASK=${OBLIQ_TASK:-descriptive}; SUBSET=${OBLIQ_SUBSET:-twitter}; INDEX=${OBLIQ_INDEX:-obliq-des-twitter}
SH=/tmp/obliq_dspy_shards/$(basename $OUT)   # per run: two concurrent runs must not share shard files
cd "$REPO_ROOT"   # original: cd to the RLM repository root
mkdir -p $SH "$(dirname "$OUT")"
# map query_ids -> dataset indices, then split
$PYV - "$IDS" "$W" "$TASK" "$SUBSET" "$SH" <<'PY'
import json,sys
from rlm_shared import obliq_data as OD   # original: sys.path.insert(<RLM root>); from examples import obliq_data
want=set(json.load(open(sys.argv[1]))); W=int(sys.argv[2])
rows=OD.load_queries(sys.argv[3],sys.argv[4])
idx=[i for i,r in enumerate(rows) if r['_id'] in want]
for w in range(W):
    open(f'{sys.argv[5]}/{w}.txt','w').write(','.join(str(i) for i in idx[w::W]))
print('mapped %d ids -> indices, %d shards'%(len(idx),W))
PY
# Deno runs dspy.RLM's sandboxed REPL; $DENO_INSTALL/bin is where deno's installer puts it (default ~/.deno).
for w in $(seq 0 $((W-1))); do
(
  PATH=${DENO_INSTALL:-$HOME/.deno}/bin:$PATH HF_HOME=$HF_HOME \
  EMBEDDING_SERVER_URL=${EMBEDDING_SERVER_URL:-http://localhost:8010} OBLIQ_INDEX=$INDEX \
  LOCAL_LLM_BASE_URL=${LOCAL_LLM_BASE_URL:-http://localhost:8030/v1} LOCAL_LLM_MODEL=${LOCAL_LLM_MODEL:-qwen3.8-27b-bf16} \
  LOCAL_LLM_EFFORT=medium LOCAL_LM_TIMEOUT=1800 PYTHONUNBUFFERED=1 \
  $PYV -u "$ALGO_DIR/src/run_obliq.py" --task $TASK --subset $SUBSET \
    --indices "$(cat $SH/${w}.txt)" \
    --max-iters 12 --max-search 8 --top-k 20 --timeout 0 \
    --out ${OUT}_s${w} > ${OUT}_s${w}.log 2>&1
) &
done
wait
echo "ALL SHARDS DONE"
