#!/usr/bin/env bash
# OBLIQ with GPT-5.2 on Gemini-Embedding-2, raw text (the OBLIQ paper's setup), 2026-10-27: the document REPL agent half
# ("fd") of the original repository's scripts/run5d/obliq_gemini.sh. The "snip" half belongs to the snippet REPL agent.
#   fd    the document REPL agent (src/run_experiment.py, design B, no sub-LLM): effort high, max 10 searches, 30 turns (as on
#         BrowseComp+ / FRAMES; with 12, all 5 smoke questions hit the turn cap before submitting), fixed k 20,
#         turn-cap fallback asks for a plain ranking (startup hook cap_ranking, via run5d); prompts
#         $RLM_DATA_DIR/obliq_fd_gemini/prompts_<task>.json (10 searches, no cutoffs), loaded through GEPA_PROMPTS.
# Search: Gemini-Embedding-2 servers (--query-format raw) :8020 math / :8021 twitter / :8022 congress (not part of
# this agent). Math goes through a filtering proxy on :8014 (drops each query's own problem); Twitter and Congress
# go through src/obliq_adapter.py on :8768 / :8769 (this harness's wire protocol).
# Rows MAT-11 / TWI-11 / CON-11. Results: $RUNS_DIR/<basename of out_dir> (original: <FULLDOC_ROOT>/runs/<name>).
# Arguments unchanged from the original; only the "fd" arm is implemented here.
#   experiments/obliq_fd.sh fd <math|twitter|congress> <ids.json> <out_dir> [workers]
set -u
ARM=$1; TASK=$2; IDS=$(realpath $3); OUT=$(realpath -m $4); W=${5:-8}
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
PYV=$PYTHON; S=$FULLDOC_ROOT                 # original: the vllm conda env's python; the document REPL agent working directory
R=${RLM_DATA_DIR:?set RLM_DATA_DIR (data/ of the original repository: obliq_math, obliq_congress, obliq_fd_gemini)}
set -a; . "${AZ_ENV:?set AZ_ENV}"; set +a    # original: AZ_ENV defaulted to a file in a session scratch directory
DEP=${AZURE_DEPLOYMENT_GPT52:?set AZURE_DEPLOYMENT_GPT52}
case $TASK in
  math)     OT=analogues;     OS=math;     IDX=obliq-ana-math;     SNIP_URL=http://localhost:8014; FD_URL=http://127.0.0.1:8014
            QF=$R/obliq_math/queries.jsonl ;;
  twitter)  OT=descriptive;   OS=twitter;  IDX=obliq-des-twitter;  SNIP_URL=http://localhost:8021; FD_URL=http://127.0.0.1:8768
            QF=$S/data/obliq_twitter/queries.jsonl ;;
  congress) OT=tip-of-tongue; OS=congress; IDX=obliq-tot-congress; SNIP_URL=http://localhost:8022; FD_URL=http://127.0.0.1:8769
            QF=$R/obliq_congress/queries.jsonl ;;
  *) echo "unknown task $TASK"; exit 2 ;;
esac
mkdir -p "$OUT"
echo "$(date '+%F %T') arm=$ARM task=$TASK ids=$(basename $IDS) workers=$W out=$OUT"
if [ "$ARM" = snip ]; then
  echo "the snip arm is the snippet REPL agent's launcher (agents/snippet_repl/experiments)"; exit 2
elif [ "$ARM" = fd ]; then
  [ "$(curl -s -m5 -o /dev/null -w "%{http_code}" ${FD_URL}/)" != "000" ] || { echo "ABORT: $FD_URL down"; exit 1; }
  need_hook run5d
  cd $S
  NAME=$(basename "$OUT")
  env -u SUP_FENCE_TOLERANT -u SUP_MAX_BLOCKS -u SUP_PRESENCE_PENALTY \
    AZURE_OPENAI_ENDPOINT="$AZ_EP/" AZURE_OPENAI_API_KEY="$AZ_KEY" AZURE_OPENAI_API_VERSION=2024-12-01-preview \
    AZURE_OPENAI_DEPLOYMENT=$DEP PYTHONUNBUFFERED=1 PYTHONPATH=$HOOKS_DIR/run5d:$RLMLIB_PATH OBLIQ_CAP_RANKING=1 RLM_USAGE_LOG=$OUT/usage.jsonl \
    HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} QUERIES_FILE=$QF GEPA_PROMPTS=$R/obliq_fd_gemini/prompts_$TASK.json \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
    $PYV -u $SRC/run_experiment.py --name "$NAME" --ids-file "$IDS" --workers $W --stagger 0.3 \
      --design B --server $FD_URL --backend azure_openai \
      --root-model gpt-5.2 --sub-model gpt-5.2 --root-deployment $DEP --sub-deployment $DEP \
      --reasoning-effort high --sub-reasoning-effort high \
      --default-k 20 --fixed-k --max-searches 10 --max-iterations 30 --seed-context --compaction \
      --lean-prologue --no-delegation --solo-prompt --no-orchestrator --compaction-threshold-pct 0.85 \
      --max-concurrent-subcalls 4 --max-timeout 3600 --lm-timeout 1800 --save-trajectory
  echo "runs dir: $RUNS_DIR/$NAME"
else
  echo "unknown arm $ARM"; exit 2
fi
echo "$(date '+%F %T') done"
