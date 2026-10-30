#!/usr/bin/env bash
# OBLIQ with GPT-5.2 (Azure ismail-gpt-5.2) on Gemini-Embedding-2, raw text (the OBLIQ paper's setup), 2026-10-27.
# One launcher for the paper table's two harnesses (DSPy is out for now):
#   snip  our snippet RLM (scripts/run_obliq_mine.py via run_obliq_shard.sh): root effort high, llm_query GPT-5.2
#         medium, top-k 20, 12 turns, no search cap; prompts without fixed score cutoffs (OBLIQ_RELATIVE_SCORES=1);
#   fd    the supervisor's full-document RLM (suprlm run_experiment.py, design B, no sub-LLM): effort high, max 10
#         searches, 30 turns (as on BrowseComp+ / FRAMES; with 12, all 5 smoke questions hit the turn cap before
#         submitting), fixed k 20, turn-cap fallback asks for a plain ranking (scripts/hooks/cap_ranking); prompts data/obliq_fd_gemini/prompts_<task>.json (10 searches, no cutoffs).
# Search: Gemini servers scripts/gemini/search_server.py (--query-format raw) :8020 math / :8021 twitter / :8022 congress;
# Math goes through scripts/obliq_math_search.py :8014 (drops each query's own problem, both wires); Twitter and
# Congress full-doc through suprlm/src/obliq_adapter.py :8768 / :8769 (supervisor wire).
#   scripts/run5d/obliq_gemini.sh <snip|fd> <math|twitter|congress> <ids.json> <out_dir> [workers]
set -u
ARM=$1; TASK=$2; IDS=$(realpath $3); OUT=$(realpath -m $4); W=${5:-8}
R=/home/extstu_00/RLM; S=/home/extstu_00/suprlm; PYV=/home/extstu_00/miniconda3/envs/vllm/bin/python
AZ_ENV=${AZ_ENV:-/tmp/claude-1003/-home-extstu-00-RLM/2a1318b5-524a-4d81-9aba-3918a191383c/scratchpad/azure.env}
set -a; . "$AZ_ENV"; set +a
DEP=ismail-gpt-5.2
case $TASK in
  math)     OT=analogues;     OS=math;     IDX=obliq-ana-math;     SNIP_URL=http://localhost:8014; FD_URL=http://127.0.0.1:8014
            QF=$R/data/obliq_math/queries.jsonl
            SNIP_FILES="OBLIQ_TASK_FILE=$R/data/obliq_math/task_snippet.txt OBLIQ_STRATEGY_FILE=$R/data/obliq_math/strategy_snippet.txt" ;;
  twitter)  OT=descriptive;   OS=twitter;  IDX=obliq-des-twitter;  SNIP_URL=http://localhost:8021; FD_URL=http://127.0.0.1:8768
            QF=$S/data/obliq_twitter/queries.jsonl; SNIP_FILES="" ;;
  congress) OT=tip-of-tongue; OS=congress; IDX=obliq-tot-congress; SNIP_URL=http://localhost:8022; FD_URL=http://127.0.0.1:8769
            QF=$R/data/obliq_congress/queries.jsonl
            SNIP_FILES="OBLIQ_TASK_FILE=$R/data/obliq_congress/task_snippet.txt OBLIQ_STRATEGY_FILE=$R/data/obliq_congress/strategy_snippet.txt" ;;
  *) echo "unknown task $TASK"; exit 2 ;;
esac
mkdir -p "$OUT"
echo "$(date '+%F %T') arm=$ARM task=$TASK ids=$(basename $IDS) workers=$W out=$OUT"
if [ "$ARM" = snip ]; then
  cd $R
  env -u AZURE_OPENAI_DEPLOYMENT -u OBLIQ_FENCE_TOLERANT -u OBLIQ_PRESENCE_PENALTY $SNIP_FILES \
    AZURE_OPENAI_API_KEY="$AZ_KEY" AZURE_OPENAI_ENDPOINT="$AZ_EP/" AZURE_OPENAI_API_VERSION=2025-04-01-preview \
    AZURE_OPENAI_MODEL=$DEP LOCAL_LLM_BASE_URL="" EXTRA_PYTHONPATH=$R/scripts/hooks/run5d \
    ROOT_EFFORT=high SUB_EFFORT=medium OBLIQ_RELATIVE_SCORES=1 RLM_USAGE_LOG=$OUT/usage.jsonl \
    OBLIQ_TASK=$OT OBLIQ_SUBSET=$OS OBLIQ_INDEX=$IDX EMBEDDING_SERVER_URL=$SNIP_URL \
    OBLIQ_REPL_CUE=1 OBLIQ_TAG_SEARCH=1 RLM_TRAJ_DIR=$OUT/traj \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
    scripts/run_obliq_shard.sh "$IDS" "$OUT/all" 20 12 $W
elif [ "$ARM" = fd ]; then
  [ "$(curl -s -m5 -o /dev/null -w "%{http_code}" ${FD_URL}/)" != "000" ] || { echo "ABORT: $FD_URL down"; exit 1; }
  cd $S
  NAME=$(basename "$OUT")
  env -u SUP_FENCE_TOLERANT -u SUP_MAX_BLOCKS -u SUP_PRESENCE_PENALTY \
    AZURE_OPENAI_ENDPOINT="$AZ_EP/" AZURE_OPENAI_API_KEY="$AZ_KEY" AZURE_OPENAI_API_VERSION=2024-12-01-preview \
    AZURE_OPENAI_DEPLOYMENT=$DEP PYTHONUNBUFFERED=1 PYTHONPATH=$R/scripts/hooks/run5d:/home/extstu_00/rlmrepo OBLIQ_CAP_RANKING=1 RLM_USAGE_LOG=$OUT/usage.jsonl \
    HF_HOME=/home/extstu_00/.cache/huggingface QUERIES_FILE=$QF GEPA_PROMPTS=$R/data/obliq_fd_gemini/prompts_$TASK.json \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
    $PYV -u src/run_experiment.py --name "$NAME" --ids-file "$IDS" --workers $W --stagger 0.3 \
      --design B --server $FD_URL --backend azure_openai \
      --root-model gpt-5.2 --sub-model gpt-5.2 --root-deployment $DEP --sub-deployment $DEP \
      --reasoning-effort high --sub-reasoning-effort high \
      --default-k 20 --fixed-k --max-searches 10 --max-iterations 30 --seed-context --compaction \
      --lean-prologue --no-delegation --solo-prompt --no-orchestrator --compaction-threshold-pct 0.85 \
      --max-concurrent-subcalls 4 --max-timeout 3600 --lm-timeout 1800 --save-trajectory
  echo "suprlm runs dir: $S/runs/$NAME"
else
  echo "unknown arm $ARM"; exit 2
fi
echo "$(date '+%F %T') done"
