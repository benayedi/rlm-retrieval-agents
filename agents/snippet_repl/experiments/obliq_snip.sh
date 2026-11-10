#!/usr/bin/env bash
# PUBLICATION COPY of the `snip` arm of scripts/run5d/obliq_gemini.sh (rows MAT-9 / TWI-9 / CON-9). The `fd` arm of that
# launcher (the document REPL agent, rows MAT-11 / TWI-11 / CON-11) belongs to agents/document_repl and is
# not reproduced here. Arguments and environment settings of the snip arm are the original's; paths, the Azure deployment
# name and the credential file come from configs/paths.env, and the OBLIQ_TASK_FILE / OBLIQ_STRATEGY_FILE prompt files
# are byte-identical copies under ../configs/obliq/<task>/ (were <original repo>/data/obliq_<task>/). The original's
# first positional argument <snip|fd> is kept, and only `snip` is accepted. See ../CHANGES_FROM_ORIGINAL.md (issues:
# the BrowseComp+ format gate stays ON here, and OBLIQ_INDEX is read by nothing).
#
# Original header (snip part):
# OBLIQ with GPT-5.2 (Azure gpt-5.2 deployment) on Gemini-Embedding-2, raw text (the OBLIQ paper's setup), 2026-10-27.
#   snip  our snippet RLM (src/run_obliq.py via obliq_shard.sh): root effort high, llm_query GPT-5.2
#         medium, top-k 20, 12 turns, no search cap; prompts without fixed score cutoffs (OBLIQ_RELATIVE_SCORES=1);
# Search: Gemini servers scripts/gemini/search_server.py (--query-format raw) :8020 math / :8021 twitter / :8022 congress;
# Math goes through scripts/obliq_math_search.py :8014 (drops each query's own problem, both wires).
#   experiments/obliq_snip.sh snip <math|twitter|congress> <ids.json> <out_dir> [workers]
set -u
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
ARM=$1; TASK=$2; IDS=$(realpath $3); OUT=$(realpath -m $4); W=${5:-8}
set -a; . "${AZ_ENV:?set AZ_ENV in configs/paths.env (file with AZ_EP and AZ_KEY)}"; set +a
DEP=${AZURE_DEPLOYMENT_GPT52:?}
P=$ALG/configs/obliq
case $TASK in
  math)     OT=analogues;     OS=math;     IDX=obliq-ana-math;     SNIP_URL=http://localhost:8014
            SNIP_FILES="OBLIQ_TASK_FILE=$P/math/task_snippet.txt OBLIQ_STRATEGY_FILE=$P/math/strategy_snippet.txt" ;;
  twitter)  OT=descriptive;   OS=twitter;  IDX=obliq-des-twitter;  SNIP_URL=http://localhost:8021
            SNIP_FILES="" ;;
  congress) OT=tip-of-tongue; OS=congress; IDX=obliq-tot-congress; SNIP_URL=http://localhost:8022
            SNIP_FILES="OBLIQ_TASK_FILE=$P/congress/task_snippet.txt OBLIQ_STRATEGY_FILE=$P/congress/strategy_snippet.txt" ;;
  *) echo "unknown task $TASK"; exit 2 ;;
esac
mkdir -p "$OUT"
echo "$(date '+%F %T') arm=$ARM task=$TASK ids=$(basename $IDS) workers=$W out=$OUT"
if [ "$ARM" = snip ]; then
  # (original: cd <original repo>; obliq_shard.sh now changes to $RUNS_DIR itself)
  env -u AZURE_OPENAI_DEPLOYMENT -u OBLIQ_FENCE_TOLERANT -u OBLIQ_PRESENCE_PENALTY $SNIP_FILES \
    AZURE_OPENAI_API_KEY="$AZ_KEY" AZURE_OPENAI_ENDPOINT="$AZ_EP/" AZURE_OPENAI_API_VERSION=2025-04-01-preview \
    AZURE_OPENAI_MODEL=$DEP LOCAL_LLM_BASE_URL="" EXTRA_PYTHONPATH=$SRC/hooks/run5d \
    ROOT_EFFORT=high SUB_EFFORT=medium OBLIQ_RELATIVE_SCORES=1 RLM_USAGE_LOG=$OUT/usage.jsonl \
    OBLIQ_TASK=$OT OBLIQ_SUBSET=$OS OBLIQ_INDEX=$IDX EMBEDDING_SERVER_URL=$SNIP_URL \
    OBLIQ_REPL_CUE=1 OBLIQ_TAG_SEARCH=1 RLM_TRAJ_DIR=$OUT/traj \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
    "$ALG/experiments/obliq_shard.sh" "$IDS" "$OUT/all" 20 12 $W
else
  echo "unknown arm $ARM (the fd arm is part of agents/document_repl)"; exit 2
fi
echo "$(date '+%F %T') done"
