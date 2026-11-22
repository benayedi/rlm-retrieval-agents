#!/bin/bash
# DSPy RLM ablation, arm 3 (2026-10-31): dspy.RLM with the snippet RLM's budgets, output limit and prompt, on the 100
# stratified BrowseComp+ questions in logs/dspy_ablation/sample100.json (dense Qwen3-Embed-8B, port 8001).
# Unchanged from the original DSPy dense run: gpt-5 medium root, gpt-5.4-mini sub-LM, top-10, 500-char snippets.
# Changed: 100 turns (was 16), no search cap (was 35), no sub-LLM cap (was 50), 60K output (was 10K, middle cut),
# prompt v21 + MINIMAL_TOOL_DOCS (was v2 + our REPL tools/rules/orchestration blocks), get_document returns into a
# variable without printing (GETDOC_INLINE_CHARS=0, as the snippet RLM). 1 h per question.
#   bash scripts/run5d/dspy_ablation_arm3.sh [indices]   (default: all 100, 5 shards of 20)
set -u
R=/home/extstu_00/RLM; cd $R
AZ_ENV=${AZ_ENV:-/tmp/claude-1003/-home-extstu-00-RLM/2a1318b5-524a-4d81-9aba-3918a191383c/scratchpad/azure.env}
set -a; . "$AZ_ENV"; set +a
export PATH=$HOME/.deno/bin:$PATH PYTHONUNBUFFERED=1
export AZURE_OPENAI_API_KEY="$AZ_KEY" AZURE_OPENAI_ENDPOINT="$AZ_EP/" AZURE_OPENAI_API_VERSION=2025-04-01-preview
export AZURE_OPENAI_MODEL=ismail-gpt-5 AZURE_OPENAI_REASONING_EFFORT=medium SUB_LM_MODEL=gpt-5.4-mini
export EMBEDDING_SERVER_URL=http://localhost:8001 RETRIEVER_TIMEOUT=300
export MINIMAL_TOOL_DOCS=1 GETDOC_INLINE_CHARS=0 RLM_MAX_OUTPUT_CHARS=60000 DSPY_MAX_LLM_CALLS=100000 DSPY_QUIET_GETDOC=1
unset SEARCH_SNIPPET_TOKENS SEARCH_SNIPPET_LEN MAX_TOP_K MAX_SEARCH_PER_TURN DISABLE_GET_DOCUMENT AZURE_OPENAI_DEPLOYMENT
PY=/home/extstu_00/miniconda3/envs/vllm/bin/python
OUT=logs/dspy_ablation/arm3; mkdir -p $OUT
run() { $PY examples/browsecomp_eval_dspy_rlm.py --indices "$1" --sig v21 --top-k 10 --max-iters 100 --max-search 1000000 \
          --timeout 3600 --out $OUT/$2 > $OUT/$2.log 2>&1; }
if [ $# -ge 1 ]; then run "$1" "${2:-test}"; exit; fi
IDX=$($PY -c "import json; print(' '.join(str(x['q_idx']) for x in json.load(open('logs/dspy_ablation/sample100.json'))['questions']))")
set -- $IDX; i=0
while [ $# -gt 0 ]; do
  chunk=$(echo "${@:1:20}" | tr ' ' ','); shift $(( $# < 20 ? $# : 20 ))
  run "$chunk" "s$i" & i=$((i+1))
done
wait; echo "arm3 done $(date -u)"
