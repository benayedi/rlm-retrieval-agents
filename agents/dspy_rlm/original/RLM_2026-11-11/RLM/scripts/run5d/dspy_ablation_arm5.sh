#!/bin/bash
# DSPy RLM ablation, arm 5 (2026-10-31, NOT YET RUN): dspy.RLM with everything we can match to the snippet RLM (BCP-9):
# gpt-5 HIGH root and gpt-5 MEDIUM sub-LM; search returns top-5 by default with snippets cut at 512 Qwen3 tokens (as
# OFFICIAL_TOP_K=5 + SEARCH_SNIPPET_TOKENS=512; the model may still ask for another top_k, as in the snippet harness);
# 100 turns, no search or sub-LLM cap, 60K output, task prompt v21 + MINIMAL_TOOL_DOCS, get_document not printed.
# Dense port 8001, 1 h per question, same 100 questions (logs/dspy_ablation/sample100.json). Per-call timeout 900 s
# (sub 600 s) and root and sub-LM max_tokens 32000 (arm 3 cut 60 sub-LM answers at 8000).
# What still differs from the snippet RLM is the library itself: DSPy's reasoning/code turn format and instructions,
# the Deno/Pyodide sandbox, SUBMIT vs answer["ready"], and the extract fallback.
#   bash scripts/run5d/dspy_ablation_arm5.sh [indices]   (default: all 100, 5 shards of 20)
set -u
R=/home/extstu_00/RLM; cd $R
AZ_ENV=${AZ_ENV:-/tmp/claude-1003/-home-extstu-00-RLM/2a1318b5-524a-4d81-9aba-3918a191383c/scratchpad/azure.env}
set -a; . "$AZ_ENV"; set +a
export PATH=$HOME/.deno/bin:$PATH PYTHONUNBUFFERED=1
export AZURE_OPENAI_API_KEY="$AZ_KEY" AZURE_OPENAI_ENDPOINT="$AZ_EP/" AZURE_OPENAI_API_VERSION=2025-04-01-preview
export AZURE_OPENAI_MODEL=ismail-gpt-5 AZURE_OPENAI_REASONING_EFFORT=high SUB_LM_MODEL=ismail-gpt-5 SUB_LM_REASONING_EFFORT=medium AZURE_OPENAI_TIMEOUT=900 SUB_LM_TIMEOUT=600 SUB_LM_MAX_TOKENS=32000 AZURE_OPENAI_MAX_TOKENS=32000
export EMBEDDING_SERVER_URL=http://localhost:8001 RETRIEVER_TIMEOUT=300
export MINIMAL_TOOL_DOCS=1 GETDOC_INLINE_CHARS=0 RLM_MAX_OUTPUT_CHARS=60000 DSPY_MAX_LLM_CALLS=100000 DSPY_QUIET_GETDOC=1 SEARCH_SNIPPET_TOKENS=512
unset SEARCH_SNIPPET_LEN MAX_TOP_K MAX_SEARCH_PER_TURN DISABLE_GET_DOCUMENT AZURE_OPENAI_DEPLOYMENT
PY=/home/extstu_00/miniconda3/envs/vllm/bin/python
OUT=logs/dspy_ablation/arm5; mkdir -p $OUT
run() { $PY examples/browsecomp_eval_dspy_rlm.py --indices "$1" --sig v21 --top-k 5 --max-iters 100 --max-search 1000000 \
          --timeout 3600 --out $OUT/$2 > $OUT/$2.log 2>&1; }
if [ $# -ge 1 ]; then run "$1" "${2:-test}"; exit; fi
IDX=$($PY -c "import json; print(' '.join(str(x['q_idx']) for x in json.load(open('logs/dspy_ablation/sample100.json'))['questions']))")
set -- $IDX; i=0
while [ $# -gt 0 ]; do
  chunk=$(echo "${@:1:20}" | tr ' ' ','); shift $(( $# < 20 ? $# : 20 ))
  run "$chunk" "s$i" & i=$((i+1))
done
wait; echo "arm5 done $(date -u)"
