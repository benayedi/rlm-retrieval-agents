#!/usr/bin/env bash
# Wait for the GPT-5 BM25 snippet run (scripts/run5d/bcp_snip.sh gpt5 ... bcp_snip_gpt5_bm25) to print its "done:" line,
# then grade it with gpt-4.1 and write the sheet numbers to logs/run5d/full/bcp_snip_gpt5_bm25.grade.txt (2026-10-28).
cd /home/extstu_00/RLM
until grep -q " done:" logs/run5d/full/bcp_snip_gpt5_bm25.out; do sleep 120; done
AZ_ENV=/tmp/claude-1003/-home-extstu-00-RLM/2a1318b5-524a-4d81-9aba-3918a191383c/scratchpad/azure.env \
  /home/extstu_00/miniconda3/envs/vllm/bin/python scripts/run5d/bcp_snip_grade.py logs/run5d/full/bcp_snip_gpt5_bm25 \
  2>&1 | grep -v "HF_TOKEN\|unauthenticated" > logs/run5d/full/bcp_snip_gpt5_bm25.grade.txt
echo "$(date '+%F %T') graded" >> logs/run5d/full/bcp_snip_gpt5_bm25.grade.txt
