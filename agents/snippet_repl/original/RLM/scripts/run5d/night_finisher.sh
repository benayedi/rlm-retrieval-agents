#!/usr/bin/env bash
# 2026-10-28 evening. Unattended follow-up for the Qwen runs (reads and grades; changes no agent settings):
#  1. when the full-doc dense run (pid of bcp_fd_qwen_dense.py) ends: grade it with gpt-4.1 -> logs/run5d/bcp_fd_qwen_dense.grade.txt
#  2. when the FRAMES Qwen snippet run (frames_snip.sh) ends: grade it -> logs/run5d/full/frames_snip_qwen.grade.txt
#  3. raise the Qwen BrowseComp+ snippet queue's xargs parallelism (GNU xargs: SIGUSR1 = +1 job) from 8 to 14 once one
#     of the two runs has ended and to 20 once both have, for every row the queue starts (bm25 -> dense -> colbert).
# Exits when the queue (qwen_bcp_snip_queue2.sh) is gone.
cd /home/extstu_00/RLM
AZ=/tmp/claude-1003/-home-extstu-00-RLM/2a1318b5-524a-4d81-9aba-3918a191383c/scratchpad/azure.env
PY=/home/extstu_00/miniconda3/envs/vllm/bin/python
alive(){ ps -eo args | grep -qE "$1"; }
declare -A BUMPED
fd_done=0; fr_done=0
while alive "^bash scripts/run5d/qwen_bcp_snip_queue2.sh"; do
  if [ $fd_done = 0 ] && ! alive "^/home/extstu_00/miniconda3/envs/vllm/bin/python -u scripts/run5d/bcp_fd_qwen_dense.py"; then
    fd_done=1; echo "$(date '+%F %T') full-doc dense ended; grading"
    AZ_ENV=$AZ $PY scripts/run5d/bcp_fd_grade.py logs/run5d/bcp_fd_qwen_dense e925f0f6c7 2>&1 | grep -v "HF_TOKEN\|unauthenticated" > logs/run5d/bcp_fd_qwen_dense.grade.txt
    echo "$(date '+%F %T') graded" >> logs/run5d/bcp_fd_qwen_dense.grade.txt
  fi
  if [ $fr_done = 0 ] && ! alive "^bash scripts/run5d/frames_snip.sh qwen"; then
    fr_done=1; echo "$(date '+%F %T') FRAMES Qwen snippet ended; grading"
    AZ_ENV=$AZ $PY scripts/frames_judge_gpt41.py logs/run5d/full/frames_snip_qwen 2>&1 | grep -v "HF_TOKEN\|unauthenticated" > logs/run5d/full/frames_snip_qwen.grade.txt
    echo "$(date '+%F %T') graded" >> logs/run5d/full/frames_snip_qwen.grade.txt
  fi
  target=$(( 6 * (fd_done + fr_done) ))
  for x in $(ps -eo pid,args | awk '/^ *[0-9]+ xargs -P 8 / && /run_bcp_snippet.py/ && /QTIMEOUT/ {print $1}'); do
    b0=${BUMPED[$x]:-0}
    while [ "${BUMPED[$x]:-0}" -lt $target ]; do
      kill -USR1 $x && BUMPED[$x]=$(( ${BUMPED[$x]:-0} + 1 ))
    done
    [ "${BUMPED[$x]:-0}" != "$b0" ] && echo "$(date '+%F %T') xargs $x raised to $((8 + ${BUMPED[$x]})) workers"
  done
  sleep 120
done
echo "$(date '+%F %T') queue finished; finisher exits"
