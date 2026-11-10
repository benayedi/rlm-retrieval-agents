#!/usr/bin/env bash
# PUBLICATION COPY of scripts/run5d/night_finisher.sh. OPERATIONAL ONLY: it changes no agent setting and is not needed to
# reproduce any row. It waited for runs to end, graded them with gpt-4.1, and raised the parallelism of the running Qwen
# BrowseComp+ snippet queue (GNU xargs -P: SIGUSR1 = one more job). Kept for provenance: on 2026-10-29 it raised the
# BCP-19 (bm25) xargs to 14 then 20 workers and BCP-15 (dense) to 20 workers (logs/run5d/night_finisher.log), so those
# rows ran with more than the 8 workers named in their launch line. Worker count changes wall-clock time only.
# Changes: process patterns name the new launcher/runner files ("^bash .*experiments/qwen_bcp_snip_queue.sh",
# "^bash .*experiments/frames_snip.sh qwen", run_browsecomp.py; the ^bash anchor keeps grep from matching itself); log paths are under $RUNS_DIR/run5d (were <original repo>/logs/run5d);
# the grading scripts (bcp_fd_grade.py, frames_judge_gpt41.py; not part of this agent) are looked up in
# $GRADERS_DIR, and the document REPL process pattern in $FD_DENSE_PATTERN (the document REPL runner belongs to
# agents/document_repl; the original pattern was "^<python> -u scripts/run5d/bcp_fd_qwen_dense.py").
#
# Original header:
# 2026-10-28 evening. Unattended follow-up for the Qwen runs (reads and grades; changes no agent settings):
#  1. when the document REPL dense run (pid of bcp_fd_qwen_dense.py) ends: grade it with gpt-4.1 -> logs/run5d/bcp_fd_qwen_dense.grade.txt
#  2. when the FRAMES Qwen snippet run (frames_snip.sh) ends: grade it -> logs/run5d/full/frames_snip_qwen.grade.txt
#  3. raise the Qwen BrowseComp+ snippet queue's xargs parallelism (GNU xargs: SIGUSR1 = +1 job) from 8 to 14 once one
#     of the two runs has ended and to 20 once both have, for every row the queue starts (bm25 -> dense -> colbert).
# Exits when the queue (qwen_bcp_snip_queue2.sh) is gone.
. "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${RUNS_DIR:?}"
AZ=${AZ_ENV:?}
PY=$PYTHON
G=${GRADERS_DIR:?directory holding bcp_fd_grade.py and frames_judge_gpt41.py}
FDP=${FD_DENSE_PATTERN:?regex matching the document REPL dense run command line}
alive(){ ps -eo args | grep -qE "$1"; }
declare -A BUMPED
fd_done=0; fr_done=0
while alive "^bash .*experiments/qwen_bcp_snip_queue.sh"; do
  if [ $fd_done = 0 ] && ! alive "$FDP"; then
    fd_done=1; echo "$(date '+%F %T') document REPL dense ended; grading"
    AZ_ENV=$AZ $PY $G/bcp_fd_grade.py run5d/bcp_fd_qwen_dense e925f0f6c7 2>&1 | grep -v "HF_TOKEN\|unauthenticated" > run5d/bcp_fd_qwen_dense.grade.txt
    echo "$(date '+%F %T') graded" >> run5d/bcp_fd_qwen_dense.grade.txt
  fi
  if [ $fr_done = 0 ] && ! alive "^bash .*experiments/frames_snip.sh qwen"; then
    fr_done=1; echo "$(date '+%F %T') FRAMES Qwen snippet ended; grading"
    AZ_ENV=$AZ $PY $G/frames_judge_gpt41.py run5d/full/frames_snip_qwen 2>&1 | grep -v "HF_TOKEN\|unauthenticated" > run5d/full/frames_snip_qwen.grade.txt
    echo "$(date '+%F %T') graded" >> run5d/full/frames_snip_qwen.grade.txt
  fi
  target=$(( 6 * (fd_done + fr_done) ))
  for x in $(ps -eo pid,args | awk '/^ *[0-9]+ xargs -P 8 / && /run_browsecomp.py/ && /QTIMEOUT/ {print $1}'); do
    b0=${BUMPED[$x]:-0}
    while [ "${BUMPED[$x]:-0}" -lt $target ]; do
      kill -USR1 $x && BUMPED[$x]=$(( ${BUMPED[$x]:-0} + 1 ))
    done
    [ "${BUMPED[$x]:-0}" != "$b0" ] && echo "$(date '+%F %T') xargs $x raised to $((8 + ${BUMPED[$x]})) workers"
  done
  sleep 120
done
echo "$(date '+%F %T') queue finished; finisher exits"
