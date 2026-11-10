#!/usr/bin/env bash
# PUBLICATION COPY of scripts/run5d/bcp_snip_gpt5_bm25_finish.sh. OPERATIONAL ONLY (grading, no agent setting): waits
# for the BCP-13 run (experiments/bcp_snip.sh gpt5 ... bcp_snip_gpt5_bm25) to print "done:", then grades it with gpt-4.1.
# Changes: paths under $RUNS_DIR/run5d (were <original repo>/logs/run5d); the grader bcp_snip_grade.py (not part of
# this agent) is looked up in $GRADERS_DIR; credentials from $AZ_ENV; interpreter $PYTHON.
#
# Original header:
# Wait for the GPT-5 BM25 snippet run (scripts/run5d/bcp_snip.sh gpt5 ... bcp_snip_gpt5_bm25) to print its "done:" line,
# then grade it with gpt-4.1 and write the sheet numbers to logs/run5d/full/bcp_snip_gpt5_bm25.grade.txt (2026-10-28).
. "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${RUNS_DIR:?}"
G=${GRADERS_DIR:?directory holding bcp_snip_grade.py}
until grep -q " done:" run5d/full/bcp_snip_gpt5_bm25.out; do sleep 120; done
AZ_ENV=${AZ_ENV:?} \
  "$PYTHON" $G/bcp_snip_grade.py run5d/full/bcp_snip_gpt5_bm25 \
  2>&1 | grep -v "HF_TOKEN\|unauthenticated" > run5d/full/bcp_snip_gpt5_bm25.grade.txt
echo "$(date '+%F %T') graded" >> run5d/full/bcp_snip_gpt5_bm25.grade.txt
