#!/usr/bin/env bash
# FRC-1, part 1 of 2: the document REPL agent on the 200 FRAMES test questions with search cap 25 and 50 turns
# (FR-13 ran cap 10 / 30 turns); everything else identical to FR-13. GPU 0, shuffled question order.
# From original/RLM_2026-11-11/RLM/scripts/frames_s25_chain.sh (arguments and environment unchanged; locations from
# configs/paths.env; the id list is the byte-identical copy configs/frames_ids/test_ids_shuffled.txt).
# Output: $RUNS_DIR/frames/sup_q36_bm25f_s25 (original: logs/frames/sup_q36_bm25f_s25).
# Part 2 (the other 624 questions): experiments/frames_qwen_cap25_rest.sh.
#   bash agents/document_repl/experiments/frames_qwen_cap25_test200.sh
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$REPO"                                   # original: cd <original repository>
export PYTHONPATH=$RLMLIB_PATH OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
say() { printf '%s  [s25] %s\n' "$(date '+%F %T')" "$*"; }
BM25_PROMPT=1 MAX_SEARCHES=25 MAX_ITERS=50 BASE_URL=http://localhost:8030/v1 W=8 \
  IDS_FILE=$ALG/configs/frames_ids/test_ids_shuffled.txt OUT=$RUNS_DIR/frames/sup_q36_bm25f_s25 \
  bash "$ALG/experiments/frames_qwen.sh"    # original: scripts/run_frames_sup.sh
# original: $PY scripts/frames_arch_compare.py -- the test200 architecture comparison of the original repository,
# which includes the FRAMES multi-step pipeline re-implementation (not part of this repository). Run only if given.
if [ -n "${FRAMES_ARCH_COMPARE:-}" ]; then $PYTHON "$FRAMES_ARCH_COMPARE"; fi
say "done"
