#!/usr/bin/env bash
# FRC-1, part 2 of 2: the document REPL agent with search cap 25 / 50 turns on the remaining 624 FRAMES questions
# (the 200 test questions are part 1, experiments/frames_qwen_cap25_test200.sh), split over both GPUs into two
# output directories, then the all-824 report (src/frames_cap25_report.py, Qwen3.6-judged graded.json).
# From original/RLM_2026-11-11/RLM/scripts/frames_s25_full.sh (arguments and environment unchanged; locations from
# configs/paths.env; id lists = byte-identical copies in configs/frames_ids/).
# Outputs: $RUNS_DIR/frames/sup_q36_bm25f_s25_rest{0,1} (original: logs/frames/...), logs as in the original.
# The reported FRC-1 accuracy is the gpt-4.1 re-judge (graded_gpt41.json, agents/shared/rlm_shared/grading/grade_frames.py).
#   bash agents/document_repl/experiments/frames_qwen_cap25_rest.sh
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$REPO"                                   # original: cd <original repository>
export PYTHONPATH=$RLMLIB_PATH OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
say() { printf '%s  [s25-full] %s\n' "$(date '+%F %T')" "$*"; }
LOGD=$RUNS_DIR/frames; mkdir -p "$LOGD"        # original: logs/frames
run() {  # gpu-port ids-file out-dir
  BM25_PROMPT=1 MAX_SEARCHES=25 MAX_ITERS=50 BASE_URL=http://localhost:$1/v1 W=8 IDS_FILE=$2 OUT=$3 bash "$ALG/experiments/frames_qwen.sh"
}
say "start: 312 q on GPU 0, 312 q on GPU 1"
run 8030 $ALG/configs/frames_ids/s25_rest_gpu0.txt $RUNS_DIR/frames/sup_q36_bm25f_s25_rest0 > $LOGD/s25_rest0.log 2>&1 &
sleep 120   # stagger the second runner's grading at the end
run 8031 $ALG/configs/frames_ids/s25_rest_gpu1.txt $RUNS_DIR/frames/sup_q36_bm25f_s25_rest1 > $LOGD/s25_rest1.log 2>&1 &
wait
FRAMES_RUNS_DIR=$RUNS_DIR/frames $PYTHON "$SRC/frames_cap25_report.py"   # original: the vllm env's python scripts/frames_s25_full_report.py
say "done"
