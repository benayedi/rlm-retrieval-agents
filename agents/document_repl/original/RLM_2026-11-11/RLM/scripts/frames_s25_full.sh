#!/usr/bin/env bash
# 25-search / 50-turn supervisor RLM agent on the remaining 624 FRAMES questions (the 200 test q are in
# logs/frames/sup_q36_bm25f_s25), split over both GPUs into separate dirs, then the full-824 report.
cd /home/extstu_00/RLM
export PYTHONPATH=/home/extstu_00/rlmrepo OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
say() { printf '%s  [s25-full] %s\n' "$(date '+%F %T')" "$*"; }
run() {  # gpu-port ids-file out-dir
  BM25_PROMPT=1 MAX_SEARCHES=25 MAX_ITERS=50 BASE_URL=http://localhost:$1/v1 W=8 IDS_FILE=$2 OUT=$3 scripts/run_frames_sup.sh
}
say "start: 312 q on GPU 0, 312 q on GPU 1"
run 8030 data/frames/gepa_splits/s25_rest_gpu0.txt logs/frames/sup_q36_bm25f_s25_rest0 > logs/frames/s25_rest0.log 2>&1 &
sleep 120   # stagger the second runner's grading at the end
run 8031 data/frames/gepa_splits/s25_rest_gpu1.txt logs/frames/sup_q36_bm25f_s25_rest1 > logs/frames/s25_rest1.log 2>&1 &
wait
/home/extstu_00/miniconda3/envs/vllm/bin/python scripts/frames_s25_full_report.py
say "done"
