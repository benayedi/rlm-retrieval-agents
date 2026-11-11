#!/usr/bin/env bash
# Budget ablation: supervisor RLM agent on the 200 FRAMES test q with 25 searches / 50 turns (baseline 10 / 30),
# everything else identical to logs/frames/sup_q36_bm25f. GPU 0. Shuffled order.
cd /home/extstu_00/RLM
export PYTHONPATH=/home/extstu_00/rlmrepo OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
say() { printf '%s  [s25] %s\n' "$(date '+%F %T')" "$*"; }
BM25_PROMPT=1 MAX_SEARCHES=25 MAX_ITERS=50 BASE_URL=http://localhost:8030/v1 W=8 \
  IDS_FILE=data/frames/gepa_splits/test_ids_shuffled.txt OUT=logs/frames/sup_q36_bm25f_s25 scripts/run_frames_sup.sh
/home/extstu_00/miniconda3/envs/vllm/bin/python scripts/frames_arch_compare.py
say "done"
