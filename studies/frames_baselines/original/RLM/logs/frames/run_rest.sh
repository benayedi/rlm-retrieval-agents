#!/bin/bash
# go/no-go first (oracle on the first 100 questions, the same 100 as closed-book's first 100), then the rest
cd /home/extstu_00/RLM
P=/home/extstu_00/miniconda3/envs/vllm/bin/python
$P -u scripts/frames_eval.py --conds oracle --limit 100 --workers 12 >> logs/frames/eval.log 2>&1
$P -u scripts/frames_eval.py --conds oracle,bm25_4,bm25_2 --workers 16 >> logs/frames/eval.log 2>&1
