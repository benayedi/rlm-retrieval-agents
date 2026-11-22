#!/usr/bin/env bash
set -uo pipefail
cd /home/extstu_00/RLM
PYV=/home/extstu_00/miniconda3/envs/vllm/bin/python
L=logs/night6; mkdir -p $L
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L/chain.log; }
export PYTHONPATH=/home/extstu_00/rlmrepo HF_HOME=/home/extstu_00/.cache/huggingface
export GEPA_PER_QUESTION_CAP=3600
say "GEPA wide-surface, cap now 3600s PER QUESTION (was 5400s per batch -- 7x too tight)"
$PYV -u scripts/gepa_wide.py --workers 16 --valset 100 --minibatch 3 --max-metric-calls 800 2>&1 | tee $L/gepa_wide.log
say "GEPA done"
