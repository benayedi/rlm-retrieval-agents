#!/usr/bin/env bash
set -uo pipefail
cd /home/extstu_00/RLM
PYV=/home/extstu_00/miniconda3/envs/vllm/bin/python
L=logs/night5; mkdir -p $L
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L/chain.log; }
export PYTHONPATH=/home/extstu_00/rlmrepo HF_HOME=/home/extstu_00/.cache/huggingface
export GEPA_SHARD_WALL_CAP=5400
say "GEPA wide-surface, resumed (cache intact), per-shard wall cap 5400s"
$PYV -u scripts/gepa_wide.py --workers 16 --valset 100 --minibatch 3 --max-metric-calls 800 2>&1 | tee $L/gepa_wide.log
say "GEPA done"
