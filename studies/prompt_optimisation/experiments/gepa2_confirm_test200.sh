#!/usr/bin/env bash
# GEPA-2 held-out confirmation: winner d911ebbe23 vs the original prompt on test200, both arms interleaved, 20 workers
# in total (2026-10-09 18:46 -> 10-10 04:57).
# RECONSTRUCTED: no wrapper script existed. The command is the usage line of original/RLM/scripts/bcp_confirm.py
#   GEPA_OUT=logs/gepa_bcp2 python scripts/bcp_confirm.py --winner logs/gepa_bcp2/winner.json --split test200
# with the run-2 environment (cache key: model + Qwen3.6 switches), as in logs/gepa_bcp2/confirm_test200.log
# ("test200 (200 q) | original ee5ea32a1c vs winner d911ebbe23 | interleaved, 20 workers total"; --workers 20 is the
# script's default). Add --report to score what is cached without rollouts.
#   bash studies/prompt_optimisation/experiments/gepa2_confirm_test200.sh [--report]
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$RUNS_DIR"
export GEPA_BCP_MODEL=qwen36-35b-a3b SUP_FENCE_TOLERANT=1 SUP_MAX_BLOCKS=5 SUP_PRESENCE_PENALTY=1.5
export GEPA_BCP_SERVERS=http://localhost:8030/v1 HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
L=$RUNS_DIR/gepa_bcp2
env -u GEPA_PROMPTS GEPA_OUT=$L $PYV -u $SRC/bcp_confirm.py --winner $L/winner.json --split test200 "$@" \
  2>&1 | tee -a $L/confirm_test200.log
