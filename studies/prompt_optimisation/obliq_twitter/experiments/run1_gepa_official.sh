#!/usr/bin/env bash
# GEPA-4 run 1 (2026-09-28, logs/gepa_official): GEPA over the snippet REPL agent on OBLIQ-Twitter, STRATEGY block only,
# Qwen3.8-27B (:8030) as agent and critic; declared VOID by run 2 (selection on a 12-question valset is noise).
# RECONSTRUCTED: no launcher survives. The driver's defaults reproduce what progress.txt shows (val 12, minibatch 6,
# budget 200); --workers is not recorded (default 6). Needs: the Qwen3.8-27B server on :8030, the Qwen3-Embedding-8B
# search server on :8010 (index obliq-des-twitter), OBLIQ_BENCH_ROOT, the `gepa` package.
#   bash studies/prompt_optimisation/obliq_twitter/experiments/run1_gepa_official.sh [--workers N]
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../../experiments/_common.sh"
cd "$RUNS_DIR"
export PYTHONPATH=$RLMLIB_PATH HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
$PYV -u $OBLIQ_SRC/gepa_official_obliq.py "$@"
