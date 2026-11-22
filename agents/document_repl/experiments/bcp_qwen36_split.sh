#!/usr/bin/env bash
# The document REPL agent with Qwen3.6-35B-A3B on one split of BrowseComp-Plus, through src/split_runner.py --seed-split
# (the unchanged seed prompt). Agent side of BCP-29 (RETRIEVAL = ColBERT :8766) and of the search-policy runs SP-1 / SP-2
# (RETRIEVAL = the rerank proxy, started by studies/search_policy/experiments/).
#
# RECONSTRUCTED: the original runs had no launcher file. They were started as one command line each, recorded in the
# Claude Code session transcript 8ba31816 (2026-10-10 11:28 test200 :8768, 18:16 test200 :8769, 2026-10-11 08:47
# test_rest :8766/:8772, 2026-10-12 20:29 pareto/feedback :8772, 2026-10-14 14:47 rest300 :8772) and in
# <original repo>/logs/obliq_rr/overnight.sh (the 2026-10-13 resume of pareto/feedback). All of them were:
#   env -u GEPA_PROMPTS GEPA_BCP_MODEL=qwen36-35b-a3b SUP_FENCE_TOLERANT=1 SUP_MAX_BLOCKS=5 SUP_PRESENCE_PENALTY=1.5 \
#     GEPA_BCP_SERVERS=http://localhost:8030/v1 GEPA_BCP_RETRIEVAL=<retrieval URL> GEPA_OUT=logs/<run> \
#     GEPA_SPLIT=data/bcp_gepa_splits/split.json HF_HOME=<hf cache> \
#     setsid nohup <python> -u scripts/gepa_bcp.py --seed-split <split> --workers <W> > logs/<run>/run.log
# (rest300: GEPA_SPLIT=data/bcp_full830/split_rest300.json; resumes appended with >>). This file sets the same variables
# and arguments; only locations differ (configs/paths.env; GEPA_OUT is resolved by split_runner.py against RUNS_DIR;
# the split files are the copies in studies/prompt_optimisation/splits/). It runs in the foreground (no setsid/nohup).
#   RETRIEVAL=http://127.0.0.1:8766 OUT=<run> SPLIT=test200 W=20 [SPLIT_FILE=<split.json>] experiments/bcp_qwen36_split.sh
# Workers in the original runs: test200 20, test_rest 12, pareto/feedback 10 (then 12 on each resume), rest300 24.
# Needs: vLLM serving qwen36-35b-a3b on :8030 (benchmarks: see GEPA_BROWSECOMP_PLAN §2 settings), the retrieval URL up.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$REPO"                                   # original: cd <original repo>
RETRIEVAL=${RETRIEVAL:-http://127.0.0.1:8766}
OUT=${OUT:?set OUT (run directory name under RUNS_DIR, e.g. bcp_policy_e2e)}
SPLIT=${SPLIT:?set SPLIT (test200 / test_rest / pareto / feedback / rest300)}
W=${W:-20}
case "$SPLIT" in
  rest300) SPLIT_FILE=${SPLIT_FILE:-$REPO/studies/prompt_optimisation/splits/bcp_full830/split_rest300.json} ;;
  *)       SPLIT_FILE=${SPLIT_FILE:-$REPO/studies/prompt_optimisation/splits/bcp_gepa_splits/split.json} ;;
esac
[ -f "$SPLIT_FILE" ] || { echo "ABORT: split file $SPLIT_FILE missing" >&2; exit 2; }
mkdir -p "$RUNS_DIR/$OUT"
env -u GEPA_PROMPTS GEPA_BCP_MODEL=qwen36-35b-a3b SUP_FENCE_TOLERANT=1 SUP_MAX_BLOCKS=5 SUP_PRESENCE_PENALTY=1.5 \
  GEPA_BCP_SERVERS=http://localhost:8030/v1 GEPA_BCP_RETRIEVAL=$RETRIEVAL GEPA_OUT=$OUT \
  GEPA_SPLIT=$SPLIT_FILE HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} \
  "$PYTHON" -u "$SRC/split_runner.py" --seed-split "$SPLIT" --workers "$W" >> "$RUNS_DIR/$OUT/run.log" 2>&1
