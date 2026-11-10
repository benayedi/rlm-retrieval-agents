#!/usr/bin/env bash
# ILLUSTRATIVE launcher for src/run_browsecomp.py (DSPy.RLM on BrowseComp-Plus).
#
# DSPy.RLM has no row in the final results table (docs/Experiments.xlsx), so this does NOT reproduce a final-table
# experiment and there is no per-experiment config. The original repository had no wrapper for this driver; the
# flags below are the driver's own CLI defaults, written out so a run is explicit. Any extra arguments are appended
# and override them (argparse keeps the last value).
#
# Needs: a retrieval server (POST /search, /document, GET /health) at EMBEDDING_SERVER_URL; Azure credentials
# AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY / AZURE_OPENAI_MODEL (root LM, sub-LM and judge; see ../configs/example.env);
# deno on PATH (dspy.RLM's sandboxed REPL); the Tevatron/browsecomp-plus dataset via HF datasets.
#
# Example (first 50 questions, signature v1, no per-question timeout):
#   EMBEDDING_SERVER_URL=http://localhost:8010 AZURE_OPENAI_MODEL=<deployment> \
#     agents/dspy_rlm/experiments/browsecomp.sh --start 0 --end 50 --out runs/dspy_rlm/bcp_q0_q50
set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ALGO_DIR="$REPO_ROOT/agents/dspy_rlm"
if [ ! -f "$REPO_ROOT/configs/paths.env" ]; then
  echo "missing $REPO_ROOT/configs/paths.env: copy configs/paths.example.env and fill it in" >&2; exit 1
fi
set -a; . "$REPO_ROOT/configs/paths.env"; set +a
export PYTHONPATH="$REPO_ROOT/agents/shared:$ALGO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
: "${EMBEDDING_SERVER_URL:?set EMBEDDING_SERVER_URL (retrieval server URL)}"
PYV=${PYTHON:?set PYTHON in configs/paths.env}
export PATH="${DENO_INSTALL:-$HOME/.deno}/bin:$PATH"
cd "$REPO_ROOT"
# Driver defaults (src/run_browsecomp.py argparse): --start 0 --end 50 --out logs/run --max-iters 16
# --max-search 35 --max-depth 5 --top-k 10 --sig v1 --timeout $BROWSECOMP_TIMEOUT (0 = none).
exec $PYV -u "$ALGO_DIR/src/run_browsecomp.py" \
  --start 0 --end 50 --out logs/run \
  --max-iters 16 --max-search 35 --max-depth 5 --top-k 10 --sig v1 \
  --timeout "${BROWSECOMP_TIMEOUT:-0}" \
  "$@"
