# Sourced by every launcher of the prompt-optimisation study (experiments/, obliq_twitter/experiments/,
# hover/experiments/); not run on its own. Locates the repository, sources configs/paths.env (or $PATHS_ENV) and
# defines the code locations. The original launchers did `cd <original repo>` and used relative scripts/, logs/ and
# data/ paths; here code paths are absolute, run outputs go to RUNS_DIR (the original logs/), and the question-id
# splits are studies/prompt_optimisation/splits/ (the original data/).
_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STUDY=$(cd "$_here/.." && pwd)                       # studies/prompt_optimisation
REPO=$(cd "$STUDY/../.." && pwd)
SRC=$STUDY/src                                       # BrowseComp-Plus / FRAMES helpers
OBLIQ_SRC=$STUDY/obliq_twitter/src                   # GEPA-4 drivers
HOVER_SRC=$STUDY/hover/src                           # GEPA-5
SPLITS=$STUDY/splits
DOC_SRC=$REPO/agents/document_repl/src               # split_runner.py (original scripts/gepa_bcp.py), run_one.py, servers
SNIP_SRC=$REPO/agents/snippet_repl/src               # run_obliq.py (original scripts/run_obliq_mine.py)
PATHS_ENV=${PATHS_ENV:-$REPO/configs/paths.env}
[ -f "$PATHS_ENV" ] || { echo "missing $PATHS_ENV (copy configs/paths.example.env and fill it in)" >&2; exit 2; }
set -a; . "$PATHS_ENV"; set +a
: "${RLMLIB_PATH:?set RLMLIB_PATH in configs/paths.env}" "${RUNS_DIR:?set RUNS_DIR in configs/paths.env}"
: "${PYTHON:?set PYTHON in configs/paths.env}"
RUNS_DIR=$(realpath -m "$RUNS_DIR"); mkdir -p "$RUNS_DIR"
PYV=$PYTHON                                          # original: the vllm conda env's python
# Helpers that the original chains called but that are not part of this repository: each launcher checks the one it
# needs, at the point it needs it, and stops with a message naming the variable.
need_file() { [ -f "${!1:-}" ] || { echo "set $1: $2" >&2; exit 2; }; }
