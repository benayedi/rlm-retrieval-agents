# Sourced by the launchers in this directory (not run on its own). Locates the repository, sources configs/paths.env
# (or $PATHS_ENV) and defines:
#   REPO    repository root          STUDY  studies/search_policy          PROXY  src/rerank_proxy.py
#   start_proxy PORT LOG   start the proxy in the background with the caller's environment (the variables that
#                          select its behaviour are set by each launcher, as in the original command lines), then wait
#                          for its READY line. If something already answers on PORT it is left alone (as the original
#                          chains did: "if ! curl ... :8042/stats; then start").
#   need_up URL WHAT       stop with a message when a server the run needs (retriever, model) is not answering.
# The original command lines ran the proxy with the vllm conda env's python; here $PYTHON (configs/paths.env).
_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STUDY=$(cd "$_here/.." && pwd)
REPO=$(cd "$STUDY/../.." && pwd)
PROXY=$STUDY/src/rerank_proxy.py
PATHS_ENV=${PATHS_ENV:-$REPO/configs/paths.env}
[ -f "$PATHS_ENV" ] || { echo "missing $PATHS_ENV (copy configs/paths.example.env and fill it in)" >&2; exit 2; }
set -a; . "$PATHS_ENV"; set +a
: "${RUNS_DIR:?set RUNS_DIR in configs/paths.env}" "${PYTHON:?set PYTHON in configs/paths.env}"
mkdir -p "$RUNS_DIR"
say() { printf '%s  [search-policy] %s\n' "$(date '+%F %T')" "$*"; }
need_up() { curl -s -m10 -o /dev/null "$1" || { say "ABORT: $2 not answering at $1"; exit 1; }; }
start_proxy() {  # PORT LOG
  if curl -s -m5 "127.0.0.1:$1/stats" >/dev/null; then say "proxy :$1 already up: $(curl -s -m5 127.0.0.1:$1/stats)"; return 0; fi
  HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} nohup "$PYTHON" -u "$PROXY" --port "$1" > "$2" 2>&1 < /dev/null &
  for _i in $(seq 1 120); do grep -aqE "READY|Traceback" "$2" 2>/dev/null && break; sleep 5; done
  grep -ao "READY.*" "$2" || { say "ABORT: proxy :$1 not ready (see $2)"; exit 1; }
}
