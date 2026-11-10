# Sourced (not executed) by every snippet REPL agent launcher in this directory and its subdirectories. Not in the original
# repository: it holds only the layout wiring that rule 5 of docs/architecture/REFACTOR_RULES.md asks every launcher for.
#   ALG  agents/snippet_repl         SRC  agents/snippet_repl/src         ART  repository root
# and loads the machine-specific settings from <repo>/configs/paths.env (copy of configs/paths.example.env).
# Snippet REPL agent PYTHONPATH (snip_pythonpath below), in order:
#   src/hooks/run5d   the startup hooks (retry429 + azure_sampling + cap_ranking + balance_local); MUST come first,
#                     Python imports only the first sitecustomize.py on sys.path
#   $RLMLIB_PATH      github.com/alexzhang13/rlm @ 854e688
#   shared            rlm_shared (BrowseComp+/OBLIQ data and grading)
#   dspy_rlm/src      dspy_rlm.agent._get_snippet_tokenizer (the 512-token snippet truncation)
ART=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
ALG=$ART/agents/snippet_repl
SRC=$ALG/src
[ -f "$ART/configs/paths.env" ] || { echo "ABORT: $ART/configs/paths.env missing (copy configs/paths.example.env and fill it in)"; exit 2; }
set -a; . "$ART/configs/paths.env"; set +a
: "${RLMLIB_PATH:?set RLMLIB_PATH in configs/paths.env (rlm checkout at 854e688)}"
: "${PYTHON:?set PYTHON in configs/paths.env}"
snip_pythonpath() {  # $1 = hook directory name under src/hooks, or "" for none
  local p="${RLMLIB_PATH}:$ART/agents/shared:$ART/agents/dspy_rlm/src"
  [ -n "$1" ] && p="$SRC/hooks/$1:$p"
  echo "$p"
}
