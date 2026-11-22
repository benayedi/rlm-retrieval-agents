#!/usr/bin/env bash
# GEPA-2: BrowseComp+ GEPA run 2 (plan A+B+D, 2026-10-08), chained after run 1. Same model/harness/cache.
#   A: minibatch gate 8 instead of 3  (run 1: lucky 3-q passes cost ~100 rollouts each)
#   B: D_pareto enriched with the seed's own failures (src/make_bcp_enriched_split.py)
#   D: critic rule against cutting search effort (in split_runner.py CRITIC_SYSTEM)
# From original/RLM/scripts/run_gepa_bcp2.sh (arguments and environment unchanged; locations from configs/paths.env).
# The frozen copy that ran (original/RLM/logs/run_gepa_bcp2_frozen.sh) differs only in the un-anchored wait pattern
# 'scripts/gepa_bcp[.]py'. The run itself was taken over by experiments/gepa2_bcp_run2_continue.sh (original
# logs/gepa_bcp2/continue_after_fix.sh) after the relative-GEPA_OUT bug of 2026-10-08. Run 2 ran on gepa_bcp.py before
# its 2026-10-12 edit (CHANGES_FROM_ORIGINAL.md F-B); this launcher runs split_runner.py, the later version.
#   bash studies/prompt_optimisation/experiments/gepa2_bcp_run2.sh
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$RUNS_DIR"                                       # original: cd <original repo>
L=$RUNS_DIR/gepa_bcp2; mkdir -p $L                   # original: L=logs/gepa_bcp2
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L/chain.log; }
export GEPA_BCP_MODEL=qwen36-35b-a3b SUP_FENCE_TOLERANT=1 SUP_MAX_BLOCKS=5 SUP_PRESENCE_PENALTY=1.5
export GEPA_BCP_SERVERS=http://localhost:8030/v1 HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
BUDGET=${BUDGET:-1200}

say "waiting for GEPA run 1 to finish"
# anchored: a shell whose cmdline merely CONTAINS the text must not match (original pattern: ... -u scripts/gepa_bcp[.]py)
while pgrep -u "$(whoami)" -f '^[^ ]*/python[0-9.]* -u [^ ]*/split_runner[.]py' >/dev/null; do sleep 60; done
say "run 1 finished -- final validation"
env -u GEPA_PROMPTS $PYV $SRC/validate_bcp.py 2>&1 | grep -v Warning | tee $L/run1_validate.log

say "seed on the 200-question feedback pool (cache-aware) to find Qwen3.6's own failures"
env -u GEPA_PROMPTS GEPA_OUT=$RUNS_DIR/gepa_bcp $PYV -u $DOC_SRC/split_runner.py --seed-split feedback --workers 20 \
  2>&1 | grep -E 'seed-split|Traceback' | tee -a $L/chain.log
$PYV $SRC/make_bcp_enriched_split.py 2>&1 | tee -a $L/chain.log
[ -s $SPLITS/bcp_gepa_splits/split_enriched_q36.json ] || { say "ABORT: enriched split not written"; exit 1; }

cp $RUNS_DIR/gepa_bcp/cache.json $L/cache.json
say "GEPA run 2: enriched D_pareto, minibatch 8, budget $BUDGET evaluations, critic claude"
env -u GEPA_PROMPTS GEPA_OUT=$L GEPA_SPLIT=$SPLITS/bcp_gepa_splits/split_enriched_q36.json \
  $PYV -u $DOC_SRC/split_runner.py --max-metric-calls $BUDGET --minibatch 8 --workers 20 > $L/gepa.log 2>&1
say "GEPA run 2 exited ($?)"
