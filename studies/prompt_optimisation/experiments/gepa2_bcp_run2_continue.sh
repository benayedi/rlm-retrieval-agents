#!/usr/bin/env bash
# GEPA-2, the continuation that actually launched run 2 (2026-10-08 19:00): after the relative-GEPA_OUT fix, let
# orphaned workers finish, recover their misfiled rollouts, finish the seed on the feedback pool, enrich, launch run 2.
# From original/RLM/logs/gepa_bcp2/continue_after_fix.sh (arguments and environment unchanged; locations from
# configs/paths.env). src/bcp_recover_misfiled.py MOVES the misfiled rollouts, as the original did.
#   bash studies/prompt_optimisation/experiments/gepa2_bcp_run2_continue.sh
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$RUNS_DIR"                                       # original: cd <original repo>
L=$RUNS_DIR/gepa_bcp2; mkdir -p $L                   # original: L=logs/gepa_bcp2 (existed already)
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $L/chain.log; }
export GEPA_BCP_MODEL=qwen36-35b-a3b SUP_FENCE_TOLERANT=1 SUP_MAX_BLOCKS=5 SUP_PRESENCE_PENALTY=1.5
export GEPA_BCP_SERVERS=http://localhost:8030/v1 HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
say "fix: GEPA_OUT made absolute; recovering misfiled rollouts (<FULLDOC_ROOT>/logs/gepa_bcp)"
$PYV $SRC/bcp_recover_misfiled.py 2>&1 | grep -v Warning >> $L/chain.log
while pgrep -u "$(whoami)" -f 'run_one[.]py --query-id' >/dev/null; do sleep 60; done
$PYV $SRC/bcp_recover_misfiled.py 2>&1 | grep -v Warning >> $L/chain.log
say "seed on the remaining feedback questions (missing/killed only)"
env -u GEPA_PROMPTS GEPA_OUT=$RUNS_DIR/gepa_bcp $PYV -u $DOC_SRC/split_runner.py --seed-split feedback --workers 20 2>&1 | grep -E 'seed-split|Traceback' >> $L/chain.log
$PYV $SRC/make_bcp_enriched_split.py >> $L/chain.log 2>&1 || { say "ABORT: enrich failing"; exit 1; }
cp $RUNS_DIR/gepa_bcp/cache.json $L/cache.json
say "GEPA run 2: enriched D_pareto, minibatch 8, budget 1200 evaluations, critic claude"
env -u GEPA_PROMPTS GEPA_OUT=$L GEPA_SPLIT=$SPLITS/bcp_gepa_splits/split_enriched_q36.json \
  $PYV -u $DOC_SRC/split_runner.py --max-metric-calls 1200 --minibatch 8 --workers 20 > $L/gepa.log 2>&1
say "GEPA run 2 exited ($?)"
