#!/usr/bin/env bash
# From original/FULLDOC/src/final830_complete.sh. Arguments unchanged; locations and credentials from configs/paths.env
# (runs/ -> $RUNS_DIR, src/ -> $SRC, .venv/bin/python -> $PYTHON). NOT the launcher of BCP-21: final830_G_high is
# Design G at HIGH effort without the solo/lean/no-delegation flags, and per launch_high400.sh it was served
# gpt-5.4 because no deployment was pinned. The launchers of BCP-21 (final100_medium, final200_medium, final100_e,
# final_rest) were not preserved; configs/BCP-21.env records their configuration from the run records.
# report.py and budget.py were not preserved either: those steps fail.
# Completion chain for the 830: wait for the running orchestrator to exit, re-run whatever is missing
# (resumable; the orchestrator skips existing results), then judge both templates and report.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
. "$(dirname "${BASH_SOURCE[0]}")/_azure.sh"
cd "$FULLDOC_ROOT"; PY=$PYTHON; NAME=final830_G_high   # original: cd <working directory>; PY=.venv/bin/python
while pgrep -f "run_experiment\.py --name ${NAME}\b" >/dev/null; do sleep 60; done
echo "main pass ended $(date '+%H:%M'); results: $(ls $RUNS_DIR/$NAME/[0-9]*.json | wc -l)/830"
for attempt in 1 2; do
  MISSING=$($PY -c "import json,os; o=json.load(open('data/prepared/final830_order.json')); print(len([q for q in o if not os.path.exists(f'$RUNS_DIR/$NAME/{q}.json')]))")
  [ "$MISSING" -eq 0 ] && break
  echo "re-run attempt $attempt for $MISSING missing questions $(date '+%H:%M')"
  ONLY=$($PY -c "import json;print(','.join(json.load(open('data/prepared/final830_order.json'))))")
  $PY $SRC/run_experiment.py --name $NAME --design B --workers 10 --max-searches 10 --seed-context --compaction --grounding-gate \
      --max-timeout 2700 --sub-model gpt-5 --reasoning-effort high --sub-reasoning-effort high --only "$ONLY" --save-trajectory --allow-code-change --workers 10 2>&1 | tee -a $RUNS_DIR/final830_rerun.log | tail -n 4
done
echo "results now: $(ls $RUNS_DIR/$NAME/[0-9]*.json | wc -l)/830"
for T in paper official; do
  $PY $SRC/judge.py --run-dir $RUNS_DIR/$NAME --expect data/prepared/sample_830_seed1337.json --template $T --votes 3 --workers 10 2>&1 | grep -E "queries|^  accuracy|recall  |avg search|avg k|flagged|timeouts|judge calls|wrote|REFUS"
done
$PY $SRC/report.py --runs eval100_G_high heldout100_G_high $NAME 2>&1 | tail -n 20
$PY $SRC/budget.py | tail -n 3
echo "FINAL830_DONE"
