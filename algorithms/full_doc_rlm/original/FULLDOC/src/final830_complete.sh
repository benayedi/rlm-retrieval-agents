#!/usr/bin/env bash
# Completion chain for the 830: wait for the running orchestrator to exit, re-run whatever is missing
# (resumable; the orchestrator skips existing results), then judge both templates and report.
set -uo pipefail; cd /home/extstu_00/suprlm; PY=.venv/bin/python; NAME=final830_G_high
while pgrep -f "run_experiment\.py --name ${NAME}\b" >/dev/null; do sleep 60; done
echo "main pass ended $(date '+%H:%M'); results: $(ls runs/$NAME/[0-9]*.json | wc -l)/830"
for attempt in 1 2; do
  MISSING=$($PY -c "import json,os; o=json.load(open('data/prepared/final830_order.json')); print(len([q for q in o if not os.path.exists(f'runs/$NAME/{q}.json')]))")
  [ "$MISSING" -eq 0 ] && break
  echo "re-run attempt $attempt for $MISSING missing questions $(date '+%H:%M')"
  ONLY=$($PY -c "import json;print(','.join(json.load(open('data/prepared/final830_order.json'))))")
  $PY src/run_experiment.py --name $NAME --design B --workers 10 --max-searches 10 --seed-context --compaction --grounding-gate \
      --max-timeout 2700 --sub-model gpt-5 --reasoning-effort high --sub-reasoning-effort high --only "$ONLY" --save-trajectory --allow-code-change --workers 10 2>&1 | tee -a logs/final830_rerun.log | tail -n 4
done
echo "results now: $(ls runs/$NAME/[0-9]*.json | wc -l)/830"
for T in paper official; do
  $PY src/judge.py --run-dir runs/$NAME --expect data/prepared/sample_830_seed1337.json --template $T --votes 3 --workers 10 2>&1 | grep -E "queries|^  accuracy|recall  |avg search|avg k|flagged|timeouts|judge calls|wrote|REFUS"
done
$PY src/report.py --runs eval100_G_high heldout100_G_high $NAME 2>&1 | tail -n 20
$PY src/budget.py | tail -n 3
echo "FINAL830_DONE"
