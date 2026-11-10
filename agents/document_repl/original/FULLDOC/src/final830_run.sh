#!/usr/bin/env bash
# THE FINAL RUN: all 830 BrowseComp-Plus queries, Design G, gpt-5 root AND sub-calls at high.
# Approved by the user 13 Sep 2026 16:40 with the budget ceiling raised to EUR 1,400.
# Order (data/prepared/final830_order.json): the 400 pre-registered held-out questions first
# (the first 100 of which are runs/heldout100_G_high, re-run here for a single-config 830),
# then the remaining 330 unseen, then the 100 development questions last.
# Resumable: run_experiment skips query ids that already have a result file.
set -uo pipefail
cd /home/extstu_00/suprlm
PY=.venv/bin/python
WORKERS=${WORKERS:-10}
NEED_EUR=${NEED_EUR:-740}
NAME=final830_G_high

REMAIN=$($PY -c "import json;print(json.load(open('runs/BUDGET.json'))['remaining_eur'])")
$PY -c "import sys; r=float('$REMAIN'); sys.exit(0 if r >= $NEED_EUR else 1)" || { echo "REFUSING: only EUR $REMAIN remaining; the 830 needs ~EUR $NEED_EUR"; exit 1; }
echo "budget ok: EUR $REMAIN remaining -> starting 830 (Design G, root+sub high, $WORKERS workers) $(date '+%H:%M')"

ONLY=$($PY -c "import json;print(','.join(json.load(open('data/prepared/final830_order.json'))))")
$PY src/run_experiment.py --name $NAME --design B --workers $WORKERS --max-searches 10 \
    --seed-context --compaction --grounding-gate --max-timeout 2700 --sub-model gpt-5 \
    --reasoning-effort high --sub-reasoning-effort high --only "$ONLY" --save-trajectory 2>&1 | tail -n 6

for T in paper official; do
  $PY src/judge.py --run-dir runs/$NAME --expect data/prepared/sample_830_seed1337.json --template $T --votes 3 --workers 10 2>&1 | grep -E "queries|accuracy|recall|avg search|avg k|judge calls|timeouts|wrote|REFUS"
done
$PY src/report.py --runs eval100_G_high heldout100_G_high $NAME 2>&1 | tail -n 18
$PY src/budget.py | tail -n 3
echo "FINAL830_DONE"
