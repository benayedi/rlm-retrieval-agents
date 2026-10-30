#!/usr/bin/env bash
# PUBLICATION COPY of scripts/run5d/question_watchdog.sh (used on BCP-13 with warn 5 / kill 10 USD). Changes: the
# process pattern names src/run_browsecomp.py (was scripts/run_bcp_snippet.py) and the interpreter is $PYTHON.
# NOTE: prices are GPT-5's ($1.25 in / $0.125 cached / $10 out per 1M tokens) whatever the model.
#
# Original header:
# Per-question cost watchdog (2026-10-27). Every 60 s, sums the logged Azure spend of every live question process of a
# run (one process per question; RLM_USAGE_LOG lines carry the pid). ALERT once when a question passes <warn_usd>;
# STOP (kill that question's process, which the runner records as an error, scored wrong) when it passes <kill_usd>.
#   experiments/question_watchdog.sh <run_dir> <warn_usd> <kill_usd>
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
D=$1; WARN=$2; KILL=$3
while pgrep -f "run_browsecomp.py.*$(basename $D)" >/dev/null; do
  "$PYTHON" - "$D" "$WARN" "$KILL" <<'PY'
import json, os, sys, collections, time
D, WARN, KILL = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
st = f"{D}/.watchdog_state.json"
seen = json.load(open(st)) if os.path.exists(st) else {"warned": [], "killed": []}
alive = {}
for p in os.listdir("/proc"):
    if p.isdigit():
        try:
            c = open(f"/proc/{p}/cmdline").read().split("\0")
        except Exception:
            continue
        if any("run_browsecomp.py" in a for a in c) and any(os.path.basename(D) in a for a in c) and "python" in c[0]:
            q = c[c.index("--start") + 1] if "--start" in c else "?"
            alive[int(p)] = q
spend = collections.defaultdict(float)
for l in open(f"{D}/usage.jsonl"):
    x = json.loads(l)
    if x["pid"] in alive:
        spend[x["pid"]] += ((x["prompt"] - x["cached"]) * 1.25 + x["cached"] * 0.125 + x["completion"] * 10) / 1e6
for pid, s in spend.items():
    q = alive[pid]
    if s > KILL and pid not in seen["killed"]:
        os.kill(pid, 15); seen["killed"].append(pid)
        print(f"{time.strftime('%F %T')} STOPPED q{q} (pid {pid}) at ${s:.2f} > ${KILL}", flush=True)
    elif s > WARN and pid not in seen["warned"]:
        seen["warned"].append(pid)
        print(f"{time.strftime('%F %T')} WARN q{q} (pid {pid}) at ${s:.2f} > ${WARN}", flush=True)
json.dump(seen, open(st, "w"))
PY
  sleep 60
done
echo "$(date '+%F %T') run finished, watchdog exits"
