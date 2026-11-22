"""Trace health for the BrowseComp+ GEPA run: are the rollouts failing in ways that are NOT the prompt?

Usage: bcp_trace_health.py [--since EPOCH] [--json]
Scans logs/gepa_bcp/rollouts/<cand>/*.json written after --since and reports, per candidate:
    n, completed, accuracy (from cache), timeouts, context overflows, other errors,
    loop turns, dead turns, >5-block turns, median wall, mean searches.
Prints ALERT lines when a threshold is crossed (exit 2), else a single OK line (exit 0).

Failure signatures, each measured on 2026-10-07 with Qwen3.6-35B-A3B:
  LOOP TURN   >8,000 chars, no code executed, and either its most common line is >15% of its
              lines ("I'll write the code." x thousands) or its thinking never closes. These
              ate 86-98% of the wall time of every timed-out question.
  OVERFLOW    "maximum context length" -- the prompt outgrew the served context.
  BIG TURN    >5 code blocks in one turn: means SUP_MAX_BLOCKS is not applied.
"""
import argparse, collections, glob, json, os, statistics as st, sys

R = "/home/extstu_00/RLM"
OUTD = os.environ.get("GEPA_OUT", f"{R}/logs/gepa_bcp")   # run directory (run 1 default; run 2 = logs/gepa_bcp2)
ROLL = f"{OUTD}/rollouts"
CACHE = f"{OUTD}/cache.json"
TH = {"timeout_rate": 0.10, "loop_turn_rate": 0.08, "dead_turn_rate": 0.25}


def loop_turn(x: dict) -> bool:
    r = x.get("response") or ""
    if len(r) < 8000 or x.get("code_blocks"):
        return False
    lines = [l.strip() for l in r.splitlines() if l.strip()]
    top = collections.Counter(lines).most_common(1)[0][1] / len(lines) if lines else 0
    unclosed = "</think>" not in r and len(r) > 30000
    return top > 0.15 or unclosed


def scan(since: float):
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    out = {}
    for d in sorted(glob.glob(f"{ROLL}/*/")):
        key = os.path.basename(d.rstrip("/"))
        files = [f for f in glob.glob(f"{d}*.json") if os.path.getmtime(f) >= since]
        if not files:
            continue
        s = collections.Counter(); walls, searches, accs = [], [], []
        for f in files:
            try:
                r = json.load(open(f))
            except Exception:
                continue
            s["n"] += 1
            err = str(r.get("error") or "")
            if r.get("status") == "completed":
                s["completed"] += 1
            elif r.get("status") == "empty_output":
                s["empty"] += 1   # agent submitted nothing: scored WRONG, not infra
            elif "Timeout" in err:
                s["timeout"] += 1
            elif "maximum context length" in err:
                s["overflow"] += 1
            else:
                s["other_err"] += 1
            for x in (r.get("trajectory") or {}).get("iterations") or []:
                s["turns"] += 1
                nb = len(x.get("code_blocks") or [])
                if nb > 5:
                    s["big_turn"] += 1
                if loop_turn(x):
                    s["loop_turn"] += 1
                elif nb == 0 and not x.get("final_answer"):
                    # Qwen3.6 restates its final answer as prose ("Exact Answer: ...") instead of
                    # setting answer["content"]; it costs turns, not answers (measured 10-07: 19/20
                    # questions, 27% of wall time, all still finished). Counted apart so the
                    # dead-turn alert keeps meaning "no code and no answer".
                    tail = (x.get("response") or "").split("</think>")[-1]
                    s["prose_answer" if "Exact Answer" in tail else "dead_turn"] += 1
            walls.append(r.get("wall_time_s") or 0)
            searches.append((r.get("tool_call_counts") or {}).get("search", 0))
            c = cache.get(f"{key}|{r.get('query_id')}")
            if c is not None:
                accs.append(1.0 if c.get("correct") else 0.0)
        s["median_wall"] = int(st.median(walls)) if walls else 0
        s["mean_search"] = round(st.mean(searches), 1) if searches else 0
        s["acc"] = round(st.mean(accs), 3) if accs else None
        out[key] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=float, default=0.0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    res = scan(a.since)
    alerts = []
    for k, s in res.items():
        n, t = max(1, s["n"]), max(1, s["turns"])
        line = (f"{k}: n={s['n']} done={s['completed']} acc={s['acc']} timeout={s['timeout']} "
                f"overflow={s['overflow']} empty={s['empty']} err={s['other_err']} | turns={s['turns']} "
                f"loop={s['loop_turn']} ({s['loop_turn']/t:.0%}) dead={s['dead_turn']} "
                f"({s['dead_turn']/t:.0%}) prose_answer={s['prose_answer']} ({s['prose_answer']/t:.0%}) big={s['big_turn']} | wall~{s['median_wall']}s "
                f"search~{s['mean_search']}")
        print(line)
        if s["overflow"]:
            alerts.append(f"ALERT {k}: {s['overflow']} context overflow(s)")
        if s["n"] >= 10 and s["empty"] / n > 0.05:
            alerts.append(f"ALERT {k}: {s['empty']} empty submissions ({s['empty']/n:.0%})")
        if s["other_err"]:
            alerts.append(f"ALERT {k}: {s['other_err']} non-timeout error(s) -- possible crash/infra")
        if s["big_turn"]:
            alerts.append(f"ALERT {k}: {s['big_turn']} turns with >5 blocks -- block cap not applied")
        if s["n"] >= 10 and s["timeout"] / n > TH["timeout_rate"]:
            alerts.append(f"ALERT {k}: timeout rate {s['timeout']/n:.0%} > {TH['timeout_rate']:.0%}")
        if s["turns"] >= 50 and s["loop_turn"] / t > TH["loop_turn_rate"]:
            alerts.append(f"ALERT {k}: loop-turn rate {s['loop_turn']/t:.0%} > {TH['loop_turn_rate']:.0%}")
        if s["turns"] >= 50 and s["dead_turn"] / t > TH["dead_turn_rate"]:
            alerts.append(f"ALERT {k}: dead-turn rate {s['dead_turn']/t:.0%} > {TH['dead_turn_rate']:.0%}")
    for x in alerts:
        print(x)
    sys.exit(2 if alerts else 0)


if __name__ == "__main__":
    main()
