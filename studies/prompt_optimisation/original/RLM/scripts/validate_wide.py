"""Pick GEPA's best candidate by its D_pareto score, then report it vs the seed.

The D_pareto score is what GEPA SELECTED on, so it is upward-biased by max-over-candidates.
The honest number is the held-out test set, which this script prepares the id files for.
"""
import json, math, statistics, sys, hashlib, os
R = "/home/extstu_00/RLM"
# per-run dir: the 27B and the model-swap runs must not overwrite each other's winner
OUT = os.environ.get("GEPA_OUT", f"{R}/logs/gepa_wide")
print("[validate] run dir", OUT)
C = json.load(open(f"{OUT}/cache.json"))
sp = json.load(open(f"{OUT}/split.json")); par = sp["pareto"]
cands = json.load(open(f"{OUT}/candidates.json"))
def sha(c): return hashlib.sha1((c["STRATEGY"] + c["TASK_SUFFIX"]).encode()).hexdigest()[:10]
rows = []
for i, c in enumerate(cands):
    s = sha(c)
    v = [(C.get(f"{s}|{q}") or {}).get("ndcg@10") for q in par]
    have = [x for x in v if x is not None]
    rows.append((i, s, len(c["STRATEGY"]), len(c["TASK_SUFFIX"]), len(have),
                 statistics.mean(have) if have else None, v))
print("GEPA WIDE — candidates scored on D_pareto (n=%d)\n" % len(par))
print("  %-4s %-12s %9s %12s %8s %10s" % ("idx","sha","STRATEGY","TASK_SUFFIX","cached","NDCG@10"))
for i, s, ls, lt, hv, mn, _ in rows:
    print("  %-4d %-12s %9d %12d %5d/%d %10s" % (i, s, ls, lt, hv, len(par),
          ("%.4f" % mn) if mn is not None else "incomplete"))
full = [r for r in rows if r[4] == len(par)]
if len(full) < 2:
    print("\n  only %d candidate(s) fully evaluated -- nothing to compare yet" % len(full)); sys.exit()
seed = full[0]
print("\nPAIRED vs seed on D_pareto (the SELECTION set -- upward biased):")
best = None
for i, s, ls, lt, hv, mn, v in full[1:]:
    d = [b - a for a, b in zip(seed[6], v) if a is not None and b is not None]
    se = statistics.stdev(d) / math.sqrt(len(d))
    print("  cand %d %s  delta %+.4f  se %.4f  t %+.2f" % (i, s, statistics.mean(d), se, statistics.mean(d)/se))
    if best is None or mn > best[5]: best = (i, s, ls, lt, hv, mn, v)
# The winner must be able to be THE SEED. Picking the best non-seed candidate even when it
# loses to the seed is wrong twice over: it reports a regression as GEPA's "result", and it
# sends a known-worse prompt into a held-out test that costs hours of GPU to confirm a
# negative. If nothing beats the seed, that IS the finding.
if best and best[5] <= seed[5]:
    print("\n  NO CANDIDATE BEAT THE SEED on D_pareto: best was cand %d (%s) at %.4f vs seed %.4f"
          % (best[0], best[1], best[5], seed[5]))
    print("  GEPA found no improvement. Not spending a held-out test on a prompt that already lost")
    print("  on the selection set -- the selection set is the generous comparison, being the one")
    print("  the candidate was chosen on.")
    for _f in ("best_strategy.txt", "best_task.txt"):
        try: os.remove(f"{OUT}/{_f}")
        except FileNotFoundError: pass
    json.dump({"verdict": "no_improvement", "seed": seed[5], "best": best[5],
               "best_sha": best[1]}, open(f"{OUT}/verdict_pareto.json", "w"), indent=1)
    best = None
if best:
    print("\n  BEST by D_pareto: cand %d (%s) at %.4f vs seed %.4f" % (best[0], best[1], best[5], seed[5]))
    json.dump(cands[best[0]], open(f"{OUT}/best_candidate_wide.json", "w"), indent=1)
    open(f"{OUT}/best_strategy.txt","w").write(cands[best[0]]["STRATEGY"])
    open(f"{OUT}/best_task.txt","w").write(cands[best[0]]["TASK_SUFFIX"])
    open(f"{OUT}/seed_strategy.txt","w").write(cands[0]["STRATEGY"])
    open(f"{OUT}/seed_task.txt","w").write(cands[0]["TASK_SUFFIX"])
    print(f"  wrote {OUT}/best_strategy.txt + best_task.txt for the held-out test")
    print("\n  *** this number is NOT the result. Run 3 scored +0.0534 here and")
    print("      -0.0030 on held-out. The 81-question test decides. ***")
