"""Pick GEPA's best candidate by its D_pareto score, then report it vs the seed.

Refactored copy of original/RLM/scripts/validate_wide.py (studies/prompt_optimisation/CHANGES_FROM_ORIGINAL.md).
Run: GEPA_OUT=<abs run dir> python obliq_twitter/src/validate_wide.py (run 4 verdict, night8_gepa27_finish.sh).
Changed: the default run directory only.
Names: 'supervisor harness' / 'full-doc RLM' in the text below is the document REPL agent (agents/document_repl);
  'official RLM harness' / 'our harness' / run_obliq_mine.py is the snippet REPL agent (agents/snippet_repl,
  src/run_obliq.py).

The D_pareto score is what GEPA SELECTED on, so it is upward-biased by max-over-candidates.
The honest number is the held-out test set, which this script prepares the id files for.
"""
import json, math, statistics, sys, hashlib, os
# Import wiring and paths (refactor, rules 2-3). Original: hard-coded locations of the original repository (R), the
# document REPL agent working directory (SUP) and the interpreter. Code now comes from this repository (computed from
# __file__), data and run outputs from configs/paths.env (rlm_shared.paths). The original logs/ directory is RUNS_DIR.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, os.pardir, os.pardir))
DOC_SRC = os.path.join(_REPO, "agents", "document_repl", "src")    # split_runner.py (was scripts/gepa_bcp.py), run_bm25.py
SPLITS = os.path.join(_REPO, "studies", "prompt_optimisation", "splits")   # original: <original repo>/data
sys.path.insert(0, os.path.join(_REPO, "agents", "shared"))
from rlm_shared.paths import path  # noqa: E402
RUNS = str(path("RUNS_DIR"))                                         # original: f"{R}/logs"
# per-run dir: the 27B and the model-swap runs must not overwrite each other's winner
OUT = os.environ.get("GEPA_OUT", f"{RUNS}/gepa_wide")
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
