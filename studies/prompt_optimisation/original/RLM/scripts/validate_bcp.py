"""After GEPA on BrowseComp-Plus: score every fully-evaluated candidate against the seed on D_pareto.

Paired per question, on BOTH accuracy (the reported metric) and the selection score
(correct + 0.5*gold_recall). A winner is written only if it beats the seed on the selection
score -- the seed can win, which is the Twitter lesson (validate_wide.py once wrote a losing
prompt as "BEST" and nearly spent 5.5 GPU-hours confirming a known negative).

D_pareto is the set GEPA SELECTED on, so the winner's margin here is biased upward. The claim
is made on test200 (run the winner there; the seed is re-run or cached per the plan).
"""
import json, math, os, statistics as st, sys
sys.path.insert(0, "/home/extstu_00/RLM/scripts")
os.environ.pop("GEPA_PROMPTS", None)
import gepa_bcp as G  # noqa: E402

OUT = os.environ.get("GEPA_OUT", G.OUT)
C = json.load(open(f"{OUT}/cache.json"))
par = json.load(open(G.SPLIT_F))["pareto"]
cands = json.load(open(f"{OUT}/candidates.json"))
seed_k = G.ckey(G.seed_blocks())


def recs(k):
    return {q: C[f"{k}|{q}"] for q in par if f"{k}|{q}" in C and not C[f"{k}|{q}"].get("_fake")}


def paired(a, b, f):
    # ALL questions, timeouts/kills counted as WRONG (acc 0, recall as recorded). Dropping them
    # (the old behaviour) rewarded prompts that CAUSE timeouts: run 2's best candidate timed out on 8
    # of 100 (original: 3), all brainstorm loops after its 10 searches -- prompt-caused, not infra.
    # Dropping them inflated its gain from +0.070 to +0.089 (2026-10-09).
    qs = [q for q in a if q in b]
    d = [f(b[q]) - f(a[q]) for q in qs]
    if len(d) < 2:
        return len(d), 0.0, 0.0, 0.0
    se = st.stdev(d) / math.sqrt(len(d))
    return len(d), st.mean(d), se, (st.mean(d) / se if se else 0.0)


acc = lambda r: 1.0 if r.get("correct") else 0.0
S = recs(seed_k)
print(f"[validate] {OUT} | seed {seed_k}: {len(S)}/{len(par)} on D_pareto, "
      f"acc {st.mean(acc(r) for r in S.values()):.3f}" if S else "[validate] seed not scored")
best = None
for i, c in enumerate(cands):
    rc, _ = G.repair(c)
    k = G.ckey(rc)
    if k == seed_k:
        continue
    X = recs(k)
    if len(X) < len(par):
        print(f"  cand {i} {k}: {len(X)}/{len(par)} scored -- incomplete, ignored"); continue
    n, da, sa, ta = paired(S, X, acc)
    _, ds, ss, ts = paired(S, X, G.score)
    print(f"  cand {i} {k}: n={n}  d_acc {da:+.3f} (se {sa:.3f}, t {ta:+.2f})  "
          f"d_score {ds:+.4f} (t {ts:+.2f})")
    if ds > 0 and (best is None or ds > best[1]):
        best = (i, ds, k)
if best is None:
    print("  NO CANDIDATE BEAT THE SEED on D_pareto -- nothing to send to test200.")
    json.dump({"verdict": "no_improvement"}, open(f"{OUT}/verdict_pareto.json", "w"))
else:
    rc, _ = G.repair(cands[best[0]])
    json.dump(rc, open(f"{OUT}/winner.json", "w"), indent=1)
    json.dump({"verdict": "candidate", "cand": best[0], "key": best[2], "d_score": best[1]},
              open(f"{OUT}/verdict_pareto.json", "w"))
    print(f"  WINNER cand {best[0]} ({best[2]}) -> {OUT}/winner.json. This margin is selection-"
          f"biased; the claim comes from test200.")
