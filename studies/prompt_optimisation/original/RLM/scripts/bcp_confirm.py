"""Held-out confirmation: GEPA winner vs the original prompt on test200 (BrowseComp-Plus, Qwen3.6).

test200 is proportional (stratified like the benchmark) and was never used by GEPA -- not for
selection, not for feedback -- and none of the winner's examples come from it (leak-checked
2026-10-09). Both arms run INTERLEAVED (two concurrent batches sharing the same server), so load
and wall-clock timeouts hit them equally. Paired per question; timeouts / kills / empty submissions
count as WRONG (they are agent outcomes -- run 2's prompts measurably cause timeouts).

Usage (run env required -- model/harness settings are part of the cache key):
  GEPA_OUT=logs/gepa_bcp2 python scripts/bcp_confirm.py --winner logs/gepa_bcp2/winner.json --split test200
  ... --report      (score what is cached, no rollouts)
"""
import argparse, json, math, os, statistics as st, sys, threading
sys.argv_saved = list(sys.argv); sys.argv = [sys.argv[0]]
sys.path.insert(0, "/home/extstu_00/RLM/scripts")
os.environ.pop("GEPA_PROMPTS", None)
import gepa_bcp as G
sys.argv = sys.argv_saved

ap = argparse.ArgumentParser()
ap.add_argument("--winner", required=True)
ap.add_argument("--split", default="test200")
ap.add_argument("--split-file", default="/home/extstu_00/RLM/data/bcp_gepa_splits/split.json")
ap.add_argument("--workers", type=int, default=20)
ap.add_argument("--report", action="store_true")
a = ap.parse_args()

G._cache_load()
seed = G.seed_blocks()
win, fixed = G.repair(json.load(open(a.winner)))
assert G.validate(win, seed) is None, G.validate(win, seed)
qids = json.load(open(a.split_file))[a.split]
ks, kw = G.ckey(seed), G.ckey(win)
G._log(f"[confirm] {a.split} ({len(qids)} q) | original {ks} vs winner {kw} | interleaved, "
       f"{a.workers} workers total | timeouts count WRONG")

if not a.report:
    th = [threading.Thread(target=G.run_batch, args=(c, qids, max(1, a.workers // 2))) for c in (seed, win)]
    for t in th: t.start()
    for t in th: t.join()

acc = lambda r: 1.0 if (r and r.get("correct")) else 0.0
rec = lambda r: (r or {}).get("gold_recall") or 0.0
S = {q: G._CACHE.get(f"{ks}|{q}") for q in qids}; W = {q: G._CACHE.get(f"{kw}|{q}") for q in qids}
both = [q for q in qids if S[q] and W[q] and S[q].get("status") != "killed" and W[q].get("status") != "killed"]
out = [f"[confirm] {a.split}: paired n={len(both)} of {len(qids)}"]
if len(both) < 2:
    G._log(out[0] + " -- not enough paired results yet"); sys.exit(0)
for nm, f in (("accuracy", acc), ("gold recall", rec)):
    d = [f(W[q]) - f(S[q]) for q in both]
    se = st.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else 0.0
    out.append(f"  {nm}: original {st.mean(f(S[q]) for q in both):.3f}  winner {st.mean(f(W[q]) for q in both):.3f}  "
               f"diff {st.mean(d):+.3f} (se {se:.3f}, t {st.mean(d)/se if se else 0:+.2f})")
wins = sum(1 for q in both if acc(W[q]) > acc(S[q])); loss = sum(1 for q in both if acc(W[q]) < acc(S[q]))
to = lambda X: sum(1 for q in both if X[q].get("status") not in ("completed", "empty_output"))
out.append(f"  questions won/lost by the winner: {wins}/{loss} | timeouts+kills: original {to(S)}, winner {to(W)}")
for l in out:
    G._log(l)
