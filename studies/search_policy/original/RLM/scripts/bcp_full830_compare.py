"""Full BrowseComp+ 830: plain ColBERT vs novelty+rerank proxy (Qwen3-Reranker-4B, depth 30), original prompt,
Qwen3.6 on the supervisor harness. Pools the four splits' caches. Killed / timed-out / missing runs count as WRONG
(same rule as the 530-held-out row in docs/rlm_results_summary.xlsx). Works mid-run (reports what is done).
Usage: python scripts/bcp_full830_compare.py
"""
import json, math, statistics as st

R = "/home/extstu_00/RLM"
SP = json.load(open(f"{R}/data/bcp_gepa_splits/split.json"))
CTRL = {"pareto": ("logs/gepa_bcp2/cache.json", "ee5ea32a1c"), "feedback": ("logs/gepa_bcp2/cache.json", "ee5ea32a1c"),
        "test200": ("logs/gepa_bcp2/cache.json", "ee5ea32a1c"), "test_rest": ("logs/bcp_testrest_ctrl/cache.json", None)}
TRT = {"pareto": "logs/bcp_rest300_rr/cache.json", "feedback": "logs/bcp_rest300_rr/cache.json",   # 10-14: merged, one 24-worker run
       "test200": "logs/bcp_policy_e2e/cache.json", "test_rest": "logs/bcp_testrest_rr/cache.json"}


def arm(path, qs, key=None):
    D = json.load(open(f"{R}/{path}"))
    if key is None:
        keys = {k.split("|")[0] for k in D if k.split("|")[1] in set(qs)}
        assert len(keys) == 1, (path, keys)
        key = keys.pop()
    return {q: D.get(f"{key}|{q}") for q in qs}


acc = lambda r: 1.0 if r and r.get("correct") and r.get("status") != "killed" else 0.0
rec = lambda r: (r.get("gold_recall") or 0.0) if r else 0.0
rows, done = [], 0
for s in ("pareto", "feedback", "test200", "test_rest"):
    qs = [str(q) for q in SP[s]]
    c, t = arm(CTRL[s][0], qs, CTRL[s][1]), arm(TRT[s], qs)
    n_t = sum(1 for q in qs if t[q]); done += n_t
    print(f"{s:10s} treatment done {n_t}/{len(qs)}  ctrl acc {st.mean(acc(c[q]) for q in qs):.3f}  "
          f"rerank acc (done only) {st.mean(acc(t[q]) for q in qs if t[q]) if n_t else 0:.3f}")
    rows += [(acc(c[q]), acc(t[q]), rec(c[q]), rec(t[q]), t[q]) for q in qs if t[q]]
print(f"\nTOTAL treatment done {done}/830; paired n={len(rows)}")
for nm, i in (("accuracy", 0), ("gold recall", 2)):
    d = [r[i + 1] - r[i] for r in rows]
    se = st.stdev(d) / math.sqrt(len(d))
    print(f"  {nm:12s} plain ColBERT {st.mean(r[i] for r in rows):.3f}  novelty+rerank {st.mean(r[i+1] for r in rows):.3f}  "
          f"diff {st.mean(d):+.3f} (t {st.mean(d)/se:+.2f})")
w = sum(1 for r in rows if r[1] > r[0]); l = sum(1 for r in rows if r[1] < r[0])
print(f"  won/lost {w}/{l} | rerank killed/timeouts: {sum(1 for r in rows if r[4].get('status') not in ('completed', 'empty_output'))}")
