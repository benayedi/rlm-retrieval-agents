"""Search-policy study, SP-1: all 830 BrowseComp-Plus questions, plain ColBERT (BCP-29) vs novelty + reranker, pooled
over the four splits' run caches.

From original/RLM/scripts/bcp_full830_compare.py; changed: import wiring, locations from configs/paths.env, this
paragraph. The cache paths keep their original "logs/<run>/cache.json" form and are read as
<LOGS_ROOT>/<run>/cache.json. The verdicts are the in-run Qwen3.6 judge's
(superseded; the reported 0.5795 -> 0.6747 is the gpt-4.1 re-judge, results/follow_up/verdicts/).
  python analysis/compare_bcp_830.py

Original docstring follows.

Full BrowseComp+ 830: plain ColBERT vs novelty+rerank proxy (Qwen3-Reranker-4B, depth 30), original prompt,
Qwen3.6 on the supervisor harness. Pools the four splits' caches. Killed / timed-out / missing runs count as WRONG
(same rule as the 530-held-out row in docs/rlm_results_summary.xlsx). Works mid-run (reports what is done).
Usage: python scripts/bcp_full830_compare.py
"""
import json, math, os, statistics as st, sys

# Import wiring: agents/shared/rlm_shared is located from this file; locations come from configs/paths.env
# (rlm_shared.paths.path). Original: absolute paths of the machine the experiments ran on.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir, "agents", "shared"))
from rlm_shared.paths import path  # noqa: E402
REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir))
# BrowseComp-Plus question splits (D_pareto / D_feedback / test200 / test_rest): GEPA_SPLIT, default the copy in
# studies/prompt_optimisation/splits/ (original: <original repo>/data/bcp_gepa_splits/split.json).
SPLIT_F = os.environ.get("GEPA_SPLIT") or os.path.join(REPO, "studies", "prompt_optimisation", "splits", "bcp_gepa_splits", "split.json")
# The cache paths below keep their original form "logs/<run>/cache.json" (relative to the original repository root,
# R = "<home>/RLM"); arm() reads them as <LOGS_ROOT>/<run>/cache.json. (In arm() the name `path` is its argument.)
LOGS = str(path("LOGS_ROOT"))
SP = json.load(open(SPLIT_F))
CTRL = {"pareto": ("logs/gepa_bcp2/cache.json", "ee5ea32a1c"), "feedback": ("logs/gepa_bcp2/cache.json", "ee5ea32a1c"),
        "test200": ("logs/gepa_bcp2/cache.json", "ee5ea32a1c"), "test_rest": ("logs/bcp_testrest_ctrl/cache.json", None)}
TRT = {"pareto": "logs/bcp_rest300_rr/cache.json", "feedback": "logs/bcp_rest300_rr/cache.json",   # 10-14: merged, one 24-worker run
       "test200": "logs/bcp_policy_e2e/cache.json", "test_rest": "logs/bcp_testrest_rr/cache.json"}


def arm(path, qs, key=None):
    D = json.load(open(os.path.join(LOGS, path[len("logs/"):])))  # original: open(f"{R}/{path}")
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
