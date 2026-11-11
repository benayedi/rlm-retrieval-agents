"""After GEPA on BrowseComp-Plus: score every fully-evaluated candidate against the seed on D_pareto.

Refactored copy of original/RLM/scripts/validate_bcp.py (studies/prompt_optimisation/CHANGES_FROM_ORIGINAL.md).
Run: with the run's environment (model and Qwen3.6 switches are part of the cache key): GEPA_OUT=<abs dir>
  [GEPA_SPLIT=...] python src/validate_bcp.py. Called by experiments/gepa2_bcp_run2.sh.
Changed: import wiring only. See PO-I4: a relative GEPA_OUT is read relative to the working directory here but
  relative to RUNS_DIR inside split_runner.
Names: 'supervisor harness' / 'full-doc RLM' in the text below is the document REPL agent (agents/document_repl);
  'official RLM harness' / 'our harness' / run_obliq_mine.py is the snippet REPL agent (agents/snippet_repl,
  src/run_obliq.py).

Paired per question, on BOTH accuracy (the reported metric) and the selection score
(correct + 0.5*gold_recall). A winner is written only if it beats the seed on the selection
score -- the seed can win, which is the Twitter lesson (validate_wide.py once wrote a losing
prompt as "BEST" and nearly spent 5.5 GPU-hours confirming a known negative).

D_pareto is the set GEPA SELECTED on, so the winner's margin here is biased upward. The claim
is made on test200 (run the winner there; the seed is re-run or cached per the plan).
"""
import json, math, os, statistics as st, sys
# Import wiring and paths (refactor, rules 2-3). Original: hard-coded locations of the original repository (R), the
# document REPL agent working directory (SUP) and the interpreter. Code now comes from this repository (computed from
# __file__), data and run outputs from configs/paths.env (rlm_shared.paths). The original logs/ directory is RUNS_DIR.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, os.pardir))
DOC_SRC = os.path.join(_REPO, "agents", "document_repl", "src")    # split_runner.py (was scripts/gepa_bcp.py), run_bm25.py
SPLITS = os.path.join(_REPO, "studies", "prompt_optimisation", "splits")   # original: <original repo>/data
sys.path.insert(0, os.path.join(_REPO, "agents", "shared"))
from rlm_shared.paths import path  # noqa: E402
RUNS = str(path("RUNS_DIR"))                                         # original: f"{R}/logs"
sys.path.insert(0, DOC_SRC)   # original: <original repo>/scripts
os.environ.pop("GEPA_PROMPTS", None)
import split_runner as G  # noqa: E402  (original: import gepa_bcp as G)

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
