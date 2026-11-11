"""Held-out confirmation: GEPA winner vs the original prompt on test200 (BrowseComp-Plus, Qwen3.6).

Refactored copy of original/RLM/scripts/bcp_confirm.py (studies/prompt_optimisation/CHANGES_FROM_ORIGINAL.md).
Run: studies/prompt_optimisation/experiments/gepa2_confirm_test200.sh (GEPA-2 held-out confirmation). GEPA_OUT is
  resolved by split_runner against RUNS_DIR (original: against the original repository), so GEPA_OUT=gepa_bcp2 or an
  absolute path.
Changed: import wiring (split_runner.py, was gepa_bcp.py) and the default split file. See PO-I3.
Names: 'supervisor harness' / 'full-doc RLM' in the text below is the document REPL agent (agents/document_repl);
  'official RLM harness' / 'our harness' / run_obliq_mine.py is the snippet REPL agent (agents/snippet_repl,
  src/run_obliq.py).

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
sys.argv_saved = list(sys.argv); sys.argv = [sys.argv[0]]
sys.path.insert(0, DOC_SRC)   # original: <original repo>/scripts
os.environ.pop("GEPA_PROMPTS", None)
import split_runner as G   # original: import gepa_bcp as G (same module, renamed)
sys.argv = sys.argv_saved

ap = argparse.ArgumentParser()
ap.add_argument("--winner", required=True)
ap.add_argument("--split", default="test200")
ap.add_argument("--split-file", default=os.path.join(SPLITS, "bcp_gepa_splits", "split.json"))
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
