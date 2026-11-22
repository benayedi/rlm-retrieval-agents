"""Naive baseline: one BM25 search with the claim as query, on the 300 val examples.

Refactored copy of original/RLM/scripts/hover_baseline.py (studies/prompt_optimisation/CHANGES_FROM_ORIGINAL.md).
Run: python hover/src/baseline.py [val|test] (single BM25 search with the claim).
Changed: import wiring and the data directory; `os` added to the import line.
Names: 'supervisor harness' / 'full-doc RLM' in the text below is the document REPL agent (agents/document_repl);
  'official RLM harness' / 'our harness' / run_obliq_mine.py is the snippet REPL agent (agents/snippet_repl,
  src/run_obliq.py).
Also checks that the numba backend returns the same rankings as bm25s's default (numpy) backend."""
import json, os, sys, time
# Import wiring and paths (refactor, rules 2-3). Original: hard-coded locations of the original repository (R), the
# document REPL agent working directory (SUP) and the interpreter. Code now comes from this repository (computed from
# __file__), data and run outputs from configs/paths.env (rlm_shared.paths). The original logs/ directory is RUNS_DIR.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, os.pardir, os.pardir))
DOC_SRC = os.path.join(_REPO, "agents", "document_repl", "src")    # split_runner.py (was scripts/gepa_bcp.py), run_bm25.py
SPLITS = os.path.join(_REPO, "studies", "prompt_optimisation", "splits")   # original: <original repo>/data
sys.path.insert(0, os.path.join(_REPO, "agents", "shared"))
from rlm_shared.paths import path  # noqa: E402
sys.path.insert(0, _HERE)   # original: <original repo>/scripts (hover_retriever.py)
import hover_retriever as hr
import numpy as np

split = sys.argv[1] if len(sys.argv) > 1 else "val"
ex = [x for x in json.load(open(f"{path('HOVER_DATA_DIR')}/splits.json")) if x["split"] == split]
s = hr._load()
K = 30
same_ids = same_titles_k = 0; mism = []
rows = []
t_np = t_nb = 0.0
for x in ex:
    tok = s["bm25s"].tokenize(x["claim"], stopwords="en", stemmer=s["stemmer"], show_progress=False)
    t = time.time(); a, sa = s["retriever"].retrieve(tok, k=K, n_threads=1, show_progress=False, backend_selection="numpy"); t_np += time.time() - t
    t = time.time(); b, sb = s["retriever"].retrieve(tok, k=K, n_threads=1, show_progress=False, backend_selection="numba"); t_nb += time.time() - t
    if list(a[0]) == list(b[0]): same_ids += 1
    elif not np.allclose(np.sort(sa[0]), np.sort(sb[0]), atol=1e-5): mism.append(x["id"])
    titles = [s["corpus"][int(d)].partition(" | ")[0] for d in a[0]]
    rows.append((titles, x["gold_titles"]))
print(f"{split}: n={len(ex)}  identical top-{K} id lists numpy vs numba: {same_ids}/{len(ex)}  score-multiset mismatches: {len(mism)} {mism[:5]}")
print(f"mean latency numpy {t_np/len(ex)*1000:.1f} ms, numba {t_nb/len(ex)*1000:.1f} ms")
for k in (7, 21, 30):
    rec = [hr.recall_gold(t[:k], g) for t, g in rows]
    af = [hr.all_found(t[:k], g) for t, g in rows]
    dist = np.bincount([round(r * 3) for r in rec], minlength=4)
    print(f"k={k:2d}: mean recall {np.mean(rec):.4f}  all-found {np.mean(af):.4f} ({sum(af)}/{len(af)})  #gold found 0/1/2/3: {dist.tolist()}")
