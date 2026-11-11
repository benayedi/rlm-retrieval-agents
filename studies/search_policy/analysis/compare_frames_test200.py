"""FRAMES test200, cap 10: BM25 (FR-13) vs novelty + reranker proxy (logs/frames/sup_q36_bm25f_rr). This is the
comparison of the dropped run SP-5 (Qwen3.6 judge only, not reported); kept because it is in the study's original/.

From original/RLM/scripts/frames_rr_compare.py; changed: import wiring, locations from configs/paths.env, this
paragraph. Reads <LOGS_ROOT>/frames/<run>/graded.json (the in-run Qwen3.6 verdicts) and
studies/prompt_optimisation/splits/frames/gepa_splits/split.json.
  python analysis/compare_frames_test200.py

Original docstring follows.

Paired comparison on the GEPA test split: BM25 baseline (sup_q36_bm25f) vs novelty+rerank proxy
(sup_q36_bm25f_rr). Same harness, seed prompt, model, caps, judge; only --server differs.
Missing results (hard-cap kills) count as wrong."""
import json, math, os, statistics as st, sys
from math import comb
# Import wiring: agents/shared/rlm_shared is located from this file; locations come from configs/paths.env
# (rlm_shared.paths.path). Original: absolute paths of the machine the experiments ran on.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir, "agents", "shared"))
from rlm_shared.paths import path  # noqa: E402
REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir))
t = json.load(open(os.path.join(REPO, "studies", "prompt_optimisation", "splits", "frames", "gepa_splits", "split.json")))["test"]  # original: <original repo>/data/frames/gepa_splits/split.json
def load(d):
    return {v["qid"]: v for v in json.load(open(f"{path('LOGS_ROOT')}/frames/{d}/graded.json")).values()}
A, B = load("sup_q36_bm25f"), load("sup_q36_bm25f_rr")
a = [1 if A.get(q, {}).get("correct") else 0 for q in t]; b = [1 if B.get(q, {}).get("correct") else 0 for q in t]
w = sum(y > x for x, y in zip(a, b)); l = sum(x > y for x, y in zip(a, b))
p = min(1, 2 * sum(comb(w + l, i) for i in range(min(w, l) + 1)) / 2 ** (w + l)) if w + l else 1
d = [y - x for x, y in zip(a, b)]; se = st.stdev(d) / math.sqrt(len(d))
gr = lambda D: st.mean(D.get(q, {}).get("gold_recall", 0) for q in t)
print(f"test n={len(t)} | BM25 {sum(a)} ({sum(a)/len(t):.3f}) | rerank {sum(b)} ({sum(b)/len(t):.3f}) | "
      f"diff {st.mean(d):+.3f} se {se:.3f} t {st.mean(d)/se:+.2f} | won {w} lost {l} McNemar p={p:.4f} | "
      f"gold recall {gr(A):.3f} -> {gr(B):.3f} | results {sum(q in B for q in t)}/{len(t)}")
