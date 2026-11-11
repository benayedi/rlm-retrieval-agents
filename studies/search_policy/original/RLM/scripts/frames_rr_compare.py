"""Paired comparison on the GEPA test split: BM25 baseline (sup_q36_bm25f) vs novelty+rerank proxy
(sup_q36_bm25f_rr). Same harness, seed prompt, model, caps, judge; only --server differs.
Missing results (hard-cap kills) count as wrong."""
import json, math, statistics as st
from math import comb
R = "/home/extstu_00/RLM"
t = json.load(open(f"{R}/data/frames/gepa_splits/split.json"))["test"]
def load(d):
    return {v["qid"]: v for v in json.load(open(f"{R}/logs/frames/{d}/graded.json")).values()}
A, B = load("sup_q36_bm25f"), load("sup_q36_bm25f_rr")
a = [1 if A.get(q, {}).get("correct") else 0 for q in t]; b = [1 if B.get(q, {}).get("correct") else 0 for q in t]
w = sum(y > x for x, y in zip(a, b)); l = sum(x > y for x, y in zip(a, b))
p = min(1, 2 * sum(comb(w + l, i) for i in range(min(w, l) + 1)) / 2 ** (w + l)) if w + l else 1
d = [y - x for x, y in zip(a, b)]; se = st.stdev(d) / math.sqrt(len(d))
gr = lambda D: st.mean(D.get(q, {}).get("gold_recall", 0) for q in t)
print(f"test n={len(t)} | BM25 {sum(a)} ({sum(a)/len(t):.3f}) | rerank {sum(b)} ({sum(b)/len(t):.3f}) | "
      f"diff {st.mean(d):+.3f} se {se:.3f} t {st.mean(d)/se:+.2f} | won {w} lost {l} McNemar p={p:.4f} | "
      f"gold recall {gr(A):.3f} -> {gr(B):.3f} | results {sum(q in B for q in t)}/{len(t)}")
