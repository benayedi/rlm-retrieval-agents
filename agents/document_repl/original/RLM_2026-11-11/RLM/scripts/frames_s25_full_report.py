"""All 824 FRAMES questions: supervisor RLM agent with 25 searches / 50 turns (three dirs merged) vs the
10-search baseline (logs/frames/sup_q36_bm25f) and single-shot BM25-4 / oracle. Missing = wrong."""
import json, math, os, statistics as st
from math import comb
R = "/home/extstu_00/RLM"
def g(d):
    p = f"{R}/logs/frames/{d}/graded.json"
    return {v["qid"]: v for v in json.load(open(p)).values()} if os.path.exists(p) else {}
S = {**g("sup_q36_bm25f_s25"), **g("sup_q36_bm25f_s25_rest0"), **g("sup_q36_bm25f_s25_rest1")}
B = g("sup_q36_bm25f")
qs = [str(q) for q in range(824)]
a = [1 if B.get(q, {}).get("correct") else 0 for q in qs]; b = [1 if S.get(q, {}).get("correct") else 0 for q in qs]
w = sum(y > x for x, y in zip(a, b)); l = sum(x > y for x, y in zip(a, b))
p = min(1, 2 * sum(comb(w + l, i) for i in range(min(w, l) + 1)) / 2 ** (w + l))
d = [y - x for x, y in zip(a, b)]; se = st.stdev(d) / math.sqrt(len(d))
acc = sum(b) / 824
print(f"25-search agent: results {len(S)}/824 | accuracy {sum(b)}/824 = {acc:.3f} (+-{1.96*math.sqrt(acc*(1-acc)/824):.3f}) | "
      f"gold recall {st.mean(S[q]['gold_recall'] for q in S):.3f} | searches {st.mean(S[q]['searches'] for q in S):.1f}")
print(f"10-search agent: {sum(a)}/824 = {sum(a)/824:.3f} | paired diff {st.mean(d)*100:+.1f} pts (t {st.mean(d)/se:+.2f}), won {w} lost {l}, McNemar p={p:.4f}")
print("paper (Gemini-Pro-1.5): BM25-4 0.474, multi-step planning 0.66; ours same model: BM25-4 0.477, oracle 0.774")
