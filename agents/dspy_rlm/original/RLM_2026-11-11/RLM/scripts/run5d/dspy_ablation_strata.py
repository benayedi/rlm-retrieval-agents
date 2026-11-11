# Draws logs/dspy_ablation/sample100.json (2026-10-31). Run from scripts/run5d/; needs per_question.json from
# scripts/failure_analysis_extract.py (path set in dspy_vs_snip.py). Writes ablation_sample100.json in the cwd.
import json, random, collections, math
exec(open("dspy_vs_snip.py").read().split("OUT = {}")[0])   # reuse loaders
row, pats = "5", ["browsecomp_v2medium_baseline_first50.json", "browsecomp_v2medium_baseline_200.json",
                  "browsecomp_v2_medium_q148_q830.json", "browsecomp_v2_medium_retry56.json"]
D = dspy(row, pats); S = {o["qid"]: o for o in PQ if o["sheet"] == "BrowseComp+" and o["row"] == 9}
G5 = {}   # GPT-5 difficulty: how many of the 6 GPT-5 rows solve it
for o in PQ:
    if o["sheet"] == "BrowseComp+" and o["row"] in (9, 11, 13, 21, 23, 25): G5[o["qid"]] = G5.get(o["qid"], 0) + bool(o["correct"])
Q = [q for q in D if q in S]
def stratum(q):
    it = int(float(D[q]["r"].get("iterations") or 0)); d, s = D[q]["ok"], bool(S[q]["correct"])
    if it >= 16: return "S1 DSPy hit turn cap, snippet right" if s else "S2 DSPy hit turn cap, snippet wrong"
    if d != s: return "S3 no cap, DSPy and snippet disagree"
    return "S4 no cap, both right" if d else "S5 no cap, both wrong"
st = {q: stratum(q) for q in Q}; N = collections.Counter(st.values())
alloc = {"S1 DSPy hit turn cap, snippet right": 35, "S2 DSPy hit turn cap, snippet wrong": 15, "S3 no cap, DSPy and snippet disagree": 20,
         "S4 no cap, both right": 20, "S5 no cap, both wrong": 10}
rng = random.Random(20261031); sample = []
for k, n in alloc.items():
    pool = sorted(q for q in Q if st[q] == k); sample += [(q, k) for q in rng.sample(pool, n)]
for k in sorted(N):
    xs = [q for q, kk in sample if kk == k]
    print(f"{k:42s} pop {N[k]:4d} ({N[k]/len(Q):5.1%})  sample {alloc[k]:3d}  weight {N[k]/alloc[k]:5.2f}  "
          f"DSPy-orig acc {sum(D[q]['ok'] for q in xs)/len(xs):.2f}  snippet acc {sum(bool(S[q]['correct']) for q in xs)/len(xs):.2f}  GPT-5 diff mean {sum(G5[q] for q in xs)/len(xs):.1f}")
idx = {q: i for i, q in enumerate(ALL)}
json.dump(dict(seed=20261031, population=len(Q), strata={k: dict(pop=N[k], n=alloc[k]) for k in N},
               questions=[dict(qid=q, q_idx=idx[q], stratum=k) for q, k in sorted(sample, key=lambda x: idx[x[0]])]),
          open("ablation_sample100.json", "w"), indent=1)
# power: McNemar, 2-sided alpha .05, 80% power; p_d = discordant share, delta = accuracy difference
def n_needed(pd, d): return math.ceil(((1.96 * math.sqrt(pd) + 0.84 * math.sqrt(pd - d * d)) / d) ** 2)
print("random-sample n for 80% power:", {f"{d:.0%} (disc {pd:.0%})": n_needed(pd, d) for d, pd in [(0.03, .11), (0.05, .12), (0.08, .15), (0.10, .17), (0.15, .20)]})
