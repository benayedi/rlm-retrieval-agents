"""Offline estimate of a NOVELTY search policy: return the top-k documents the agent has NOT seen yet.

Measured 2026-10-10 (Qwen3.6, original prompt, 2,273 agent-written searches): 33% of the documents a
search returns were already returned earlier in the same run (median 20%; 32% of searches are
>= half repeats). Under design B every returned doc is a full page, so repeats are wasted slots.

For each cached query (data/bcp_policy/lists.jsonl) we rebuild the docs already returned earlier in
the SAME rollout (from the raw rollout files) and compare, at the agent's k:
  production : ColBERT top-k as returned (repeats included)
  novelty    : first k ColBERT docs (from top-100) not yet seen in this run
Metric: does the search surface a NEW gold doc (a gold doc not already seen)? -- the event that adds
evidence. Off-policy caveat: later searches would differ in reality; this is per-search only.
"""
import glob, json, statistics as st

R = "/home/extstu_00/RLM"
SRC = ["/home/extstu_00/suprlm/runs/qwen27b_colbert8192_first20/*.json",
       "/home/extstu_00/suprlm/runs/qwen27b_colbert8192_rest810/*.json",
       f"{R}/logs/gepa_bcp/rollouts/*/*.json", f"{R}/logs/gepa_bcp2/rollouts/*/*.json"]

hist = {}   # (qid, query) -> docs returned earlier in the first rollout where this query appears
for pat in SRC:
    for f in glob.glob(pat):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        q = str(r.get("query_id")); seen = set()
        for s in r.get("search_log") or []:
            key = (q, str(s.get("query") or "").strip())
            hist.setdefault(key, set(seen))
            seen.update(str(x) for x in s.get("docids") or [])

recs = [json.loads(l) for l in open(f"{R}/data/bcp_policy/lists.jsonl")]
# save the per-query "already seen" sets for the cached queries (re-used by bcp_rerank_eval.py --novelty)
json.dump({f"{r['qid']}\t{r['query']}": sorted(hist.get((r["qid"], r["query"]), set())) for r in recs},
          open(f"{R}/data/bcp_policy/hist.json", "w"))
prod, nov, n_rep = [], [], []
for r in recs:
    if r.get("seed_search"):
        continue
    seen = hist.get((r["qid"], r["query"]), set())
    k = max(1, int(r.get("agent_k") or 5)); g = set(r["gold"]) - seen
    p = r["colbert"][:k]; n = [d for d in r["colbert"] if d not in seen][:k]
    prod.append(bool(g & set(p))); nov.append(bool(g & set(n)))
    n_rep.append(sum(1 for d in p if d in seen) / len(p))
print(f"agent-written cached searches: {len(prod)} | repeats in production top-k: {st.mean(n_rep):.3f}")
print(f"surfaces a NEW gold doc:  production {st.mean(prod):.3f}  |  novelty {st.mean(nov):.3f}  "
      f"(+{st.mean(nov)-st.mean(prod):.3f}, {sum(nov)-sum(prod):+d} searches)")
