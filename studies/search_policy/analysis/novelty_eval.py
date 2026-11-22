"""Search-policy study: offline estimate of the novelty filter (drop documents the run was already shown).

From original/RLM/scripts/bcp_novelty_eval.py; changed: import wiring, locations from configs/paths.env, this
paragraph. Runs at import (a script). Reads the rollouts named in policy_replay.py and
<RLM_DATA_DIR>/bcp_policy/lists.jsonl; writes <RLM_DATA_DIR>/bcp_policy/hist.json (read by rerank_eval.py --novelty).
  python analysis/novelty_eval.py

Original docstring follows.

Offline estimate of a NOVELTY search policy: return the top-k documents the agent has NOT seen yet.

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
import glob, json, os, statistics as st, sys

# Import wiring: agents/shared/rlm_shared is located from this file; locations come from configs/paths.env
# (rlm_shared.paths.path). Original: absolute paths of the machine the experiments ran on.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir, "agents", "shared"))
from rlm_shared.paths import path  # noqa: E402
REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir))
POLICY_DIR = os.path.join(str(path("RLM_DATA_DIR")), "bcp_policy")  # original: <original repo>/data/bcp_policy
SRC = [f"{path('FULLDOC_ROOT')}/runs/qwen27b_colbert8192_first20/*.json",   # original: <home>/suprlm/runs/...
       f"{path('FULLDOC_ROOT')}/runs/qwen27b_colbert8192_rest810/*.json",
       f"{path('LOGS_ROOT')}/gepa_bcp/rollouts/*/*.json", f"{path('LOGS_ROOT')}/gepa_bcp2/rollouts/*/*.json"]  # original: f"{R}/logs/..."

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

recs = [json.loads(l) for l in open(f"{POLICY_DIR}/lists.jsonl")]
# save the per-query "already seen" sets for the cached queries (re-used by bcp_rerank_eval.py --novelty)
json.dump({f"{r['qid']}\t{r['query']}": sorted(hist.get((r["qid"], r["query"]), set())) for r in recs},
          open(f"{POLICY_DIR}/hist.json", "w"))
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
