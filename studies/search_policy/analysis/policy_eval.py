"""Search-policy study: score retrieval policies (ColBERT, BM25, RRF fusions, interleaving) offline on the cached lists.

From original/RLM/scripts/bcp_policy_eval.py; changed: import wiring, the input location (RLM_DATA_DIR), this
paragraph. Reads <RLM_DATA_DIR>/bcp_policy/lists.jsonl; no network.
  python analysis/policy_eval.py [--split val|train|all]

Original docstring follows.

Offline evaluation of search policies on cached deep lists (data/bcp_policy/lists.jsonl).

A policy maps (colbert_top100, bm25_top100, query, k) -> ranked doc ids; it is scored at the k the
agent actually asked for, so it changes WHICH k documents the agent sees, not how many.

Metrics (per query, then averaged; val split unless --split):
  hit@k       any gold document among the k returned        (what turns into accuracy)
  recall@k    share of the question's gold documents returned
  new_gold    for the agent's own (non-seed) queries only
Headroom: gold in ColBERT top-100 but not in top-k -- the most a re-ranking/fusion policy can recover.
"""
import argparse, json, os, statistics as st, sys

# Import wiring: agents/shared/rlm_shared is located from this file; locations come from configs/paths.env
# (rlm_shared.paths.path). Original: absolute paths of the machine the experiments ran on.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir, "agents", "shared"))
from rlm_shared.paths import path  # noqa: E402
REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir))


def rrf(lists, k=10, weights=None):
    sc = {}
    for li, l in enumerate(lists):
        w = (weights or [1.0] * len(lists))[li]
        for r, d in enumerate(l):
            sc[d] = sc.get(d, 0.0) + w / (k + r + 1)
    return [d for d, _ in sorted(sc.items(), key=lambda x: -x[1])]


POLICIES = {
    "colbert (production)": lambda c, b, q, k: c,
    "bm25": lambda c, b, q, k: b,
    "rrf k=10 (1:1)": lambda c, b, q, k: rrf([c, b], 10),
    "rrf k=60 (1:1)": lambda c, b, q, k: rrf([c, b], 60),
    "rrf k=10 (colbert 2:1)": lambda c, b, q, k: rrf([c, b], 10, [2, 1]),
    "interleave c,b": lambda c, b, q, k: [d for pair in zip(c, b) for d in pair],
    "colbert top-(k-2) + bm25 top-2 new": lambda c, b, q, k: c[:max(1, k - 2)] + [d for d in b if d not in set(c[:max(1, k - 2)])][:2],
}


def dedup(l):
    seen, out = set(), []
    for d in l:
        if d not in seen:
            seen.add(d); out.append(d)
    return out


def score(policy, recs):
    hit, rec, new = [], [], []
    for r in recs:
        k = max(1, int(r.get("agent_k") or 5))
        top = dedup(policy(r["colbert"], r["bm25"], r["query"], k))[:k]
        g = set(r["gold"])
        h = bool(g & set(top)); hit.append(h); rec.append(len(g & set(top)) / len(g) if g else 0)
        if not r.get("seed_search"):
            new.append(h)
    return st.mean(hit), st.mean(rec), (st.mean(new) if new else 0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="val"); a = ap.parse_args()
    recs = [json.loads(l) for l in open(os.path.join(str(path("RLM_DATA_DIR")), "bcp_policy", "lists.jsonl"))]
    recs = [r for r in recs if a.split == "all" or r["split"] == a.split]
    print(f"{len(recs)} cached queries ({a.split}), {len({r['qid'] for r in recs})} questions, "
          f"median agent k {st.median(int(r.get('agent_k') or 5) for r in recs)}")
    head = [bool(set(r['gold']) & set(r['colbert'])) and not (set(r['gold']) & set(r['colbert'][:max(1, int(r.get('agent_k') or 5))])) for r in recs]
    bm_only = [bool(set(r['gold']) & set(r['bm25'][:max(1, int(r.get('agent_k') or 5))])) and not (set(r['gold']) & set(r['colbert'][:max(1, int(r.get('agent_k') or 5))])) for r in recs]
    print(f"HEADROOM: gold in ColBERT top-100 but not top-k: {st.mean(head):.3f} of queries | "
          f"gold in BM25 top-k but not ColBERT top-k: {st.mean(bm_only):.3f}")
    print(f"{'policy':38s} {'hit@k':>7s} {'recall@k':>9s} {'hit@k agent-written':>20s}")
    for n, p in POLICIES.items():
        h, rc, nw = score(p, recs)
        print(f"{n:38s} {h:7.3f} {rc:9.3f} {nw:20.3f}")


if __name__ == "__main__":
    main()
