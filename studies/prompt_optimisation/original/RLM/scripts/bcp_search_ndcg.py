"""Per-search nDCG@10 of the agent's OWN queries, per GEPA candidate, paired against the seed.

BrowseComp-Plus reports nDCG@10 only for retrievers (one query = the question). An agent
issues many queries, so we score each one: the ranked docids in search_log[i] vs the gold and
the evidence docs (binary relevance, the paper's Table-2 convention), then average per
question. The FIRST search is the harness's seed search (the question verbatim) and is
identical for every prompt, so it is excluded -- what remains measures how good the queries
the agent itself wrote are, which accuracy and gold recall only see indirectly.

Usage: bcp_search_ndcg.py   (reads logs/gepa_bcp/rollouts + cache; D_pareto questions only)
"""
import glob, json, math, os, statistics as st, sys

R = "/home/extstu_00/RLM"
sys.path.insert(0, R)
from examples.browsecomp_data import get_gold_doc_ids, get_evidence_doc_ids  # noqa: E402

SEED = "ee5ea32a1c"


def ndcg10(ranked, rel):
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked[:10]) if d in rel)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(10, len(rel))))
    return dcg / idcg if idcg else 0.0


def main():
    from datasets import load_dataset
    rows = {str(r["query_id"]): r for r in load_dataset("Tevatron/browsecomp-plus", split="test")}
    par = set(json.load(open(f"{R}/data/bcp_gepa_splits/split.json"))["pareto"])
    per = {}   # cand -> qid -> (gold ndcg, evid ndcg, best gold ndcg, n searches)
    for d in glob.glob(f"{R}/logs/gepa_bcp/rollouts/*/"):
        k = os.path.basename(d.rstrip("/"))
        for f in glob.glob(f"{d}*.json"):
            q = os.path.basename(f)[:-5]
            if q not in par:
                continue
            log = (json.load(open(f)).get("search_log") or [])[1:]   # drop the seed search
            if not log:
                continue
            g = set(map(str, get_gold_doc_ids(rows[q]))); e = set(map(str, get_evidence_doc_ids(rows[q])))
            gs = [ndcg10([str(x) for x in s.get("docids") or []], g) for s in log]
            es = [ndcg10([str(x) for x in s.get("docids") or []], e) for s in log]
            per.setdefault(k, {})[q] = (st.mean(gs), st.mean(es), max(gs), len(log))
    S = per.get(SEED, {})
    print(f"{'candidate':<12} {'n':>3} {'nDCG@10 gold':>13} {'evidence':>9} {'best/q gold':>12} {'agent searches':>15}   paired vs seed (gold, mean per search)")
    for k, v in sorted(per.items(), key=lambda kv: (kv[0] != SEED, -len(kv[1]))):
        if len(v) < 20:
            continue
        m = lambda i: st.mean(x[i] for x in v.values())
        line = f"{k:<12} {len(v):>3} {m(0):>13.3f} {m(1):>9.3f} {m(2):>12.3f} {m(3):>15.1f}"
        if k != SEED:
            qs = [q for q in v if q in S]
            dd = [v[q][0] - S[q][0] for q in qs]
            se = st.stdev(dd) / math.sqrt(len(dd)) if len(dd) > 1 else 0
            line += f"   n={len(qs)} diff {st.mean(dd):+.3f} (t {st.mean(dd)/se if se else 0:+.2f})"
        print(line)


if __name__ == "__main__":
    main()
