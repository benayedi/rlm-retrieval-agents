"""Per-search nDCG@10 of the agent's OWN queries, per GEPA candidate, paired against the seed.

Refactored copy of original/RLM/scripts/bcp_search_ndcg.py (studies/prompt_optimisation/CHANGES_FROM_ORIGINAL.md).
Run: python src/bcp_search_ndcg.py (GEPA-1 rollouts under RUNS_DIR/gepa_bcp; needs the BrowseComp-Plus dataset from
  Hugging Face, decrypted in memory only).
Changed: import wiring and paths.
Names: 'supervisor harness' / 'full-doc RLM' in the text below is the document REPL agent (agents/document_repl);
  'official RLM harness' / 'our harness' / run_obliq_mine.py is the snippet REPL agent (agents/snippet_repl,
  src/run_obliq.py).

BrowseComp-Plus reports nDCG@10 only for retrievers (one query = the question). An agent
issues many queries, so we score each one: the ranked docids in search_log[i] vs the gold and
the evidence docs (binary relevance, the paper's Table-2 convention), then average per
question. The FIRST search is the harness's seed search (the question verbatim) and is
identical for every prompt, so it is excluded -- what remains measures how good the queries
the agent itself wrote are, which accuracy and gold recall only see indirectly.

Usage: bcp_search_ndcg.py   (reads logs/gepa_bcp/rollouts + cache; D_pareto questions only)
"""
import glob, json, math, os, statistics as st, sys

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
from rlm_shared.browsecomp_data import get_gold_doc_ids, get_evidence_doc_ids  # noqa: E402  (original: examples.browsecomp_data)

SEED = "ee5ea32a1c"


def ndcg10(ranked, rel):
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked[:10]) if d in rel)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(10, len(rel))))
    return dcg / idcg if idcg else 0.0


def main():
    from datasets import load_dataset
    rows = {str(r["query_id"]): r for r in load_dataset("Tevatron/browsecomp-plus", split="test")}
    par = set(json.load(open(f"{SPLITS}/bcp_gepa_splits/split.json"))["pareto"])
    per = {}   # cand -> qid -> (gold ndcg, evid ndcg, best gold ndcg, n searches)
    for d in glob.glob(f"{RUNS}/gepa_bcp/rollouts/*/"):
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
