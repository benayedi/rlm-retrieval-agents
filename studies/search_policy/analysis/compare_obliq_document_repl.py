"""Search-policy study, SP-7 vs TWQ-2: OBLIQ-Twitter (281), document REPL agent, Qwen3.6, plain dense search vs the
novelty + reranker proxy. NDCG@10 is recomputed from each record's ranking (no judge).

From original/RLM/scripts/obliq_rr_compare.py; changed: import wiring (examples.obliq_data is
agents/shared/rlm_shared/obliq_data.py, which reads OBLIQ_BENCH_ROOT), locations from configs/paths.env, this
paragraph. Defaults: <FULLDOC_ROOT>/runs/obliq_q36_ctrl and <FULLDOC_ROOT>/runs/obliq_q36_rr; dev100 split
studies/prompt_optimisation/splits/obliq_splits/twitter_dev.json.
  python analysis/compare_obliq_document_repl.py [ctrl_dir] [treat_dir]

Original docstring follows.

OBLIQ Twitter, supervisor harness + Qwen3.6: plain dense search (:8767) vs novelty+rerank proxy (:8774).

Paired over questions finished in both arms; timeouts/errors score NDCG 0 (they are reported too).
Usage: python scripts/obliq_rr_compare.py [ctrl_dir] [treat_dir]
"""
import collections, glob, json, math, os, statistics, sys
# Import wiring: agents/shared/rlm_shared is located from this file; locations come from configs/paths.env
# (rlm_shared.paths.path). Original: absolute paths of the machine the experiments ran on.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir, "agents", "shared"))
from rlm_shared.paths import path  # noqa: E402
REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir))
from rlm_shared import obliq_data as OD  # noqa: E402  (original: from examples import obliq_data)

QR = OD.load_qrels('descriptive', 'twitter')
DEV = set(json.load(open(os.path.join(REPO, "studies", "prompt_optimisation", "splits", "obliq_splits", "twitter_dev.json"))))  # original: <original repo>/data/obliq_splits/twitter_dev.json
C = sys.argv[1] if len(sys.argv) > 1 else f"{path('FULLDOC_ROOT')}/runs/obliq_q36_ctrl"
T = sys.argv[2] if len(sys.argv) > 2 else f"{path('FULLDOC_ROOT')}/runs/obliq_q36_rr"


def ndcg10(ranked, gold):
    d = sum(1 / math.log2(i + 2) for i, x in enumerate(ranked[:10]) if x in gold)
    i_ = sum(1 / math.log2(i + 2) for i in range(min(10, len(gold))))
    return d / i_ if i_ else 0.0


def load(d):
    o = {}
    for f in glob.glob(d + '/q*.json'):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        q = r['query_id']; g = set(map(str, QR.get(q, set())))
        if not g:
            continue
        res = (r.get('result') or [{}])[0].get('output', '') or ''
        ranked = [l.strip() for l in res.splitlines() if l.strip().isdigit()]
        got = set(map(str, r.get('retrieved_docids') or []))
        o[q] = dict(nd=ndcg10(ranked, g) if r.get('status') == 'completed' else 0.0,
                    lat=r.get('wall_time_s') or 0, it=r.get('n_iterations') or 0,
                    srch=(r.get('tool_call_counts') or {}).get('search', 0), nsub=len(ranked),
                    gr=len(g & got) / len(g), status=r.get('status'))
    return o


c, t = load(C), load(T)
for name, sel in [('ALL', lambda q: True), ('dev100', lambda q: q in DEV), ('heldout', lambda q: q not in DEV)]:
    k = sorted(q for q in set(c) & set(t) if sel(q))
    if len(k) < 2:
        continue
    d = [t[q]['nd'] - c[q]['nd'] for q in k]
    se = statistics.stdev(d) / math.sqrt(len(d)) or 1e-9
    print(f"== {name}: paired n={len(k)}  (done: ctrl {len(c)}, rerank {len(t)})")
    print(f"  {'':16s} {'dense':>9s} {'nov+rerank':>11s}")
    for lab, key in [('NDCG@10', 'nd'), ('gold recall', 'gr'), ('searches', 'srch'),
                     ('iterations', 'it'), ('ids submitted', 'nsub'), ('latency s', 'lat')]:
        print(f"  {lab:16s} {statistics.mean(c[q][key] for q in k):9.4f} {statistics.mean(t[q][key] for q in k):11.4f}")
    dg = [t[q]['gr'] - c[q]['gr'] for q in k]
    print(f"  delta NDCG {statistics.mean(d):+.4f}  se {se:.4f}  t {statistics.mean(d)/se:+.2f}  "
          f"w/l/tie {sum(x > 1e-9 for x in d)}/{sum(x < -1e-9 for x in d)}/{sum(abs(x) <= 1e-9 for x in d)}")
    sg = statistics.stdev(dg) / math.sqrt(len(dg)) or 1e-9
    print(f"  delta recall {statistics.mean(dg):+.4f}  t {statistics.mean(dg)/sg:+.2f}")
    print(f"  status ctrl {dict(collections.Counter(c[q]['status'] for q in k))}  "
          f"rerank {dict(collections.Counter(t[q]['status'] for q in k))}")
print("refs (dev100): 27B supervisor harness 0.1248; our harness+27B 0.1302; paper multi-hop 0.141")
