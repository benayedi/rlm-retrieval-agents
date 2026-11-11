"""Search-policy study, SP-6 vs TWQ-1 on the 100 dev questions: OBLIQ-Twitter, snippet REPL agent, Qwen3.6, plain
dense search vs the novelty + reranker proxy (WIRE=mine), read from the shard result files.

From original/RLM/scripts/obliq_mine_rr_compare.py; changed: locations from configs/paths.env, this paragraph.
Default directory <LOGS_ROOT>/obliq_mine_rr; the 10-04 reference run is <LOGS_ROOT>/q36_base_*_s*.json (not part of
this repository; skipped when absent).
  python analysis/compare_obliq_snippet_dev100.py [dir]

Original docstring follows.

OBLIQ Twitter dev100, OUR harness (run_obliq_mine) + Qwen3.6 + presence penalty:
plain dense search (:8010) vs novelty+rerank proxy (:8775, WIRE=mine). Paired over questions done in both.
Reads the per-question checkpoints, so it works mid-run. Errors score NDCG 0 (as the harness records them).
Usage: python scripts/obliq_mine_rr_compare.py [dir]
"""
import glob, json, math, os, statistics, sys

# Import wiring: agents/shared/rlm_shared is located from this file; locations come from configs/paths.env
# (rlm_shared.paths.path). Original: absolute paths of the machine the experiments ran on.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir, "agents", "shared"))
from rlm_shared.paths import path  # noqa: E402
REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir))

D = sys.argv[1] if len(sys.argv) > 1 else f"{path('LOGS_ROOT')}/obliq_mine_rr"


def load(pattern):
    o = {}
    for f in glob.glob(pattern):
        try:
            for r in json.load(open(f)):
                o[r['query_id']] = r
        except Exception:
            pass
    return o


def arm(name):
    o = load(f'{D}/{name}_dev100_s*_checkpoint.json')
    o.update(load(f'{D}/{name}_dev100_s[0-9].json'))
    return o


c, t = arm('ctrl'), arm('rr')
old = load(f"{path('LOGS_ROOT')}/q36_base_*_s*.json")      # 10-04 run: same harness, no presence penalty
k = sorted(set(c) & set(t))
print(f"done: ctrl {len(c)}/100, rerank {len(t)}/100, paired {len(k)}")
if len(k) >= 2:
    rows = [('NDCG@10', 'ndcg@10'), ('recall@10', 'recall@10'), ('gold recall', 'gold_recall'),
            ('searches', 'search_calls'), ('get_document', 'get_document_calls'), ('LM calls', 'lm_calls'),
            ('ids parsed', 'n_parsed'), ('latency s', 'latency_s')]
    print(f"  {'':14s} {'dense':>9s} {'nov+rerank':>11s}")
    for lab, key in rows:
        m = lambda a: statistics.mean(float(a[q].get(key) or 0) for q in k)
        print(f"  {lab:14s} {m(c):9.4f} {m(t):11.4f}")
    for lab, key in [('NDCG@10', 'ndcg@10'), ('gold recall', 'gold_recall')]:
        d = [float(t[q].get(key) or 0) - float(c[q].get(key) or 0) for q in k]
        se = (statistics.stdev(d) / math.sqrt(len(d))) or 1e-9
        print(f"  delta {lab}: {statistics.mean(d):+.4f}  se {se:.4f}  t {statistics.mean(d)/se:+.2f}  "
              f"w/l/tie {sum(x > 1e-9 for x in d)}/{sum(x < -1e-9 for x in d)}/{sum(abs(x) <= 1e-9 for x in d)}")
    print(f"  errors: ctrl {sum(1 for q in k if c[q].get('error'))}  rerank {sum(1 for q in k if t[q].get('error'))}")
ko = sorted(set(old) & set(c))
if ko:
    print(f"  ref: 10-04 run (no presence penalty) on the same {len(ko)} q: "
          f"{statistics.mean(old[q].get('ndcg@10', 0) for q in ko):.4f} vs ctrl now "
          f"{statistics.mean(c[q].get('ndcg@10', 0) for q in ko):.4f}")
print("refs (dev100): Qwen3.6 10-04 0.0780 | sup harness Qwen3.6 0.048 dense / 0.070 rerank | 27B ours 0.1302")
