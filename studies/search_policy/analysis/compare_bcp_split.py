"""Search-policy study: paired comparison of one split, plain ColBERT (control) vs the proxy arm, from the run caches.

From original/RLM/scripts/bcp_policy_compare.py; changed: import wiring, locations from configs/paths.env, this
paragraph. The verdicts it reads are the in-run Qwen3.6 judge's (superseded; the reported numbers are the gpt-4.1
re-judge in results/follow_up/verdicts/). Defaults: control <LOGS_ROOT>/gepa_bcp2/cache.json under key ee5ea32a1c
(BCP-29 on test200), treatment <LOGS_ROOT>/bcp_policy_e2e/cache.json (SP-1 test200). CTRL_CACHE / TREAT_CACHE given
as relative paths are read relative to the current directory, as in the original.
  [SPLIT=test_rest CTRL_CACHE=<cache.json> TREAT_CACHE=<cache.json>] python analysis/compare_bcp_split.py

Original docstring follows.

End-to-end: original prompt + novelty/rerank search proxy vs original prompt + plain ColBERT, on test200.

Control  = cached rollouts of the original prompt on test200 (logs/gepa_bcp2, run during the run-2
           confirmation, 2026-10-09/10; retrieval :8766 ColBERT).
Treatment= the same prompt, model, harness settings and judge, with --server pointed at
           scripts/bcp_rerank_proxy.py (POLICY=novelty+rerank, Qwen3-Reranker-4B, depth 30).
Paired per question; timeouts / kills / empty submissions count as WRONG.
Caveat: the arms ran at different times (server load differs), which only affects wall-clock timeouts.
"""
import json, math, statistics as st

import os
import sys
# Import wiring: agents/shared/rlm_shared is located from this file; locations come from configs/paths.env
# (rlm_shared.paths.path). Original: absolute paths of the machine the experiments ran on.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir, "agents", "shared"))
from rlm_shared.paths import path  # noqa: E402
REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir))
# BrowseComp-Plus question splits (D_pareto / D_feedback / test200 / test_rest): GEPA_SPLIT, default the copy in
# studies/prompt_optimisation/splits/ (original: <original repo>/data/bcp_gepa_splits/split.json).
SPLIT_F = os.environ.get("GEPA_SPLIT") or os.path.join(REPO, "studies", "prompt_optimisation", "splits", "bcp_gepa_splits", "split.json")
# SPLIT / CTRL_CACHE / TREAT_CACHE env: e.g. SPLIT=test_rest CTRL_CACHE=logs/bcp_testrest_ctrl/cache.json
#   TREAT_CACHE=logs/bcp_testrest_rr/cache.json (both arms run concurrently, 2026-10-11)
test = json.load(open(SPLIT_F))[os.environ.get("SPLIT", "test200")]


def arm(path, key=None):
    D = json.load(open(path))
    if key is None:
        keys = {k.split("|")[0] for k in D if k.split("|")[1] in set(test)}
        assert len(keys) == 1, keys
        key = keys.pop()
    return key, {q: D.get(f"{key}|{q}") for q in test}


_, ctrl = arm(os.environ["CTRL_CACHE"]) if "CTRL_CACHE" in os.environ else arm(f"{path('LOGS_ROOT')}/gepa_bcp2/cache.json", "ee5ea32a1c")
tk, trt = arm(os.environ.get("TREAT_CACHE", f"{path('LOGS_ROOT')}/bcp_policy_e2e/cache.json"))
ok = lambda r: r and r.get("status") != "killed"
both = [q for q in test if ok(ctrl[q]) and ok(trt[q])]
acc = lambda r: 1.0 if r.get("correct") else 0.0
rec = lambda r: r.get("gold_recall") or 0.0
print(f"paired n={len(both)} of {len(test)} (treatment key {tk})")
for nm, f in (("accuracy", acc), ("gold recall", rec)):
    d = [f(trt[q]) - f(ctrl[q]) for q in both]
    se = st.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else 0
    print(f"  {nm:12s} control {st.mean(f(ctrl[q]) for q in both):.3f}  proxy {st.mean(f(trt[q]) for q in both):.3f}  "
          f"diff {st.mean(d):+.3f} (se {se:.3f}, t {st.mean(d)/se if se else 0:+.2f})")
w = sum(1 for q in both if acc(trt[q]) > acc(ctrl[q])); l = sum(1 for q in both if acc(trt[q]) < acc(ctrl[q]))
to = lambda X: sum(1 for q in both if X[q].get("status") not in ("completed", "empty_output"))
srch = lambda X: st.mean((X[q].get("searches") or 0) for q in both)
print(f"  won/lost {w}/{l} | timeouts: control {to(ctrl)} proxy {to(trt)} | searches/q: control {srch(ctrl):.1f} proxy {srch(trt):.1f}")
