"""Search-policy study, SP-6 vs TWQ-1 on all 281 OBLIQ-Twitter questions (NDCG@10 from the shard result files). The
same report also pairs two arms that are not migrated (dspy.RLM on Qwen3.6, the OBLIQ paper's multi-hop agent
re-implemented); with their logs absent they print "nothing yet".

From original/RLM/scripts/obliq_q36_next_report.py; changed: locations from configs/paths.env, this paragraph.
Reads <LOGS_ROOT>/obliq_mine_rr (ctrl_dev100, snip_rest181_g*, rr_dev100, rr_rest181_g*),
<RLM_DATA_DIR>/obliq_fd/all281.json and studies/prompt_optimisation/splits/obliq_splits/twitter_dev.json.
  python analysis/compare_obliq_snippet_281.py [--final]

Original docstring follows.

OBLIQ Twitter (281), Qwen3.6, 2026-10-21: two arms paired against the snippet RLM's full 281 run (0.0759).
  RR   = snippet RLM + novelty + Qwen3-Reranker-4B proxy (:8775); dev100 from 10-13 (logs/obliq_mine_rr/rr_dev100.json),
         rest181 from 10-21 (rr_rest181_g*_s*.json). Everything else identical to the snippet run.
  DSPY / DSPYFIX = dspy.RLM implicit stance agent (examples/obliq_eval_dspy_rlm.py, ObliqTwitterSig, 12 iters / 8 searches / top-k 20)
         on Qwen3.6 with its sub-LM local (LOCAL_SUB_LM=1) and presence_penalty 1.5 (logs/obliq_q36_next/dspy_q36_281_s*, as-is, stopped at 184;
         dspy_q36fix_281_s* with DSPY_FORMAT_FIX=1).
Reads checkpoints too, so it works mid-run (paired over the questions done). Missing = 0 only in the final report.
  python scripts/obliq_q36_next_report.py [--final]
"""
import glob, json, math, os, statistics as st, sys

# Import wiring: agents/shared/rlm_shared is located from this file; locations come from configs/paths.env
# (rlm_shared.paths.path). Original: absolute paths of the machine the experiments ran on.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir, "agents", "shared"))
from rlm_shared.paths import path  # noqa: E402
REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir))
L = f"{path('LOGS_ROOT')}/obliq_mine_rr"   # original: <home>/RLM/logs/obliq_mine_rr
N = f"{path('LOGS_ROOT')}/obliq_q36_next"
ALL = [str(x) for x in json.load(open(f"{path('RLM_DATA_DIR')}/obliq_fd/all281.json"))]
DEV = set(str(x) for x in json.load(open(os.path.join(REPO, "studies", "prompt_optimisation", "splits", "obliq_splits", "twitter_dev.json"))))  # original: <original repo>/data/obliq_splits/twitter_dev.json
FINAL = "--final" in sys.argv


def snip_rows(files):
    o = {}
    for f in files:
        try:
            for x in json.load(open(f)):
                o.setdefault(str(x["query_id"]), x)
        except Exception:
            pass
    return o


def finals(pattern):  # final shard files first, then checkpoints for questions not yet in a final file
    fs = sorted(glob.glob(pattern))
    return [f for f in fs if not f.endswith("_checkpoint.json")] + [f for f in fs if f.endswith("_checkpoint.json")]


SNIP = snip_rows([f"{L}/ctrl_dev100.json"] + finals(f"{L}/snip_rest181_g*_s*.json"))
RR = snip_rows([f"{L}/rr_dev100.json"] + finals(f"{L}/rr_rest181_g*_s*.json"))


def dspy_rows(prefix):
    o = {}
    for f in finals(f"{N}/{prefix}_s*.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for r in (d.get("results", []) if isinstance(d, dict) else d):
            o.setdefault(str(r["q_id"]), r)
    return o


DSPY = dspy_rows("dspy_q36_281")        # as-is: stopped at 184/281 on 10-21 (18 AdapterParseError zeros)
DSPYFIX = dspy_rows("dspy_q36fix_281")  # DSPY_FORMAT_FIX=1: reasoning/code cue + parse retry
# MH = the paper's 4-hop agent re-implemented on our retriever + Qwen3.6 (scripts/obliq_multihop.py)
MH = {}
if os.path.exists(f"{path('LOGS_ROOT')}/obliq_multihop/q36.jsonl"):
    for l in open(f"{path('LOGS_ROOT')}/obliq_multihop/q36.jsonl"):
        r = json.loads(l); MH[r["query_id"]] = dict(r, latency_s=r.get("sec"))

nd = {"SNIP": lambda r: r.get("ndcg@10") or 0.0, "RR": lambda r: r.get("ndcg@10") or 0.0,
      "DSPY": lambda r: (r.get("answer_ndcg") or {}).get("10", 0.0) or 0.0}
nd["DSPYFIX"] = nd["DSPY"]; nd["MH"] = nd["SNIP"]
lat = {k: (lambda r: r.get("latency_s")) for k in ("SNIP", "RR", "DSPY", "DSPYFIX", "MH")}
print(f"snippet RLM reference: {len(SNIP)}/281, NDCG@10 {st.mean(nd['SNIP'](SNIP.get(q, {})) for q in ALL):.4f}")
for name, D in (("RR", RR), ("DSPY", DSPY), ("DSPYFIX", DSPYFIX), ("MH", MH)):
    ks = ALL if FINAL and name != "DSPY" else [q for q in ALL if q in D]
    if not ks:
        print(f"{name}: nothing yet"); continue
    a = [nd["SNIP"](SNIP.get(q, {})) for q in ks]; b = [nd[name](D.get(q, {})) for q in ks]
    d = [y - x for x, y in zip(a, b)]; se = st.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
    w = sum(y > x for x, y in zip(a, b)); l_ = sum(x > y for x, y in zip(a, b))
    ls = [lat[name](D[q]) for q in ks if q in D and lat[name](D[q])]
    dv = [q for q in ks if q in DEV]; rs = [q for q in ks if q not in DEV]
    part = lambda qs: st.mean(nd[name](D.get(q, {})) for q in qs) if qs else float("nan")
    print(f"{name}: {sum(q in D for q in ALL)}/281 done | paired n={len(ks)}: {st.mean(b):.4f} vs snippet {st.mean(a):.4f} "
          f"diff {st.mean(d):+.4f} (t {st.mean(d)/se:+.2f}) won/lost {w}/{l_} | dev100 {part(dv):.4f} rest {part(rs):.4f} | "
          f"mean latency {st.mean(ls):.0f} s" if ls else "")
