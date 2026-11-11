"""OBLIQ-Twitter, Qwen3.6: pick the best agent version on the 100 dev questions, then report all 281 (TWQ-1).

From original/RLM_2026-11-11/RLM/scripts/obliq_best_full.py; changed: import wiring, locations from configs/paths.env
(LOGS_ROOT for the snippet REPL runs, FULLDOC_ROOT/runs for the document REPL runs, the repository copy of
twitter_dev.json), this paragraph (../CHANGES_FROM_ORIGINAL.md is not changed; the mapping is in
studies/search_policy/CHANGES_FROM_ORIGINAL.md). On 2026-10-20 `pick` chose SNIP (the snippet REPL agent, 12 turns),
which is then run on the other 181 questions (experiments/obliq_twitter_qwen36.sh ARM=ctrl SLICE=rest181) and reported
with `report SNIP`. Versions A and B are document REPL agent runs that are not part of this repository.
  python src/obliq_pick_best.py pick [A B SNIP ...]      python src/obliq_pick_best.py report SNIP

Original docstring follows ("full-doc RLM" = the document REPL agent, "snippet RLM" = the snippet REPL agent).

OBLIQ Twitter (281), Qwen3.6: pick the best version on dev100 and report the full 281.
  pick [V ...] -> prints the best of the given versions (default A B SNIP) by NDCG@10 on dev100; ties go to
                  the earlier (cheaper) one
  report  -> NDCG@10 on all 281 for the chosen version (dev100 results + rest181 results), missing = 0
Versions: A = full-doc RLM + format cue, matched budget, 12 turns (suprlm/runs/obliq_q36_fdA_*)
          B = same, 20 turns (obliq_q36_fdB_*)
          SNIP = snippet RLM, 12 turns (logs/obliq_mine_rr/ctrl_dev100.json; rest in logs/obliq_mine_rr/snip_rest181_*)
          SNIP20 = snippet RLM, 20 turns (only run if B beats A; logs/obliq_mine_rr/snip20_dev100_*, snip20_rest181_*)"""
import glob, json, math, os, statistics as st, sys
# Import wiring: agents/shared/rlm_shared located from this file (original: the original repository root on sys.path
# for examples.obliq_data, which reads the OBLIQ-Bench data from OBLIQ_BENCH_ROOT here).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "shared"))
from rlm_shared.paths import path  # noqa: E402
from rlm_shared import obliq_data as OD  # noqa: E402
REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir))
QR = OD.load_qrels("descriptive", "twitter")
ALL = [str(x) for x in json.load(open(f"{path('LOGS_ROOT')}/obliq_rr/ids281.json"))]
DEV = [str(x) for x in json.load(open(os.path.join(REPO, "studies", "prompt_optimisation", "splits", "obliq_splits", "twitter_dev.json")))]
RUNS = f"{path('FULLDOC_ROOT')}/runs"  # original: <home>/suprlm/runs (the document REPL agent's run outputs)
def ndcg10(r, g):
    d = sum(1 / math.log2(i + 2) for i, x in enumerate(r[:10]) if x in g); i_ = sum(1 / math.log2(i + 2) for i in range(min(10, len(g))))
    return d / i_ if i_ else 0.0
def fd(prefix):
    o = {}
    for f in glob.glob(f"{RUNS}/{prefix}*/q*.json"):
        try: r = json.load(open(f))
        except Exception: continue
        q = str(r["query_id"]); res = (r.get("result") or [{}])[0].get("output", "") or ""
        ranked = [l.strip() for l in res.splitlines() if l.strip().isdigit()]
        o[q] = ndcg10(ranked, set(map(str, QR[q]))) if r.get("status") == "completed" else 0.0
    return o
def snip(dev_files, rest_glob):
    o = {}
    for f in dev_files + glob.glob(rest_glob):
        if f.endswith("_checkpoint.json"): continue
        try: rows = json.load(open(f))
        except Exception: continue
        for x in rows:
            o.setdefault(str(x["query_id"]), x.get("ndcg@10") or 0.0)
    return o
L = f"{path('LOGS_ROOT')}/obliq_mine_rr"  # original: <home>/RLM/logs/obliq_mine_rr
V = {"A": lambda: fd("obliq_q36_fdA_"), "B": lambda: fd("obliq_q36_fdB_"),
     "SNIP": lambda: snip([f"{L}/ctrl_dev100.json"], f"{L}/snip_rest181_g*_s*.json"),
     "SNIP20": lambda: snip(glob.glob(f"{L}/snip20_dev100_g*_s*.json"), f"{L}/snip20_rest181_g*_s*.json")}
if sys.argv[1] == "pick":
    order = sys.argv[2:] or ["A", "B", "SNIP"]
    sc = {k: st.mean(V[k]().get(q, 0.0) for q in DEV) for k in order}
    best = max(order, key=lambda k: (round(sc[k], 6), -order.index(k)))
    print(json.dumps({"dev100": sc, "best": best}), file=sys.stderr); print(best)
else:
    k = sys.argv[2]; D = V[k]()
    orig = fd("obliq_q36_ctrl")
    have = sum(1 for q in ALL if q in D)
    m = st.mean(D.get(q, 0.0) for q in ALL); o = st.mean(orig.get(q, 0.0) for q in ALL)
    d = [D.get(q, 0.0) - orig.get(q, 0.0) for q in ALL]; se = st.stdev(d) / math.sqrt(len(d))
    print(f"OBLIQ Twitter 281, best version {k}: results {have}/281 | NDCG@10 {m:.4f} | original full-doc run {o:.4f} | "
          f"diff {m-o:+.4f} (t {(m-o)/se:+.2f}) | dev100 {st.mean(D.get(q,0) for q in DEV):.4f}, rest181 "
          f"{st.mean(D.get(q,0) for q in ALL if q not in set(DEV)):.4f} | paper GPT-5.2 Multi-Hop 0.141")
