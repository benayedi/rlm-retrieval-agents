"""Grade DSPy ablation arm 3 with gpt-4.1 and compare it, question by question, with the original DSPy dense run and the
snippet RLM (BCP-9) on the 100 stratified questions of logs/dspy_ablation/sample100.json (2026-10-31).
Accuracy is reported per stratum and reweighted to the 775-question population (weight = stratum size / sample size).
A question with no result (killed at the 1 h limit or crashed) counts wrong.
  AZ_ENV=... python scripts/run5d/dspy_ablation_report.py [arm dir]   -> <arm dir>/report.json"""
import glob, json, os, sys, collections
from math import comb
R = "/home/extstu_00/RLM"; L = f"{R}/logs"
sys.path.insert(0, f"{R}/scripts")
from bcp_judge import grade_outputs  # noqa: E402
from datasets import load_dataset  # noqa: E402

ARM = sys.argv[1] if len(sys.argv) > 1 else f"{L}/dspy_ablation/arm3"
ALL = [str(r["query_id"]) for r in load_dataset("Tevatron/browsecomp-plus", split="test")]
SMP = json.load(open(f"{L}/dspy_ablation/sample100.json"))
QS = [x["qid"] for x in SMP["questions"]]; STR = {x["qid"]: x["stratum"] for x in SMP["questions"]}
W = {k: v["pop"] / v["n"] for k, v in SMP["strata"].items()}; POP = SMP["population"]

recs = {}
for f in sorted(glob.glob(f"{ARM}/s*.json"), key=os.path.getmtime):
    for r in json.load(open(f)):
        recs[ALL[int(r["q_idx"])]] = r
outs = {q: (str(recs[q].get("predicted") or "") if q in recs and not recs[q].get("error") else "") for q in QS}
new = grade_outputs(outs, f"{ARM}/_judge_cache_gpt41.json")

old = json.load(open(f"{L}/bcp_rejudge_gpt41/5.json"))["verdicts"]
snip = {o["qid"]: o for o in json.load(open(sys.argv[2] if len(sys.argv) > 2 else
        "/home/extstu_00/RLM/logs/dspy_ablation/snippet_bcp9_per_question.json"))["qa"]
        if o["sheet"] == "BrowseComp+" and o["row"] == 9}
olddsp = {}
for p in ["browsecomp_v2medium_baseline_first50.json", "browsecomp_v2medium_baseline_200.json",
          "browsecomp_v2_medium_q148_q830.json", "browsecomp_v2_medium_retry56.json"]:
    for r in json.load(open(f"{L}/{p}")):
        olddsp[ALL[int(r["q_idx"])]] = r


def f(x):
    try: return float(x)
    except Exception: return 0.0


def mcnemar(b, c):
    n = b + c
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n) if n else 1.0


def acc(sel, v):
    return sum(bool(v[q]) for q in sel) / len(sel)


def wacc(v):
    return sum(W[STR[q]] * bool(v[q]) for q in QS) / POP


A3 = {q: bool(new[q]) for q in QS}; O = {q: bool(old[q]) for q in QS}; S = {q: bool(snip[q]["correct"]) for q in QS}
rep = dict(n=len(QS), results=sum(q in recs for q in QS), arm3_raw=acc(QS, A3), old_dspy_raw=acc(QS, O), snippet_raw=acc(QS, S),
           arm3_weighted=wacc(A3), old_dspy_weighted=wacc(O), snippet_weighted=wacc(S), strata={})
for name, ref in (("vs_old_dspy", O), ("vs_snippet", S)):
    b = sum(A3[q] and not ref[q] for q in QS); c = sum(ref[q] and not A3[q] for q in QS)
    rep[name] = dict(arm3_only=b, other_only=c, p=mcnemar(b, c))
it = {q: f(recs[q].get("iterations")) for q in QS if q in recs}; sc = {q: f(recs[q].get("search_calls")) for q in QS if q in recs}
rep["arm3_mean_turns"] = sum(it.values()) / max(1, len(it)); rep["arm3_mean_searches"] = sum(sc.values()) / max(1, len(sc))
rep["arm3_searches_per_turn"] = sum(sc.values()) / max(1, sum(it.values()))
rep["old_mean_turns"] = sum(f(olddsp[q].get("iterations")) for q in QS) / len(QS)
rep["old_searches_per_turn"] = sum(f(olddsp[q].get("search_calls")) for q in QS) / sum(f(olddsp[q].get("iterations")) for q in QS)
rep["snip_mean_turns"] = sum(snip[q]["turns"] or 0 for q in QS) / len(QS)
rep["arm3_over_16_turns"] = sum(v > 16 for v in it.values()); rep["arm3_no_result"] = len(QS) - len(it)
rep["arm3_mean_latency_s"] = sum(f(recs[q].get("latency_s")) for q in recs if q in STR) / max(1, len(it))
for k in sorted(W):
    sel = [q for q in QS if STR[q] == k]
    rep["strata"][k] = dict(n=len(sel), arm3=acc(sel, A3), old_dspy=acc(sel, O), snippet=acc(sel, S),
                            arm3_turns=sum(it.get(q, 0) for q in sel) / len(sel))
json.dump(rep, open(f"{ARM}/report.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
