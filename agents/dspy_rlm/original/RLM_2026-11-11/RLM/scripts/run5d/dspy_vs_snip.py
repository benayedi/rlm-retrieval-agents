"""Paired comparison: DSPy RLM (GPT-5 medium, rows 5 / 8 of the old summary) vs snippet RLM (GPT-5 high, BCP-9 / BCP-11),
same questions, gpt-4.1 verdicts. Focus: how much of the gap is the DSPy 16-turn cap."""
import glob, json, os, re, collections
from math import comb
L = "/home/extstu_00/RLM/logs"
from datasets import load_dataset
ALL = [str(r["query_id"]) for r in load_dataset("Tevatron/browsecomp-plus", split="test")]
PQ = json.load(open("/tmp/claude-1003/-home-extstu-00-RLM/b03f7805-ba55-4536-8604-e7fcccc7103b/scratchpad/per_question.json"))["qa"]
ABST = re.compile(r"\b(unknown|not (?:be )?(?:found|determin|identif|availab)|unable|cannot|can't|insufficient|no (?:answer|definitive)|n/?a)\b", re.I)
CAP, SCAP = 16, 35


def mcnemar(b, c):
    n = b + c
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n) if n else 1.0


def dspy(row, pats):
    fs = sorted({f for p in pats for f in glob.glob(f"{L}/{p}")}, key=os.path.getmtime); m = {}
    for f in fs:
        for r in json.load(open(f)):
            m[ALL[int(r["q_idx"])]] = r
    J = json.load(open(f"{L}/bcp_rejudge_gpt41/{row}.json"))
    return {q: dict(r=m[q], ok=bool(J["verdicts"][q])) for q in J["graded"] if q in m}


def f(x):
    try: return float(x)
    except Exception: return None


OUT = {}
for name, row, pats, srow in [("Qwen3-Embed-8B (dense)", "5", ["browsecomp_v2medium_baseline_first50.json", "browsecomp_v2medium_baseline_200.json",
                                "browsecomp_v2_medium_q148_q830.json", "browsecomp_v2_medium_retry56.json"], 9),
                               ("Reason-ModernColBERT", "8", ["dl4096_s*.json", "dl4096x_s*.json"], 11)]:
    D = dspy(row, pats)
    S = {o["qid"]: o for o in PQ if o["sheet"] == "BrowseComp+" and o["row"] == srow}
    Q = [q for q in D if q in S]
    it = {q: int(f(D[q]["r"].get("iterations")) or 0) for q in Q}
    sc = {q: f(D[q]["r"].get("search_calls")) or 0 for q in Q}
    gr = {q: f(D[q]["r"].get("gold_recall")) for q in Q}
    ab = {q: bool(ABST.search(str(D[q]["r"].get("predicted") or "")) and len(str(D[q]["r"].get("predicted") or "")) < 120) for q in Q}
    dok = {q: D[q]["ok"] for q in Q}; sok = {q: bool(S[q]["correct"]) for q in Q}
    st = {q: S[q]["turns"] for q in Q}
    both = sum(dok[q] and sok[q] for q in Q); oD = [q for q in Q if dok[q] and not sok[q]]; oS = [q for q in Q if sok[q] and not dok[q]]
    capped = [q for q in Q if it[q] >= CAP]

    def kind(q):   # one bucket per DSPy loss, first match wins
        if it[q] >= CAP: return "A hit the 16-turn cap"
        if sc[q] >= SCAP: return "B hit the 35-search cap"
        if ab[q]: return "C gave up (Unknown) before the caps"
        if gr[q] == 0: return "D never retrieved gold"
        if gr[q] is not None and gr[q] < 1: return "E partial gold, wrong"
        return "F all gold, wrong"
    lossk = collections.Counter(kind(q) for q in oS)
    winsk = collections.Counter(kind(q) for q in oD)
    res = dict(
        n=len(Q), acc_dspy=sum(dok.values()) / len(Q), acc_snip=sum(sok.values()) / len(Q), both=both, only_dspy=len(oD), only_snip=len(oS),
        neither=len(Q) - both - len(oD) - len(oS), p=mcnemar(len(oD), len(oS)),
        dspy_mean_turns=sum(it.values()) / len(Q), snip_mean_turns=sum(st[q] for q in Q if st[q] is not None) / sum(st[q] is not None for q in Q),
        dspy_capped=len(capped), dspy_capped_acc=sum(dok[q] for q in capped) / max(1, len(capped)),
        dspy_uncapped_acc=sum(dok[q] for q in Q if it[q] < CAP) / max(1, len(Q) - len(capped)),
        snip_acc_on_dspy_capped=sum(sok[q] for q in capped) / max(1, len(capped)),
        snip_acc_on_dspy_uncapped=sum(sok[q] for q in Q if it[q] < CAP) / max(1, len(Q) - len(capped)),
        snip_turns_on_dspy_capped=sorted(st[q] for q in capped if st[q] is not None),
        dspy_search_capped=sum(sc[q] >= SCAP for q in Q),
        snip_right_after_16_turns=sum(1 for q in Q if sok[q] and st[q] is not None and st[q] > CAP),
        losses_by_kind=dict(sorted(lossk.items())), wins_by_kind=dict(sorted(winsk.items())),
        loss_snip_turns_le16=sum(1 for q in oS if st[q] is not None and st[q] <= CAP),
        # counterfactual: DSPy with no turn cap matches the snippet RLM on every question it capped on
        acc_dspy_if_cap_lifted_upper=(sum(dok.values()) + sum(1 for q in capped if sok[q] and not dok[q]) - sum(1 for q in capped if dok[q] and not sok[q])) / len(Q),
        dspy_abst=sum(ab.values()), snip_abst=sum(1 for q in Q if S[q]["category"].startswith("F2")),
        dspy_gold0=sum(1 for q in Q if gr[q] == 0), snip_gold0=sum(1 for q in Q if S[q]["recall"] == 0),
        dspy_recall=sum(gr[q] or 0 for q in Q) / len(Q), snip_recall=sum(S[q]["recall"] or 0 for q in Q) / len(Q),
        dspy_searches=sum(sc.values()) / len(Q), snip_searches=sum(S[q]["searches"] or 0 for q in Q) / len(Q),
    )
    OUT[name] = res
    print("==", name); [print(f"  {k}: {v if not isinstance(v, float) else round(v, 4)}") for k, v in res.items() if k != "snip_turns_on_dspy_capped"]
    t = res["snip_turns_on_dspy_capped"]; print("  snip turns on DSPy-capped questions: median", t[len(t) // 2], "share <=16:", round(sum(x <= 16 for x in t) / len(t), 3))
json.dump(OUT, open("/tmp/claude-1003/-home-extstu-00-RLM/b03f7805-ba55-4536-8604-e7fcccc7103b/scratchpad/dspy_vs_snip.json", "w"), indent=1)
