"""Per-question records for every filled row of docs/Experiments.xlsx -> per_question.json (failure analysis)."""
import glob, json, os, re, sys, time
R = "/home/extstu_00/RLM"; L = f"{R}/logs"; F = f"{L}/run5d/full"
sys.path.insert(0, R)
from datasets import load_dataset
from examples.browsecomp_data import get_gold_answer, get_question
from examples import obliq_data as OD

BCP = list(load_dataset("Tevatron/browsecomp-plus", split="test"))
QID = [str(r["query_id"]) for r in BCP]
BGOLD = {str(r["query_id"]): get_gold_answer(r) for r in BCP}
BQ = {str(r["query_id"]): get_question(r) for r in BCP}
FR = json.load(open(f"{R}/data/frames/frames_rows.json"))
FGOLD = {str(r["i"]): r["answer"] for r in FR}; FTYPE = {str(r["i"]): r["reasoning_types"] for r in FR}
FQ = {str(r["i"]): r["question"] for r in FR}
EV = json.load(open("/home/extstu_00/suprlm/data/prepared/qrel_evidence.json"))
ABST = re.compile(r"\b(unknown|not (?:be )?(?:found|determin|identif|availab)|unable|cannot|can't|insufficient|no (?:answer|definitive)|n/?a)\b", re.I)


def exact(t):
    t = t or ""
    m = re.findall(r"Exact Answer:\s*\**\s*(.+)", t)
    return m[-1].strip()[:300] if m else ""


def conf(t):
    m = re.findall(r"Confidence:\s*\**\s*(\d{1,3})", t or "")
    return int(m[-1]) if m else None


def load(p):
    x = json.load(open(p)); return x[0] if isinstance(x, list) else x


def B(v):
    return v is True or str(v).lower() == "true"


def flt(v):
    try:
        return float(v)
    except Exception:
        return None


OUT = []


def add(sheet, row, run, qid, *, status, correct, answer, recall, recall_kind, searches=None, iters=None, wall=None,
        gold=None, question=None, rtype=None, cap=None):
    ea = exact(answer) if answer else ""
    if answer and not ea and status == "ok":
        ea = answer.strip().splitlines()[-1][:300] if answer.strip() else ""
    answered = bool(ea) and status == "ok"
    abst = bool(answered and ABST.search(ea) and len(ea) < 120)
    hit_cap = bool(cap and iters is not None and iters >= cap)
    if correct:
        cat = "Correct"
    elif status != "ok":
        cat = "F1 Run did not finish"
    elif not answered or abst:
        cat = "F2 No / abstained answer"
    elif recall is None:
        cat = "F6 Wrong, recall not logged"
    elif recall <= 0:
        cat = "F3 Retrieval miss (no gold doc)"
    elif recall < 1:
        cat = "F4 Partial evidence, wrong answer"
    else:
        cat = "F5 All gold found, wrong answer"
    OUT.append(dict(sheet=sheet, row=row, run=run, qid=str(qid), status=status, correct=int(bool(correct)),
                    answered=int(answered), abstained=int(abst), hit_turn_cap=int(hit_cap), recall=recall,
                    recall_kind=recall_kind, searches=searches, iters=iters, wall=wall, confidence=conf(answer),
                    exact_answer=ea, gold=gold, question=(question or "")[:300], rtype=rtype, category=cat))


# ---------------- BrowseComp+ ----------------
def sept(prefixes, verd, row, run):
    V = json.load(open(f"{L}/bcp_rejudge_gpt41/{verd}.json"))["verdicts"]
    recs = {}
    for pre in prefixes:
        for p in sorted(glob.glob(f"{L}/{pre}_q*.json"), key=os.path.getmtime):
            x = load(p); recs[QID[int(x["q_idx"])]] = x
    for q in QID:
        x = recs.get(q)
        if x is None:
            add("BrowseComp+", row, run, q, status="no_result", correct=False, answer="", recall=None, recall_kind="gold", gold=BGOLD[q], question=BQ[q]); continue
        err = x.get("error") not in (None, "None")
        add("BrowseComp+", row, run, q, status="error" if err else "ok", correct=B(V.get(q, x.get("correct"))),
            answer=x.get("predicted") or "", recall=flt(x.get("gold_recall")), recall_kind="gold",
            searches=flt(x.get("search_calls")), wall=flt(x.get("latency_s")), gold=BGOLD[q], question=BQ[q])


def run5d_snip(d, row, run, killed_after=3600):
    G = json.load(open(f"{d}/graded_gpt41.json"))
    stop = set(re.findall(r"q(\d+) stopped", open(f"{d}/WATCHDOG_STOPPED.txt").read())) if os.path.exists(f"{d}/WATCHDOG_STOPPED.txt") else set()
    recs = {}
    for p in glob.glob(f"{d}/q*.json"):
        x = load(p); recs[QID[int(x["q_idx"])]] = x
    now = time.time()
    for i, q in enumerate(QID):
        x = recs.get(q)
        if x is None:
            lg = f"{d}/q{i}.log"
            if str(i) in stop:
                st = "watchdog_stop"
            elif os.path.exists(lg) and now - os.path.getmtime(lg) > 600:
                st = "timeout"            # log exists, no result, not written for >10 min: killed by the time limit
            elif os.path.exists(lg):
                continue                  # still in flight
            elif "qwen" in d:
                continue                  # not started yet (partial run)
            else:
                st = "no_result"
            add("BrowseComp+", row, run, q, status=st, correct=False, answer="", recall=None, recall_kind="gold", gold=BGOLD[q], question=BQ[q]); continue
        if q not in G:
            continue                      # result written after the last grading pass
        err = x.get("error") not in (None, "None")
        add("BrowseComp+", row, run, q, status="error" if err else "ok", correct=G[q]["correct"], answer=x.get("predicted") or "",
            recall=flt(x.get("gold_recall")), recall_kind="gold", searches=flt(x.get("search_calls")),
            wall=flt(x.get("latency_s")), gold=BGOLD[q], question=BQ[q])


def sup(names, row, run):
    d = json.loads(re.search(r'<script id="trace"[^>]*>(.*?)</script>', open(f"{R}/docs/supervisor_artifact.html").read(), re.S).group(1))
    qs = {str(x["qid"]): x for r in d["runs"] if r["run"] in names for x in r["questions"]}
    for q in QID:
        x = qs.get(q)
        if x is None:
            add("BrowseComp+", row, run, q, status="no_result", correct=False, answer="", recall=None, recall_kind="evidence", gold=BGOLD[q], question=BQ[q]); continue
        a = x.get("answer_given") or ""
        add("BrowseComp+", row, run, q, status="ok", correct=B(x["correct"]), answer=a,
            recall=(flt(x.get("final_recall")) or 0) / 100, recall_kind="evidence", searches=flt(x.get("searches")),
            iters=flt(x.get("iters")), wall=flt(x.get("wall")), gold=BGOLD[q], question=BQ[q], cap=30)


def gpt5_bm25(row, run):
    V = json.load(open(f"{L}/bcp_rejudge_gpt41/bm25_gpt5.json"))["verdicts"]
    recs = {}
    for f in glob.glob(f"{L}/bcp_gpt5_bm25/runs/*.json"):
        if os.path.basename(f).startswith("_"):
            continue
        x = json.load(open(f)); recs[str(x["query_id"])] = x
    for q in QID:
        x = recs.get(q)
        if x is None or x.get("status") != "completed":
            add("BrowseComp+", row, run, q, status="timeout" if x else "no_result", correct=False, answer="", recall=None,
                recall_kind="evidence", iters=flt((x or {}).get("n_iterations")), gold=BGOLD[q], question=BQ[q], cap=30); continue
        out = ((x.get("result") or [{}])[0] or {}).get("output", "")
        got = set(map(str, x.get("server_docids") or x.get("retrieved_docids") or []))
        ev = EV.get(q) or []
        add("BrowseComp+", row, run, q, status="ok", correct=B(V.get(q)), answer=out,
            recall=len(got & set(ev)) / len(ev) if ev else None, recall_kind="evidence",
            searches=flt((x.get("tool_call_counts") or {}).get("search", 0)), iters=flt(x.get("n_iterations")),
            wall=flt(x.get("wall_time_s")), gold=BGOLD[q], question=BQ[q], cap=30)


def qwen_fd(parts, verdicts, row, run):
    C = {}
    for d, key in parts:
        for k, v in json.load(open(f"{L}/{d}/cache.json")).items():
            if k.startswith(key + "|") and k.split("|", 1)[1] not in C:
                C[k.split("|", 1)[1]] = v
    for q in QID:
        x = C.get(q)
        if x is None or x.get("status") != "completed":
            add("BrowseComp+", row, run, q, status=("timeout" if x else "no_result"), correct=False, answer="", recall=None,
                recall_kind="gold", gold=BGOLD[q], question=BQ[q], cap=30, iters=flt((x or {}).get("iters"))); continue
        add("BrowseComp+", row, run, q, status="ok", correct=bool(verdicts.get(q)), answer=x.get("answer") or "",
            recall=flt(x.get("gold_recall")), recall_kind="gold", searches=flt(x.get("searches")), iters=flt(x.get("iters")),
            wall=flt(x.get("wall")), gold=BGOLD[q], question=BQ[q], cap=30)


sept(["rlmfix32"], "10", 9, "Snippet RLM · GPT-5 high · Qwen3-Embed-8B")
sept([f"rlmfix{n}" for n in (24, 25, 29, 30, 31)], "9", 11, "Snippet RLM · GPT-5 high · Reason-ModernColBERT")
run5d_snip(f"{F}/bcp_snip_gpt5_bm25", 13, "Snippet RLM · GPT-5 high · BM25")
run5d_snip(f"{F}/bcp_snip_qwen_dense", 15, "Snippet RLM · Qwen3.6 · Qwen3-Embed-8B")
run5d_snip(f"{F}/bcp_snip_qwen_colbert", 17, "Snippet RLM · Qwen3.6 · Reason-ModernColBERT")
run5d_snip(f"{F}/bcp_snip_qwen_bm25", 19, "Snippet RLM · Qwen3.6 · BM25")
sup({"final100_medium", "final200_medium", "final100_e", "final_rest"}, 21, "Full-Doc RLM · GPT-5 medium · Qwen3-Embed-8B")
sup({"colbert8192_high400", "colbert8192_high100b", "colbert8192_high100c", "colbert8192_high_rest230"}, 23, "Full-Doc RLM · GPT-5 high · Reason-ModernColBERT")
gpt5_bm25(25, "Full-Doc RLM · GPT-5 medium · BM25")
G27 = json.load(open(f"{L}/run5d/bcp_fd_qwen_dense/graded_gpt41.json"))
qwen_fd([("run5d/bcp_fd_qwen_dense", "e925f0f6c7")], {q: v["correct"] for q, v in G27.items()}, 27, "Full-Doc RLM · Qwen3.6 · Qwen3-Embed-8B")
qwen_fd([("gepa_bcp2", "ee5ea32a1c"), ("bcp_testrest_ctrl", "ee5ea32a1c"), ("gepa_bcp", "ee5ea32a1c")],
        json.load(open(f"{L}/bcp_rejudge_gpt41/16.json"))["verdicts"], 29, "Full-Doc RLM · Qwen3.6 · Reason-ModernColBERT")
qwen_fd([("bcp_bm25_test200", "11d0c5bc6b"), ("bcp_bm25_rest_gpu0", "11d0c5bc6b"), ("bcp_bm25_rest_gpu1", "11d0c5bc6b")],
        json.load(open(f"{L}/bcp_rejudge_gpt41/18.json"))["verdicts"], 31, "Full-Doc RLM · Qwen3.6 · BM25")

# ---------------- FRAMES ----------------
FIDS = [str(r["i"]) for r in FR]


def frames_snip(d, row, run, timeout_label):
    G = json.load(open(f"{d}/graded_gpt41.json"))
    recs = {}
    for p in glob.glob(f"{d}/q*.json"):
        x = load(p); recs[str(int(x["q_idx"]))] = x
    for q in FIDS:
        x = recs.get(q)
        if x is None:
            add("FRAMES", row, run, q, status=timeout_label, correct=False, answer="", recall=None, recall_kind="gold",
                gold=FGOLD[q], question=FQ[q], rtype=FTYPE[q]); continue
        err = x.get("error") not in (None, "None")
        add("FRAMES", row, run, q, status="error" if err else "ok", correct=bool((G.get(q) or {}).get("correct")),
            answer=x.get("predicted") or "", recall=flt(x.get("gold_recall")), recall_kind="gold",
            searches=flt(x.get("search_calls")), wall=flt(x.get("latency_s")), gold=FGOLD[q], question=FQ[q], rtype=FTYPE[q])


def frames_fd(files, G, row, run):
    recs = {}
    for f in files:
        x = json.load(open(f)); recs[str(x["query_id"])] = x
    for q in FIDS:
        x = recs.get(q); g = G.get(q) or {}
        if x is None or x.get("status") != "completed":
            add("FRAMES", row, run, q, status="timeout" if x else "no_result", correct=False, answer="", recall=None,
                recall_kind="gold", gold=FGOLD[q], question=FQ[q], rtype=FTYPE[q], cap=30, iters=flt((x or {}).get("n_iterations"))); continue
        out = ((x.get("result") or [{}])[0] or {}).get("output", "")
        add("FRAMES", row, run, q, status="ok", correct=bool(g.get("correct")), answer=out, recall=flt(g.get("gold_recall")),
            recall_kind="gold", searches=flt(g.get("searches")), iters=flt(x.get("n_iterations")), wall=flt(x.get("wall_time_s")),
            gold=FGOLD[q], question=FQ[q], rtype=FTYPE[q], cap=30)


frames_snip(f"{F}/frames_snip_gpt5", 7, "Snippet RLM · GPT-5 high · BM25", "no_result")
frames_snip(f"{F}/frames_snip_qwen", 9, "Snippet RLM · Qwen3.6 · BM25", "timeout")
frames_fd([f for f in glob.glob(f"{F}/frames_fd_gpt5/*.json") if re.match(r"\d+\.json$", os.path.basename(f))],
          json.load(open(f"{F}/frames_fd_gpt5/graded_gpt41.json")), 11, "Full-Doc RLM · GPT-5 high · BM25")
frames_fd(glob.glob(f"{L}/frames/sup_q36_bm25f/[0-9]*.json"), json.load(open(f"{L}/frames/sup_q36_bm25f/graded_gpt41.json")),
          13, "Full-Doc RLM · Qwen3.6 · BM25")

# ---------------- OBLIQ ----------------
OBL = []
for task, T in (("Math", ("analogues", "math")), ("Twitter", ("descriptive", "twitter")), ("Congress", ("tip-of-tongue", "congress"))):
    G = OD.load_qrels(*T)
    for arm in ("snip", "fd"):
        row = 9 if arm == "snip" else 11
        run = ("Snippet RLM" if arm == "snip" else "Full-Doc RLM") + " · GPT-5.2 high · Gemini-Embedding-2"
        rec = {}
        if arm == "snip":
            for x in json.load(open(f"{F}/obliq_snip_{task.lower()}/all.json")):
                q = str(x["query_id"])
                out = str(x.get("predicted") or "") if x.get("error") in (None, "None") else ""
                rec[q] = dict(out=out, err=x.get("error") not in (None, "None"), seen=flt(x.get("gold_recall")),
                              searches=flt(x.get("search_calls")), wall=flt(x.get("latency_s")), iters=None)
        else:
            for f in glob.glob(f"/home/extstu_00/suprlm/runs/obliq_fd_{task.lower()}/q*.json"):
                x = json.load(open(f)); q = str(x["query_id"])
                out = ((x.get("result") or [{}])[0] or {}).get("output", "") if x.get("status") == "completed" else ""
                got = set(map(str, x.get("server_docids") or x.get("retrieved_docids") or []))
                rec[q] = dict(out=out or "", err=x.get("status") != "completed",
                              seen=len(got & G[q]) / len(G[q]) if G.get(q) else None,
                              searches=flt((x.get("tool_call_counts") or {}).get("search", 0)),
                              wall=flt(x.get("wall_time_s")), iters=flt(x.get("n_iterations")))
        for q in sorted(G):
            r = rec.get(q); g = G[q]
            if r is None:
                OBL.append(dict(sheet=f"OBLIQ - {task}", row=row, run=run, qid=q, status="no_result", n_gold=len(g), n_ranked=0,
                                ndcg10=0.0, recall10=0.0, seen=None, n_gold_top10=0, searches=None, iters=None, wall=None,
                                category="O1 No usable ranking")); continue
            ids = OD.extract_doc_ids_from_answer(r["out"], T[1])
            nd, rc = OD.ndcg_at(ids, g, 10), OD.recall_at(ids, g, 10)
            top = len(set(ids[:10]) & g)
            if r["err"] or len(ids) == 0:
                cat = "O1 No usable ranking"
            elif top > 0:
                cat = "Hit (≥1 gold in top 10)"
            elif not r["seen"]:
                cat = "O2 Gold never retrieved"
            else:
                cat = "O3 Gold retrieved, not ranked in top 10"
            OBL.append(dict(sheet=f"OBLIQ - {task}", row=row, run=run, qid=q, status="error" if r["err"] else "ok",
                            n_gold=len(g), n_ranked=len(ids), ndcg10=nd, recall10=rc, seen=r["seen"], n_gold_top10=top,
                            searches=r["searches"], iters=r["iters"], wall=r["wall"], category=cat))

json.dump({"qa": OUT, "obliq": OBL}, open(sys.argv[1], "w"))
import collections
for (s, r), c in sorted(collections.Counter((o["sheet"], o["row"]) for o in OUT).items()):
    xs = [o for o in OUT if o["sheet"] == s and o["row"] == r]
    print(s, r, c, "acc", round(sum(o["correct"] for o in xs) / len(xs), 4), collections.Counter(o["category"] for o in xs).most_common())
for (s, r), c in sorted(collections.Counter((o["sheet"], o["row"]) for o in OBL).items()):
    xs = [o for o in OBL if o["sheet"] == s and o["row"] == r]
    print(s, r, c, "ndcg", round(sum(o["ndcg10"] for o in xs) / len(xs), 4), collections.Counter(o["category"] for o in xs).most_common())
