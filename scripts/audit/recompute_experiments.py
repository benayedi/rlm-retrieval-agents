"""Result-integrity audit: recompute the headline numbers of Experiments.xlsx from the per-question records the runs
wrote, with code written for this audit (agents/shared/rlm_shared/metrics.py), and compare them with the workbook.

Read-only. It makes no API call and starts no model: correctness comes from the gpt-4.1 verdicts already on disk.
For each row it reports accuracy (correct / all questions), mean recall over questions with a result and over all
questions (the workbook does not say which; the audit finds out), mean search calls, the number of questions with a
result, duplicate or missing records, and verdicts that are not plain booleans.

usage: python scripts/audit/recompute_experiments.py WORKBOOK.xlsx LOGS_ROOT FULLDOC_ROOT OUT.json
  LOGS_ROOT     the original repository root holding logs/ and docs/supervisor_artifact.html
  FULLDOC_ROOT  the document REPL agent directory holding runs/obliq_fd_*
Needs the HF dataset Tevatron/browsecomp-plus (for the question order) and OBLIQ_BENCH_ROOT (OBLIQ qrels)."""
import glob, json, os, re, sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "agents", "shared"))
from rlm_shared import metrics as M  # noqa: E402
from rlm_shared import obliq_data as OD  # noqa: E402

WB, ROOT, FD, OUT = sys.argv[1:5]
L = f"{ROOT}/logs"; F = f"{L}/run5d/full"


def jload(p):
    x = json.load(open(p))
    return x[0] if isinstance(x, list) and x and isinstance(x[0], dict) else x


def head_field(p, name, n=600_000):
    """A top-level field from the first n bytes of a (possibly 70 MB) JSON record."""
    h = open(p, errors="ignore").read(n); i = h.find(f'"{name}":')
    if i < 0:
        return None
    j = i + len(name) + 3
    while h[j] in " \n":
        j += 1
    try:
        return json.JSONDecoder().raw_decode(h, j)[0]
    except Exception:
        return None


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def truth(v):
    return v is True or str(v).lower() == "true"


from datasets import load_dataset  # noqa: E402
BQ = [str(r["query_id"]) for r in load_dataset("Tevatron/browsecomp-plus", split="test")]
FQ = [str(i) for i in range(824)]


def row(qids, verdict, recall, searches, note=""):
    """verdict/recall/searches: {qid: value} for questions with a result."""
    odd = Counter(type(v).__name__ for v in verdict.values() if not isinstance(v, bool))
    return dict(n_questions=len(qids), n_results=len(recall),
                accuracy=M.accuracy({q: verdict.get(q) for q in qids}, qids),
                recall_over_results=M.mean(recall.values()),
                recall_over_all=sum((recall.get(q) or 0) for q in qids) / len(qids),
                searches_over_results=M.mean(searches.values()),
                non_bool_verdicts=dict(odd), note=note)


R = {}

# ---- BrowseComp+ snippet REPL agent, September runs (verdicts re-judged with gpt-4.1)
def sept(prefixes, vfile):
    rec = {}
    for pre in prefixes:
        for p in sorted(glob.glob(f"{L}/{pre}_q*.json"), key=os.path.getmtime):
            x = jload(p); rec[BQ[int(x["q_idx"])]] = x        # a later retry replaces an earlier one
    V = json.load(open(f"{L}/bcp_rejudge_gpt41/{vfile}"))["verdicts"]
    return row(BQ, {q: V.get(q) for q in rec}, {q: fnum(x.get("gold_recall")) for q, x in rec.items()},
               {q: fnum(x.get("search_calls")) for q, x in rec.items()}, f"{len(rec)} records")


R["BCP-9"] = sept(["rlmfix32"], "10.json")
R["BCP-11"] = sept([f"rlmfix{n}" for n in (24, 25, 29, 30, 31)], "9.json")

# ---- run5d graded directories (snippet BCP and FRAMES, document REPL FRAMES / Qwen dense)
def graded(d, qids, rec_key="gold_recall"):
    G = json.load(open(f"{d}/graded_gpt41.json"))
    return row(qids, {q: v.get("correct") for q, v in G.items()}, {q: fnum(v.get(rec_key)) for q, v in G.items()},
               {q: fnum(v.get("searches")) for q, v in G.items()}, f"{len(G)} graded")


R["BCP-13"] = graded(f"{F}/bcp_snip_gpt5_bm25", BQ)
R["BCP-15"] = graded(f"{F}/bcp_snip_qwen_dense", BQ) if os.path.exists(f"{F}/bcp_snip_qwen_dense/graded_gpt41.json") else None
R["BCP-17"] = graded(f"{F}/bcp_snip_qwen_colbert", BQ) if os.path.exists(f"{F}/bcp_snip_qwen_colbert/graded_gpt41.json") else None
R["BCP-19"] = graded(f"{F}/bcp_snip_qwen_bm25", BQ)
R["BCP-27"] = graded(f"{L}/run5d/bcp_fd_qwen_dense", BQ)
_G27 = json.load(open(f"{L}/run5d/bcp_fd_qwen_dense/graded_gpt41.json"))
_written = [v for v in _G27.values() if v.get("status") != "killed"]          # killed runs wrote no rollout record
R["BCP-27"]["recall_workbook_definition"] = M.mean(fnum(v.get("gold_recall")) or 0 for v in _written)
R["BCP-27"]["searches_workbook_definition"] = M.mean(fnum(v.get("searches")) or 0 for v in _written)
R["BCP-27"]["recall_workbook_definition_is"] = f"mean over the {len(_written)} records written (killed runs excluded)"
R["FR-7"] = graded(f"{F}/frames_snip_gpt5", FQ)
R["FR-9"] = graded(f"{F}/frames_snip_qwen", FQ)
R["FR-11"] = graded(f"{F}/frames_fd_gpt5", FQ)
R["FR-13"] = graded(f"{L}/frames/sup_q36_bm25f", FQ)

# ---- BrowseComp+ document REPL GPT-5 dense / ColBERT: the trace embedded in the results page
T = json.loads(re.search(r'<script id="trace"[^>]*>(.*?)</script>', open(f"{ROOT}/docs/supervisor_artifact.html").read(), re.S).group(1))
for rid, names in (("BCP-21", {"final100_medium", "final200_medium", "final100_e", "final_rest"}),
                   ("BCP-23", {"colbert8192_high400", "colbert8192_high100b", "colbert8192_high100c", "colbert8192_high_rest230"})):
    qs = {}
    for r in T["runs"]:
        if r["run"] in names:
            for x in r["questions"]:
                qs.setdefault(str(x["qid"]), []).append(x)
    dup = sum(1 for v in qs.values() if len(v) > 1)
    one = {q: v[-1] for q, v in qs.items()}
    R[rid] = row(BQ, {q: truth(x["correct"]) for q, x in one.items()}, {q: (fnum(x.get("final_recall")) or 0) / 100 for q, x in one.items()},
                 {q: fnum(x.get("searches")) for q, x in one.items()}, f"{len(one)} questions in trace, {dup} duplicated across runs; recall = evidence recall")

# ---- BrowseComp+ document REPL GPT-5 BM25
V = json.load(open(f"{L}/bcp_rejudge_gpt41/bm25_gpt5.json"))["verdicts"]
recs = {}
for p in glob.glob(f"{L}/bcp_gpt5_bm25/runs/*.json"):
    if not os.path.basename(p).startswith("_"):
        qid = str(head_field(p, "query_id")); recs[qid] = p
import importlib.util  # noqa: E402
_bd = importlib.util.spec_from_file_location("bd", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "agents", "shared", "rlm_shared", "browsecomp_data.py"))
BD = importlib.util.module_from_spec(_bd); _bd.loader.exec_module(BD)
EV = {str(r["query_id"]): BD.get_evidence_doc_ids(r) for r in load_dataset("Tevatron/browsecomp-plus", split="test")}
GD = {str(r["query_id"]): BD.get_gold_doc_ids(r) for r in load_dataset("Tevatron/browsecomp-plus", split="test")}
rc_e, rc_g, sc = {}, {}, {}
for q, p in recs.items():
    if head_field(p, "status") != "completed":
        continue
    got = set(map(str, head_field(p, "server_docids", 3_000_000) or head_field(p, "retrieved_docids", 3_000_000) or []))
    rc_e[q] = M.gold_recall(got, EV[q]); rc_g[q] = M.gold_recall(got, GD[q])
    sc[q] = fnum((head_field(p, "tool_call_counts") or {}).get("search", 0))
_all_gold = []
for q, p in recs.items():
    got = set(map(str, head_field(p, "server_docids", 3_000_000) or head_field(p, "retrieved_docids", 3_000_000) or []))
    _all_gold.append(M.gold_recall(got, GD[q]))
R["BCP-25"] = row(BQ, V, rc_e, sc, f"{len(recs)} records, {len(rc_e)} completed; recall = evidence recall (gold-doc recall {M.mean(rc_g.values()):.4f})")

R["BCP-25"]["recall_workbook_definition"] = M.mean(_all_gold)
R["BCP-25"]["recall_workbook_definition_is"] = "gold-document recall, mean over all 830 records incl. unfinished"

# ---- BrowseComp+ document REPL Qwen3.6 ColBERT / BM25: cache.json keyed '<key>|<qid>', verdicts re-judged
def caches(parts, vfile):
    C = {}
    for d, key in parts:
        for k, v in json.load(open(f"{L}/{d}/cache.json")).items():
            if k.startswith(key + "|"):
                C.setdefault(k.split("|", 1)[1], v)
    V = json.load(open(f"{L}/bcp_rejudge_gpt41/{vfile}"))["verdicts"]
    ok = {q: x for q, x in C.items() if x.get("status") == "completed"}
    r = row(BQ, V, {q: fnum(x.get("gold_recall")) for q, x in ok.items()}, {q: fnum(x.get("searches")) for q, x in ok.items()},
            f"{len(C)} cache entries, {len(ok)} completed")
    r["recall_workbook_definition"] = M.mean(fnum(x.get("gold_recall")) or 0 for x in C.values())
    r["recall_workbook_definition_is"] = "gold recall, mean over all cache entries incl. errored/killed runs"
    return r


R["BCP-29"] = caches([("gepa_bcp2", "ee5ea32a1c"), ("bcp_testrest_ctrl", "ee5ea32a1c"), ("gepa_bcp", "ee5ea32a1c")], "16.json")
R["BCP-31"] = caches([("bcp_bm25_test200", "11d0c5bc6b"), ("bcp_bm25_rest_gpu0", "11d0c5bc6b"), ("bcp_bm25_rest_gpu1", "11d0c5bc6b")], "18.json")

# ---- OBLIQ: NDCG@10 / Recall@10 on gold qrels, recomputed from the answer text
for task, (t, sub) in (("MAT", ("analogues", "math")), ("TWI", ("descriptive", "twitter")), ("CON", ("tip-of-tongue", "congress"))):
    gold = OD.load_qrels(t, sub)
    for arm, row_id in (("snip", 9), ("fd", 11)):
        ans = {}
        if arm == "snip":
            for x in json.load(open(f"{F}/obliq_snip_{ {'MAT': 'math', 'TWI': 'twitter', 'CON': 'congress'}[task] }/all.json")):
                ans[str(x["query_id"])] = (str(x.get("predicted") or "") if x.get("error") in (None, "None") else "", fnum(x.get("search_calls")), x.get("ndcg@10"))
        else:
            for p in glob.glob(f"{FD}/runs/obliq_fd_{ {'MAT': 'math', 'TWI': 'twitter', 'CON': 'congress'}[task] }/q*.json"):
                x = jload(p)
                out = ((x.get("result") or [{}])[0] or {}).get("output", "") if x.get("status") == "completed" else ""
                ans[str(x["query_id"])] = (out or "", fnum((x.get("tool_call_counts") or {}).get("search", 0)), None)
        nd = [M.ndcg_at_k(OD.extract_doc_ids_from_answer(ans.get(q, ("",))[0], sub), gold[q], 10) for q in gold]
        rc = [M.recall_at_k(OD.extract_doc_ids_from_answer(ans.get(q, ("",))[0], sub), gold[q], 10) for q in gold]
        logged = [a[2] for a in ans.values() if a[2] is not None]
        R[f"{task}-{row_id}"] = dict(n_questions=len(gold), n_results=len(ans), ndcg10=sum(nd) / len(gold),
                                     recall10=sum(rc) / len(gold), searches_over_results=M.mean(a[1] for a in ans.values()),
                                     ndcg10_as_logged=(sum(logged) / len(logged)) if logged else None,
                                     note=f"{len(ans)} records; missing questions score 0")

# ---- compare with the workbook
from openpyxl import load_workbook  # noqa: E402
wb = load_workbook(WB, data_only=True)
SHEET = {"BCP": "BrowseComp+", "FR": "FRAMES", "MAT": "OBLIQ - Math", "TWI": "OBLIQ - Twitter", "CON": "OBLIQ - Congress"}
report = {}
for rid, r in R.items():
    if r is None:
        report[rid] = {"status": "no graded results yet"}; continue
    ws = wb[SHEET[rid.split("-")[0]]]; n = int(rid.split("-")[1])
    hdr = {ws.cell(3, c).value: c for c in range(1, ws.max_column + 1) if ws.cell(3, c).value}   # header row 3; values sit in merged 2-column cells
    cells = {h: ws.cell(n, c).value for h, c in hdr.items()}
    report[rid] = dict(workbook=cells, recomputed=r)
json.dump(report, open(OUT, "w"), indent=1, default=str)


def first_float(v):
    m = re.search(r"\d*\.\d+|\d+", str(v)) if v is not None else None
    return float(m.group()) if m else None


print(f"{'row':7} {'workbook acc/ndcg':>18} {'recomputed':>11} {'wb recall':>10} {'rec/results':>11} {'rec/all':>8} {'wb srch':>8} {'srch':>7}  note")
for rid, x in report.items():
    if "recomputed" not in x:
        print(rid, x["status"]); continue
    w, r = x["workbook"], x["recomputed"]
    if "ndcg10" in r:
        print(f"{rid:7} {str(w.get('NDCG@10 (G / P)', w.get('NDCG@10')))[:18]:>18} {r['ndcg10']:11.4f} {str(w.get('Recall@10 (G)', w.get('Recall@10')))[:10]:>10} {r['recall10']:11.4f} {'':>8} {str(w.get('Avg. Search Calls Per Query'))[:8]:>8} {r['searches_over_results'] or 0:7.2f}  {r['note']}")
    else:
        print(f"{rid:7} {str(w.get('Accuracy'))[:18]:>18} {r['accuracy']:11.4f} {str(w.get('Recall'))[:10]:>10} {r['recall_over_results'] or 0:11.4f} {r['recall_over_all']:8.4f} "
              f"{str(w.get('Avg. Search Calls Per Query'))[:8]:>8} {r['searches_over_results'] or 0:7.2f}  {r['note']} {r['non_bool_verdicts'] or ''}")
        if "recall_workbook_definition" in r:
            print(f"{'':7} {'':>18} {'':>11} {'':>10} {r['recall_workbook_definition']:11.4f} {'':>8} {'':>8} "
                  f"{r.get('searches_workbook_definition') or r['searches_over_results'] or 0:7.2f}  <- {r['recall_workbook_definition_is']}")
