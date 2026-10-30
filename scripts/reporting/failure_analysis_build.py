"""Build docs/Failure_Analysis.xlsx from per_question.json (extract.py). Summary tables are COUNTIFS/AVERAGEIFS formulas
over the 'Detail QA' / 'Detail OBLIQ' sheets, so every number traces back to a question.

Refactored copy of scripts/failure_analysis_build.py of the original repository, unchanged apart from this docstring (see
scripts/reporting/README.md for the pipeline order).
  python scripts/reporting/failure_analysis_build.py <per_question.json> <out.xlsx> \
      scripts/reporting/failure_analysis_findings.py scripts/reporting/failure_analysis_extras.py \
      scripts/reporting/failure_analysis_turns_sheet.py
"""
import difflib, json, re, sys
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter as CL
from openpyxl.comments import Comment

D = json.load(open(sys.argv[1])); OUTP = sys.argv[2]
QA, OB = D["qa"], D["obliq"]
FONT = "Arial"
HF = PatternFill("solid", fgColor="1F3864"); SF = PatternFill("solid", fgColor="D9E1F2"); NF = PatternFill("solid", fgColor="F2F2F2")
thin = Side(style="thin", color="BFBFBF"); BOX = Border(top=thin, bottom=thin, left=thin, right=thin)
SHORT = {"BrowseComp+": "BCP", "FRAMES": "FR"}


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).strip()


def near(a, g):
    a, g = norm(a), norm(g)
    if not a or not g:
        return 0
    return int(a in g or g in a or difflib.SequenceMatcher(None, a, g).ratio() >= 0.6)


for o in QA:
    o["exp"] = f"{SHORT[o['sheet']]}-{o['row']}"
    o["near"] = near(o["exact_answer"], o["gold"]) if (not o["correct"] and o["answered"] and not o["abstained"]) else 0
    h, m, r = (o["run"].replace(" (partial)", "").split(" · ") + ["", "", ""])[:3]
    o["harness"], o["model"], o["retr"] = h, m, r
for o in OB:
    o["exp"] = f"{o['sheet'].split(' - ')[1][:3].upper()}-{o['row']}"
    o["seen_n"] = round((o["seen"] or 0) * o["n_gold"])

wb = Workbook()


def style_header(ws, r, c1, c2, fill=HF, color="FFFFFF"):
    for c in range(c1, c2 + 1):
        x = ws.cell(r, c); x.font = Font(name=FONT, bold=True, color=color); x.fill = fill
        x.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center"); x.border = BOX


def title(ws, text, sub=None):
    ws["A1"] = text; ws["A1"].font = Font(name=FONT, bold=True, size=14, color="1F3864")
    if sub:
        ws["A2"] = sub; ws["A2"].font = Font(name=FONT, italic=True, size=9, color="595959")
        ws["A2"].alignment = Alignment(wrap_text=False)


def body(ws, r1, r2, c1, c2, fmt=None):
    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            x = ws.cell(r, c); x.font = Font(name=FONT, size=10, bold=x.font.bold if x.font else False); x.border = BOX
            if fmt and c in fmt:
                x.number_format = fmt[c]


def widths(ws, w):
    for i, v in enumerate(w, 1):
        ws.column_dimensions[CL(i)].width = v


# ---------------------------------------------------------------- Detail QA
dq = wb.active; dq.title = "Detail QA"
COLS = ["Exp ID", "Dataset", "Sheet row", "Harness", "Model", "Retriever", "QID", "Status", "Correct", "Answered",
        "Abstained", "Hit turn cap", "Gold recall", "Recall kind", "Searches", "Turns", "Turn limit", "Turn outcome", "Time (s)",
        "Stated confidence", "Near miss", "Failure category", "Reasoning type (FRAMES)", "Exact answer", "Gold answer", "Question"]
dq.append(COLS); style_header(dq, 1, 1, len(COLS))
for o in QA:
    dq.append([o["exp"], o["sheet"], o["row"], o["harness"], o["model"], o["retr"], o["qid"], o["status"], o["correct"],
               o["answered"], o["abstained"], o["hit_turn_cap"], o["recall"], o["recall_kind"], o["searches"], o["turns"],
               o["turn_cap"], o["turn_outcome"], o["wall"], o["confidence"], o["near"], o["category"], o["rtype"], o["exact_answer"], o["gold"], o["question"]])
NQ = len(QA) + 1
for row in dq.iter_rows(min_row=2, max_row=NQ):
    for x in row:
        x.font = Font(name=FONT, size=9)
    row[12].number_format = "0.00"
widths(dq, [9, 11, 7, 13, 14, 22, 7, 12, 7, 8, 9, 8, 8, 9, 8, 7, 7, 30, 8, 9, 7, 30, 26, 40, 30, 60])
dq.freeze_panes = "A2"; dq.auto_filter.ref = f"A1:{CL(len(COLS))}{NQ}"
C = {n: CL(i + 1) for i, n in enumerate(COLS)}


def rq(col):   # bounded range on Detail QA
    return f"'Detail QA'!${C[col]}$2:${C[col]}${NQ}"


# ---------------------------------------------------------------- Detail OBLIQ
do = wb.create_sheet("Detail OBLIQ")
OC = ["Exp ID", "Task", "Sheet row", "Harness", "QID", "Status", "# gold", "# ids ranked", "NDCG@10", "Recall@10",
      "Gold seen in search (share)", "Gold seen (count)", "Gold in top 10 (count)", "Searches", "Turns", "Turn limit", "Hit turn cap",
      "Time (s)", "Failure category"]
do.append(OC); style_header(do, 1, 1, len(OC))
for o in OB:
    do.append([o["exp"], o["sheet"].split(" - ")[1], o["row"], o["run"].split(" · ")[0], o["qid"], o["status"], o["n_gold"],
               o["n_ranked"], o["ndcg10"], o["recall10"], o["seen"], o["seen_n"], o["n_gold_top10"], o["searches"],
               o["turns"], o["turn_cap"], o["hit_turn_cap"], o["wall"], o["category"]])
NO = len(OB) + 1
for row in do.iter_rows(min_row=2, max_row=NO):
    for x in row:
        x.font = Font(name=FONT, size=9)
    for i in (8, 9, 10):
        row[i].number_format = "0.000"
widths(do, [9, 10, 7, 13, 10, 8, 7, 8, 8, 8, 10, 9, 9, 8, 7, 7, 7, 8, 34])
do.freeze_panes = "A2"; do.auto_filter.ref = f"A1:{CL(len(OC))}{NO}"
OCc = {n: CL(i + 1) for i, n in enumerate(OC)}


def ro(col):
    return f"'Detail OBLIQ'!${OCc[col]}$2:${OCc[col]}${NO}"


# ---------------------------------------------------------------- experiment list
EXPS = []
for o in QA:
    if o["exp"] not in [e[0] for e in EXPS]:
        EXPS.append((o["exp"], o["sheet"], o["row"], o["harness"], o["model"], o["retr"]))
EXPS.sort(key=lambda e: (e[1] != "BrowseComp+", e[2]))
CATS = [("F1", "F1 Run did not finish", "Killed at the time limit, crashed, stopped by the cost watchdog or rate-limit error: no answer reached the judge."),
        ("F2", "F2 No / abstained answer", "Run finished but the final answer is empty or an abstention ('Not found in the provided corpus', 'Unable to determine')."),
        ("F3", "F3 Retrieval miss (no gold doc)", "Answered wrongly and none of the question's gold documents was ever retrieved: the agent never saw the evidence."),
        ("F4", "F4 Partial evidence, wrong answer", "Answered wrongly with some, but not all, gold documents retrieved: a hop of the chain was missing or mis-resolved."),
        ("F5", "F5 All gold found, wrong answer", "Answered wrongly although every gold document was retrieved: a reading / reasoning / verification error.")]

# ---------------------------------------------------------------- QA Summary
qs = wb.create_sheet("QA Summary", 0)
title(qs, "Failure breakdown per experiment — BrowseComp+ and FRAMES",
      "One row per filled row of docs/Experiments.xlsx. Counts are COUNTIFS over 'Detail QA'. Shares in the right block are shares of the failed questions.")
H = ["Exp ID", "Dataset", "Exp. row", "Harness", "Model", "Retriever", "Questions", "Correct", "Accuracy", "Failed",
     "F1 did not finish", "F2 no / abstained", "F3 retrieval miss", "F4 partial evidence", "F5 all gold, wrong",
     "F1 %", "F2 %", "F3 %", "F4 %", "F5 %", "Retrieval-side share (F3+F4)", "Reasoning-side share (F5)",
     "Wrong & confident ≥90%", "Wrong answers w/ confidence logged", "Overconfident share", "Near-miss wrong answers",
     "Turn-cap hits", "Mean recall, correct", "Mean recall, wrong", "Mean searches, correct", "Mean searches, wrong"]
qs.append([]); qs.append(H); style_header(qs, 4, 1, len(H))
r0 = 5
for i, e in enumerate(EXPS):
    r = r0 + i; A = f"$A{r}"
    row = list(e) + [
        f"=COUNTIFS({rq('Exp ID')},{A})", f"=COUNTIFS({rq('Exp ID')},{A},{rq('Correct')},1)", f"=IFERROR(H{r}/G{r},0)", f"=G{r}-H{r}"]
    row += [f'=COUNTIFS({rq("Exp ID")},{A},{rq("Failure category")},"{k}*")' for k, _, _ in CATS]
    row += [f"=IFERROR({CL(11 + j)}{r}/$J{r},0)" for j in range(5)]
    row += [f"=R{r}+S{r}", f"=T{r}",
            f'=COUNTIFS({rq("Exp ID")},{A},{rq("Correct")},0,{rq("Answered")},1,{rq("Abstained")},0,{rq("Stated confidence")},">=90")',
            f'=COUNTIFS({rq("Exp ID")},{A},{rq("Correct")},0,{rq("Answered")},1,{rq("Abstained")},0,{rq("Stated confidence")},">=0")',
            f"=IFERROR(W{r}/X{r},0)",
            f'=COUNTIFS({rq("Exp ID")},{A},{rq("Near miss")},1)',
            f'=COUNTIFS({rq("Exp ID")},{A},{rq("Hit turn cap")},1)',
            f'=IFERROR(AVERAGEIFS({rq("Gold recall")},{rq("Exp ID")},{A},{rq("Correct")},1),"—")',
            f'=IFERROR(AVERAGEIFS({rq("Gold recall")},{rq("Exp ID")},{A},{rq("Correct")},0),"—")',
            f'=IFERROR(AVERAGEIFS({rq("Searches")},{rq("Exp ID")},{A},{rq("Correct")},1),"—")',
            f'=IFERROR(AVERAGEIFS({rq("Searches")},{rq("Exp ID")},{A},{rq("Correct")},0),"—")']
    for j, v in enumerate(row, 1):
        qs.cell(r, j, v)
rN = r0 + len(EXPS) - 1
body(qs, r0, rN, 1, len(H), {9: "0.0%", **{c: "0.0%" for c in range(16, 23)}, 25: "0.0%", 28: "0.00", 29: "0.00", 30: "0.0", 31: "0.0"})
for r in range(r0, rN + 1):
    for c in range(16, 23):
        qs.cell(r, c).fill = NF
widths(qs, [8, 11, 6, 12, 13, 22, 9, 8, 9, 7, 9, 9, 9, 9, 9, 7, 7, 7, 7, 7, 11, 11, 10, 11, 10, 10, 8, 9, 9, 9, 9])
qs.row_dimensions[4].height = 58; qs.freeze_panes = "G5"
n = rN + 2
qs.cell(n, 1, "Failure categories (assigned in this order; the first that applies wins)").font = Font(name=FONT, bold=True)
for k, (_, name, desc) in enumerate(CATS):
    qs.cell(n + 1 + k, 1, name).font = Font(name=FONT, bold=True, size=9)
    qs.cell(n + 1 + k, 4, desc).font = Font(name=FONT, size=9)
notes = ["Gold recall: share of the question's gold documents the agent retrieved at any point. BrowseComp+ snippet rows and Qwen3.6 full-doc rows log recall of the answer-bearing gold_docs; "
         "the full-doc GPT-5 dense and ColBERT rows (BCP-21, BCP-23) and BCP-25 only have evidence-doc recall (evidence ⊇ gold, so 0 and 1 mean the same, the partial band differs).",
         "Near miss: a wrong, non-abstained answer whose normalised text contains / is contained in the gold answer, or has string similarity ≥ 0.6 (difflib). Candidates for partial answers or judge strictness, not proven judge errors.",
         "Overconfident: a wrong, non-abstained answer whose 'Confidence:' line is ≥ 90%. Denominator: wrong answers that state a confidence.",
         "BCP-19 (Qwen3.6 snippet, BM25) is still running: only attempted questions are counted (graded 2026-10-29 with gpt-4.1); a question whose log stopped for >10 min without a result is counted as killed at the 1 h limit (F1)."]
for k, t in enumerate(notes):
    c = qs.cell(n + 7 + k, 1, t); c.font = Font(name=FONT, size=9, italic=True)

# ---------------------------------------------------------------- By Recall
br = wb.create_sheet("By Recall", 1)
title(br, "Accuracy given how much gold evidence was retrieved",
      "Questions with a finished run only. If accuracy at recall = 1 is high and at recall = 0 near zero, the errors are retrieval errors; the gap between the 'all gold' column and 100% is the reading / reasoning loss.")
H2 = ["Exp ID", "Dataset", "Harness", "Model", "Retriever", "n recall = 0", "n 0 < recall < 1", "n recall = 1",
      "Acc. recall = 0", "Acc. 0 < recall < 1", "Acc. recall = 1", "Share of questions with all gold", "Share with no gold"]
br.append([]); br.append(H2); style_header(br, 4, 1, len(H2))
for i, e in enumerate(EXPS):
    r = 5 + i; A = f"$A{r}"; ok = f'{rq("Exp ID")},{A},{rq("Status")},"ok"'
    vals = [e[0], e[1], e[3], e[4], e[5],
            f'=COUNTIFS({ok},{rq("Gold recall")},0)', f'=COUNTIFS({ok},{rq("Gold recall")},">0",{rq("Gold recall")},"<1")',
            f'=COUNTIFS({ok},{rq("Gold recall")},">=1")',
            f'=IFERROR(COUNTIFS({ok},{rq("Gold recall")},0,{rq("Correct")},1)/F{r},"—")',
            f'=IFERROR(COUNTIFS({ok},{rq("Gold recall")},">0",{rq("Gold recall")},"<1",{rq("Correct")},1)/G{r},"—")',
            f'=IFERROR(COUNTIFS({ok},{rq("Gold recall")},">=1",{rq("Correct")},1)/H{r},"—")',
            f"=IFERROR(H{r}/(F{r}+G{r}+H{r}),0)", f"=IFERROR(F{r}/(F{r}+G{r}+H{r}),0)"]
    for j, v in enumerate(vals, 1):
        br.cell(r, j, v)
body(br, 5, 4 + len(EXPS), 1, len(H2), {c: "0.0%" for c in range(9, 14)})
widths(br, [8, 11, 12, 13, 22, 9, 10, 9, 9, 10, 9, 11, 10]); br.row_dimensions[4].height = 45

# ---------------------------------------------------------------- By Searches
bs = wb.create_sheet("By Searches", 2)
title(bs, "Accuracy by number of searches the agent made",
      "Finished questions only. Long search sequences are mostly a symptom (the agent is not finding the evidence), not a cure.")
BK = [("1-5", ">=1", "<=5"), ("6-10", ">=6", "<=10"), ("11-25", ">=11", "<=25"), ("26-50", ">=26", "<=50"), ("> 50", ">50", None)]
H3 = ["Exp ID", "Dataset", "Harness", "Model", "Retriever"] + [f"n {b[0]}" for b in BK] + [f"Acc. {b[0]}" for b in BK]
bs.append([]); bs.append(H3); style_header(bs, 4, 1, len(H3))
for i, e in enumerate(EXPS):
    r = 5 + i; A = f"$A{r}"; ok = f'{rq("Exp ID")},{A},{rq("Status")},"ok"'
    vals = [e[0], e[1], e[3], e[4], e[5]]
    crit = [f'{rq("Searches")},"{lo}"' + (f',{rq("Searches")},"{hi}"' if hi else "") for _, lo, hi in BK]
    vals += [f"=COUNTIFS({ok},{c})" for c in crit]
    vals += [f'=IFERROR(COUNTIFS({ok},{c},{rq("Correct")},1)/{CL(6 + k)}{r},"—")' for k, c in enumerate(crit)]
    for j, v in enumerate(vals, 1):
        bs.cell(r, j, v)
body(bs, 5, 4 + len(EXPS), 1, len(H3), {c: "0.0%" for c in range(11, 16)})
widths(bs, [8, 11, 12, 13, 22] + [8] * 10); bs.row_dimensions[4].height = 30
bs.cell(6 + len(EXPS), 1, "Searches are not logged per question for BCP-21 / BCP-23 bucket ≤ 10 (hard cap 10). Full-doc rows are capped at 10 searches; snippet GPT rows on BrowseComp+ have no cap, FRAMES snippet rows 25.").font = Font(name=FONT, size=9, italic=True)

# ---------------------------------------------------------------- wide per-question sheets
def wide(name, sheet, exps):
    ws = wb.create_sheet(name)
    ids = sorted({o["qid"] for o in QA if o["sheet"] == sheet}, key=int)
    by = {(o["exp"], o["qid"]): o for o in QA if o["sheet"] == sheet}
    hdr = ["QID"] + [f"{e} correct" for e in exps] + [f"{e} category" for e in exps]
    ws.append(hdr); style_header(ws, 1, 1, len(hdr))
    for q in ids:
        row = [int(q)] + [by[(e, q)]["correct"] if (e, q) in by else None for e in exps] + \
              [by[(e, q)]["category"][:2] if (e, q) in by else None for e in exps]
        ws.append(row)
    n = len(ids) + 1
    for row in ws.iter_rows(min_row=2, max_row=n):
        for x in row:
            x.font = Font(name=FONT, size=9)
    ws.freeze_panes = "B2"; widths(ws, [7] + [9] * (2 * len(exps)))
    cols = {e: (CL(2 + k), CL(2 + len(exps) + k)) for k, e in enumerate(exps)}
    return ws, n, cols


BEXPS = [e[0] for e in EXPS if e[1] == "BrowseComp+"]; FEXPS = [e[0] for e in EXPS if e[1] == "FRAMES"]
wbq, NB, BC = wide("BCP by Question", "BrowseComp+", BEXPS)
GPT5 = ["BCP-9", "BCP-11", "BCP-13", "BCP-21", "BCP-23", "BCP-25"]
sc = len(BEXPS) * 2 + 2
wbq.cell(1, sc, "GPT-5 rows solved (of 6)"); wbq.cell(1, sc + 1, "Qwen3.6 full-doc rows solved (of 3)")
style_header(wbq, 1, sc, sc + 1)
for r in range(2, NB + 1):
    wbq.cell(r, sc, "=" + "+".join(f"{BC[e][0]}{r}" for e in GPT5)).font = Font(name=FONT, size=9)
    wbq.cell(r, sc + 1, "=" + "+".join(f"{BC[e][0]}{r}" for e in ("BCP-27", "BCP-29", "BCP-31"))).font = Font(name=FONT, size=9)
SOLV = CL(sc); SOLQ = CL(sc + 1)
wfq, NF_, FC = wide("FRAMES by Question", "FRAMES", FEXPS)


def wr(ws, n, col):
    return f"'{ws.title}'!${col}$2:${col}${n}"


# ---------------------------------------------------------------- Pairwise
pw = wb.create_sheet("Pairwise", 3)
title(pw, "Paired comparisons on the same questions — who loses where, and why",
      "'Only A right' = A correct and B wrong. The right block shows the failure category of B on the questions only A got right (and the reverse below it).")
PAIRS = [("Snippet vs full-doc, GPT-5, Qwen3-Embed-8B", "BCP-9", "BCP-21", "B"),
         ("Snippet vs full-doc, GPT-5, Reason-ModernColBERT", "BCP-11", "BCP-23", "B"),
         ("Snippet vs full-doc, GPT-5, BM25", "BCP-13", "BCP-25", "B"),
         ("GPT-5 vs Qwen3.6, full-doc, Qwen3-Embed-8B", "BCP-21", "BCP-27", "B"),
         ("GPT-5 vs Qwen3.6, full-doc, Reason-ModernColBERT", "BCP-23", "BCP-29", "B"),
         ("GPT-5 vs Qwen3.6, full-doc, BM25", "BCP-25", "BCP-31", "B"),
         ("ColBERT vs BM25, full-doc, GPT-5", "BCP-23", "BCP-25", "B"),
         ("ColBERT vs BM25, snippet, GPT-5", "BCP-11", "BCP-13", "B"),
         ("ColBERT vs dense, snippet, GPT-5", "BCP-11", "BCP-9", "B"),
         ("Snippet vs full-doc, GPT-5, BM25 (FRAMES)", "FR-7", "FR-11", "F"),
         ("Snippet vs full-doc, Qwen3.6, BM25 (FRAMES)", "FR-9", "FR-13", "F"),
         ("GPT-5 vs Qwen3.6, full-doc (FRAMES)", "FR-11", "FR-13", "F"),
         ("GPT-5 vs Qwen3.6, snippet (FRAMES)", "FR-7", "FR-9", "F")]
H4 = ["Comparison", "A", "B", "Both right", "Only A right", "Only B right", "Both wrong", "Net A − B", "McNemar p (exact, two-sided)",
      "B's failure when only A right: F1", "F2", "F3", "F4", "F5",
      "A's failure when only B right: F1", "F2", "F3", "F4", "F5"]
pw.append([]); pw.append(H4); style_header(pw, 4, 1, len(H4))
for i, (lab, a, b, ds) in enumerate(PAIRS):
    r = 5 + i
    ws, n, cc = (wbq, NB, BC) if ds == "B" else (wfq, NF_, FC)
    ac, bc = wr(ws, n, cc[a][0]), wr(ws, n, cc[b][0]); acat, bcat = wr(ws, n, cc[a][1]), wr(ws, n, cc[b][1])
    vals = [lab, a, b, f"=COUNTIFS({ac},1,{bc},1)", f"=COUNTIFS({ac},1,{bc},0)", f"=COUNTIFS({ac},0,{bc},1)",
            f"=COUNTIFS({ac},0,{bc},0)", f"=E{r}-F{r}",
            f"=IF(E{r}+F{r}=0,1,MIN(1,2*BINOMDIST(MIN(E{r},F{r}),E{r}+F{r},0.5,TRUE)))"]
    vals += [f'=COUNTIFS({ac},1,{bc},0,{bcat},"{k}")' for k, _, _ in CATS]
    vals += [f'=COUNTIFS({ac},0,{bc},1,{acat},"{k}")' for k, _, _ in CATS]
    for j, v in enumerate(vals, 1):
        pw.cell(r, j, v)
body(pw, 5, 4 + len(PAIRS), 1, len(H4), {9: "0.0E+00"})
widths(pw, [44, 7, 7, 8, 8, 8, 8, 8, 11] + [9] * 10); pw.row_dimensions[4].height = 58
pw.cell(6 + len(PAIRS), 1, "BCP-19 (Qwen3.6 snippet) is left out: it is still running. Pairs only count questions present in both runs (all 830 / 824 for complete rows).").font = Font(name=FONT, size=9, italic=True)

# ---------------------------------------------------------------- BCP Difficulty
bd = wb.create_sheet("BCP Difficulty", 4)
title(bd, "BrowseComp+: how many configurations solve each question",
      "Counts over 'BCP by Question'. 'Hard core' = questions no GPT-5 configuration (6 complete rows) answers correctly.")
bd.append([]); bd.append(["GPT-5 rows that solve the question", "Questions", "Share"]); style_header(bd, 4, 1, 3)
for k in range(7):
    r = 5 + k
    bd.cell(r, 1, k); bd.cell(r, 2, f"=COUNTIF({wr(wbq, NB, SOLV)},A{r})"); bd.cell(r, 3, f"=B{r}/SUM($B$5:$B$11)")
body(bd, 5, 11, 1, 3, {3: "0.0%"})
bd.cell(13, 1, "Hard core (0 of 6 GPT-5 rows): failure category per row").font = Font(name=FONT, bold=True)
bd.append(["Exp ID"] + [c[1] for c in CATS] + ["Correct"]); style_header(bd, 14, 1, 7)
for i, e in enumerate(BEXPS):
    r = 15 + i; bd.cell(r, 1, e)
    for k, (code, _, _) in enumerate(CATS):
        bd.cell(r, 2 + k, f'=COUNTIFS({wr(wbq, NB, SOLV)},0,{wr(wbq, NB, BC[e][1])},"{code}")')
    bd.cell(r, 7, f'=COUNTIFS({wr(wbq, NB, SOLV)},0,{wr(wbq, NB, BC[e][0])},1)')
body(bd, 15, 14 + len(BEXPS), 1, 7)
r = 16 + len(BEXPS)
bd.cell(r, 1, "Qwen3.6 full-doc rows vs GPT-5 difficulty").font = Font(name=FONT, bold=True)
bd.cell(r + 1, 1, "GPT-5 rows that solve the question"); bd.cell(r + 1, 2, "Questions"); bd.cell(r + 1, 3, "Qwen3.6 full-doc solve rate (of 3 rows)")
style_header(bd, r + 1, 1, 3)
for k in range(7):
    rr = r + 2 + k; bd.cell(rr, 1, k); bd.cell(rr, 2, f"=COUNTIF({wr(wbq, NB, SOLV)},A{rr})")
    bd.cell(rr, 3, f'=IFERROR(SUMIF({wr(wbq, NB, SOLV)},A{rr},{wr(wbq, NB, SOLQ)})/(3*B{rr}),0)')
body(bd, r + 2, r + 8, 1, 3, {3: "0.0%"})
widths(bd, [34, 16, 20, 20, 20, 20, 10]); bd.row_dimensions[14].height = 30

# ---------------------------------------------------------------- FRAMES types
ft = wb.create_sheet("FRAMES Types", 5)
title(ft, "FRAMES: accuracy and main failure by reasoning type",
      "A question can carry several reasoning types (multi-label), so the columns overlap. COUNTIFS with wildcards on 'Reasoning type (FRAMES)'.")
TYPES = ["Multiple constraints", "Numerical reasoning", "Temporal reasoning", "Tabular reasoning", "Post processing"]
H5 = ["Exp ID", "Harness", "Model"] + [f"Acc. {t}" for t in TYPES] + [f"F5 share of failures, {t}" for t in TYPES]
ft.append([]); ft.append(H5); style_header(ft, 4, 1, len(H5))
FE = [e for e in EXPS if e[1] == "FRAMES"]
ft.cell(5, 1, "Questions"); [ft.cell(5, 4 + k, f'=COUNTIFS({rq("Exp ID")},"FR-7",{rq("Reasoning type (FRAMES)")},"*{t}*")') for k, t in enumerate(TYPES)]
for i, e in enumerate(FE):
    r = 6 + i; A = f"$A{r}"
    ft.cell(r, 1, e[0]); ft.cell(r, 2, e[3]); ft.cell(r, 3, e[4])
    for k, t in enumerate(TYPES):
        ty = f'{rq("Reasoning type (FRAMES)")},"*{t}*"'
        ft.cell(r, 4 + k, f'=IFERROR(COUNTIFS({rq("Exp ID")},{A},{ty},{rq("Correct")},1)/COUNTIFS({rq("Exp ID")},{A},{ty}),0)')
        ft.cell(r, 9 + k, f'=IFERROR(COUNTIFS({rq("Exp ID")},{A},{ty},{rq("Failure category")},"F5*")/COUNTIFS({rq("Exp ID")},{A},{ty},{rq("Correct")},0),0)')
body(ft, 5, 5 + len(FE), 1, len(H5), {c: "0.0%" for c in range(4, 14)})
for c in range(4, 14):
    ft.cell(5, c).number_format = "0"
widths(ft, [9, 13, 14] + [12] * 10); ft.row_dimensions[4].height = 45

# ---------------------------------------------------------------- OBLIQ Summary
os_ = wb.create_sheet("OBLIQ Summary", 6)
title(os_, "OBLIQ (Math, Twitter, Congress): where the ranking fails",
      "Per question: O1 no usable ranking (no doc ids could be parsed from the final answer, or an error); O2 no gold document ever came up in the agent's searches; O3 gold was retrieved but not put in the top 10; Hit = at least one gold in the top 10.")
OE = []
for o in OB:
    if o["exp"] not in [x[0] for x in OE]:
        OE.append((o["exp"], o["sheet"].split(" - ")[1], o["run"].split(" · ")[0]))
H6 = ["Exp ID", "Task", "Harness", "Questions", "Mean NDCG@10 (gold)", "Hit", "O1 no usable ranking", "O2 gold never retrieved",
      "O3 retrieved, not top 10", "Hit %", "O1 %", "O2 %", "O3 %", "Gold docs total", "Gold seen in search", "Gold placed in top 10",
      "Seen gold that reached top 10", "Mean searches", "Mean NDCG@10 if O1 were scored as a perfect parse? (n/a)"]
H6 = H6[:-1]
os_.append([]); os_.append(H6); style_header(os_, 4, 1, len(H6))
for i, e in enumerate(OE):
    r = 5 + i; A = f"$A{r}"
    vals = list(e) + [f"=COUNTIFS({ro('Exp ID')},{A})", f"=AVERAGEIFS({ro('NDCG@10')},{ro('Exp ID')},{A})"]
    vals += [f'=COUNTIFS({ro("Exp ID")},{A},{ro("Failure category")},"{k}*")' for k in ("Hit", "O1", "O2", "O3")]
    vals += [f"=IFERROR({CL(6 + k)}{r}/$D{r},0)" for k in range(4)]
    vals += [f"=SUMIFS({ro('# gold')},{ro('Exp ID')},{A})", f"=SUMIFS({ro('Gold seen (count)')},{ro('Exp ID')},{A})",
             f"=SUMIFS({ro('Gold in top 10 (count)')},{ro('Exp ID')},{A})", f"=IFERROR(P{r}/O{r},0)",
             f'=IFERROR(AVERAGEIFS({ro("Searches")},{ro("Exp ID")},{A}),"—")']
    for j, v in enumerate(vals, 1):
        os_.cell(r, j, v)
body(os_, 5, 4 + len(OE), 1, len(H6), {5: "0.0000", 10: "0.0%", 11: "0.0%", 12: "0.0%", 13: "0.0%", 17: "0.0%", 18: "0.0"})
widths(os_, [8, 9, 12, 9, 10, 7, 9, 9, 9, 7, 7, 7, 7, 8, 9, 9, 10, 9]); os_.row_dimensions[4].height = 58
nr = 6 + len(OE)
for k, t in enumerate([
        "Gold seen in search: snippet rows use the harness's own gold_recall (gold among all documents its searches returned); full-doc rows use retrieved_docids ∩ gold. "
        "'Seen gold that reached top 10' is gold placed in the top 10 over gold ever retrieved: the ranking / selection step's yield.",
        "An O1 snippet answer is typically a final answer that is still a ```repl code block or prose without doc ids (the snippet harness has a 12-turn cap and no forced ranking at the cap)."]):
    os_.cell(nr + k, 1, t).font = Font(name=FONT, size=9, italic=True)

# ---------------------------------------------------------------- Error samples
es = wb.create_sheet("Error Samples", 7)
title(es, "Sample wrong answers for manual review",
      "Up to 25 per (experiment, category), F3–F5 plus near misses, from the best GPT-5 rows. Gold answers are decrypted from BrowseComp-Plus: keep this file out of public places.")
H7 = ["Exp ID", "QID", "Failure category", "Gold recall", "Searches", "Stated confidence", "Near miss", "Exact answer", "Gold answer", "Question"]
es.append([]); es.append(H7); style_header(es, 4, 1, len(H7))
r = 5
for exp in ("BCP-23", "BCP-11", "BCP-25", "FR-11", "FR-7"):
    for cat in ("F5", "F4", "F3"):
        xs = [o for o in QA if o["exp"] == exp and o["category"].startswith(cat)]
        xs.sort(key=lambda o: (-o["near"], int(o["qid"])))
        for o in xs[:25]:
            for j, v in enumerate([o["exp"], o["qid"], o["category"], o["recall"], o["searches"], o["confidence"], o["near"],
                                   o["exact_answer"], o["gold"], o["question"]], 1):
                c = es.cell(r, j, v); c.font = Font(name=FONT, size=9); c.alignment = Alignment(wrap_text=True, vertical="top")
            r += 1
widths(es, [8, 6, 26, 8, 8, 9, 7, 40, 30, 80]); es.freeze_panes = "A5"; es.auto_filter.ref = f"A4:J{r - 1}"

# ---------------------------------------------------------------- Read Me
rm = wb.create_sheet("Read Me", 0)
title(rm, "Failure analysis of docs/Experiments.xlsx")
lines = [
    ("What this is", "A per-question failure analysis of every filled result row of docs/Experiments.xlsx (BrowseComp+, FRAMES, OBLIQ Math / Twitter / Congress). "
     "WildChat and Writing have no runs yet. Paper baselines and the OBLIQ oracle have no per-question data and are not analysed."),
    ("How it is built", "Detail QA / Detail OBLIQ hold one line per (experiment row, question), extracted from the run logs. All other sheets are formulas over them, "
     "so filtering Detail QA reproduces any number. Each row's accuracy / NDCG@10 equals the Experiments.xlsx value (checked)."),
    ("Exp ID", "Dataset prefix + row number in Experiments.xlsx: BCP-23 = BrowseComp+ sheet row 23 (Full-Doc RLM, GPT-5 high, Reason-ModernColBERT), FR-7 = FRAMES row 7, "
     "MAT/TWI/CON-9 = OBLIQ snippet RLM, -11 = OBLIQ full-doc RLM."),
    ("Judge", "BrowseComp+ and FRAMES correctness: gpt-4.1, official BrowseComp-Plus template, 3 votes (logs/bcp_rejudge_gpt41/, logs/run5d/full/*/graded_gpt41.json, "
     "the trace in docs/supervisor_artifact.html for BCP-21/23). OBLIQ: NDCG@10 against gold qrels, ranking re-parsed with examples/obliq_data.py."),
    ("Sheets", "Key Findings · QA Summary (failure categories per row) · By Recall (accuracy given retrieved evidence) · By Searches · Pairwise (snippet vs full-doc, model and retriever swaps) · "
     "BCP Difficulty · FRAMES Types · OBLIQ Summary · Error Samples · BCP / FRAMES by Question (wide) · Detail QA · Detail OBLIQ · By Turns (when the answer was reached, turn-limit exhaustion)."),
    ("Sources per row", "BCP-9 logs/rlmfix32_q*.json · BCP-11 logs/rlmfix{24,25,29,30,31}_q*.json · BCP-13 logs/run5d/full/bcp_snip_gpt5_bm25 · BCP-19 logs/run5d/full/bcp_snip_qwen_bm25 (running) · "
     "BCP-21/23 docs/supervisor_artifact.html · BCP-25 logs/bcp_gpt5_bm25/runs · BCP-27 logs/run5d/bcp_fd_qwen_dense · BCP-29 logs/{gepa_bcp2,bcp_testrest_ctrl,gepa_bcp}/cache.json · "
     "BCP-31 logs/bcp_bm25_{test200,rest_gpu0,rest_gpu1}/cache.json · FR-7/9 logs/run5d/full/frames_snip_{gpt5,qwen} · FR-11 logs/run5d/full/frames_fd_gpt5 · FR-13 logs/frames/sup_q36_bm25f · "
     "OBLIQ logs/run5d/full/obliq_snip_*/all.json and suprlm/runs/obliq_fd_*."),
    ("Turns", "Turns = root-model REPL turns until the final answer. Snippet rows: counted from the trajectory logs (the harness's own 'iterations' field is broken), "
     "BCP-9 / BCP-11 from the September token maps (scripts/workbook/tokmaps; 3 questions each unmatched). Full-doc rows: n_iterations. Turn limits: full-doc 30; "
     "snippet BrowseComp+ GPT-5 100, Qwen3.6 50; snippet FRAMES 50; snippet OBLIQ 12. Built by scripts/failure_analysis_turns.py + failure_analysis_turns_merge.py."),
    ("Caveats", "Recall kind differs by row (see QA Summary notes). "
     "Abstention detection is a regex on the 'Exact Answer' line. BCP-19 is partial (455 attempted at extraction time, 2026-10-29)."),
    ("Built", "2026-10-29, by scratchpad scripts extract.py + build.py (Claude Code session)."),
]
for k, (a, b) in enumerate(lines):
    rm.cell(3 + k, 1, a).font = Font(name=FONT, bold=True); c = rm.cell(3 + k, 2, b)
    c.font = Font(name=FONT, size=10); c.alignment = Alignment(wrap_text=True, vertical="top")
    rm.cell(3 + k, 1).alignment = Alignment(vertical="top")
    rm.row_dimensions[3 + k].height = max(18, 15 * (len(b) // 120 + 1))
widths(rm, [18, 140])

for _x in sys.argv[3:]:
    exec(open(_x).read())
wb.calculation.fullCalcOnLoad = True
wb.save(OUTP)
json.dump({"exps": EXPS, "pairs": PAIRS, "oe": OE}, open(OUTP + ".meta.json", "w"))
print("saved", OUTP, len(QA), len(OB))
