# exec'd by build.py after extras.py: the 'By Turns' sheet (when the answer was reached, turn-limit exhaustion, turn
# counts) and its charts on 'Charts'. Needs per_question.json merged by failure_analysis_turns_merge.py.
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.chart.series import SeriesLabel

TOUT = [("T1", "Answered before the limit, correct", "B4B2AA"), ("T2", "Answered before the limit, wrong", "E87BA4"),
        ("T3", "Gave up before the limit (abstained)", "EB6834"), ("T4", "Used every turn, still correct", "6F6D66"),
        ("T5", "Used every turn, no correct answer", "EDA100"), ("T6", "Stopped by time limit / error", "2A78D6"),
        ("T0", "Turns not logged", "E5E4DF")]
TB = [("1-3", ">=1", "<=3"), ("4-6", ">=4", "<=6"), ("7-10", ">=7", "<=10"), ("11-15", ">=11", "<=15"),
      ("16-20", ">=16", "<=20"), ("21-29", ">=21", "<=29"), ("30+", ">=30", None)]

bt = wb.create_sheet("By Turns", wb.sheetnames.index("By Searches") + 1)
title(bt, "Turns: when the answer was reached, and when the turn budget ran out",
      "A turn is one root-model REPL step. 'Used every turn' = the run reached its turn limit (full-doc 30; snippet BrowseComp+ GPT-5 100, Qwen3.6 50; snippet FRAMES 50; snippet OBLIQ 12). "
      "At the limit the harness forces a final answer, so a run can still be right there.")

# ---- block A: turn outcome per run
HA = ["Exp ID", "Dataset", "Harness", "Model", "Retriever", "Turn limit", "Questions", "Turns logged", "Mean turns",
      "Mean turns, correct", "Mean turns, not correct"] + [f"{k} {n}" for k, n, _ in TOUT] + \
     ["Used every turn (T4+T5)", "Used every turn, share", "Used every turn, no correct answer: share of questions",
      "Accuracy when every turn was used", "Accuracy when answered before the limit"]
bt.append([]); bt.append(HA); style_header(bt, 4, 1, len(HA))
for i, e in enumerate(EXPS):
    r = 5 + i; A = f"$A{r}"; ex = f'{rq("Exp ID")},{A}'
    v = [e[0], e[1], e[3], e[4], e[5], f'=AVERAGEIFS({rq("Turn limit")},{ex})', f"=COUNTIFS({ex})",
         f'=COUNTIFS({ex},{rq("Turns")},">=0")',
         f'=IFERROR(AVERAGEIFS({rq("Turns")},{ex}),"—")',
         f'=IFERROR(AVERAGEIFS({rq("Turns")},{ex},{rq("Correct")},1),"—")',
         f'=IFERROR(AVERAGEIFS({rq("Turns")},{ex},{rq("Correct")},0),"—")']
    v += [f'=COUNTIFS({ex},{rq("Turn outcome")},"{k}*")' for k, _, _ in TOUT]
    v += [f"=O{r}+P{r}", f"=IFERROR(S{r}/G{r},0)", f"=IFERROR(P{r}/G{r},0)", f'=IFERROR(O{r}/S{r},"—")',
          f'=IFERROR(L{r}/(L{r}+M{r}+N{r}),"—")']
    for j, x in enumerate(v, 1):
        bt.cell(r, j, x)
rA = 4 + len(EXPS)
body(bt, 5, rA, 1, len(HA), {9: "0.0", 10: "0.0", 11: "0.0", 20: "0.0%", 21: "0.0%", 22: "0.0%", 23: "0.0%"})
for k, (_, _, col) in enumerate(TOUT):
    bt.cell(4, 12 + k).fill = PatternFill("solid", fgColor=col)
    bt.cell(4, 12 + k).font = Font(name=FONT, bold=True, color="000000" if col in ("B4B2AA", "EDA100", "E5E4DF", "E87BA4") else "FFFFFF")

# ---- block B: turn at which the answer was given (finished questions)
rB = rA + 3
bt.cell(rB - 1, 1, "Turn at which the run ended (finished questions): how many, and how many of them were right").font = Font(name=FONT, bold=True)
HB = ["Exp ID", "Turn limit"] + [f"n {b[0]}" for b in TB] + [f"Acc. {b[0]}" for b in TB]
for j, x in enumerate(HB, 1):
    bt.cell(rB, j, x)
style_header(bt, rB, 1, len(HB))
for i, e in enumerate(EXPS):
    r = rB + 1 + i; ok = f'{rq("Exp ID")},$A{r},{rq("Status")},"ok"'
    crit = [f'{rq("Turns")},"{lo}"' + (f',{rq("Turns")},"{hi}"' if hi else "") for _, lo, hi in TB]
    v = [e[0], f"=F{5 + i}"] + [f"=COUNTIFS({ok},{c})" for c in crit]
    v += [f'=IFERROR(COUNTIFS({ok},{c},{rq("Correct")},1)/{CL(3 + k)}{r},"—")' for k, c in enumerate(crit)]
    for j, x in enumerate(v, 1):
        bt.cell(r, j, x)
rBN = rB + len(EXPS)
body(bt, rB + 1, rBN, 1, len(HB), {c: "0.0%" for c in range(10, 17)})
bt.cell(rBN + 1, 2, "Turns").font = Font(name=FONT, size=8, italic=True, color="595959")
for k, b_ in enumerate(TB):   # plain bucket labels for the chart axes
    bt.cell(rBN + 1, 3 + k, b_[0]).font = Font(name=FONT, size=8, italic=True, color="595959")

# ---- block C: OBLIQ
rC = rBN + 3
bt.cell(rC - 1, 1, "OBLIQ: turn use and ranking quality").font = Font(name=FONT, bold=True)
HC = ["Exp ID", "Task", "Harness", "Turn limit", "Questions", "Mean turns", "Used every turn", "Used every turn, share",
      "Mean NDCG@10, before the limit", "Mean NDCG@10, every turn used", "Hit %, before the limit", "Hit %, every turn used",
      "O1 (no usable ranking) with every turn used"]
for j, x in enumerate(HC, 1):
    bt.cell(rC, j, x)
style_header(bt, rC, 1, len(HC))
for i, e in enumerate(OE):
    r = rC + 1 + i; ex = f'{ro("Exp ID")},$A{r}'
    v = list(e) + [f'=AVERAGEIFS({ro("Turn limit")},{ex})', f"=COUNTIFS({ex})", f'=IFERROR(AVERAGEIFS({ro("Turns")},{ex}),"—")',
                   f'=COUNTIFS({ex},{ro("Hit turn cap")},1)', f"=IFERROR(G{r}/E{r},0)",
                   f'=IFERROR(AVERAGEIFS({ro("NDCG@10")},{ex},{ro("Hit turn cap")},0),"—")',
                   f'=IFERROR(AVERAGEIFS({ro("NDCG@10")},{ex},{ro("Hit turn cap")},1),"—")',
                   f'=IFERROR(COUNTIFS({ex},{ro("Hit turn cap")},0,{ro("Failure category")},"Hit*")/COUNTIFS({ex},{ro("Hit turn cap")},0),"—")',
                   f'=IFERROR(COUNTIFS({ex},{ro("Hit turn cap")},1,{ro("Failure category")},"Hit*")/G{r},"—")',
                   f'=COUNTIFS({ex},{ro("Hit turn cap")},1,{ro("Failure category")},"O1*")']
    for j, x in enumerate(v, 1):
        bt.cell(r, j, x)
rCN = rC + len(OE)
body(bt, rC + 1, rCN, 1, len(HC), {6: "0.0", 8: "0.0%", 9: "0.000", 10: "0.000", 11: "0.0%", 12: "0.0%"})
for k, t in enumerate([
        "Turn outcome (Detail QA column 'Turn outcome'): T1-T3 ended before the limit, T4-T5 reached it, T6 was killed (time limit, watchdog or error) before an answer, "
        "T0 has no trajectory to count (3 questions each in BCP-9 / BCP-11).",
        "BCP-21 / BCP-23 run with a 30-turn limit like the other full-doc rows. Snippet GPT-5 rows never come near their 100- or 50-turn limit; "
        "their long tail is in searches, not turns (see By Searches).",
        "Block B: '30+' is the turn limit for full-doc rows (30, sometimes logged as 31 with the forced final turn); for snippet rows it is just a long run."]):
    bt.cell(rCN + 2 + k, 1, t).font = Font(name=FONT, size=9, italic=True)
widths(bt, [8, 11, 12, 13, 22, 7, 9, 8, 8, 9, 9] + [11] * 7 + [10, 10, 12, 11, 11])
bt.row_dimensions[4].height = 72; bt.row_dimensions[rB].height = 30; bt.row_dimensions[rC].height = 58
bt.freeze_panes = "B5"

# ---- charts on 'Charts'
ch.cell(82, 1, "Turns: when the answer was reached and when the turn budget ran out (By Turns sheet)").font = Font(name=FONT, bold=True, size=12, color="1F3864")


def turn_outcome_chart(r1, r2, ttl):
    c = BarChart(); c.type = "bar"; c.grouping = "percentStacked"; c.overlap = 100; c.gapWidth = 40
    c.title = ttl; c.y_axis.number_format = "0%"; c.y_axis.majorGridlines = None
    for k in range(len(TOUT)):
        c.add_data(Reference(bt, min_col=12 + k, min_row=r1, max_row=r2), titles_from_data=False)
    for s, (_, nm, _) in zip(c.series, TOUT):
        s.tx = SeriesLabel(v=nm)
    c.set_categories(Reference(bt, min_col=1, min_row=r1, max_row=r2)); color_series(c, [t[2] for t in TOUT])
    c.x_axis.scaling.orientation = "maxMin"; c.legend.position = "b"; c.height = 9 + 0.5 * (r2 - r1); c.width = 22
    return c


ch.add_chart(turn_outcome_chart(5, 14, "BrowseComp+: did the run answer before the turn limit?"), "A84")
ch.add_chart(turn_outcome_chart(15, 18, "FRAMES: did the run answer before the turn limit?"), "N84")

c = BarChart(); c.type = "bar"; c.title = "Mean turns: correct vs not-correct questions"; c.gapWidth = 50
c.add_data(Reference(bt, min_col=10, max_col=11, min_row=4, max_row=rA), titles_from_data=True)
c.set_categories(Reference(bt, min_col=1, min_row=5, max_row=rA)); color_series(c, [GREY, PAL[1]])
c.x_axis.scaling.orientation = "maxMin"; c.legend.position = "b"; c.height = 11; c.width = 16
ch.add_chart(c, "A106")

PICK = ["BCP-23", "BCP-25", "BCP-31", "FR-11", "FR-13"]
c = BarChart(); c.type = "col"; c.title = "Turn at which the run ended (finished questions)"; c.gapWidth = 40
for k, e in enumerate(PICK):
    r = rB + 1 + BEXPS.index(e) if e in BEXPS else rB + 1 + len(BEXPS) + FEXPS.index(e)
    c.add_data(Reference(bt, min_col=3, max_col=9, min_row=r, max_row=r), from_rows=True, titles_from_data=False)
for s, e in zip(c.series, PICK):
    s.tx = SeriesLabel(v=e)
c.set_categories(Reference(bt, min_col=3, max_col=9, min_row=rBN + 1, max_row=rBN + 1)); color_series(c, PAL)
c.x_axis.title = "Turns"; c.y_axis.title = "Questions"; c.legend.position = "b"; c.height = 11; c.width = 22
ch.add_chart(c, "K106")

c = LineChart(); c.title = "Accuracy by the turn the run ended on"; c.y_axis.number_format = "0%"; c.y_axis.scaling.min = 0; c.y_axis.scaling.max = 1
for k, e in enumerate(PICK):
    r = rB + 1 + BEXPS.index(e) if e in BEXPS else rB + 1 + len(BEXPS) + FEXPS.index(e)
    c.add_data(Reference(bt, min_col=10, max_col=16, min_row=r, max_row=r), from_rows=True, titles_from_data=False)
for s, e, col in zip(c.series, PICK, PAL):
    s.tx = SeriesLabel(v=e); s.graphicalProperties.line.solidFill = col; s.graphicalProperties.line.width = 28000; s.smooth = False
c.set_categories(Reference(bt, min_col=3, max_col=9, min_row=rBN + 1, max_row=rBN + 1))
c.x_axis.title = "Turns"; c.legend.position = "b"; c.height = 9; c.width = 22
ch.add_chart(c, "A130")

c = BarChart(); c.type = "bar"; c.title = "OBLIQ: share of questions that used every turn"; c.gapWidth = 40
c.add_data(Reference(bt, min_col=8, min_row=rC, max_row=rCN), titles_from_data=True)
c.set_categories(Reference(bt, min_col=1, min_row=rC + 1, max_row=rCN)); color_series(c, [PAL[3]])
c.y_axis.number_format = "0%"; c.y_axis.scaling.min = 0; c.y_axis.scaling.max = 1; c.legend = None
c.x_axis.scaling.orientation = "maxMin"; c.height = 8; c.width = 14
ch.add_chart(c, "N130")
