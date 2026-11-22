# exec'd by build.py after findings.py: 'Failure Cases' (descriptions + real examples) and 'Charts' (native Excel charts).
from openpyxl.chart import BarChart, Reference
from openpyxl.chart.label import DataLabelList

PAL = ["2A78D6", "EB6834", "1BAF7A", "EDA100", "E87BA4"]; GREY = "B4B2AA"

# ---------------------------------------------------------------- Failure Cases
fc = wb.create_sheet("Failure Cases", 2)
title(fc, "Failure cases: what each category means, how it shows up, and real examples",
      "One block per category. Where it dominates = rows of QA Summary / OBLIQ Summary with the highest share. Examples are real questions from the runs (Detail QA / Detail OBLIQ).")


def pick(exp, cat, n=2, near_first=False, maxlen=160):
    xs = [o for o in QA if o["exp"] == exp and o["category"].startswith(cat) and len(o["question"]) <= 300]
    xs.sort(key=lambda o: (-(o["near"] if near_first else 0), len(o["exact_answer"] or "") > maxlen, int(o["qid"])))
    return xs[:n]


DESC = [
 ("F1 Run did not finish", "#2A78D6",
  "The question produced no answer the judge could grade. Causes: killed at the per-question time limit (Qwen3.6 runs: 1 h for BrowseComp+ snippet, 2 h for FRAMES snippet, runaway kill for full-doc), the $10 cost watchdog, an API error (Azure 429 rate limit, 400 content filter).",
  "Detected from the run logs: no result file, status != completed, or an error field.",
  "Qwen3.6 only, in practice: BCP-19 (243 questions), BCP-15 (190), BCP-27 (121), FR-9 (75), BCP-17 (68), BCP-29 (53). GPT-5 rows have at most 9.",
  "Long Qwen3.6 trajectories: many turns at 8k output tokens each, plus servers shared by 8-20 workers. The model rarely stops on its own when the evidence is missing.",
  "Cap turns / tokens and force a final answer at the cap instead of killing the process; give Qwen3.6 an explicit 'answer now' turn.",
  [("BCP-27", "F1"), ("FR-9", "F1")]),
 ("F2 No / abstained answer", "#EB6834",
  "The run finished but gave no usable answer: the 'Exact Answer' line is empty, or the model explicitly abstained ('Not found in the provided corpus', 'Unable to determine from the retrieved documents').",
  "Regex on the final 'Exact Answer:' line (unknown / not found / unable / cannot / insufficient / N/A, under 120 characters).",
  "BCP-25 full-doc BM25 (118; 43 of them at the 30-turn cap), BCP-13 snippet BM25 (57), BCP-9 snippet dense (45).",
  "The agent knows it lacks the evidence. In full-doc runs, the 10-search cap is spent; in snippet runs, the model gives up even with budget left.",
  "Reject an abstention while search budget remains and ask for new query angles; make the search cap adaptive for full-doc.",
  [("BCP-25", "F2"), ("BCP-9", "F2")]),
 ("F3 Retrieval miss (no gold doc)", "#1BAF7A",
  "A confident wrong answer, and none of the question's gold documents was ever retrieved. The agent never saw the evidence, so it answered from a wrong candidate.",
  "Gold recall = 0 for the question (documents retrieved over the whole trajectory vs the gold qrels).",
  "Weak retrievers and weak searchers: BCP-31 Qwen BM25 (543), BCP-27 (260), BCP-29 (203), BCP-13 (123), BCP-25 (112). Small in GPT-5 dense / ColBERT runs (24-58).",
  "BM25 needs the exact vocabulary of the gold page. Qwen3.6 issues fewer, less varied queries. The model then settles on a near-match entity.",
  "Better first-stage retriever (ColBERT > dense > BM25 here), a reranker, query decomposition per clue; require a citation that contains the answer.",
  [("BCP-23", "F3"), ("FR-11", "F3")]),
 ("F4 Partial evidence, wrong answer", "#EDA100",
  "Some, but not all, gold documents were retrieved. One hop of the multi-hop chain was missing or resolved to the wrong entity, so the final answer is wrong.",
  "0 < gold recall < 1.",
  "The largest category for the strongest runs: BCP-21 (77), BCP-23 (54), FR-13 (192), FR-9 (146), FR-7 (99), FR-11 (85).",
  "The agent stops once one chain looks plausible and does not check every constraint of the question against a document.",
  "Per-constraint evidence table before answering (each clue → document id); search again for any clue without a source.",
  [("BCP-23", "F4"), ("FR-11", "F4")]),
 ("F5 All gold found, wrong answer", "#E87BA4",
  "Every gold document was retrieved and the answer is still wrong: a reading, counting, arithmetic or verification error. Near-miss answers (close to the gold string) are flagged; some may be judge strictness.",
  "Gold recall = 1. 'Near miss' flag = string containment or similarity ≥ 0.6 with the gold answer.",
  "Snippet RLM on BrowseComp+ (BCP-9: 43, BCP-11: 39, BCP-13: 38 vs 15-16 for full-doc GPT-5); FRAMES everywhere (55-67), mostly numerical and tabular questions.",
  "Snippets show only the first 512 tokens; the detail sits deeper in the document. On FRAMES, arithmetic over tables and dates done in the model's head.",
  "Read the whole gold document before answering (get_document + llm_query on the full text); compute numbers in Python from extracted rows.",
  [("BCP-11", "F5"), ("FR-11", "F5")]),
]
fc.append([]); fc.append(["Category", "What it means", "How it is detected", "Where it dominates", "Typical cause", "What would fix it"])
style_header(fc, 4, 1, 6)
r = 5
for name, color, mean, det, where, cause, fix, ex in DESC:
    for j, v in enumerate([name, mean, det, where, cause, fix], 1):
        c = fc.cell(r, j, v); c.font = Font(name=FONT, size=10, bold=(j == 1), color=("FFFFFF" if j == 1 else "000000"))
        c.alignment = Alignment(wrap_text=True, vertical="top"); c.border = BOX
    fc.cell(r, 1).fill = PatternFill("solid", fgColor=color.strip("#"))
    fc.row_dimensions[r].height = 120
    r += 1
    fc.cell(r, 2, "Examples").font = Font(name=FONT, size=9, bold=True, italic=True)
    hdr = ["", "Run · QID", "Question (shortened)", "Model's answer", "Gold answer", "Recall · searches · confidence"]
    for j, v in enumerate(hdr[1:], 2):
        fc.cell(r, j, v).font = Font(name=FONT, size=9, bold=True); fc.cell(r, j).fill = SF
    r += 1
    for exp, cat in ex:
        for o in pick(exp, cat, 2, near_first=(cat == "F5")):
            vals = [f"{o['exp']} · q{o['qid']} ({o['run'].replace(' (partial)', '')})", o["question"][:300],
                    o["exact_answer"] or f"(no answer: {o['status']})", o["gold"],
                    f"recall {o['recall'] if o['recall'] is not None else '—'} · {o['searches'] if o['searches'] is not None else '—'} searches · conf {o['confidence'] if o['confidence'] is not None else '—'}"]
            for j, v in enumerate(vals, 2):
                c = fc.cell(r, j, v); c.font = Font(name=FONT, size=9); c.alignment = Alignment(wrap_text=True, vertical="top")
            fc.row_dimensions[r].height = 62; r += 1
    r += 1

ODESC = [
 ("O1 No usable ranking", "#2A78D6", "The final answer contains no document ids the scorer can parse, or the run errored. NDCG@10 = 0 regardless of what was found.",
  "Snippet RLM: Math 14, Twitter 37, Congress 39 questions; full-doc: 1 each.",
  "The snippet harness has a 12-turn cap; at the cap the last turn is often still a ```repl code block or prose instead of the ranking.",
  "Force a ranking at the turn cap (the full-doc harness does this); validate the output format before submitting."),
 ("O2 Gold never retrieved", "#EB6834", "No gold document ever came up in any of the agent's searches.",
  "Congress: 83% of questions (full-doc), 56% (snippet). Twitter full-doc 30%.",
  "Tip-of-the-tongue descriptions (tone, rhetorical move) share almost no words with the passage; 10 searches are too few.",
  "More, more varied searches (snippet runs 54.5 per question and sees twice the gold); paraphrase the recollection into likely spoken phrases."),
 ("O3 Gold retrieved, not ranked in top 10", "#1BAF7A", "Gold appeared in the search results, but the agent's final top 10 left it out.",
  "Math (29%), Twitter (30-35%). Only 24-34% of the gold ever retrieved reaches the top 10.",
  "Relevance is implicit (same technique, implied stance); the model ranks surface-similar documents higher.",
  "A listwise rerank over all retrieved candidates before answering (the paper's oracle tournament shows the headroom: NDCG@10 0.28-0.33)."),
]
fc.cell(r, 1, "OBLIQ categories").font = Font(name=FONT, bold=True, size=12, color="1F3864"); r += 1
for j, v in enumerate(["Category", "What it means", "Where it dominates", "Typical cause", "What would fix it"], 1):
    fc.cell(r, j, v)
style_header(fc, r, 1, 5); r += 1
for name, color, mean, where, cause, fix in ODESC:
    for j, v in enumerate([name, mean, where, cause, fix], 1):
        c = fc.cell(r, j, v); c.font = Font(name=FONT, size=10, bold=(j == 1), color=("FFFFFF" if j == 1 else "000000"))
        c.alignment = Alignment(wrap_text=True, vertical="top"); c.border = BOX
    fc.cell(r, 1).fill = PatternFill("solid", fgColor=color.strip("#")); fc.row_dimensions[r].height = 80; r += 1
widths(fc, [26, 46, 40, 40, 40, 40]); fc.freeze_panes = "A5"

# ---------------------------------------------------------------- Charts
ch = wb.create_sheet("Charts", 3)
title(ch, "Charts", "Native Excel charts over the summary sheets; they update when the summaries change. Colours: F1 blue, F2 orange, F3 aqua, F4 yellow, F5 pink, correct grey.")
Q = wb["QA Summary"]; BRs = wb["By Recall"]; PWs = wb["Pairwise"]; BDs = wb["BCP Difficulty"]; FTs = wb["FRAMES Types"]; OSs = wb["OBLIQ Summary"]


def color_series(c, cols):
    for s, col in zip(c.series, cols):
        s.graphicalProperties.solidFill = col; s.graphicalProperties.line.solidFill = col


def outcome_chart(r1, r2, ttl):
    c = BarChart(); c.type = "bar"; c.grouping = "percentStacked"; c.overlap = 100; c.gapWidth = 40
    c.title = ttl; c.y_axis.number_format = "0%"; c.y_axis.majorGridlines = None
    for col in [8] + list(range(11, 16)):
        ref = Reference(Q, min_col=col, min_row=r1, max_row=r2)
        c.add_data(ref, titles_from_data=False)
    from openpyxl.chart.series import SeriesLabel
    for s, nm in zip(c.series, ["Correct"] + [x[1] for x in CATS]):
        s.tx = SeriesLabel(v=nm)
    c.set_categories(Reference(Q, min_col=1, min_row=r1, max_row=r2))
    color_series(c, [GREY] + PAL); c.x_axis.scaling.orientation = "maxMin"
    c.legend.position = "b"; c.height = 11; c.width = 22
    return c


ch.add_chart(outcome_chart(5, 4 + len(BEXPS), "BrowseComp+: outcome of every question, per run"), "A4")
ch.add_chart(outcome_chart(5 + len(BEXPS), 4 + len(EXPS), "FRAMES: outcome of every question, per run"), "N4")

c = BarChart(); c.type = "col"; c.title = "Accuracy given retrieved gold evidence"; c.y_axis.number_format = "0%"; c.y_axis.scaling.min = 0; c.y_axis.scaling.max = 1
c.add_data(Reference(BRs, min_col=9, max_col=11, min_row=4, max_row=4 + len(EXPS)), titles_from_data=True)
c.set_categories(Reference(BRs, min_col=1, min_row=5, max_row=4 + len(EXPS))); color_series(c, [PAL[2], PAL[3], PAL[0]])
c.legend.position = "b"; c.height = 9; c.width = 22
ch.add_chart(c, "A26")

c = BarChart(); c.type = "bar"; c.title = "Paired runs: questions only A / only B gets right"; c.gapWidth = 60
c.add_data(Reference(PWs, min_col=5, max_col=6, min_row=4, max_row=4 + len(PAIRS)), titles_from_data=True)
c.set_categories(Reference(PWs, min_col=1, min_row=5, max_row=4 + len(PAIRS))); color_series(c, [PAL[0], PAL[1]])
c.x_axis.scaling.orientation = "maxMin"; c.legend.position = "b"; c.height = 11; c.width = 22
ch.add_chart(c, "N26")

c = BarChart(); c.type = "col"; c.title = "BrowseComp+: how many of the 6 GPT-5 runs solve a question"; c.gapWidth = 30
c.add_data(Reference(BDs, min_col=2, min_row=4, max_row=11), titles_from_data=True)
c.set_categories(Reference(BDs, min_col=1, min_row=5, max_row=11)); color_series(c, [PAL[0]])
c.legend = None; c.x_axis.title = "GPT-5 runs that solve it"; c.y_axis.title = "Questions"; c.height = 8; c.width = 16
c.dataLabels = DataLabelList(); c.dataLabels.showVal = True
ch.add_chart(c, "A46")

c = BarChart(); c.type = "col"; c.title = "FRAMES: accuracy by reasoning type"; c.y_axis.number_format = "0%"; c.y_axis.scaling.min = 0.4; c.y_axis.scaling.max = 0.9
for rr in range(6, 10):
    c.add_data(Reference(FTs, min_col=4, max_col=8, min_row=rr, max_row=rr), from_rows=True, titles_from_data=False)
from openpyxl.chart.series import SeriesLabel
for s, rr in zip(c.series, range(6, 10)):
    s.tx = SeriesLabel(v=FTs.cell(rr, 1).value)
c.set_categories(Reference(FTs, min_col=4, max_col=8, min_row=4, max_row=4)); color_series(c, PAL[:4])
c.legend.position = "b"; c.height = 8; c.width = 22
ch.add_chart(c, "N46")

c = BarChart(); c.type = "bar"; c.grouping = "percentStacked"; c.overlap = 100; c.gapWidth = 40; c.title = "OBLIQ: outcome per question"
c.add_data(Reference(OSs, min_col=6, max_col=9, min_row=4, max_row=10), titles_from_data=True)
c.set_categories(Reference(OSs, min_col=1, min_row=5, max_row=10)); color_series(c, [GREY] + PAL[:3])
c.y_axis.number_format = "0%"; c.x_axis.scaling.orientation = "maxMin"; c.legend.position = "b"; c.height = 8; c.width = 16
ch.add_chart(c, "A64")
