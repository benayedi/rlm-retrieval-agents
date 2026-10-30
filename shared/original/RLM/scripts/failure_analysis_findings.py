# exec'd by build.py: the Key Findings sheet. Each finding cites a live cell (Evidence column) on the sheets it summarises.
kf = wb.create_sheet("Key Findings", 1)
title(kf, "Key findings", "Narrative numbers come from the cited cells (column D shows the live value). Rows of QA Summary / By Recall: BCP-9 r5, -11 r6, -13 r7, -19 r8, -21 r9, -23 r10, -25 r11, -27 r12, -29 r13, -31 r14, FR-7 r15, -9 r16, -11 r17, -13 r18.")
F = [
 ("BrowseComp+", "Retrieval, not reading, limits the best runs.",
  "In the best run (BCP-23, full-doc, GPT-5, ColBERT), 75% of the 104 failures are on the retrieval side (24 never saw a gold document, 54 saw only part of the evidence). Only 15 (14%) are wrong although every evidence document was retrieved. When all evidence is retrieved, the full-doc GPT-5 runs are right 96% of the time.",
  "='QA Summary'!U10", "Retrieval-side share of BCP-23 failures", "QA Summary U10; By Recall K9:K10",
  "Spend effort on retrieval (reranker, query reformulation, more searches when evidence is partial) rather than on the reading prompt."),
 ("BrowseComp+", "BM25 fails through retrieval and runs out of budget.",
  "Swapping ColBERT for BM25 in the full-doc GPT-5 harness costs 255 questions and wins back 27. On the 255, BM25's run abstains or hits the turn cap on 93 ('Unable to determine from the retrieved documents'), never sees gold on 87 and sees partial evidence on 69. Questions with no gold retrieved rise from 31 to 219.",
  "=Pairwise!E11", "Questions only ColBERT gets right (BCP-23 vs BCP-25)", "Pairwise row 11; By Recall F10:F11",
  "With BM25, lift the 10-search cap or add a reranker. The abstentions show the model knows it is missing evidence but cannot search further."),
 ("BrowseComp+", "Snippet RLM loses on reading; full-doc RLM loses on search budget.",
  "With the dense retriever, snippet RLM makes 43 'all gold found, wrong' errors against 16 for full-doc, and abstains 45 times against 20. At recall = 1 it is right 91% of the time against 96% for full-doc: 512-token snippets hide details. With BM25 it is the other way round: snippet wins 136 to 74, because full-doc's 10-search cap leaves it abstaining (52 of its 136 losses) or without gold (37).",
  "='QA Summary'!O5", "Snippet dense (BCP-9): F5 errors", "QA Summary O5 / O9; By Recall K5 vs K9; Pairwise rows 5, 7",
  "Hybrid: give the snippet RLM a full-document read (or llm_query on the whole document) before it answers, and give full-doc an adaptive search cap."),
 ("BrowseComp+", "Snippet RLM gives up too early.",
  "Snippet GPT-5 runs abstain on 29-57 questions ('Not found in the provided corpus') even though they have no search cap. That is about a quarter of their failures (26% for BCP-9, 25% for BCP-11).",
  "='QA Summary'!Q5", "Abstention share of BCP-9 failures", "QA Summary L5:L7, Q5:Q7",
  "Add a prompt rule or verifier that rejects 'not found' while the budget is unused, and asks for new query angles."),
 ("BrowseComp+", "Qwen3.6 has a stamina and search problem, not a reading problem.",
  "Qwen3.6 full-doc runs fail mostly by never retrieving gold (260 / 203 / 543 questions on dense / ColBERT / BM25) and by not finishing (121 / 53 / 18 killed). When Qwen3.6 does have every gold document, it is right 89% of the time (GPT-5: 96%). The running Qwen snippet BM25 row has 137 of 455 attempted questions killed at the 1 h limit.",
  "='By Recall'!K12", "Qwen3.6 full-doc dense, accuracy at recall = 1", "QA Summary K12:M14, K8; By Recall K12:K14",
  "For Qwen3.6, invest in search guidance (query decomposition, reranker; the reranker already gave +9.5 points) and wall-clock control, not in reading."),
 ("BrowseComp+", "There is a small hard core of questions.",
  "31 questions (3.7%) are solved by none of the 6 GPT-5 configurations, and 339 (41%) by all 6. On the hard core, every run splits its failures between abstaining, retrieval misses and wrong readings. Qwen3.6 full-doc solves only 54% of the questions all GPT-5 runs solve.",
  "='BCP Difficulty'!B5", "Questions no GPT-5 row solves", "BCP Difficulty B5:C11, rows 15-24, C34",
  "Use the hard-core list (BCP by Question: 'GPT-5 rows solved' = 0) as a focused dev set for retrieval work."),
 ("BrowseComp+ / FRAMES", "Confidence is not a reliable error signal for Qwen3.6.",
  "Qwen3.6 states ≥90% confidence on 82-84% of its wrong FRAMES answers and on 36-42% of its wrong BrowseComp+ answers. GPT-5 snippet runs do so on 5-12% of wrong BrowseComp+ answers.",
  "='QA Summary'!Y16", "FR-9 overconfident share of wrong answers", "QA Summary W:Y",
  "Do not use self-reported confidence to trigger re-tries for Qwen3.6. Use evidence checks instead (e.g. gold-recall proxies such as citation verification)."),
 ("FRAMES", "FRAMES errors are mainly incomplete evidence and wrong reasoning.",
  "In the best FRAMES run (FR-11, full-doc GPT-5), 53% of the 159 failures have partial evidence and 35% have all gold articles but a wrong answer. Only 15 questions saw no gold article. Even with no gold article retrieved, 71% are answered correctly (the model's own knowledge), so recall says less about FRAMES than about BrowseComp+.",
  "='QA Summary'!S17", "FR-11 partial-evidence share of failures", "QA Summary R17:T17; By Recall I15:K18",
  "Prompt for explicit per-hop evidence (one article per constraint) before answering, and add a calculation / table-reading step."),
 ("FRAMES", "Numerical and tabular questions are the weak spot.",
  "Full-doc GPT-5 gets 81% of multiple-constraint and temporal questions right, but 73% of numerical and 72% of tabular ones. Numerical and tabular are also among the three weakest types on every other FRAMES row; on Qwen3.6 full-doc, post-processing is lowest. Snippet RLM's 74 losses to full-doc are mostly partial evidence (30) and wrong reading with full evidence (28): the answer sits deeper in the article than the snippet reaches.",
  "='FRAMES Types'!G8", "FR-11 accuracy, tabular reasoning", "FRAMES Types D6:M9; Pairwise row 14",
  "Route numeric / table questions through a Python computation over the extracted table rows instead of in-head arithmetic."),
 ("OBLIQ Congress", "Congress fails at search, not ranking.",
  "Full-doc never retrieves the gold passage on 83% of Congress questions. When it does see the passage, it ranks it in the top 10 90% of the time. The snippet RLM sees more gold (79 vs 42 passages) through 5× more searches (54.5 vs 10), which is why it wins, but 15% of its answers contain no parsable ranking.",
  "='OBLIQ Summary'!L10", "CON full-doc: gold never retrieved", "OBLIQ Summary rows 9-10",
  "More diverse searches (paraphrases of tone / rhetorical move), plus a forced final ranking for the snippet RLM."),
 ("OBLIQ Math / Twitter", "Math and Twitter fail at ranking.",
  "On Math and Twitter, 29-35% of questions retrieve gold but do not place it in the top 10, and only 24-34% of all gold ever retrieved reaches the top 10. The snippet RLM also loses 9-13% of questions to unusable final answers (a ```repl block or prose without IDs at the 12-turn cap).",
  "='OBLIQ Summary'!Q7", "TWI snippet: seen gold that reaches top 10", "OBLIQ Summary rows 5-8",
  "Add a listwise rerank step over all retrieved candidates before the final answer (the paper's oracle tournament shows the headroom), and force a ranking at the turn cap."),
 ("BrowseComp+ / FRAMES / OBLIQ", "Running out of turns is a BM25 and Qwen3.6 problem, and long runs are failing runs.",
  "Full-doc GPT-5 with dense or ColBERT retrieval almost never uses all 30 turns (4 and 2 questions). With BM25 it does on 20% of questions, and only 35% of those end correct. "
  "Qwen3.6 full-doc uses every turn on 33% (ColBERT), 39% (dense) and 70% (BM25) of questions; with BM25 only 6% of those are right. Accuracy falls with the turn a run ends on: "
  "BCP-23 is right 98% of the time when it answers within 4-6 turns and 41% at 21-29. Snippet GPT-5 rows stop far below their limit (4.7-8.4 turns on average, limit 100). "
  "The OBLIQ snippet RLM uses all 12 turns on 78-87% of questions, and all of its unusable rankings (O1) come from those runs.",
  "='By Turns'!T14", "BCP-31: share of questions that used every turn", "By Turns rows 5-18 (S:W), turn-bucket block, OBLIQ block",
  "Treat a long trajectory as a failure signal: after ~15 turns without the evidence, switch strategy (new query angles, reranker) instead of continuing; force Qwen3.6 to answer earlier; give OBLIQ snippet a forced ranking at the limit."),
]
kf.append([]); kf.append(["Dataset", "Finding", "Detail", "Evidence (live)", "Evidence is", "Where to look", "Implication / next step"])
style_header(kf, 4, 1, 7)
for i, f in enumerate(F):
    r = 5 + i
    for j, v in enumerate(f, 1):
        c = kf.cell(r, j, v); c.font = Font(name=FONT, size=10, bold=(j == 2)); c.alignment = Alignment(wrap_text=True, vertical="top"); c.border = BOX
    kf.cell(r, 4).number_format = "0.0%" if i not in (1, 2, 5) else "0"
    kf.row_dimensions[r].height = 105
widths(kf, [13, 30, 70, 10, 22, 22, 45])
kf.freeze_panes = "A5"
