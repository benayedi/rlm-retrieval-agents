"""Write one Markdown document per experiment (docs/experiments/<agent>/<ID>_*.md) and the index
docs/experiments/README.md.

Facts come from experiments/registry.yaml. Every number comes from an evidence file:
- results/original_baselines/Experiments.xlsx (reported values);
- docs/verification/data/recompute_experiments.json (the independent recomputation);
- results/original_baselines/Failure_Analysis.xlsx (failure categories, recall-conditional accuracy, turn outcomes,
  paired comparisons).

Nothing is typed in by hand, so a document cannot disagree with the evidence. Text that interprets the numbers is
generated from them, with the rule that produced it stated alongside.

usage: python scripts/docs/make_experiment_docs.py
"""
import json, os, re
from pathlib import Path

import yaml
from openpyxl import load_workbook

from naming import paper_names

ROOT = Path(__file__).resolve().parents[2]
REG = yaml.safe_load(open(ROOT / "experiments/registry.yaml"))
DEF = REG["defaults"]
XL = load_workbook(ROOT / "results/original_baselines/Experiments.xlsx", data_only=True)
FA = load_workbook(ROOT / "results/original_baselines/Failure_Analysis.xlsx", data_only=True)
AUDIT = json.load(open(ROOT / "docs/verification/data/recompute_experiments.json"))
OUT = ROOT / "docs/experiments"
ALGO_NAME = {"snippet_repl": "Snippet REPL agent", "document_repl": "Document REPL agent", "dspy_rlm": "DSPy.RLM"}


def rel(target, frm):
    """Relative link from document `frm` to repo path `target`."""
    return os.path.relpath(ROOT / target, (ROOT / frm).parent)


def workbook_row(e):
    ws = XL[e["sheet"]]
    hdr = {ws.cell(3, c).value: c for c in range(1, ws.max_column + 1) if ws.cell(3, c).value}
    return {h: ws.cell(e["row"], c).value for h, c in hdr.items()}


def footnotes(sheet):
    ws = XL[sheet]; out = []
    for r in range(ws.max_row - 12, ws.max_row + 1):
        v = ws.cell(r, 1).value
        if isinstance(v, str) and (v.startswith("*") or v.startswith("†") or v.startswith("Input") or v.startswith("Paper") or v.startswith("Oracle")):
            out.append(v.strip())
    return out


def fa_row(sheet, exp_id, header_row=4):
    ws = FA[sheet]
    hdr = [c.value for c in ws[header_row]]
    for r in ws.iter_rows(min_row=header_row + 1, values_only=True):
        if r[0] == exp_id:
            return {h: v for h, v in zip(hdr, r) if h}
    return None


def turn_buckets(exp_id):
    ws = FA["By Turns"]
    for r in range(20, 40):
        if ws.cell(r, 1).value == "Exp ID" and str(ws.cell(r, 3).value).startswith("n "):
            hdr = [ws.cell(r, c).value for c in range(1, 17)]
            for rr in range(r + 1, r + 20):
                if ws.cell(rr, 1).value == exp_id:
                    return {h: ws.cell(rr, c + 1).value for c, h in enumerate(hdr) if h}
    return None


def pairs(exp_id):
    ws = FA["Pairwise"]; out = []
    for r in range(5, 30):
        a, b = ws.cell(r, 2).value, ws.cell(r, 3).value
        if exp_id in (a, b):
            out.append(dict(label=ws.cell(r, 1).value, a=a, b=b, both=ws.cell(r, 4).value, onlyA=ws.cell(r, 5).value,
                            onlyB=ws.cell(r, 6).value, neither=ws.cell(r, 7).value, p=ws.cell(r, 9).value))
    return out


def pct(x, d=1):
    return f"{100 * x:.{d}f}%" if isinstance(x, (int, float)) else "—"


def num(x, d=4):
    return f"{x:.{d}f}" if isinstance(x, float) else ("—" if x is None else str(x))


def table(rows, head):
    s = "| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n"
    return s + "".join("| " + " | ".join(str(c) for c in r) + " |\n" for r in rows)


def exists(p):
    return (ROOT / p).exists()


def doc(e):
    algo = e["agent"]; eid = e["id"]
    slug = re.sub(r"[^a-z0-9]+", "-", f"{e['benchmark'].split('(')[0]} {e['model'].split('(')[0]} {e['retriever'].split(', ')[0].split('(')[0]}".lower()).strip("-")
    path = f"docs/experiments/{algo}/{eid}_{slug}.md"
    w = workbook_row(e); a = AUDIT.get(eid, {}); rec = a.get("recomputed") or {}
    qa = e["sheet"] in ("BrowseComp+", "FRAMES")
    L = []
    L.append(f"# {eid}: {ALGO_NAME[algo]} · {e['benchmark'].split(' (')[0]} · {e['model'].split(' (')[0]} · {e['retriever'].split(', ')[0]}\n")
    status = {"complete": "complete", "running": "**running** at the time of writing (2026-10-30); numbers below are partial",
              "queued": "**not run yet** (queued)"}[e["status"]]
    L.append(f"Status: {status}. Row {e['row']} of the `{e['sheet']}` sheet of "
             f"[Experiments.xlsx]({rel('results/original_baselines/Experiments.xlsx', path)}).\n")

    # 1 overview
    L.append("## 1. Experiment overview\n")
    L.append(table([["Identifier", f"`{eid}` (sheet `{e['sheet']}`, row {e['row']})"],
                    ["Agent", f"[{ALGO_NAME[algo]}]({rel(f'agents/{algo}/README.md', path)})"],
                    ["Benchmark", e["benchmark"]], ["Model", e["model"]], ["Retriever", e["retriever"]],
                    ["Research question", e["question"]],
                    ["Original execution", e["dates"]], ["Run name in the logs", f"`{e['run_name']}`"]], ["", ""]))

    # 2 motivation
    L.append("## 2. Research motivation\n")
    L.append(f"{e['question']} This row is one cell of the comparison matrix in Experiments.xlsx: the same benchmark is run "
             f"with each agent × model × retriever combination, so this row is read against its neighbours (section 8). "
             f"See [AGENT_COMPARISON.md]({rel('docs/experiments/AGENT_COMPARISON.md', path)}) for the matrix and "
             f"[research_overview.md]({rel('docs/research_overview.md', path)}) for the research questions.\n")

    # 3 methodology
    L.append("## 3. Experimental methodology\n")
    if algo == "snippet_repl":
        L.append("The snippet REPL agent answers each question by writing Python in a persistent REPL, one turn at a time. "
                 "Its tools return search results as short snippets (the first tokens of each document) into a variable; "
                 "it can open a whole document and pass text to a sub-LLM (`llm_query`) to read it. "
                 "A run ends when the model sets a final answer or the turn limit is reached.")
    else:
        L.append("The document REPL agent answers each question by writing Python in a persistent REPL. Each search puts the "
                 "**whole** retrieved documents into a `DOCS` variable; the model reads slices of them itself (prints "
                 "are capped per code block). Searches are capped client-side; at the turn limit the harness forces a "
                 "final answer.")
    L.append(f" Agent details: [agents/{algo}/README.md]({rel(f'agents/{algo}/README.md', path)}).\n")
    L.append("Per question: the agent runs on one benchmark question with the budgets in section 4; the run record "
             "(answer, retrieved document ids, tool counts, turns, time, tokens) is written to the run directory.")
    if qa:
        L.append(f" Correctness is judged afterwards by {DEF['judge_qa']}. Recall is the share of the question's gold "
                 "documents retrieved at any point of the run (definitions per row: "
                 f"[RESULT_INTEGRITY.md]({rel('docs/verification/RESULT_INTEGRITY.md', path)}) §R1).\n")
    else:
        L.append(f" Scoring: {DEF['judge_obliq']}. A question without a parsable ranking scores 0.\n")

    # 4 configuration
    L.append("## 4. Configuration\n")
    b = e["budgets"]
    rows = [["Model", e["model"]], ["Reasoning effort requested", e["effort_requested"]],
            ["Reasoning effort actually sent", e["effort_sent"]], ["Retriever / index", e["retriever"]]]
    rows += [[k.replace("_", " ").capitalize(), v] for k, v in b.items()]
    rows += [["Random seed", "none set (API / vLLM sampling defaults); runs are not bit-reproducible"]]
    if e.get("harness_version"):
        rows.append(["Harness version", e["harness_version"]])
    L.append(table(rows, ["Setting", "Value"]))
    cfg = e["config"]
    if exists(cfg):
        L.append(f"\nExact environment and arguments ([{cfg}]({rel(cfg, path)})):\n\n```bash\n{(ROOT / cfg).read_text().strip()}\n```\n")
    else:
        L.append(f"\nConfiguration file `{cfg}`: **missing** (see verification status).\n")

    # 5 execution
    L.append("## 5. Execution instructions (reproduction, not executed for this documentation)\n")
    L.append("These commands launch the full experiment: hundreds of model calls and hours of GPU or API time. "
             "They were **not** run while preparing this repository.\n")
    L.append(f"```bash\nset -a; . configs/paths.env; . {cfg}; set +a\n# then the launcher; its header documents its arguments:\n{e['launcher']}\n```\n")
    L.append(f"Launcher: [{e['launcher']}]({rel(e['launcher'], path)}){'' if exists(e['launcher']) else ' (**missing**)'}. "
             "It reads the benchmark questions, talks to the retrieval server named in the config and writes one record per "
             "question (plus trajectories) under `$RUNS_DIR`. Grade afterwards with the matching script in "
             f"[agents/shared/rlm_shared/grading/]({rel('agents/shared/rlm_shared/grading', path)}).\n")

    # 6 implementation references
    L.append("## 6. Implementation references\n")
    src = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / f"agents/{algo}/src").rglob("*.py")) if exists(f"agents/{algo}/src") else []
    L.append(f"- Agent source: " + (", ".join(f"[{Path(p).name}]({rel(p, path)})" for p in src[:12]) or "—") + (" …" if len(src) > 12 else ""))
    L.append(f"- Original code that produced the result (verbatim, SHA-256 manifest): [agents/{algo}/original/]({rel(f'agents/{algo}/original/MANIFEST.json', path)})")
    L.append(f"- Configuration: [{cfg}]({rel(cfg, path)}) · Launcher: [{e['launcher']}]({rel(e['launcher'], path)})")
    L.append(f"- Data loaders: [agents/shared/rlm_shared/]({rel('agents/shared/rlm_shared/__init__.py', path)}) (`browsecomp_data.py`, `obliq_data.py`)")
    L.append(f"- Grading: [agents/shared/rlm_shared/grading/]({rel('agents/shared/rlm_shared/grading', path)}) · Metrics: [metrics.py]({rel('agents/shared/rlm_shared/metrics.py', path)})")
    L.append(f"- Tests: [agents/{algo}/tests/]({rel(f'agents/{algo}/tests', path)}), [tests/]({rel('tests', path)})")
    tm = ROOT / f"trajectories/{algo}/{eid}/MANIFEST.json"
    if tm.exists():
        tx = json.loads(tm.read_text())
        L.append(f"- **Per-question outputs in this repository** (Git LFS): [trajectories/{algo}/{eid}/]({rel(f'trajectories/{algo}/{eid}/MANIFEST.json', path)}). "
                 f"{tx['members']:,} files, {tx['member_bytes'] / 1e9:.2f} GB as written by the run"
                 f"{', encrypted with the BrowseComp canary key' if tx['encrypted'] else ''}; "
                 f"extract with `python scripts/provenance/unpack_trajectories.py {eid} OUT_DIR` "
                 f"([format notes]({rel('trajectories/README.md', path)}))")
    else:
        L.append(f"- Per-question outputs: **not in this repository** (see [trajectories/README.md]({rel('trajectories/README.md', path)}) for why)")
    L.append("- Original locations on the machine that ran the experiments: " + "; ".join(f"`{o}`" for o in e["original_outputs"]) + "\n")

    # 7 results
    L.append("## 7. Existing experimental results\n")
    L.append("Reported in Experiments.xlsx (every non-empty column of the row):\n")
    L.append(table([[k, v] for k, v in w.items() if k not in ("Harness", "Model (Reasoning)", "Retrieval Signal") and v not in (None, "")], ["Metric", "Value"]))
    model_cell = str(w.get("Model (Reasoning)") or ""); acc_cell = str(w.get("Accuracy") or "")
    notes = []
    for f in footnotes(e["sheet"]):
        tag = re.match(r"\*\s*(.+?\)\*)", f)
        if tag and tag.group(1) in model_cell and ("Snippet" in f) == (algo == "snippet_repl"):
            notes.append(f)
        elif f.startswith("†") and "†" in acc_cell:
            notes.append(f)
        elif f.startswith("Input"):
            notes.append(f)
    if notes:
        L.append("\nWorkbook footnotes that apply to this row:\n")
        L += [f"> {n}\n" for n in notes]
    if "deployment default" in e["effort_sent"] and "(high)*" in model_cell and algo == "snippet_repl":
        L.append("> **Correction (this audit):** the launcher of this run loaded no hook that forwards `reasoning_effort` to "
                 "Azure, so both the root model and the sub-LLM ran at the deployment's default effort (measured as medium "
                 f"for gpt-5), not high/medium as the footnote says. See [KNOWN_ISSUES.md]({rel('docs/verification/KNOWN_ISSUES.md', path)}) K3.\n")
    if e["status"] == "complete" and qa:
        fr = fa_row("QA Summary", eid); br = fa_row("By Recall", eid); bt = fa_row("By Turns", eid)
        if fr:
            L.append("\n**Failure breakdown** (from [Failure_Analysis.xlsx]"
                     f"({rel('results/original_baselines/Failure_Analysis.xlsx', path)}), QA Summary; first matching category wins):\n")
            cats = ["F1 did not finish", "F2 no / abstained", "F3 retrieval miss", "F4 partial evidence", "F5 all gold, wrong"]
            failed = fr["Failed"] or 0
            L.append(table([[c, fr[c], pct(fr[c] / failed if failed else 0)] for c in cats] + [["**Failed in total**", failed, ""]],
                           ["Category", "Questions", "Share of failures"]))
            if eid == "BCP-19":
                L.append("\n> The failure analysis was extracted on 2026-10-30 while this run was still going, from the 455 questions "
                         "attempted by then. The accuracy above (Experiments.xlsx) covers all 830.\n")
        if br:
            L.append("\n**Accuracy given the retrieved evidence** (By Recall sheet):\n")
            L.append(table([["no gold retrieved", br["n recall = 0"], pct(br["Acc. recall = 0"]) if isinstance(br["Acc. recall = 0"], float) else "—"],
                            ["some gold", br["n 0 < recall < 1"], pct(br["Acc. 0 < recall < 1"]) if isinstance(br["Acc. 0 < recall < 1"], float) else "—"],
                            ["all gold", br["n recall = 1"], pct(br["Acc. recall = 1"]) if isinstance(br["Acc. recall = 1"], float) else "—"]],
                           ["Evidence retrieved", "Questions", "Accuracy"]))
        if bt:
            L.append(f"\n**Turns** (By Turns sheet): limit {bt['Turn limit']:.0f}, mean {num(bt['Mean turns'], 1)} "
                     f"(correct {num(bt['Mean turns, correct'], 1)}, not correct {num(bt['Mean turns, not correct'], 1)}); "
                     f"{bt['Used every turn (T4+T5)']} questions used every turn "
                     f"({pct(bt['Used every turn, share'])}), accuracy when answered before the limit "
                     f"{pct(bt['Accuracy when answered before the limit']) if isinstance(bt['Accuracy when answered before the limit'], float) else '—'}.\n")
    if e["status"] == "complete" and not qa:
        orow = fa_row("OBLIQ Summary", eid)
        if orow:
            L.append("\n**Per-question outcome** (Failure_Analysis.xlsx, OBLIQ Summary):\n")
            L.append(table([["Hit (≥ 1 gold in top 10)", orow["Hit"], pct(orow["Hit %"])],
                            ["O1 no usable ranking", orow["O1 no usable ranking"], pct(orow["O1 %"])],
                            ["O2 gold never retrieved", orow["O2 gold never retrieved"], pct(orow["O2 %"])],
                            ["O3 retrieved, not ranked in top 10", orow["O3 retrieved, not top 10"], pct(orow["O3 %"])]],
                           ["Outcome", "Queries", "Share"]))
            L.append(f"\nGold documents retrieved at any point: {orow['Gold seen in search']} of {orow['Gold docs total']}; "
                     f"placed in the top 10: {orow['Gold placed in top 10']}.\n")

    # 8 interpretation
    L.append("## 8. Results interpretation\n")
    if e["status"] != "complete":
        L.append("Not interpreted: the run is not complete.\n")
    else:
        P = pairs(eid)
        if P:
            L.append("**Paired comparisons on the same questions** (Pairwise sheet; exact McNemar test on the discordant pairs):\n")
            rows = []
            for p in P:
                me_a = p["a"] == eid; other = p["b"] if me_a else p["a"]
                mine, theirs = (p["onlyA"], p["onlyB"]) if me_a else (p["onlyB"], p["onlyA"])
                rows.append([p["label"], other, mine, theirs, f"{p['p']:.2g}" if isinstance(p["p"], float) else p["p"],
                             "significant at 0.05" if isinstance(p["p"], float) and p["p"] < 0.05 else "not significant"])
            L.append(table(rows, ["Comparison", "Other row", f"Only {eid} right", "Only other right", "p", "Reading"]))
        L.append("")
        if qa:
            fr = fa_row("QA Summary", eid)
            if fr and fr["Failed"]:
                cats = {"F1 did not finish": "unfinished runs", "F2 no / abstained": "abstentions", "F3 retrieval miss": "never retrieving a gold document",
                        "F4 partial evidence": "partial evidence", "F5 all gold, wrong": "wrong answers despite all gold evidence"}
                top = max(cats, key=lambda c: fr[c] or 0)
                L.append(f"- **Largest failure mode:** {cats[top]} ({fr[top]} of {fr['Failed']} failures, {pct(fr[top] / fr['Failed'])}). "
                         f"Retrieval-side failures (F3 + F4) are {pct(fr['Retrieval-side share (F3+F4)'])} of failures; "
                         f"reading-side (F5) {pct(fr['Reasoning-side share (F5)'])}. *(Measured.)*")
            br = fa_row("By Recall", eid)
            if br and isinstance(br.get("Acc. recall = 1"), float):
                L.append(f"- **With all gold evidence retrieved** the run is right {pct(br['Acc. recall = 1'])} of the time "
                         f"({br['n recall = 1']} questions); with none, {pct(br['Acc. recall = 0']) if isinstance(br['Acc. recall = 0'], float) else '—'}. "
                         "The gap between the two bounds how much better retrieval alone could help. *(Measured; causal reading is a hypothesis.)*")
        L.append("- Statements about *why* a configuration wins or loses are hypotheses unless the comparison isolates "
                 "one variable; see the confounds in section 10.\n")

    # 9 verification
    L.append("## 9. Verification status\n")
    vrows = [["Experiment previously executed", "yes" if e["status"] == "complete" else e["status"]],
             ["Original outputs found", "yes (on the original machine; paths in section 6)" if e["status"] != "queued" else "no (not run)"],
             ["Configuration identified", "yes" if exists(cfg) else "**no**"],
             ["Implementation reviewed", f"yes ([agent review]({rel(f'docs/verification/REVIEW_{algo}.md', path)}))"],
             ["Lightweight tests", f"unit tests with mocks ([TESTING_REPORT.md]({rel('docs/verification/TESTING_REPORT.md', path)})); the experiment itself was not re-run"]]
    if rec:
        if qa:
            ok = abs(rec["accuracy"] - float(re.search(r"[\d.]+", str(w.get("Accuracy"))).group())) < 5e-4 if e["status"] == "complete" else None
            vrows.append(["Numbers independently recomputed", ("**matches the workbook** (accuracy, recall, searches)" if ok else "partial / see audit")
                          + f": accuracy {rec['accuracy']:.4f} from {rec['n_results']} result records "
                          f"([audit]({rel('docs/verification/RESULT_INTEGRITY.md', path)}))"])
        else:
            vrows.append(["Numbers independently recomputed", f"**matches the workbook**: NDCG@10 {rec['ndcg10']:.4f}, Recall@10 {rec['recall10']:.4f} "
                          f"recomputed from the answer text ([audit]({rel('docs/verification/RESULT_INTEGRITY.md', path)}))"])
    else:
        vrows.append(["Numbers independently recomputed", "no (run not complete)"])
    vrows.append(["Reproduced by re-running", "**no**: no experiment was re-run; full experimental equivalence of the refactored code is not established"])
    L.append(table(vrows, ["Check", "Result"]))

    # 10 limitations
    L.append("\n## 10. Limitations\n")
    lim = ["No random seed; model sampling is stochastic, so a re-run gives different per-question results "
           "(two identical 830-question snippet REPL agent runs disagreed on 17.8% of questions in the earlier campaign).",
           f"Reasoning effort actually sent: {e['effort_sent']}.",
           "The judge (gpt-4.1) is an LLM; its verdicts were not audited by hand." if qa else "Binary relevance on gold qrels only (graded relevance ignored).",
           "Comparisons with other rows can differ in more than one variable (search cap, turn limit, time limit, effort, index document length); "
           f"see [AGENT_COMPARISON.md]({rel('docs/experiments/AGENT_COMPARISON.md', path)})."]
    if "trace" in " ".join(e["original_outputs"]):
        lim.append("Per-question data exist only as the trace embedded in the results page; raw run records were not available for audit.")
    if algo == "document_repl":
        lim.append("The number of documents per search is chosen by the model (up to 50), not fixed at 5 like the snippet REPL agent; "
                   "the `llm_query` sub-LLM functions remain installed in the REPL (only the prompt says not to use them).")
    L += [f"- {x}" for x in lim]

    # 11 related
    L.append("\n## 11. Related files\n")
    L.append(f"- [Experiments.xlsx]({rel('results/original_baselines/Experiments.xlsx', path)}) · "
             f"[Failure_Analysis.xlsx]({rel('results/original_baselines/Failure_Analysis.xlsx', path)}) · "
             f"[registry.yaml]({rel('experiments/registry.yaml', path)})")
    L.append(f"- [Known issues]({rel('docs/verification/KNOWN_ISSUES.md', path)}) · "
             f"[Result integrity]({rel('docs/verification/RESULT_INTEGRITY.md', path)}) · "
             f"[Changes from the original code]({rel(f'agents/{algo}/CHANGES_FROM_ORIGINAL.md', path)})")
    (ROOT / path).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / path).write_text(paper_names("\n".join(L) + "\n"))
    return path


if __name__ == "__main__":
    follow_up = {e["id"] for e in REG.get("follow_up_studies", [])}   # written by make_follow_up_docs.py: keep
    for algo in ALGO_NAME:                      # documents are fully generated: drop stale ones (e.g. renamed slugs)
        for old in (OUT / algo).glob("*.md"):
            if old.stem not in follow_up:
                old.unlink()
    made = [(e, doc(e)) for e in REG["experiments"]]
    idx = ["# Experiments\n", "One document per row of [Experiments.xlsx](../../results/original_baselines/Experiments.xlsx) that this "
           "project ran. Generated by `scripts/docs/make_experiment_docs.py` from `experiments/registry.yaml` and the evidence files. "
           "DSPy.RLM has no row in the final table (it was evaluated in an earlier campaign with a different judge; see "
           "[its README](../../agents/dspy_rlm/README.md)).\n",
           "Cross-row comparison: [AGENT_COMPARISON.md](AGENT_COMPARISON.md).\n",
           "Work after the results table (search policy, GEPA, the FRAMES search cap, the DSPy.RLM ablation, run-to-run "
           "variation, FRAMES error analysis): [FOLLOW_UP_STUDIES.md](FOLLOW_UP_STUDIES.md), one document per study ID, "
           "generated by `scripts/docs/make_follow_up_docs.py`.\n"]
    idx.append(table([[e["id"], ALGO_NAME[e["agent"]], e["benchmark"].split(" (")[0], e["model"].split(" (")[0], e["retriever"].split(", ")[0],
                       e["status"], f"[doc]({os.path.relpath(ROOT / p, OUT)})"] for e, p in made],
                     ["ID", "Agent", "Benchmark", "Model", "Retriever", "Status", "Document"]))
    (OUT / "README.md").write_text(paper_names("\n".join(idx)))
    print(len(made), "experiment documents written")
