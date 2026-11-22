"""Build docs/experiments/AGENT_COMPARISON.md and its figures (results/comparisons/*.svg) from the evidence files only:
results/original_baselines/Experiments.xlsx (reported values) and Failure_Analysis.xlsx (paired McNemar tests).

Figures are plain SVG written here (no plotting dependency). Colours: categorical slots 1-2 of the dataviz reference
palette (#2a78d6 snippet REPL agent, #eb6834 document REPL agent), validated for colour-vision deficiency; every bar is also
labelled with its value, so identity never rests on colour alone.

usage: python scripts/docs/make_comparison.py
"""
import re
from pathlib import Path

from naming import paper_names

from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[2]
XL = load_workbook(ROOT / "results/original_baselines/Experiments.xlsx", data_only=True)
FA = load_workbook(ROOT / "results/original_baselines/Failure_Analysis.xlsx", data_only=True)
COL = {"Snippet RLM": "#2a78d6", "Full-Doc RLM": "#eb6834"}
INK, MUTED, GRID = "#1f2328", "#59636e", "#d8dee4"


def rows(sheet):
    ws = XL[sheet]
    hdr = {ws.cell(3, c).value: c for c in range(1, ws.max_column + 1) if ws.cell(3, c).value}
    out = []
    for r in range(4, ws.max_row + 1):
        h = ws.cell(r, 1).value
        if h in ("Snippet RLM", "Full-Doc RLM"):
            out.append({"row": r, **{k: ws.cell(r, c).value for k, c in hdr.items()}})
    return out


def val(x):
    m = re.match(r"\s*(\d*\.\d+|\d+)", str(x)) if x not in (None, "—", "to run") else None
    return float(m.group(1)) if m else None


def model_of(r):
    return "Qwen3.6" if "Qwen" in str(r["Model (Reasoning)"]) else ("GPT-5.2" if "5.2" in str(r["Model (Reasoning)"]) else "GPT-5")


def retr_of(r):
    s = str(r["Retrieval Signal"])
    return "BM25" if "BM25" in s else "ColBERT" if "ColBERT" in s else "Qwen3-Embed-8B" if "Qwen3" in s else s


# ---------------------------------------------------------------- SVG grouped bars, small multiples
def svg_panels(title, panels, ymax, yfmt, fname, note=""):
    """panels: [(panel title, [(group label, {harness: (value|None, label)})])]"""
    pw, ph, top, left, bottom = 330, 250, 60, 46, 58
    W = left + len(panels) * (pw + 24); H = top + ph + bottom + (16 if note else 0)
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Helvetica, Arial, sans-serif">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         f'<text x="{left}" y="24" font-size="15" font-weight="600" fill="{INK}">{title}</text>']
    lx = left
    for h, c in COL.items():                                   # legend (always present for 2 series)
        s.append(f'<rect x="{lx}" y="36" width="11" height="11" rx="2" fill="{c}"/><text x="{lx + 16}" y="46" font-size="12" fill="{INK}">{h}</text>')
        lx += 170
    for pi, (ptitle, groups) in enumerate(panels):
        x0 = left + pi * (pw + 24); y0 = top + 18
        s.append(f'<text x="{x0}" y="{top + 8}" font-size="13" font-weight="600" fill="{INK}">{ptitle}</text>')
        plot_h = ph - 30
        for t in range(0, 5):                                    # recessive grid
            v = ymax * t / 4; y = y0 + plot_h - plot_h * t / 4
            s.append(f'<line x1="{x0}" x2="{x0 + pw}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
            if pi == 0:
                s.append(f'<text x="{x0 - 6}" y="{y + 4:.1f}" font-size="11" fill="{MUTED}" text-anchor="end">{yfmt(v)}</text>')
        gw = pw / max(1, len(groups)); bw = min(34, gw / 3)
        for gi, (glabel, bars) in enumerate(groups):
            gx = x0 + gi * gw + gw / 2
            for bi, h in enumerate(COL):
                v, lab = bars.get(h, (None, "not run"))
                bx = gx - bw - 1 + bi * (bw + 2)                  # 2px gap between adjacent bars
                if v is None:
                    s.append(f'<text x="{bx + bw / 2:.1f}" y="{y0 + plot_h - 6:.1f}" font-size="10" fill="{MUTED}" text-anchor="middle" '
                             f'transform="rotate(-90 {bx + bw / 2:.1f} {y0 + plot_h - 6:.1f})">{lab}</text>')
                    continue
                bh = plot_h * v / ymax; by = y0 + plot_h - bh
                s.append(f'<path d="M{bx:.1f},{y0 + plot_h:.1f} V{by + 4:.1f} Q{bx:.1f},{by:.1f} {bx + 4:.1f},{by:.1f} H{bx + bw - 4:.1f} '
                         f'Q{bx + bw:.1f},{by:.1f} {bx + bw:.1f},{by + 4:.1f} V{y0 + plot_h:.1f} Z" fill="{COL[h]}"><title>{h}, {glabel}: {lab}</title></path>')
                s.append(f'<text x="{bx + bw / 2:.1f}" y="{by - 4:.1f}" font-size="10" fill="{INK}" text-anchor="middle">{lab}</text>')
            s.append(f'<text x="{gx:.1f}" y="{y0 + plot_h + 16:.1f}" font-size="11.5" fill="{INK}" text-anchor="middle">{glabel}</text>')
        s.append(f'<line x1="{x0}" x2="{x0 + pw}" y1="{y0 + plot_h}" y2="{y0 + plot_h}" stroke="{MUTED}" stroke-width="1"/>')
    if note:
        s.append(f'<text x="{left}" y="{H - 10}" font-size="11" fill="{MUTED}">{note}</text>')
    s.append("</svg>")
    out = ROOT / "results/comparisons" / fname
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(paper_names("\n".join(s)))
    return f"../../results/comparisons/{fname}"


def pct(v):
    return f"{100 * v:.1f}%"


# ---------------------------------------------------------------- data
B, F = rows("BrowseComp+"), rows("FRAMES")
OB = {t: rows(f"OBLIQ - {t}") for t in ("Math", "Twitter", "Congress")}


def cell(rs, harness, model, retr):
    for r in rs:
        if r["Harness"] == harness and model_of(r) == model and retr_of(r) == retr:
            return r


figs = {}
panels = []
for model in ("GPT-5", "Qwen3.6"):
    groups = []
    for retr in ("Qwen3-Embed-8B", "ColBERT", "BM25"):
        bars = {}
        for h in COL:
            r = cell(B, h, model, retr)
            a = val(r["Accuracy"]) if r else None
            partial = r is not None and "†" in str(r["Accuracy"])
            bars[h] = (None, "running" if partial else "not run") if (a is None or partial) else (a, f"{a:.3f}")
        groups.append((retr, bars))
    panels.append((model, groups))
_open = any(b[0] is None for _, gs in panels for _, bs in gs for b in bs.values())   # a bar shown as running / not run
figs["bcp"] = svg_panels("BrowseComp-Plus: accuracy (gpt-4.1 judge, 830 questions)", panels, 1.0, lambda v: f"{v:.2f}", "bcp_accuracy.svg",
                         "Qwen3.6 Snippet dense and ColBERT rows were still running or queued on 2026-10-30." if _open else "")
groups = []
for model in ("GPT-5", "Qwen3.6"):
    groups.append((model, {h: ((lambda a: (a, f"{a:.3f}"))(val(cell(F, h, model, "BM25")["Accuracy"])) if cell(F, h, model, "BM25") else (None, "not run")) for h in COL}))
figs["frames"] = svg_panels("FRAMES: accuracy (gpt-4.1 judge, 824 questions, BM25)", [("BM25 over Wikipedia", groups)], 1.0, lambda v: f"{v:.2f}", "frames_accuracy.svg")
groups = []
for t, rs in OB.items():
    bars = {}
    for h in COL:
        r = next((x for x in rs if x["Harness"] == h), None)
        nd = val(r["NDCG@10 (G / P)"] if "NDCG@10 (G / P)" in r else r.get("NDCG@10")) if r else None
        bars[h] = (nd, f"{nd:.3f}") if nd is not None else (None, "not run")
    groups.append((t, bars))
figs["obliq"] = svg_panels("OBLIQ-Bench: NDCG@10 on gold qrels (GPT-5.2, Gemini-Embedding-2)", [("", groups)], 0.25, lambda v: f"{v:.2f}", "obliq_ndcg10.svg")

# paired tests
PW = []
ws = FA["Pairwise"]
for r in range(5, 30):
    if ws.cell(r, 1).value and ws.cell(r, 2).value and ws.cell(r, 3).value:
        PW.append(dict(label=ws.cell(r, 1).value, a=ws.cell(r, 2).value, b=ws.cell(r, 3).value, onlyA=ws.cell(r, 5).value,
                       onlyB=ws.cell(r, 6).value, p=ws.cell(r, 9).value))


def table(rws, head):
    return "| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n" + "".join("| " + " | ".join(str(c) for c in r) + " |\n" for r in rws)


def qa_table(rs):
    out = []
    for r in rs:
        if r["Accuracy"] in (None, ""):
            out.append([r["Harness"], model_of(r), retr_of(r), "not run (queued)", "", "", "", "", ""]); continue
        out.append([f"{r['Harness']}", model_of(r), retr_of(r), r["Accuracy"], r["Recall"], r["Avg. Search Calls Per Query"],
                    r["Avg. Iterations Per Query"], r["Time Per Query"], r["Uncached Cost Per Query"]])
    return table(out, ["Agent", "Model", "Retriever", "Accuracy", "Recall*", "Searches / q", "Turns / q", "Time / q", "Uncached cost / q ($)"])


md = ROOT / "docs/experiments/AGENT_COMPARISON.md"
tmpl = (ROOT / "scripts/docs/AGENT_COMPARISON.template.md").read_text()
pair_rows = [[p["label"], p["a"], p["b"], p["onlyA"], p["onlyB"], f"{p['p']:.2g}" if isinstance(p["p"], float) else p["p"],
              "yes" if isinstance(p["p"], float) and p["p"] < 0.05 else "no"] for p in PW]
OBL = []
for t, rs in OB.items():
    for r in rs:
        OBL.append([t, r["Harness"], r.get("Recall@10 (G)", r.get("Recall@10")), r.get("NDCG@10 (G / P)", r.get("NDCG@10")),
                    r.get("NDCG@50 (G / P)", r.get("NDCG@50")), r["Avg. Search Calls Per Query"], r["Avg. Iterations Per Query"], r["Time Per Query"],
                    r["Uncached Cost Per Query"]])
md.write_text(paper_names(tmpl.replace("{{FIG_BCP}}", f"![BrowseComp-Plus accuracy]({figs['bcp']})")
              .replace("{{FIG_FRAMES}}", f"![FRAMES accuracy]({figs['frames']})")
              .replace("{{FIG_OBLIQ}}", f"![OBLIQ NDCG@10]({figs['obliq']})")
              .replace("{{TABLE_BCP}}", qa_table(B)).replace("{{TABLE_FRAMES}}", qa_table(F))
              .replace("{{TABLE_OBLIQ}}", table(OBL, ["Task", "Agent", "Recall@10 (G)", "NDCG@10 (G / P)", "NDCG@50 (G / P)", "Searches / q", "Turns / q", "Time / q", "Uncached cost / q ($)"]))
              .replace("{{TABLE_PAIRS}}", table(pair_rows, ["Comparison", "A", "B", "Only A right", "Only B right", "McNemar p", "p < 0.05"]))))
print("wrote", md.relative_to(ROOT), "and", len(figs), "figures")
