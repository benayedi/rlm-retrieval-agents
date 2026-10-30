"""Make the publishable copy of Failure_Analysis.xlsx: remove BrowseComp-Plus plain text (questions, gold answers,
model answers) while keeping every count, formula and chart. FRAMES and OBLIQ text is kept.

Input is the formula workbook before cached values were injected (failure_analysis_build.py output) plus the evaluated
values (failure_analysis_eval.py output); the redacted cells are all plain text, never formula inputs, so the injected
values stay valid. Output must then pass scripts/provenance/bcp_leak_scan.py.

usage: python sanitize_failure_analysis.py FA_formulas.xlsx vals.json OUT.xlsx"""
import json, subprocess, sys
from pathlib import Path
from openpyxl import load_workbook

SRC, VALS, OUT = sys.argv[1:4]
R = "[BrowseComp-Plus text removed: the benchmark is distributed encrypted]"
wb = load_workbook(SRC)


def header(ws, row):
    return {str(c.value): c.column for c in ws[row] if c.value is not None}


# Detail QA: one row per (run, question)
ws = wb["Detail QA"]; h = header(ws, 1)
n = 0
for r in range(2, ws.max_row + 1):
    if ws.cell(r, h["Dataset"]).value == "BrowseComp+":
        for col in ("Exact answer", "Gold answer", "Question"):
            ws.cell(r, h[col]).value = R if col == "Question" else None
        n += 1
print("Detail QA rows redacted:", n)

# Error Samples: header on row 4; BrowseComp+ rows have Exp ID BCP-*
ws = wb["Error Samples"]; h = header(ws, 4)
for r in range(5, ws.max_row + 1):
    if str(ws.cell(r, h["Exp ID"]).value or "").startswith("BCP"):
        for col in ("Exact answer", "Gold answer"):
            ws.cell(r, h[col]).value = None
        ws.cell(r, h["Question"]).value = R
ws["A2"] = "Up to 25 per (experiment, category). BrowseComp-Plus questions and answers are removed from this copy (the benchmark is distributed encrypted); FRAMES rows are complete."

# Failure Cases: example rows whose 'Run · QID' cell starts with BCP-
ws = wb["Failure Cases"]
for r in range(1, ws.max_row + 1):
    if str(ws.cell(r, 2).value or "").startswith("BCP-"):
        ws.cell(r, 3).value = R
        ws.cell(r, 4).value = None
        ws.cell(r, 5).value = None

# Read Me / Error Samples notes that mention decrypted gold
for ws in wb:
    for row in ws.iter_rows():
        for c in row:
            if isinstance(c.value, str) and "keep this file out of public places" in c.value:
                c.value = c.value.replace("Gold answers are decrypted from BrowseComp-Plus: keep this file out of public places.",
                                          "BrowseComp-Plus text is removed in this copy.")

tmp = Path(OUT).with_suffix(".formulas.xlsx")
wb.save(tmp)
inject = Path(__file__).resolve().parents[1] / "reporting" / "failure_analysis_inject.py"
subprocess.run([sys.executable, str(inject), str(tmp), VALS, OUT], check=True)
tmp.unlink()
print("wrote", OUT)
