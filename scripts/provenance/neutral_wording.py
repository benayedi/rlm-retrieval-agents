"""Reword origin notes in the text of the published workbooks so the document REPL agent is described neutrally, as the rest of
the repository does. Only text strings change (cell comments and text cells); every number and formula stays as it
was, which --check verifies against a reference copy.

usage: python scripts/provenance/neutral_wording.py FILE.xlsx            (rewrites in place)
       python scripts/provenance/neutral_wording.py --check NEW.xlsx REF.xlsx
"""
import re, shutil, sys, tempfile, zipfile

# (pattern, replacement). The file name docs/supervisor_artifact.html is a real path in the original repository and
# stays as it is.
RULES = [
    (r"Supervisor's full-doc run", "Full-doc run"),
    (r"the supervisor artifact's", "the results page's"),
    (r"supervisor artifact for BCP-21/23", "the embedded trace of docs/supervisor_artifact.html for BCP-21/23"),
    (r"the supervisor's full-doc rows", "the full-doc GPT-5 dense and ColBERT rows"),
    (r"for the supervisor's BCP-21", "for BCP-21"),
    (r"gpt-4\.1 judge \(his\)", "gpt-4.1 judge (as graded with the run)"),
    (r"\(equals his cost_per_q\)", "(equals the cost_per_q of the results page)"),
]


def rewrite(path):
    tmp = tempfile.mktemp(suffix=".xlsx")
    n = 0
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename.endswith(".xml"):
                t = data.decode("utf-8")
                for pat, rep in RULES:
                    t, k = re.subn(pat, rep, t); n += k
                data = t.encode("utf-8")
            zout.writestr(item, data)
    shutil.move(tmp, path)
    print(f"{path}: {n} replacements")


def check(new, ref):
    from openpyxl import load_workbook
    bad = 0
    for data_only in (True, False):
        a, b = load_workbook(new, data_only=data_only), load_workbook(ref, data_only=data_only)
        for ws in b:
            for row in ws.iter_rows():
                for c in row:
                    v, w = c.value, a[ws.title][c.coordinate].value
                    if isinstance(v, str) and not str(v).startswith("="):
                        continue                                  # text may differ by the rules above
                    if v != w:
                        bad += 1; print("DIFF", ws.title, c.coordinate, v, w)
    print("numbers and formulas identical" if not bad else f"{bad} differences")
    return bad


if __name__ == "__main__":
    if sys.argv[1] == "--check":
        sys.exit(1 if check(sys.argv[2], sys.argv[3]) else 0)
    for f in sys.argv[1:]:
        rewrite(f)
