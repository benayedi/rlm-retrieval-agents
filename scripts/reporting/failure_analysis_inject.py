"""Write the evaluated formula values (vals.json from failure_analysis_eval.py) into the workbook as cached values.

Refactored copy of scripts/failure_analysis_inject.py of the original repository, unchanged apart from this docstring (see
scripts/reporting/README.md for the pipeline order).
  python scripts/reporting/failure_analysis_inject.py <out.xlsx> <vals.json> <final.xlsx>
"""
import json, re, sys, zipfile
from xml.sax.saxutils import escape
src, vals, dst = sys.argv[1:4]
V = json.load(open(vals)); zin = zipfile.ZipFile(src)
rels = dict(re.findall(r'Target="/?xl/(worksheets/sheet\d+\.xml)" Id="(rId\d+)"', zin.read("xl/_rels/workbook.xml.rels").decode()))
rels = {v: k for k, v in rels.items()}
m = {"xl/" + rels[i]: n for n, i in re.findall(r'<sheet name="([^"]+)" sheetId="\d+" state="visible" r:id="(rId\d+)"', zin.read("xl/workbook.xml").decode())}
out = zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED); n = 0
for it in zin.infolist():
    data = zin.read(it.filename); sh = m.get(it.filename)
    if sh in V:
        vv = V[sh]
        def rep(mo):
            global n
            ref, attrs, f = mo.groups(); v = vv.get(ref)
            if v is None:
                return mo.group(0)
            n += 1
            if isinstance(v, bool): v = int(v)
            if isinstance(v, (int, float)):
                return f'<c r="{ref}"{attrs}><f>{f}</f><v>{v!r}</v></c>'
            return f'<c r="{ref}"{attrs} t="str"><f>{f}</f><v>{escape(str(v))}</v></c>'
        data = re.sub(r'<c r="([A-Z]+\d+)"([^>]*)><f>(.*?)</f><v ?/></c>', rep, data.decode()).encode()
    out.writestr(it, data)
out.close(); print("cached values written:", n)
