"""Minimal evaluator for the formula subset used by build.py (no LibreOffice here). Prints every formula cell's value
of the summary sheets as JSON {sheet: {cell: value}} and counts errors."""
import json, math, re, sys
from fnmatch import fnmatchcase
from openpyxl import load_workbook
from openpyxl.utils import column_index_from_string as CI, get_column_letter as CL

wb = load_workbook(sys.argv[1])
cache = {}


class Err(Exception):
    pass


def colvals(sheet, a, b):
    m1, m2 = re.match(r"([A-Z]+)(\d+)", a), re.match(r"([A-Z]+)(\d+)", b)
    c, r1, r2 = m1.group(1), int(m1.group(2)), int(m2.group(2))
    k = (sheet, c, r1, r2)
    if k not in cache:
        cache[k] = [val(sheet, f"{c}{r}") for r in range(r1, r2 + 1)]
    return cache[k]


vcache = {}


def val(sheet, ref):
    k = (sheet, ref)
    if k in vcache:
        return vcache[k]
    v = wb[sheet][ref].value
    if isinstance(v, str) and v.startswith("="):
        try:
            v = ev(sheet, v[1:])
        except Err:
            v = "#ERR"
        except ZeroDivisionError:
            v = "#DIV/0!"
    vcache[k] = v
    return v


def match(x, crit):
    if isinstance(crit, (int, float)):
        return isinstance(x, (int, float)) and not isinstance(x, bool) and x == crit
    m = re.match(r"(>=|<=|<>|>|<|=)?(.*)", str(crit)); op, rhs = m.group(1), m.group(2)
    try:
        num = float(rhs)
    except ValueError:
        num = None
    if op and num is not None:
        if not isinstance(x, (int, float)):
            return False
        return {">=": x >= num, "<=": x <= num, ">": x > num, "<": x < num, "=": x == num, "<>": x != num}[op]
    if x is None:
        return False
    return fnmatchcase(str(x).lower(), rhs.lower())


def ifs(pairs):
    n = len(pairs[0][0]); keep = [True] * n
    for rng, c in pairs:
        for i in range(n):
            if keep[i] and not match(rng[i], c):
                keep[i] = False
    return keep


def COUNTIFS(*a):
    return sum(ifs(list(zip(a[::2], a[1::2]))))


def COUNTIF(r, c):
    return COUNTIFS(r, c)


def AVERAGEIFS(t, *a):
    k = ifs(list(zip(a[::2], a[1::2]))); xs = [v for v, kk in zip(t, k) if kk and isinstance(v, (int, float))]
    if not xs:
        raise ZeroDivisionError
    return sum(xs) / len(xs)


def SUMIFS(t, *a):
    k = ifs(list(zip(a[::2], a[1::2]))); return sum(v for v, kk in zip(t, k) if kk and isinstance(v, (int, float)))


def SUMIF(r, c, t):
    return SUMIFS(t, r, c)


def SUM(*a):
    s = 0
    for x in a:
        s += sum(v for v in x if isinstance(v, (int, float))) if isinstance(x, list) else x
    return s


def MIN(*a):
    return min(a)


def IF(c, a, b):
    return a if c else b


def BINOMDIST(k, n, p, cum):
    return sum(math.comb(int(n), i) * p ** i * (1 - p) ** (n - i) for i in range(int(k) + 1))


def ev(sheet, f):
    # IFERROR needs lazy evaluation: rewrite IFERROR(x,y) -> _IFE(lambda: x, lambda: y)
    s = f
    s = re.sub(r"'([^']+)'!\$?([A-Z]+)\$?(\d+):\$?([A-Z]+)\$?(\d+)", lambda m: f'_R("{m.group(1)}","{m.group(2)}{m.group(3)}","{m.group(4)}{m.group(5)}")', s)
    s = re.sub(r"(?:'([^']+)'|([A-Za-z]+))!\$?([A-Z]{1,2})\$?(\d+)(?![\d:])", lambda m: f'_V("{m.group(1) or m.group(2)}","{m.group(3)}{m.group(4)}")', s)
    s = re.sub(r"(?<![A-Za-z_\"])\$?([A-Z]{1,2})\$?(\d+):\$?([A-Z]{1,2})\$?(\d+)(?![\w(])", lambda m: f'_R(_S,"{m.group(1)}{m.group(2)}","{m.group(3)}{m.group(4)}")', s)
    s = re.sub(r'(?<![A-Za-z_"\w])\$?([A-Z]{1,2})\$?(\d+)(?![\w(",])', lambda m: f'_V(_S,"{m.group(1)}{m.group(2)}")', s)
    s = re.sub(r'(?<![A-Za-z_"\w])\$?([A-Z]{1,2})\$?(\d+)(?=[,)])', lambda m: f'_V(_S,"{m.group(1)}{m.group(2)}")', s)
    s = s.replace("TRUE", "True").replace("FALSE", "False")
    s = re.sub(r"(?<![<>=!])=(?!=)", "==", s)
    s = s.replace("<>", "!=")
    # lazy IFERROR
    s = s.replace("IFERROR(", "_IFE(lambda: ")
    s = re.sub(r"_IFE\(lambda: ", "_IFE(lambda: ", s)
    s = _split_ife(s)
    env = dict(COUNTIFS=COUNTIFS, COUNTIF=COUNTIF, AVERAGEIFS=AVERAGEIFS, SUMIFS=SUMIFS, SUMIF=SUMIF, SUM=SUM, MIN=MIN, IF=IF,
               BINOMDIST=BINOMDIST, _R=colvals, _V=lambda sh, r: _num(val(sh, r)), _S=sheet, _IFE=_IFE)
    return eval(s, env)


def _num(v):
    if isinstance(v, str) and v.startswith("#"):
        raise Err(v)
    return 0 if v is None else v


def _IFE(f, g):
    try:
        r = f()
        if isinstance(r, str) and r.startswith("#"):
            return g()
        return r
    except (Err, ZeroDivisionError):
        return g()


def _split_ife(s):
    # turn "_IFE(lambda: A, B)" into "_IFE(lambda: A, lambda: B)" at the top-level comma of each _IFE call
    out, i = "", 0
    while True:
        j = s.find("_IFE(lambda: ", i)
        if j < 0:
            return out + s[i:]
        out += s[i:j] + "_IFE(lambda: "; k = j + len("_IFE(lambda: "); depth = 0; q = False
        while k < len(s):
            ch = s[k]
            if ch == '"':
                q = not q
            elif not q and ch == "(":
                depth += 1
            elif not q and ch == ")":
                depth -= 1
            elif not q and ch == "," and depth == 0:
                break
            k += 1
        inner = _split_ife(s[j + len("_IFE(lambda: "):k])
        out += inner + ", lambda: "; i = k + 1


res, errs = {}, []
for ws in wb:
    if ws.title.startswith("Detail"):
        continue
    d = {}
    for row in ws.iter_rows():
        for c in row:
            if isinstance(c.value, str) and c.value.startswith("="):
                v = val(ws.title, c.coordinate); d[c.coordinate] = v
                if isinstance(v, str) and v.startswith("#"):
                    errs.append((ws.title, c.coordinate, c.value[:120]))
    res[ws.title] = d
json.dump(res, open(sys.argv[2], "w"), default=str)
print("formulas", sum(len(v) for v in res.values()), "errors", len(errs)); print(errs[:10])
