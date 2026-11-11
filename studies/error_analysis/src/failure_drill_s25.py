"""ERR-1, search cap 25: drill-down of the failure buckets written by failure_taxonomy.py --s25 (retrieval misses: was
the missed article's title in the agent's own queries, how long is it; beyond-the-model and found-all-gold failures:
which kind of answer). Prints only.

From original/RLM/scripts/frames_s25_failure_drill.py. Changed only: the original machine's paths (LOGS_ROOT,
FRAMES_DATA_DIR) and this documentation (studies/error_analysis/CHANGES_FROM_ORIGINAL.md). Run:
  set -a; . configs/paths.env; set +a
  python studies/error_analysis/src/failure_drill_s25.py

Original docstring follows (paths in it are relative to the original repository).

Drill-down of the 25-search full-doc RLM's FRAMES failures (logs/frames/failure_taxonomy_s25.json, from
scripts/frames_failure_taxonomy.py --s25), to find what a scaffold change could fix.
  D (retrieval miss): for each missed gold article -- was its exact title in one of the agent's queries? how long
      is it vs the gold articles that were found? how many searches had the agent used?
  C (oracle also wrong) / E (found all, still wrong): answer type (number/date vs name), reasoning types, turn cap.
"""
import json, os, re, statistics as st, sys
import numpy as np

# Refactor (paths): was R = "<original repository>"; logs/ of it is LOGS_ROOT, data/frames/ is FRAMES_DATA_DIR.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "agents", "shared"))
from rlm_shared.paths import path  # noqa: E402
L, FRD = str(path("LOGS_ROOT")), str(path("FRAMES_DATA_DIR"))
TX = json.load(open(f"{L}/frames/failure_taxonomy_s25.json"))
rows = {str(r["i"]): r for r in json.load(open(f"{FRD}/frames_rows.json"))}
meta = np.load(f"{FRD}/corpus_meta.npz", allow_pickle=True)
TITLES, OFF = meta["titles"], meta["offsets"]
DIRS = [f"{L}/frames/sup_q36_bm25f_s25{x}" for x in ("", "_rest0", "_rest1")]
graded = {}
for D in DIRS:
    graded.update({v["qid"]: v for v in json.load(open(f"{D}/graded.json")).values()})


def raw(q):
    for D in DIRS:
        try:
            return json.load(open(f"{D}/{int(q):03d}.json"))
        except FileNotFoundError:
            pass


def doclen(i):
    i = int(i)
    return int(OFF[i + 1] - OFF[i]) if i + 1 < len(OFF) else None


norm = lambda s: re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()
NUM = re.compile(r"^[\s$£€~≈-]*[\d.,]+\s*(%|percent|years?|km|miles?|m|feet|ft|kg|million|billion|thousand)?\s*$", re.I)
DATE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b|january|february|march|april|may|june|july|august|september|october|november|december", re.I)

# --- D: retrieval misses
miss, found_len, miss_len = [], [], []
for q, r in TX.items():
    if not r["bucket"].startswith("D"):
        continue
    x = raw(q) or {}
    got = set(map(str, x.get("retrieved_docids", [])))
    queries = [s.get("query", "") for s in x.get("search_log", [])]
    qtok = [set(norm(s)) for s in queries]
    for g in map(str, rows[q]["gold_ids"]):
        L = doclen(g)
        if g in got:
            found_len.append(L); continue
        t = str(TITLES[int(g)]); tt = set(norm(t))
        in_q = any(tt and tt <= s for s in qtok)
        miss.append({"q": q, "title": t, "title_in_query": in_q, "len": L, "searches": len(queries)})
        miss_len.append(L)
print(f"D questions: {sum(1 for r in TX.values() if r['bucket'].startswith('D'))} | missed gold articles: {len(miss)}")
print(f"  missed article's full title was in one of the agent's queries: {sum(m['title_in_query'] for m in miss)} "
      f"({sum(m['title_in_query'] for m in miss)/len(miss):.0%})")
print(f"  median length, missed gold {st.median(miss_len)/1000:.1f}k chars vs found gold (same questions) {st.median(found_len)/1000:.1f}k")
print(f"  searches used on D questions: median {st.median(m['searches'] for m in miss)}, at cap 25: "
      f"{sum(1 for q, r in TX.items() if r['bucket'].startswith('D') and (r['searches'] or 0) >= 25)}")
print("  examples (title in query, still missed):", "; ".join(m["title"] for m in miss if m["title_in_query"])[:600])

# --- C and E: what kind of answers fail
def kind(a):
    a = a.strip()
    return "number" if NUM.match(a) else ("date/year" if DATE.search(a) and len(a) < 40 else "name/text")


for B in ("C", "E", "solved"):
    qs = [q for q, r in TX.items() if r["bucket"].startswith(B)]
    ks = [kind(rows[q]["answer"]) for q in qs]
    sh = {k: ks.count(k) / len(ks) for k in ("number", "date/year", "name/text")}
    cap = sum(1 for q in qs if (TX[q]["iters"] or 0) >= 50) / len(qs)
    tab = sum(1 for q in qs if "Tabular" in rows[q]["reasoning_types"]) / len(qs)
    print(f"{B:7s} n={len(qs):3d} gold answer: number {sh['number']:.0%} date {sh['date/year']:.0%} name {sh['name/text']:.0%} | "
          f"tabular {tab:.0%} | hit 50-turn cap {cap:.0%} | mean gold recall {st.mean(TX[q]['recall'] or 0 for q in qs):.2f}")

# C: how close was the agent? numeric answers off by a little vs far
off = []
for q, r in TX.items():
    if r["bucket"].startswith("C") and kind(rows[q]["answer"]) == "number":
        a = re.sub(r"[^\d.]", "", (graded.get(q) or {}).get("answer", "").split("Exact Answer:")[-1][:40])
        g = re.sub(r"[^\d.]", "", rows[q]["answer"])
        try:
            off.append(abs(float(a) - float(g)) / max(1e-9, abs(float(g))))
        except ValueError:
            pass
if off:
    print(f"C numeric answers parsed {len(off)}: within 5% of gold {sum(o <= .05 for o in off)}, within 20% {sum(o <= .2 for o in off)}, "
          f"further {sum(o > .2 for o in off)}")
