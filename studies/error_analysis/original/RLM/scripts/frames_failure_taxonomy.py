"""Failure taxonomy of the supervisor-harness agent on all 824 FRAMES questions
(logs/frames/sup_q36_bm25f: Qwen3.6, BM25, facts-only prompt, 10 searches / 30 turns).

Each WRONG question goes to exactly one bucket, first match wins:
  A  no usable answer       time-cap kill (no result) | harness timeout | empty submission | template "..."
  B  corpus gap             a gold article is missing from the June 2023 dump
  C  beyond the model       oracle (same Qwen3.6, all gold articles in the prompt) is ALSO wrong:
                            better retrieval cannot fix it with this model
  D  retrieval miss         oracle right, agent found only some / none of the gold articles
  E  found all, still wrong oracle right and agent retrieved every gold article: the failure is in how
                            the agent reads / reasons inside the harness (or run-to-run noise)
Oracle and closed-book come from scripts/frames_eval.py (logs/frames/llm_*.jsonl), single runs each.

  python scripts/frames_failure_taxonomy.py                      # 10 searches / 30 turns (default)
  python scripts/frames_failure_taxonomy.py --s25                # 25 searches / 50 turns, three dirs merged
"""
import ast, collections, csv, json, os, re, statistics as st, sys

R = "/home/extstu_00/RLM"
S25 = "--s25" in sys.argv
DIRS = ([f"{R}/logs/frames/sup_q36_bm25f_s25{x}" for x in ("", "_rest0", "_rest1")] if S25
        else [f"{R}/logs/frames/sup_q36_bm25f"])
TURN_CAP, SEARCH_CAP = (50, 25) if S25 else (30, 10)
OUT = f"{R}/logs/frames/failure_taxonomy{'_s25' if S25 else ''}.json"
rows = {str(r["i"]): r for r in json.load(open(f"{R}/data/frames/frames_rows.json"))}
tsv = {str(i): r for i, r in enumerate(csv.DictReader(open(f"{R}/data/frames/test.tsv"), delimiter="\t"))}
graded = {}
for D in DIRS:
    graded.update({v["qid"]: v for v in json.load(open(f"{D}/graded.json")).values()})


def raw_file(q):
    for D in DIRS:
        if os.path.exists(f"{D}/{int(q):03d}.json"):
            return f"{D}/{int(q):03d}.json"

ora = {str(x["i"]): x for x in map(json.loads, open(f"{R}/logs/frames/llm_oracle.jsonl"))}
clo = {str(x["i"]): x for x in map(json.loads, open(f"{R}/logs/frames/llm_closed.jsonl"))}


def n_links(q):
    try:
        links = ast.literal_eval(tsv[q]["wiki_links"])
    except Exception:
        links = [v for k, v in tsv[q].items() if k.startswith("wikipedia_link") and v]
    return len({re.sub(r"#.*$", "", l.split("/wiki/")[-1]).lower() for l in links if "/wiki/" in l})


out, rec = collections.Counter(), {}
for q in map(str, range(824)):
    g = graded.get(q); f = raw_file(q)
    raw = json.load(open(f)) if f else None
    ans = (g or {}).get("answer", "")
    types = rows[q]["reasoning_types"]
    if g and g["correct"]:
        b = "solved"
    elif raw is None:
        b = "A1 no answer: killed at time cap"
    elif raw.get("status") == "error":
        b = "A2 no answer: harness timeout"
    elif raw.get("status") == "empty_output" or not ans.strip():
        b = "A3 no answer: empty submission"
    elif re.search(r"Exact Answer:\s*\.\.\.", ans):
        b = "A4 no answer: submitted the template"
    elif len(rows[q]["gold_ids"]) < n_links(q):
        b = "B  corpus gap: gold article not in dump"
    elif not ora.get(q, {}).get("correct"):
        b = "C  beyond the model: oracle also wrong"
    elif (g or {}).get("gold_recall", 0) < 0.999:
        b = ("D1 retrieval miss: no gold found" if (g or {}).get("gold_recall", 0) == 0
             else "D2 retrieval miss: some gold found")
    else:
        b = "E  found all gold, still wrong"
    out[b] += 1
    rec[q] = {"bucket": b, "types": types, "recall": (g or {}).get("gold_recall"),
              "searches": (raw or {}).get("tool_call_counts", {}).get("search") if raw else None,
              "iters": (raw or {}).get("n_iterations") if raw else None,
              "oracle": ora.get(q, {}).get("correct"), "closed": clo.get(q, {}).get("correct")}

wrong = sum(v for k, v in out.items() if k != "solved")
print(f"824 questions: solved {out['solved']} ({out['solved']/824:.1%}), wrong {wrong}\n")
for k in sorted(out):
    if k == "solved":
        continue
    qs = [q for q in rec if rec[q]["bucket"] == k]
    caps = sum(1 for q in qs if (rec[q]["iters"] or 0) >= TURN_CAP); bud = sum(1 for q in qs if (rec[q]["searches"] or 0) >= SEARCH_CAP)
    kc = sum(1 for q in qs if rec[q]["closed"])
    print(f"{k:42s} {out[k]:4d}  {out[k]/wrong:5.1%} of wrong | hit {TURN_CAP}-turn cap {caps:3d} | used all {SEARCH_CAP} searches {bud:3d} | closed-book knew it {kc:3d}")

# reasoning-type breakdown (types are multi-label)
print("\nby FRAMES reasoning type (multi-label): accuracy | share of its failures in C (beyond model) / D (retrieval) / E (found all)")
for t in ["Numerical reasoning", "Tabular reasoning", "Multiple constraints", "Temporal reasoning", "Post processing"]:
    qs = [q for q in rec if t in rec[q]["types"]]
    w = [q for q in qs if rec[q]["bucket"] != "solved"]
    sh = lambda p: sum(1 for q in w if rec[q]["bucket"].startswith(p)) / max(1, len(w))
    print(f"  {t:22s} n={len(qs):3d} acc {1-len(w)/len(qs):.3f} | C {sh('C'):.0%}  D {sh('D'):.0%}  E {sh('E'):.0%}  A {sh('A'):.0%}  B {sh('B'):.0%}")

# solved-by matrix vs single-shot conditions
bm4 = {str(x["i"]): x for x in map(json.loads, open(f"{R}/logs/frames/llm_bm25_4.jsonl"))}
agent = {q for q in rec if rec[q]["bucket"] == "solved"}
oset = {q for q in ora if ora[q].get("correct")}; b4 = {q for q in bm4 if bm4[q].get("correct")}
print(f"\nagent solved {len(agent)} | oracle {len(oset)} | BM25-4 single shot {len(b4)}")
print(f"  oracle right, agent wrong {len(oset - agent)} | agent right, oracle wrong {len(agent - oset)}")
print(f"  BM25-4 right, agent wrong {len(b4 - agent)} | agent right, BM25-4 wrong {len(agent - b4)}")
json.dump(rec, open(OUT, "w"), indent=0)
