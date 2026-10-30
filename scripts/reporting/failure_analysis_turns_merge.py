"""Add per-question turn counts, the turn budget and a turn outcome to per_question.json (in place).
Snippet rows: turns from failure_analysis_turns.py outputs (run5d) or the September token maps (BCP-9 / BCP-11, which
were counted from the now-deleted logs/rlm_traj32 and rlm_traj24). Full-doc rows: n_iterations, already in 'iters'.

usage: failure_analysis_turns_merge.py <per_question.json> <turns_dir>

Refactored copy of scripts/failure_analysis_turns_merge.py of the original repository; see scripts/reporting/README.md for the pipeline order
and shared/CHANGES_FROM_ORIGINAL.md for the changes (the original repository root from ORIGINAL_REPO_ROOT,
run logs from LOGS_ROOT).
"""
import json, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared"))   # <repo>/shared
from rlm_shared.paths import path
R = str(path("ORIGINAL_REPO_ROOT"))   # refactor: was the absolute path of the original repository
P, T = sys.argv[1], sys.argv[2]
D = json.load(open(P))

def norm(s, n):   # tokmap keys keep \\u escapes and collapse spaces; compare letters and digits only
    s = re.sub(r"\\+u[0-9a-fA-F]{4}", "", s or "")
    return re.sub(r"[^a-z0-9]", "", s.lower().encode("ascii", "ignore").decode())[:n]


def find(src, q):   # keys and questions are truncated at different lengths: match on the shorter prefix, if unique
    if q in src:
        return src[q]
    hit = [v for k, v in src.items() if k.startswith(q) or q.startswith(k)]
    return hit[0] if len(hit) == 1 else None


TOK = lambda n: {k[:60]: dict(turns=v["iters"]) for k, v in json.load(open(f"{R}/scripts/workbook/tokmaps/root_tok_rlm_traj{n}.json")).items()}
TJ = lambda n: json.load(open(f"{T}/{n}.json"))
# (sheet, row) -> (turn source keyed by question prefix, normalised prefix length, budget)
SNIP = {("BrowseComp+", 9): (TOK(32), 40, 100), ("BrowseComp+", 11): (TOK(24), 40, 100),
        ("BrowseComp+", 13): (TJ("bcp_snip_gpt5_bm25"), 40, 100), ("BrowseComp+", 19): (TJ("bcp_snip_qwen_bm25"), 40, 50),
        ("FRAMES", 7): (TJ("frames_snip_gpt5"), 250, 50), ("FRAMES", 9): (TJ("frames_snip_qwen"), 250, 50)}
SNIP = {k: ({norm(q, n): v for q, v in src.items()}, n, cap) for k, (src, n, cap) in SNIP.items()}

OUTCOMES = ["T1 Answered before the limit, correct", "T2 Answered before the limit, wrong", "T3 Gave up before the limit (abstained)",
            "T4 Used every turn, still correct", "T5 Used every turn, no correct answer", "T6 Stopped by time limit / error",
            "T0 Turns not logged"]


def outcome(o):
    if o["status"] != "ok":
        return OUTCOMES[5]
    if o["turns"] is None:
        return OUTCOMES[6]
    if o["turns"] >= o["turn_cap"]:
        return OUTCOMES[3] if o["correct"] else OUTCOMES[4]
    return OUTCOMES[0] if o["correct"] else (OUTCOMES[2] if o["abstained"] else OUTCOMES[1])


miss = {}
for o in D["qa"]:
    k = (o["sheet"], o["row"])
    if k in SNIP:
        src, n, cap = SNIP[k]
        x = find(src, norm(o["question"], n))
        o["turns"], o["turn_cap"] = (x["turns"] if x else None), cap
        if x is None and o["status"] == "ok":
            miss[k] = miss.get(k, 0) + 1
    else:
        o["turns"], o["turn_cap"] = (int(o["iters"]) if o["iters"] is not None else None), 30
    o["iters"] = o["turns"]
    o["hit_turn_cap"] = int(o["turns"] is not None and o["turns"] >= o["turn_cap"])
    o["turn_outcome"] = outcome(o)

for task in ("math", "twitter", "congress"):
    src = {norm(k, 250): v for k, v in TJ(f"obliq_snip_{task}").items()}
    qtext = {str(x["query_id"]): norm(x["question"], 250)
             for x in json.load(open(f"{path('LOGS_ROOT')}/run5d/full/obliq_snip_{task}/all.json"))}
    for o in D["obliq"]:
        if o["sheet"] != f"OBLIQ - {task.capitalize()}":
            continue
        if o["row"] == 9:
            x = find(src, qtext.get(o["qid"], "-"))
            o["turns"], o["turn_cap"] = (x["turns"] if x else None), 12
            if x is None:
                miss[(o["sheet"], 9)] = miss.get((o["sheet"], 9), 0) + 1
        else:
            o["turns"], o["turn_cap"] = (int(o["iters"]) if o["iters"] is not None else None), 30
        o["iters"] = o["turns"]
        o["hit_turn_cap"] = int(o["turns"] is not None and o["turns"] >= o["turn_cap"])

json.dump(D, open(P, "w"))
print("finished questions without a turn count:", miss)
