"""Average iterations (root-model REPL turns) per question for every filled row of docs/Experiments.xlsx (2026-10-29).
Snippet RLM: counted from the trajectory logs (one 'iteration' record per root turn; the harness's own `iterations`
field is broken, HANDOFF Part II §4), one trajectory per question (the latest, when a question was retried; keyed by
the question message). Full-doc RLM: n_iterations of each run_one record (head-parsed; records can be 70 MB). Earlier
rows reuse the stored, already trajectory-counted values (rlm_results_summary.xlsx Experiments column S).
Prints {sheet: {row: value}}.

Refactored copy of scripts/run5d/iteration_counts.py of the original repository; see scripts/reporting/README.md for the pipeline order
and shared/CHANGES_FROM_ORIGINAL.md for the changes (run-log paths from LOGS_ROOT and FULLDOC_ROOT).
  python scripts/reporting/iteration_counts.py > iterations.json"""
import glob, json, os, re, statistics as st, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared"))   # <repo>/shared
from rlm_shared.paths import path
# Refactor: was R = "<original repo>"; F = f"{R}/logs/run5d/full". Every f"{R}/logs/..." below is now f"{L}/...".
L = str(path("LOGS_ROOT")); F = f"{L}/run5d/full"
DEC = json.JSONDecoder()


def traj_iters(d):
    by = {}
    for f in sorted(glob.glob(f"{d}/*.jsonl"), key=os.path.getmtime):
        n = 0; key = None
        with open(f) as fh:
            for line in fh:
                if '"type": "iteration"' not in line[:40]:
                    continue
                n += 1
                if key is None:
                    p = json.loads(line)["prompt"]
                    msgs = p if isinstance(p, list) else []
                    users = [m.get("content", "") for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
                    key = (users[0] if users else str(p))[:400]
        if n:
            by[key or f] = n
    return st.mean(by.values()), len(by)


def head_field(f, name):
    head = open(f).read(400_000); i = head.find(f'"{name}":')
    if i < 0:
        return None
    j = i + len(name) + 3
    while head[j] in " \n":
        j += 1
    return DEC.raw_decode(head, j)[0]


def records(files):
    v = [head_field(f, "n_iterations") for f in files]
    v = [x for x in v if isinstance(x, (int, float))]
    return st.mean(v), len(v)


OUT = {"BrowseComp+": {}, "FRAMES": {}, "OBLIQ - Math": {}, "OBLIQ - Twitter": {}, "OBLIQ - Congress": {}}
B = OUT["BrowseComp+"]
B[9], B[11] = (5.6, "stored"), (4.7, "stored")
B[13] = traj_iters(f"{F}/bcp_snip_gpt5_bm25/traj")
B[19] = traj_iters(f"{F}/bcp_snip_qwen_bm25/traj")
B[21], B[23] = (11.2, "stored"), (10.8, "stored")
B[25] = records([f for f in glob.glob(f"{L}/bcp_gpt5_bm25/runs/*.json") if not os.path.basename(f).startswith("_")])
B[27] = records(glob.glob(f"{L}/run5d/bcp_fd_qwen_dense/rollouts/e925f0f6c7/*.json"))
B[29], B[31] = (20.9, "stored"), (27.3, "stored")
FR = OUT["FRAMES"]
FR[7] = traj_iters(f"{F}/frames_snip_gpt5/traj")
FR[9] = traj_iters(f"{F}/frames_snip_qwen/traj")
FR[11] = records([f for f in glob.glob(f"{F}/frames_fd_gpt5/*.json") if not os.path.basename(f).startswith(("_", "graded"))])
FR[13] = records([f for f in glob.glob(f"{L}/frames/sup_q36_bm25f/[0-9]*.json")])
for t in ("math", "twitter", "congress"):
    O = OUT[f"OBLIQ - {t.capitalize()}"]
    O[7] = traj_iters(f"{F}/obliq_snip_{t}/traj")
    O[9] = records(glob.glob(f"{path('FULLDOC_ROOT')}/runs/obliq_fd_{t}/q*.json"))
json.dump({s: {r: [round(v[0], 1), v[1]] for r, v in d.items()} for s, d in OUT.items()}, sys.stdout)
