"""Per-question turn counts for the snippet-RLM runs, counted from the trajectory JSONL (one 'iteration' record per root
turn; the harness's own `iterations` field is broken, HANDOFF Part II §4). Also records the turn budget (metadata
max_iterations) and whether the last turn carried a final answer. Keyed by the first 400 characters of the question.
Full-doc runs need none of this: their n_iterations field is correct and failure_analysis_extract.py already reads it.

usage: failure_analysis_turns.py <traj_dir> <out.json>

Refactored copy of scripts/failure_analysis_turns.py of the original repository, unchanged apart from this docstring (see
scripts/reporting/README.md for the pipeline order).
  python scripts/reporting/failure_analysis_turns.py <traj_dir> <out.json>   (one call per snippet run)
"""
import glob, json, os, re, sys

d, out = sys.argv[1], sys.argv[2]
res = {}
for f in sorted(glob.glob(f"{d}/*.jsonl"), key=os.path.getmtime):   # later retries overwrite earlier ones
    cap = None; n = 0; key = None; last = None; first_final = None
    with open(f, "rb") as fh:
        for line in fh:
            head = line[:60]
            if b'"type": "metadata"' in head:
                cap = json.loads(line).get("max_iterations"); continue
            if b'"type": "iteration"' not in head:
                continue
            n += 1
            if key is None:
                p = json.loads(line)["prompt"]
                users = [m.get("content", "") for m in p if isinstance(m, dict) and m.get("role") == "user"] if isinstance(p, list) else [str(p)]
                key = re.sub(r"^Answer the following:\s*", "", users[0] if users else "").strip()[:400]
            i = line.rfind(b'"final_answer": ')
            last = line[i + 16:i + 20] if i >= 0 else b"null"
            if first_final is None and not last.startswith(b"null"):
                first_final = n
    if n and key:
        res[key] = dict(turns=n, cap=cap, final_on_last=not last.startswith(b"null"), final_turn=first_final,
                        traj=os.path.basename(f))
json.dump(res, open(out, "w"))
print(d, len(res))
