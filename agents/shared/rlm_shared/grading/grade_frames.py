"""Grade FRAMES run directories (either harness) with gpt-4.1 and print the Experiments.xlsx numbers.

Origin: scripts/frames_judge_gpt41.py of the original repository. Changed only: import wiring, the FRAMES question file
read from FRAMES_DATA_DIR, and this documentation (agents/shared/CHANGES_FROM_ORIGINAL.md). Run as a script; importing it
loads frames_rows.json and imports bcp_judge (see its import side effects), and grades every directory in sys.argv[1:].
  set -a; . configs/paths.env; set +a
  python agents/shared/rlm_shared/grading/grade_frames.py <run_dir> [<run_dir> ...]
KNOWN ISSUES (not fixed): no retry round when all votes of a question fail (it is reported as "no vote" and counted
wrong), the content-filter test matches only "content management policy", and gold_ids strings are parsed with eval();
see agents/shared/CHANGES_FROM_ORIGINAL.md.

Original docstring follows (script paths in it are those of the original repository).

Grade FRAMES (824 q) answers with gpt-4.1, the same judge as every BrowseComp+ cell (2026-10-27): the document REPL agent's
OFFICIAL_TEMPLATE, 3 votes, majority, via scripts/bcp_judge.py (Azure deployment gpt-4.1). Gold answers and gold
article ids from data/frames/frames_rows.json. A run without a completed, non-empty answer is wrong without judging;
a prompt Azure's content filter blocks is reported and counted wrong. Reads either harness's output:
  Document REPL   <dir>/<qid>.json   (suprlm run_one record: status, result[0].output, server/retrieved docids)
  snippet    <dir>/q###.json    (examples/frames_eval_rlm.py: [{q_idx, predicted, gold_recall, ...}])
Writes <dir>/graded_gpt41.json and prints accuracy, gold-article recall, searches.
  AZ_ENV=/path/azure.env python scripts/frames_judge_gpt41.py <dir> [<dir> ...]
"""
import glob, hashlib, json, os, statistics as st, sys, threading
from concurrent.futures import ThreadPoolExecutor

# Refactor (import wiring): was sys.path.insert(0, "<original repo>/scripts"); from bcp_judge import ...
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # <repo>/agents/shared
from rlm_shared.grading.bcp_judge import CLI, MODEL, TEMPLATE, TMPL, parse_judge  # noqa: E402
from rlm_shared.paths import path  # noqa: E402

# Refactor: was the absolute path <original repo>/data/frames/frames_rows.json.
ROWS = {str(r["i"]): r for r in json.load(open(path("FRAMES_DATA_DIR") / "frames_rows.json"))}
_LK = threading.Lock()


def load(d):
    """{qid: (answer, gold_recall, searches)}"""
    out = {}
    for f in glob.glob(f"{d}/*.json"):
        b = os.path.basename(f)
        if b.startswith(("_", "graded")):
            continue
        x = json.load(open(f))
        if isinstance(x, list):                           # snippet harness
            r = x[0]
            q = str(r["q_idx"]); ans = "" if r.get("error") not in (None, "None") else (r.get("predicted") or "")
            out[q] = (ans, float(r.get("gold_recall") or 0), int(r.get("search_calls") or 0))
        else:                                             # suprlm run_one record
            q = str(x["query_id"])
            ans = ((x.get("result") or [{}])[0] or {}).get("output", "") if x.get("status") == "completed" else ""
            got = set(map(str, x.get("server_docids") or x.get("retrieved_docids") or []))
            g = set(map(str, eval(ROWS[q]["gold_ids"]) if isinstance(ROWS[q]["gold_ids"], str) else ROWS[q]["gold_ids"]))
            out[q] = (ans or "", len(got & g) / len(g) if g else 0.0, (x.get("tool_call_counts") or {}).get("search", 0))
    return out


def grade(q, ans, cache):
    if not ans.strip():
        return q, False
    key = hashlib.sha256(f"frames|{TEMPLATE}|{MODEL}|{q}|{' '.join(ans.lower().split())}".encode()).hexdigest()[:24]
    with _LK:
        if key in cache:
            return q, cache[key]
    prompt = TMPL.format(question=ROWS[q]["question"], response=ans, correct_answer=ROWS[q]["answer"])
    votes = []
    for _ in range(3):
        for _try in range(4):
            try:
                r = CLI.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}],
                                                max_completion_tokens=2048)
                v = parse_judge(r.choices[0].message.content or "")
                if not v.get("parse_error"):
                    votes.append(v.get("correct") == "yes"); break
            except Exception as e:  # noqa: BLE001
                if "content management policy" in str(e):
                    return q, "filtered"
                print(f"[judge] {q}: {type(e).__name__} {str(e)[:150]}", file=sys.stderr, flush=True)
    if not votes:
        return q, None
    ok = sum(votes) * 2 > len(votes)
    with _LK:
        cache[key] = ok
    return q, ok


for d in sys.argv[1:]:
    runs = load(d)
    cp = f"{d}/_judge_cache_gpt41.json"
    cache = json.load(open(cp)) if os.path.exists(cp) else {}
    with ThreadPoolExecutor(32) as ex:
        res = dict(ex.map(lambda kv: grade(kv[0], kv[1][0], cache), runs.items()))
    json.dump(cache, open(cp, "w"))
    failed = [q for q, v in res.items() if v is None]; filt = [q for q, v in res.items() if v == "filtered"]
    ok = sum(v is True for v in res.values())
    json.dump({q: {"correct": res.get(q) is True, "verdict": res.get(q), "gold_recall": runs[q][1], "searches": runs[q][2]}
               for q in runs}, open(f"{d}/graded_gpt41.json", "w"), indent=1)
    print(f"{d}: {len(runs)}/824 results | accuracy {ok}/824 = {ok / 824:.4f} (missing = wrong) | gold recall "
          f"{sum(r[1] for r in runs.values()) / 824:.4f} | searches {st.mean(r[2] for r in runs.values()) if runs else 0:.2f}"
          f" | no vote {len(failed)} | content-filtered {len(filt)} {filt[:5]}")
