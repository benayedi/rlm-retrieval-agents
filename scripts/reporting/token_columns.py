"""Input and output tokens per question for every filled row of docs/Experiments.xlsx (2026-10-28), so rows can be
compared by work done instead of wall time (wall time depends on workers, rate limits and server load).
Sources, per row:
  run5d GPT runs          <run>/usage.jsonl (one line per Azure call, exact; includes retried attempts, like the cost cells)
  suprlm full-doc runs    per-question records, usage.model_usage_summaries (exact; no sub-LLM in design B)
  Full-Doc GPT-5 docs/supervisor_artifact.html, questions[].tok_in / tok_out (exact; its cost_per_q = these x
                          $1.25 / $10 per 1M, i.e. uncached)
  DSPy rows               logs/<files>.json input_tokens / output_tokens as logged by the harness
  Sept snippet rows       ESTIMATE: the JSON token fields hold only the sub-LM's usage when llm_query ran. Input = root
                          prompt chars / 4 from the trajectories (scripts/workbook/tokmaps) + logged sub input; output =
                          logged sub output + root iterations x the output per root iteration of the questions without a
                          sub-LM. Checked on the 2026-10-28 BM25 snippet run, which has exact usage: 8% low on input,
                          11% low on output.
Prints {sheet: {row: (input, output, kind)}} as JSON; kind is exact | estimate.

Refactored copy of scripts/run5d/token_columns.py of the original repository; see scripts/reporting/README.md and
shared/CHANGES_FROM_ORIGINAL.md (changes: the original repository root from ORIGINAL_REPO_ROOT, run logs from
LOGS_ROOT). Paths in the text above are relative to the original repository.
  python scripts/reporting/token_columns.py > tokens.json
"""
import glob, json, os, re, statistics as st, sys, ast
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared"))   # <repo>/shared
from rlm_shared.paths import path
R = str(path("ORIGINAL_REPO_ROOT")); L = str(path("LOGS_ROOT"))   # refactor: was R = "<original repo>"; L = f"{R}/logs"
sys.path.insert(0, f"{R}/scripts/workbook")   # unused: nothing is imported from scripts/workbook (kept, as in the original)


def usage(d, n):
    U = [json.loads(l) for l in open(f"{d}/usage.jsonl")]
    return sum(x["prompt"] for x in U) / n, sum(x["completion"] for x in U) / n


def sup_records(files):
    tin, tout = [], []
    for f in files:   # records can be 70 MB (full trajectory); "usage" sits in the first few KB, so read only the head
        with open(f) as fh:
            head = fh.read(200_000)
        m = re.search(r'"usage":\s*(\{"model_usage_summaries":\s*\{.*?\}\}\}?)', head, re.S)
        if not m:
            continue
        try:
            u = json.loads(m.group(1))
        except Exception:
            try:
                u = json.loads(m.group(1) + "}")
            except Exception:
                continue
        s = (u or {}).get("model_usage_summaries") or {}
        tin.append(sum(m["total_input_tokens"] for m in s.values())); tout.append(sum(m["total_output_tokens"] for m in s.values()))
    return st.mean(tin), st.mean(tout), len(tin)


def artifact(names):
    d = json.loads(re.search(r'<script id="trace"[^>]*>(.*?)</script>', open(f"{R}/docs/supervisor_artifact.html").read(), re.S).group(1))
    q = [x for r in d["runs"] if r["run"] in names for x in r["questions"]]
    return st.mean(x["tok_in"] for x in q), st.mean(x["tok_out"] for x in q), len(q)


def dspy(patterns):
    fs = sorted({f for p in patterns for f in glob.glob(f"{L}/{p}")}, key=os.path.getmtime)
    m = {}
    for f in fs:
        for r in json.load(open(f)):
            m[r["q_idx"]] = r
    v = [r for r in m.values() if r.get("input_tokens")]   # the first 50 dense questions were logged without tokens;
    return st.mean(r["input_tokens"] for r in v), st.mean(r.get("output_tokens") or 0 for r in v), len(v)  # sheet cost uses the same 775


def rollouts(parts):
    """Qwen3.6 full-doc rows: first completed rollout per question across the run's cache dirs."""
    seen = {}
    for d, key in parts:
        for f in glob.glob(f"{L}/{d}/rollouts/{key}/*.json"):
            q = os.path.basename(f)[:-5]
            if q not in seen:
                seen[q] = f
    return sup_records(seen.values())


def sept_snippet(prefixes, tokmap):
    def load(p):
        d = json.load(open(p)); return d[0] if isinstance(d, list) else d
    tm = json.load(open(f"{R}/scripts/workbook/tokmaps/root_tok_{tokmap}.json"))
    recs = {}
    for pre in prefixes:
        for p in glob.glob(f"{L}/{pre}_q*.json"):
            recs[int(p.split("_q")[1].split(".")[0])] = load(p)
    k = lambda q: re.sub(r"\s+", " ", q)[:60]
    sub = lambda r: (r.get("llm_query_calls") or 0) + (r.get("llm_query_batched_calls") or 0)
    nos = [(r.get("output_tokens") or 0, tm[k(r["question"])]["iters"]) for r in recs.values() if k(r["question"]) in tm and not sub(r)]
    rate = sum(o for o, _ in nos) / sum(i for _, i in nos)
    tin = tout = 0.0
    for r in recs.values():
        t = tm.get(k(r["question"])); li, lo = r.get("input_tokens") or 0, r.get("output_tokens") or 0
        if t and sub(r):
            tin += t["prompt_chars"] / 4.0 + li; tout += lo + rate * t["iters"]
        else:
            tin += li; tout += lo
    return tin / len(recs), tout / len(recs), len(recs)


F = f"{L}/run5d/full"
OUT = {
 "BrowseComp+": {
  9:  (*dspy(["browsecomp_v2medium_baseline_first50.json", "browsecomp_v2medium_baseline_200.json",
              "browsecomp_v2_medium_q148_q830.json", "browsecomp_v2_medium_retry56.json"])[:2], "exact"),
  11: (*dspy(["dl4096_s*.json", "dl4096x_s*.json"])[:2], "exact"),
  21: (*sept_snippet(["rlmfix32"], "rlm_traj32")[:2], "estimate"),
  23: (*sept_snippet([f"rlmfix{n}" for n in (24, 25, 29, 30, 31)], "rlm_traj24")[:2], "estimate"),
  25: (*usage(f"{F}/bcp_snip_gpt5_bm25", 828), "exact"),
  33: (*artifact({"final100_medium", "final200_medium", "final100_e", "final_rest"})[:2], "exact"),
  35: (*artifact({"colbert8192_high400", "colbert8192_high100b", "colbert8192_high100c", "colbert8192_high_rest230"})[:2], "exact"),
  37: (*sup_records(glob.glob(f"{L}/bcp_gpt5_bm25/runs/*.json"))[:2], "exact"),
  41: (*rollouts([("gepa_bcp2", "ee5ea32a1c"), ("bcp_testrest_ctrl", "ee5ea32a1c"), ("gepa_bcp", "ee5ea32a1c")])[:2], "exact"),
  43: (*rollouts([("bcp_bm25_test200", "11d0c5bc6b"), ("bcp_bm25_rest_gpu0", "11d0c5bc6b"), ("bcp_bm25_rest_gpu1", "11d0c5bc6b")])[:2], "exact"),
 },
 "FRAMES": {
  11: (*usage(f"{F}/frames_snip_gpt5", 824), "exact"),
  15: (*usage(f"{F}/frames_fd_gpt5", 824), "exact"),
  17: (*sup_records(glob.glob(f"{L}/frames/sup_q36_bm25f/[0-9]*.json"))[:2], "exact"),
 },
}
for task, n in (("Math", 151), ("Twitter", 281), ("Congress", 254)):
    OUT[f"OBLIQ - {task}"] = {9: (*usage(f"{F}/obliq_snip_{task.lower()}", n), "exact"),
                              11: (*usage(f"{F}/obliq_fd_{task.lower()}", n), "exact")}
if __name__ == "__main__":
    json.dump({s: {r: [round(a), round(b), k] for r, (a, b, k) in v.items()} for s, v in OUT.items()}, sys.stdout, indent=1)
