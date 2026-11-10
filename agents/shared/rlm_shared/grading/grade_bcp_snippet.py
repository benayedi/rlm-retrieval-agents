"""Grade a BrowseComp-Plus snippet-REPL-agent run directory with gpt-4.1 and print the Experiments.xlsx numbers.

Origin: scripts/run5d/bcp_snip_grade.py of the original repository. Changed only: import wiring and this
documentation (agents/shared/CHANGES_FROM_ORIGINAL.md). Run as a script; importing it runs it (module-level code, as in the
original), loads the BrowseComp-Plus test split and calls the judge.
  set -a; . configs/paths.env; set +a
  python agents/shared/rlm_shared/grading/grade_bcp_snippet.py <run_dir>
KNOWN ISSUE (not fixed): "correct": bool(ok[q]) turns the judge's "filtered" verdict into True; see
agents/shared/CHANGES_FROM_ORIGINAL.md, "Issues found, not fixed".

Original docstring follows (script paths in it are those of the original repository).

Grade a BrowseComp-Plus snippet-RLM run directory (scripts/run5d/bcp_snip.sh: one q<idx>.json per question, idx =
position in the Tevatron/browsecomp-plus test split) with gpt-4.1 (scripts/bcp_judge.py: official template, 3 votes,
majority; verdicts cached in <dir>/_judge_cache_gpt41.json) and print the sheet numbers (2026-10-28).
A question without a result file, with an error, or stopped by the watchdog counts wrong. Accuracy is over all 830.
Recall, searches and latency are means over the questions with a result. Cost comes from <dir>/usage.jsonl (RLM_USAGE_LOG
lines; gpt-5 $1.25 / $0.125 cached / $10, gpt-5.2 $1.75 / $0.175 / $14 per 1M tokens) divided by the questions with a result.
Writes <dir>/graded_gpt41.json.
  AZ_ENV=/path/azure.env python scripts/run5d/bcp_snip_grade.py <dir>
"""
import glob, json, os, statistics as st, sys

# Refactor (import wiring): was sys.path.insert(0, "<original repo>/scripts"); from bcp_judge import ...
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # <repo>/agents/shared
from rlm_shared.grading.bcp_judge import grade_outputs  # noqa: E402
from datasets import load_dataset  # noqa: E402

D = sys.argv[1].rstrip("/")
QID = [str(r["query_id"]) for r in load_dataset("Tevatron/browsecomp-plus", split="test")]
recs = {}
for f in glob.glob(f"{D}/q*.json"):
    x = json.load(open(f)); x = x[0] if isinstance(x, list) else x
    recs[QID[x["q_idx"]]] = x
outs = {q: "" if x.get("error") not in (None, "None") else (x.get("predicted") or "") for q, x in recs.items()}
ok = grade_outputs(outs, f"{D}/_judge_cache_gpt41.json")
g = {q: {"correct": bool(ok[q]), "old_correct": bool(x.get("correct")), "gold_recall": x.get("gold_recall"),
         "searches": x.get("search_calls"), "latency_s": x.get("latency_s"), "iterations": x.get("iterations"),
         "llm_query_calls": x.get("llm_query_calls"), "error": x.get("error")} for q, x in recs.items()}
json.dump(g, open(f"{D}/graded_gpt41.json", "w"), indent=0)
N, n = len(QID), len(recs)
c = sum(v["correct"] for v in g.values()); old = sum(v["old_correct"] for v in g.values())
errs = sum(1 for v in g.values() if v["error"] not in (None, "None"))
line = (f"{D}: {n}/{N} results ({errs} with error) | accuracy gpt-4.1 {c}/{N} = {c / N:.4f} (over results {c / n:.4f}; "
        f"harness judge {old / N:.4f}) | gold recall {st.mean(v['gold_recall'] or 0 for v in g.values()):.4f} | "
        f"searches {st.mean(v['searches'] or 0 for v in g.values()):.2f} | time {st.mean(v['latency_s'] or 0 for v in g.values()):.1f} s "
        f"({st.mean(v['latency_s'] or 0 for v in g.values()) / 60:.1f} min) | llm_query on {sum(1 for v in g.values() if v['llm_query_calls']) / n:.0%}")
if os.path.exists(f"{D}/usage.jsonl"):
    cached = unc = 0.0
    for l in open(f"{D}/usage.jsonl"):
        x = json.loads(l); a, cc, o = (1.75, 0.175, 14) if "5.2" in x["model"] else (1.25, 0.125, 10)
        cached += ((x["prompt"] - x["cached"]) * a + x["cached"] * cc + x["completion"] * o) / 1e6
        unc += (x["prompt"] * a + x["completion"] * o) / 1e6
    line += f" | cost/q cached ${cached / n:.4f} uncached ${unc / n:.4f} (total ${cached:.0f})"
else:
    line += " | local GPU (no API cost)"
print(line)
