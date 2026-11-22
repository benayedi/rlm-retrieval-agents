"""Re-judge the two remaining Qwen3.6-judged BrowseComp-Plus arms of the search-policy / GEPA study with gpt-4.1
(2026-11-10), so every BrowseComp-Plus number in the thesis uses the paper's judge.

Arms (document REPL agent, Qwen3.6, Reason-ModernColBERT 8192, test200):
  rerank_only   logs/bcp_policy_e2e_rerankonly/rollouts/108e3760dc   (GEPA_BROWSECOMP_PLAN.md §7d, re-rank-only ablation)
  gepa_winner   logs/gepa_bcp2/rollouts/d911ebbe23                    (§7b, GEPA run-2 winner vs the original prompt)
Their controls are already judged with gpt-4.1: the original prompt's test200 rollouts (gepa_bcp2/rollouts/ee5ea32a1c)
are part of row 16 in logs/bcp_rejudge_gpt41/16.json. Judge and template are unchanged: scripts/bcp_judge.py
(OFFICIAL_TEMPLATE, 3 votes, majority); a non-completed run counts as wrong without judging.

  AZ_ENV=/path/azure.env python scripts/bcp_rejudge_gpt41_rq4.py
Only test200 questions are judged and compared. Writes logs/bcp_rejudge_gpt41/rq4_<arm>.json (verdicts + paired
summary vs row 16 and row 17) and rq4_summary.json.
"""
import json
import os
import sys

R = "/home/extstu_00/RLM"
sys.path.insert(0, f"{R}/scripts")
from bcp_judge import grade_outputs, run_outputs  # noqa: E402

L = f"{R}/logs"
OUT = f"{L}/bcp_rejudge_gpt41"
ARMS = {
    "rerank_only": f"{L}/bcp_policy_e2e_rerankonly/rollouts/108e3760dc",
    "gepa_winner": f"{L}/gepa_bcp2/rollouts/d911ebbe23",
}
CONTROL = json.load(open(f"{OUT}/16.json"))["verdicts"]  # gpt-4.1 verdicts of the Qwen3.6 ColBERT control, all 830
FULL_POLICY = json.load(open(f"{OUT}/17.json"))["verdicts"]  # gpt-4.1 verdicts, novelty + re-rank, all 830
# The winner's rollout dir also holds GEPA's own evaluation rollouts (D_pareto etc.): only test200 is held out.
TEST200 = {str(q) for q in json.load(open(f"{R}/data/bcp_gepa_splits/split.json"))["test200"]}

summary = {}
for arm, d in ARMS.items():
    outs = {q: a for q, a in run_outputs(d).items() if q in TEST200}
    new = grade_outputs(outs, f"{OUT}/_cache_rq4.json")
    filtered = sorted(q for q, v in new.items() if v == "filtered")
    v = {q: (x is True) for q, x in new.items()}
    paired = sorted(q for q in v if q in CONTROL)
    s = {
        "rollout_dir": os.path.relpath(d, R), "n": len(v), "empty_or_unfinished": sum(not a.strip() for a in outs.values()),
        "content_filtered_counted_wrong": filtered,
        "correct": sum(v.values()), "acc": round(sum(v.values()) / len(v), 4),
        "paired_n": len(paired),
        "paired_acc_arm": round(sum(v[q] for q in paired) / len(paired), 4),
        "paired_acc_control_row16": round(sum(bool(CONTROL[q]) for q in paired) / len(paired), 4),
        "won": sum(v[q] and not CONTROL[q] for q in paired), "lost": sum(CONTROL[q] and not v[q] for q in paired),
        "paired_acc_full_policy_row17": round(sum(bool(FULL_POLICY.get(q)) for q in paired) / len(paired), 4),
    }
    summary[arm] = s
    json.dump({"verdicts": v, "summary": s}, open(f"{OUT}/rq4_{arm}.json", "w"), indent=1)
    print(arm, s, flush=True)
json.dump(summary, open(f"{OUT}/rq4_summary.json", "w"), indent=1)
