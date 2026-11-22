"""Re-judge, with gpt-4.1, the two BrowseComp-Plus arms of the follow-up studies that still had only Qwen3.6 verdicts:
SP-2 (reranker only, test200) and GEPA-2 (the GEPA run-2 winner d911ebbe23 on test200). Run 2026-11-10/11.

From original/RLM_2026-11-11/RLM/scripts/bcp_rejudge_gpt41_rq4.py of agents/shared (a second snapshot: the file was
written after the first migration). Changed only: import wiring (bcp_judge.py of this package, unchanged judge and
template), the original machine's paths (rollouts and earlier verdicts under LOGS_ROOT; the split file from
studies/prompt_optimisation/splits/) and this documentation (agents/shared/CHANGES_FROM_ORIGINAL.md). The verdict files
it wrote are in results/follow_up/verdicts/. Run as a script (module-level code, as in the original):
  set -a; . configs/paths.env; set +a
  python agents/shared/rlm_shared/grading/rejudge_bcp_rq4.py

Original docstring follows (paths in it are relative to the original repository).

Re-judge the two remaining Qwen3.6-judged BrowseComp-Plus arms of the search-policy / GEPA study with gpt-4.1
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

# Refactor (import wiring and paths): was R = <original repo>; sys.path.insert(0, f"{R}/scripts"); from bcp_judge ...;
# L = f"{R}/logs". The split file is the byte-identical copy in studies/prompt_optimisation/splits/.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))   # <repo>/agents/shared
from rlm_shared.grading.bcp_judge import grade_outputs, run_outputs  # noqa: E402
from rlm_shared.paths import path  # noqa: E402

R = str(path("ORIGINAL_REPO_ROOT"))   # only for the relative rollout_dir written to the summary ("logs/...")
L = str(path("LOGS_ROOT"))
SPLIT = os.path.join(_HERE, "..", "..", "..", "..", "studies", "prompt_optimisation", "splits", "bcp_gepa_splits", "split.json")
OUT = f"{L}/bcp_rejudge_gpt41"
ARMS = {
    "rerank_only": f"{L}/bcp_policy_e2e_rerankonly/rollouts/108e3760dc",
    "gepa_winner": f"{L}/gepa_bcp2/rollouts/d911ebbe23",
}
CONTROL = json.load(open(f"{OUT}/16.json"))["verdicts"]  # gpt-4.1 verdicts of the Qwen3.6 ColBERT control, all 830
FULL_POLICY = json.load(open(f"{OUT}/17.json"))["verdicts"]  # gpt-4.1 verdicts, novelty + re-rank, all 830
# The winner's rollout dir also holds GEPA's own evaluation rollouts (D_pareto etc.): only test200 is held out.
TEST200 = {str(q) for q in json.load(open(SPLIT))["test200"]}

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
