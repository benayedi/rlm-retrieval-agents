"""Search-policy study, SP-3: grade the GPT-5 reranker pilot (100 BrowseComp-Plus questions) with the gpt-4.1 judge and
compare it with the stored answers of BCP-23 (0.8747).

From original/RLM/scripts/bcp_pilot_grade.py; changed: import wiring (bcp_judge is
agents/shared/rlm_shared/grading/bcp_judge.py), locations from configs/paths.env, this paragraph. CALLS THE AZURE
gpt-4.1 JUDGE (AZ_ENV) for every answer not in the judge cache. Reads <LOGS_ROOT>/bcp_gpt5_rr/rr and
<RLM_DATA_DIR>/bcp_gpt5_rr/{pilot100.json, control_gpt5_colbert8192_high.json}; writes the judge cache and
pilot_report.json into <LOGS_ROOT>/bcp_gpt5_rr (the run directory, as in the original). The reported SP-3 verdicts
are logs/bcp_rejudge_gpt41/pilot_rr.json (rlm_shared.grading.rejudge_bcp); see ../CHANGES_FROM_ORIGINAL.md SP-I6.
  AZ_ENV=<file> python analysis/grade_gpt5_pilot.py

Original docstring follows.

Grade and report the BrowseComp-Plus reranker pilot (2026-10-27): gpt-5 high, full-doc RLM, ColBERT + novelty +
Qwen3-Reranker-4B (logs/bcp_gpt5_rr/rr) on 100 questions = 50 the stored 0.8747 run got wrong + 50 it got right
(data/bcp_gpt5_rr/pilot100.json). Compared with the STORED 0.8747 answers (no fresh control, user's choice 10-26).

Judge: the supervisor's suprlm/src/judge.py (paper template, 3 votes) with gpt-4.1 (ours); the stored run was judged by
gpt-4.1, which this Azure resource does not have. Two views:
  (a) reranker arm (gpt-4.1 (ours)) vs the stored run's original gpt-4.1 verdicts, all 100;
  (b) both judged by gpt-4.1 (ours), on the stored answers that are complete. The supervisor artifact cuts each answer at
      1,500 chars; where that removed the "Exact Answer:" line the stored answer is not re-judged (8 of the 100).
gpt-4.1 (ours) vs gpt-4.1 agreement on the stored answers is printed, so the judge effect is visible.
Fixes on the 50 previously-wrong questions are an UPPER bound: some would flip on any re-run.
  AZ_ENV=/path/azure.env python scripts/bcp_pilot_grade.py
"""
import json, os, sys

# Import wiring: agents/shared/rlm_shared is located from this file; locations come from configs/paths.env
# (rlm_shared.paths.path). Original: absolute paths of the machine the experiments ran on.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir, "agents", "shared"))
from rlm_shared.paths import path  # noqa: E402
REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir))
D = f"{path('LOGS_ROOT')}/bcp_gpt5_rr"  # original: f"{R}/logs/bcp_gpt5_rr"
P = json.load(open(f"{path('RLM_DATA_DIR')}/bcp_gpt5_rr/pilot100.json"))
C = json.load(open(f"{path('RLM_DATA_DIR')}/bcp_gpt5_rr/control_gpt5_colbert8192_high.json"))   # qid -> [run, correct, wall, searches, answer]
IDS = P["wrong"] + P["right"]
from rlm_shared.grading.bcp_judge import grade_outputs, run_outputs  # noqa: E402  (original: scripts/bcp_judge.py)

CP = f"{D}/_judge_cache_gpt41.json"
complete = [q for q in IDS if "Exact Answer" in (C[q][4] or "")]   # stored answers cut at 1500 chars lose this line


rr = grade_outputs(run_outputs(f"{D}/rr"), CP)
st = grade_outputs({q: C[q][4] for q in complete}, CP)
missing = [q for q in IDS if q not in rr]
print(f"reranker arm: {len(rr)} of 100 judged" + (f"  (MISSING {len(missing)}: scored wrong)" if missing else ""))
old = {q: bool(C[q][1]) for q in IDS}
new = {q: rr.get(q, False) for q in IDS}


def view(name, base, ids):
    for lab, s in (("previously WRONG", [q for q in ids if q in P["wrong"]]), ("previously RIGHT", [q for q in ids if q in P["right"]])):
        b = sum(base[q] for q in s); n = sum(new[q] for q in s)
        fix = sum(new[q] and not base[q] for q in s); brk = sum(base[q] and not new[q] for q in s)
        print(f"  {name:34s} {lab:17s} n={len(s):3d}  stored {b:3d} -> reranker {n:3d}   fixed {fix:2d}  broken {brk:2d}")


print("\n(a) reranker (gpt-4.1 (ours)) vs stored verdicts (gpt-4.1), all 100")
view("", old, IDS)
print(f"\n(b) both judged by gpt-4.1 (ours), {len(complete)} complete stored answers")
view("", {q: st[q] for q in complete}, complete)
agree = sum(st[q] == old[q] for q in complete)
print(f"\njudge agreement on the stored answers: gpt-4.1 (ours) = gpt-4.1 on {agree}/{len(complete)}"
      f"  (gpt-4.1 (ours) says right where 4.1 said wrong: {sum(st[q] and not old[q] for q in complete)},"
      f" the reverse: {sum(old[q] and not st[q] for q in complete)})")

# reweight the two strata to the full 830 (view a): est = acc + P(wrong) * fix_rate - P(right) * break_rate
nw, nr = P["n_wrong_830"], P["n_right_830"]
fr = sum(new[q] for q in P["wrong"]) / len(P["wrong"])
br = sum(not new[q] for q in P["right"]) / len(P["right"])
acc = nr / (nw + nr)
est = acc + nw / (nw + nr) * fr - nr / (nw + nr) * br
print(f"\nre-weighted to all 830 (view a): stored {acc:.4f} -> reranker estimate {est:.4f} ({est - acc:+.4f});"
      f" fix rate {fr:.2f} on the {nw} wrong, break rate {br:.2f} on the {nr} right."
      f"\nThe fix rate is an upper bound (no fresh control), the break rate measures run-to-run loss plus any harm.")
json.dump({"rr_gpt41": rr, "stored_rejudged_gpt41": st, "stored_gpt41": old, "estimate_830": est},
          open(f"{D}/pilot_report.json", "w"), indent=1)
