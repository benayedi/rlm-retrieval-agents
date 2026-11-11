"""FRC-1 report (Qwen3.6 judge, superseded): all 824 FRAMES questions, document REPL agent with search cap 25 / 50 turns
(three run directories merged) vs search cap 10 (FR-13), paired, missing = wrong. Reads each directory's graded.json,
i.e. the Qwen3.6 verdicts; the reported FRC-1 numbers are the gpt-4.1 re-judge (graded_gpt41.json, results/follow_up/).

From original/RLM_2026-11-11/RLM/scripts/frames_s25_full_report.py. Changed only: the location of the run
directories and this documentation (agents/document_repl/CHANGES_FROM_ORIGINAL.md). The last line prints fixed numbers
of the original repository's documents. Run: python agents/document_repl/src/frames_cap25_report.py

Original docstring follows (paths in it are relative to the original repository).

All 824 FRAMES questions: supervisor RLM agent with 25 searches / 50 turns (three dirs merged) vs the
10-search baseline (logs/frames/sup_q36_bm25f) and single-shot BM25-4 / oracle. Missing = wrong."""
import json, math, os, statistics as st
from math import comb
# Refactor (paths): was R = "<original repository>" and p = f"{R}/logs/frames/{d}/graded.json". The run directories are
# read from FRAMES_RUNS_DIR (set by experiments/frames_qwen_cap25_rest.sh to $RUNS_DIR/frames), else from the original
# location LOGS_ROOT/frames.
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared"))
from rlm_shared.paths import path  # noqa: E402
FRD = os.environ.get("FRAMES_RUNS_DIR") or os.path.join(str(path("LOGS_ROOT")), "frames")
def g(d):
    p = f"{FRD}/{d}/graded.json"
    return {v["qid"]: v for v in json.load(open(p)).values()} if os.path.exists(p) else {}
S = {**g("sup_q36_bm25f_s25"), **g("sup_q36_bm25f_s25_rest0"), **g("sup_q36_bm25f_s25_rest1")}
B = g("sup_q36_bm25f")
qs = [str(q) for q in range(824)]
a = [1 if B.get(q, {}).get("correct") else 0 for q in qs]; b = [1 if S.get(q, {}).get("correct") else 0 for q in qs]
w = sum(y > x for x, y in zip(a, b)); l = sum(x > y for x, y in zip(a, b))
p = min(1, 2 * sum(comb(w + l, i) for i in range(min(w, l) + 1)) / 2 ** (w + l))
d = [y - x for x, y in zip(a, b)]; se = st.stdev(d) / math.sqrt(len(d))
acc = sum(b) / 824
print(f"25-search agent: results {len(S)}/824 | accuracy {sum(b)}/824 = {acc:.3f} (+-{1.96*math.sqrt(acc*(1-acc)/824):.3f}) | "
      f"gold recall {st.mean(S[q]['gold_recall'] for q in S):.3f} | searches {st.mean(S[q]['searches'] for q in S):.1f}")
print(f"10-search agent: {sum(a)}/824 = {sum(a)/824:.3f} | paired diff {st.mean(d)*100:+.1f} pts (t {st.mean(d)/se:+.2f}), won {w} lost {l}, McNemar p={p:.4f}")
print("paper (Gemini-Pro-1.5): BM25-4 0.474, multi-step planning 0.66; ours same model: BM25-4 0.477, oracle 0.774")
