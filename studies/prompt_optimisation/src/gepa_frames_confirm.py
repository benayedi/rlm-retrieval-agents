"""Confirm GEPA's best FRAMES prompt on the 324 questions no GEPA split used (not feedback, pareto

Refactored copy of original/RLM/scripts/gepa_frames_confirm.py (studies/prompt_optimisation/CHANGES_FROM_ORIGINAL.md).
Run: studies/prompt_optimisation/experiments/gepa3_frames_confirm.sh (GEPA-3 confirmation), or python
  src/gepa_frames_confirm.py [--report].
Changed: paths only (gepa_frames.py next to this file, outputs and the GEPA-3 best candidate under RUNS_DIR, the split
  from studies/prompt_optimisation/splits/). See PO-I3: the paired n excludes killed records.
Names: 'supervisor harness' / 'full-doc RLM' in the text below is the document REPL agent (agents/document_repl);
  'official RLM harness' / 'our harness' / run_obliq_mine.py is the snippet REPL agent (agents/snippet_repl,
  src/run_obliq.py).
or test), paired against the seed's cached records from the 824-question seed run.

The candidate is logs/gepa_frames/best_candidate.json with one edit: the phrase "gold article, and
retrieving it counts toward your score" (GEPA leaking its own objective, correct + 0.5 x gold
recall, into the prompt) becomes "article; retrieve it". Everything else is gepa_frames.py's
(rollout path, harness flags, judge). Missing / killed results count as WRONG for both arms.
  python scripts/gepa_frames_confirm.py            # run (resumable) + report
  python scripts/gepa_frames_confirm.py --report
"""
import json, math, os, random, statistics as st, sys
from math import comb
import importlib.util

# Import wiring and paths (refactor, rules 2-3). Original: hard-coded locations of the original repository (R), the
# document REPL agent working directory (SUP) and the interpreter. Code now comes from this repository (computed from
# __file__), data and run outputs from configs/paths.env (rlm_shared.paths). The original logs/ directory is RUNS_DIR.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, os.pardir))
DOC_SRC = os.path.join(_REPO, "agents", "document_repl", "src")    # split_runner.py (was scripts/gepa_bcp.py), run_bm25.py
SPLITS = os.path.join(_REPO, "studies", "prompt_optimisation", "splits")   # original: <original repo>/data
sys.path.insert(0, os.path.join(_REPO, "agents", "shared"))
from rlm_shared.paths import path  # noqa: E402
RUNS = str(path("RUNS_DIR"))                                         # original: f"{R}/logs"
# GPU 0 ONLY. gepa_frames.py defaults to :8030 + :8031; the first launch (2026-10-17 07:18) sent every
# second rollout to :8031 before that server was up (29 APIConnectionError records, removed) and later
# onto GPU 1, which runs BrowseComp-Plus. Must be set before gepa_frames.py is imported.
os.environ["GEPA_BCP_SERVERS"] = "http://localhost:8030/v1"
_s = importlib.util.spec_from_file_location("gf", os.path.join(_HERE, "gepa_frames.py"))
gf = importlib.util.module_from_spec(_s); _s.loader.exec_module(gf)
G = gf.G
G.OUT = f"{RUNS}/gepa_frames_confirm"; G.PROG = f"{G.OUT}/progress.txt"; G.CACHE_F = f"{G.OUT}/cache.json"

OLD = ("gold article, and retrieving it counts toward your score even when you already know the fact")
NEW = ("article; retrieve it even when you already know the fact")


def candidate() -> dict:
    c = json.load(open(f"{RUNS}/gepa_frames/best_candidate.json"))
    assert c["LEAN_CLAUSE"].count(OLD) == 1, "leak phrase not found"
    c["LEAN_CLAUSE"] = c["LEAN_CLAUSE"].replace(OLD, NEW)
    assert "your score" not in c["LEAN_CLAUSE"].lower() and "gold article" not in c["LEAN_CLAUSE"].lower()
    return c


def main():
    sp = json.load(open(f"{SPLITS}/frames/gepa_splits/split.json"))
    used = set(sp["feedback"]) | set(sp["pareto"]) | set(sp["test"])
    ids = [str(q) for q in range(824) if str(q) not in used]
    random.Random(0).shuffle(ids)
    seed = gf.seed_blocks(); sk = G.ckey(seed)
    cand = candidate(); ck = G.ckey(cand)
    G._cache_load()
    gf.preload_seed(sk)
    if "--report" not in sys.argv:
        G._log(f"[confirm] candidate {ck} (best 598a758e3a minus the score phrase) vs seed {sk} on {len(ids)} unused questions")
        G.run_batch(cand, ids, 8)
    rec = lambda k, q: G._CACHE.get(f"{k}|{q}") or {}
    done = [q for q in ids if rec(ck, q).get("status") not in (None, "killed")]
    a = [1 if rec(sk, q).get("correct") else 0 for q in done]; b = [1 if rec(ck, q).get("correct") else 0 for q in done]
    w = sum(y > x for x, y in zip(a, b)); l = sum(x > y for x, y in zip(a, b))
    p = min(1, 2 * sum(comb(w + l, i) for i in range(min(w, l) + 1)) / 2 ** (w + l)) if w + l else 1
    d = [y - x for x, y in zip(a, b)]; se = st.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else 0
    G._log(f"[confirm] unused n={len(done)}/{len(ids)} | seed {sum(a)} ({st.mean(a):.3f}) | candidate {sum(b)} ({st.mean(b):.3f}) "
           f"| diff {st.mean(d):+.3f} se {se:.3f} t {st.mean(d)/se if se else 0:+.2f} | won {w} lost {l} McNemar p={p:.3f}")


if __name__ == "__main__":
    main()
