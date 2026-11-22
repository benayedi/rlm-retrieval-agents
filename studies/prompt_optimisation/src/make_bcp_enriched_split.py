"""Failure-enriched selection set for GEPA run 2 on BrowseComp-Plus (plan B, 2026-10-08).

Refactored copy of original/RLM/scripts/bcp_enrich_split.py (studies/prompt_optimisation/CHANGES_FROM_ORIGINAL.md).
Run: python src/make_bcp_enriched_split.py (after the seed ran on the feedback pool; GEPA-2 chain). Rewrites
  studies/prompt_optimisation/splits/bcp_gepa_splits/split_enriched_q36.json in place.
Changed: paths only (`os` added to the import line for them).
Names: 'supervisor harness' / 'full-doc RLM' in the text below is the document REPL agent (agents/document_repl);
  'official RLM harness' / 'our harness' / run_obliq_mine.py is the snippet REPL agent (agents/snippet_repl,
  src/run_obliq.py).

WHY. In run 1 the 100-question D_pareto was proportional to the benchmark, so ~66 questions
were answered correctly by every prompt and only ~34 could ever separate two candidates: a
paired accuracy difference had se ~0.04-0.05 and nothing under ~8 points was visible. The
headroom is in this model's failures (wrong, or low gold recall), so the new D_pareto takes
70 questions the SEED (Qwen3.6, presence_penalty 1.5) got wrong or found <50% of the gold for,
plus 30 it got right with full recall as a REGRESSION GUARD.

CONSEQUENCE, stated up front: means on this set are NOT benchmark estimates. Candidate-vs-seed
deltas here show where a prompt helps or hurts; the claim is made on test200 (proportional,
untouched). The population weights of the two strata are written out so a reweighted estimate
can be reported alongside.

Inputs: the seed's cached records for the old pareto (100) and feedback (200) questions.
Output: data/bcp_gepa_splits/split_enriched_q36.json  (pareto 100 / feedback 200 / test200 / test_rest)
"""
import json, os, random, sys

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
SEED_KEY = "ee5ea32a1c"   # Qwen3.6 seed, fingerprint 29587b4e
old = json.load(open(f"{SPLITS}/bcp_gepa_splits/split.json"))
C = json.load(open(f"{RUNS}/gepa_bcp/cache.json"))
pool = old["pareto"] + old["feedback"]
recs = {q: C.get(f"{SEED_KEY}|{q}") for q in pool}
missing = [q for q, r in recs.items() if r is None or r.get("status") == "killed"]   # killed = never finished
if missing:
    sys.exit(f"{len(missing)} pool questions have no finished seed record (missing or killed) -- run --seed-split feedback first")


def fail(r):
    return (not r.get("correct")) or (r.get("gold_recall") or 0) < 0.5


fails = sorted([q for q, r in recs.items() if fail(r)], key=int)
easy = sorted([q for q, r in recs.items() if r.get("correct") and (r.get("gold_recall") or 0) >= 0.999], key=int)
rng = random.Random(20261008)
rng.shuffle(fails); rng.shuffle(easy)
n_fail = min(70, len(fails)); n_easy = 100 - n_fail
pareto = sorted(fails[:n_fail] + easy[:n_easy], key=int)
feedback = sorted([q for q in pool if q not in set(pareto)], key=int)
split = {"seed": 20261008, "seed_key": SEED_KEY, "pareto": pareto, "feedback": feedback,
         "test200": old["test200"], "test_rest": old["test_rest"],
         "strata": {"fail": sorted(fails[:n_fail], key=int), "guard": sorted(easy[:n_easy], key=int)},
         "population_share": {"fail": len(fails) / len(pool), "guard_like": 1 - len(fails) / len(pool)}}
json.dump(split, open(f"{SPLITS}/bcp_gepa_splits/split_enriched_q36.json", "w"), indent=1)
acc = lambda qs: sum(1 for q in qs if recs[q].get("correct")) / len(qs)
print(f"pool {len(pool)}: failures {len(fails)} ({len(fails)/len(pool):.0%}), easy {len(easy)}")
print(f"new D_pareto: {n_fail} failures + {n_easy} guard = {len(pareto)}  (seed acc {acc(pareto):.3f})")
print(f"new D_feedback: {len(feedback)}  (seed acc {acc(feedback):.3f})")
