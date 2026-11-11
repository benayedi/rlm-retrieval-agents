"""Failure-enriched selection set for GEPA run 2 on BrowseComp-Plus (plan B, 2026-10-08).

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
import json, random, sys

R = "/home/extstu_00/RLM"
SEED_KEY = "ee5ea32a1c"   # Qwen3.6 seed, fingerprint 29587b4e
old = json.load(open(f"{R}/data/bcp_gepa_splits/split.json"))
C = json.load(open(f"{R}/logs/gepa_bcp/cache.json"))
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
json.dump(split, open(f"{R}/data/bcp_gepa_splits/split_enriched_q36.json", "w"), indent=1)
acc = lambda qs: sum(1 for q in qs if recs[q].get("correct")) / len(qs)
print(f"pool {len(pool)}: failures {len(fails)} ({len(fails)/len(pool):.0%}), easy {len(easy)}")
print(f"new D_pareto: {n_fail} failures + {n_easy} guard = {len(pareto)}  (seed acc {acc(pareto):.3f})")
print(f"new D_feedback: {len(feedback)}  (seed acc {acc(feedback):.3f})")
