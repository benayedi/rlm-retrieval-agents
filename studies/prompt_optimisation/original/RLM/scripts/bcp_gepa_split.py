"""Stratified splits of BrowseComp-Plus for GEPA, drawn from the 830-question Qwen baseline.

WHY STRATIFIED. A simple random 100 can easily over- or under-draw the questions where the
agent already fails, and those are where a prompt change acts. We stratify on the seed's own
per-question outcome in the completed 830 run (suprlm/runs/_graded_cache.json, Qwen3.8-27B,
design B, Reason-ModernColBERT-8192, 79.4% accuracy):

    stratum = correct/wrong  x  gold-recall bucket (0 | partial | full)   +  errors

and allocate PROPORTIONALLY, so each split's mean estimates the full-830 mean without
reweighting. Inside each stratum questions are sorted by wall time and drawn SYSTEMATICALLY
(every k-th from a random start), which also balances difficulty/latency -- the strongest
other predictor of outcome (recall-0 questions average 3,177 s, full-recall ones 694 s).

SPLITS (disjoint):
    pareto    100  GEPA's selection set: every accepted candidate is scored on all of it
    feedback  200  GEPA's minibatch pool: only 3 are drawn per iteration, so its size is free
    test200   200  final confirmation of a winner (proportional, from what is left)
    test_rest  ~330 everything else, untouched

Deterministic: the same --seed gives the same split.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import statistics as st

GRADED = "/home/extstu_00/suprlm/runs/_graded_cache.json"
OUT = "/home/extstu_00/RLM/data/bcp_gepa_splits"


def stratum(r: dict) -> str:
    if r.get("status") != "completed":
        return "error"
    g = r.get("gold_recall") or 0.0
    rb = "r0" if g == 0 else ("rfull" if g >= 0.999 else "rpart")
    return ("ok_" if r.get("correct") else "wrong_") + rb


def allocate(sizes: dict[str, int], n: int) -> dict[str, int]:
    """Proportional allocation with largest-remainder rounding (sums to exactly n)."""
    tot = sum(sizes.values())
    raw = {k: n * v / tot for k, v in sizes.items()}
    out = {k: math.floor(x) for k, x in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[: n - sum(out.values())]:
        out[k] += 1
    return out


def systematic(qids: list[str], k: int, rng: random.Random, wall: dict[str, float]) -> list[str]:
    if k <= 0:
        return []
    q = sorted(qids, key=lambda x: wall[x])
    step = len(q) / k
    start = rng.random() * step
    return [q[min(len(q) - 1, int(start + i * step))] for i in range(k)]


def draw(pool: dict[str, list[str]], n: int, rng, wall) -> list[str]:
    alloc = allocate({k: len(v) for k, v in pool.items() if v}, n)
    picked = []
    for k, m in alloc.items():
        s = systematic(pool[k], m, rng, wall)
        picked += s
        pool[k] = [x for x in pool[k] if x not in set(s)]
    return sorted(picked, key=int)


def describe(name: str, qids: list[str], G: dict) -> str:
    rs = [G[q] for q in qids]
    acc = st.mean(1.0 if r["correct"] else 0.0 for r in rs)
    rec = st.mean(r["gold_recall"] or 0.0 for r in rs)
    wall = st.median(r["wall"] or 0.0 for r in rs)
    c = collections.Counter(stratum(r) for r in rs)
    return (f"{name:<9} n={len(qids):>3}  acc={acc:.3f}  gold_recall={rec:.3f}  "
            f"median_wall={wall:.0f}s  strata={dict(sorted(c.items()))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--pareto", type=int, default=100)
    ap.add_argument("--feedback", type=int, default=200)
    ap.add_argument("--test", type=int, default=200)
    a = ap.parse_args()

    G = json.load(open(GRADED))
    wall = {q: (r.get("wall") or 0.0) for q, r in G.items()}
    pool = collections.defaultdict(list)
    for q, r in G.items():
        pool[stratum(r)].append(q)
    rng = random.Random(a.seed)

    pareto = draw(pool, a.pareto, rng, wall)
    feedback = draw(pool, a.feedback, rng, wall)
    test200 = draw(pool, a.test, rng, wall)
    rest = sorted((q for v in pool.values() for q in v), key=int)
    assert not (set(pareto) & set(feedback) | set(pareto) & set(test200) | set(feedback) & set(test200))
    assert len(pareto) + len(feedback) + len(test200) + len(rest) == len(G)

    os.makedirs(OUT, exist_ok=True)
    split = {"seed": a.seed, "source": GRADED, "pareto": pareto, "feedback": feedback,
             "test200": test200, "test_rest": rest}
    json.dump(split, open(f"{OUT}/split.json", "w"), indent=1)
    lines = [describe("ALL 830", list(G), G), describe("pareto", pareto, G),
             describe("feedback", feedback, G), describe("test200", test200, G),
             describe("test_rest", rest, G)]
    open(f"{OUT}/split_summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"-> {OUT}/split.json")


if __name__ == "__main__":
    main()
