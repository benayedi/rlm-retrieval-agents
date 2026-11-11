"""GEPA run 2 over the official RLM harness on OBLIQ Twitter -- CORRECTED.

Run 1 (logs/gepa_official/, 2026-09-28) is VOID. It reported best_batch=0.2474 against a
0.1302 baseline, an apparent +90%. It was noise. Per-question NDCG@10 has sd .2058 with 62%
exact zeros, so a 6-question minibatch mean has se .0840 -- 65% of the mean being measured.
Simulating the UNCHANGED seed over 30 minibatches of 6 and taking the max gives an expected
best of .3200; run 1's .2474 sits at the 6th percentile of doing nothing (P=.94).

FIVE CORRECTIONS, each aimed at one defect of run 1:

 1. PAIRED, FIXED EVALUATION SET. minibatch == |dataset|, so EVERY candidate is scored on the
    SAME questions. Run 1 resampled 6 questions per batch, so candidates were compared across
    different question sets and nearly all the observed variance was question difficulty.

 2. DENSER SELECTION SIGNAL. score = NDCG@10 + LAMBDA*gold_recall. NDCG@10 alone is zero on
    62/100 questions; the composite is non-zero on 65/100 and cuts the coefficient of
    variation from 32% to 24%. NDCG@10 is still reported separately -- it remains the metric.

 3. INFRA FAILURES ARE IMPUTED, NEVER ZEROED. Run 1 returned (0.0, {"infra_failure": True});
    the flag was informational and GEPA averaged the zero in regardless. A timeout is not a
    prompt defect. Here a failed rollout takes the candidate's mean over its COMPLETED
    questions, which is neutral by construction.

 4. A RESULT CACHE. Keyed (sha1(strategy), qid) and persisted. The seed's 50 are pre-loaded
    from the existing dev100_fixed run, so the seed costs ZERO new rollouts.

 5. A NULL-MODEL GUARD, LOGGED EVERY BATCH. We print what an UNCHANGED prompt would score on
    this same fixed set, so "is this candidate beating noise?" is answerable while it runs,
    not months later.

WHAT IS OPTIMISED: one block, STRATEGY. The other five paragraphs of the official system
prompt are environment mechanics and stay frozen.

HONEST POWER STATEMENT: paired at n=50 the se of a delta is ~.029, so only effects above
roughly +.06 NDCG are detectable. This run can confirm a large win or rule one out. It cannot
certify a small one, and no affordable run on 100 dev questions can.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time as _t
import uuid

sys.path.insert(0, "/home/extstu_00/RLM")
from examples import obliq_data as OD  # noqa: E402

R = "/home/extstu_00/RLM"
PYV = "/home/extstu_00/miniconda3/envs/vllm/bin/python"
TASK, SUBSET = "descriptive", "twitter"
QRELS = OD.load_qrels(TASK, SUBSET)
ROWS = {r["_id"]: r for r in OD.load_queries(TASK, SUBSET)}
BASELINE = 0.1302
LAMBDA = 0.5

OUT = f"{R}/logs/gepa_official_v2"
PROG = f"{OUT}/progress.txt"
CACHE_F = f"{OUT}/cache.json"
_CACHE: dict[str, dict] = {}


def _log(msg: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    line = f"{_t.strftime('%F %T')}  {msg}"
    with open(PROG, "a") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def _sha(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:10]


def _cache_load() -> None:
    global _CACHE
    if os.path.exists(CACHE_F):
        _CACHE = json.load(open(CACHE_F))
        _log(f"[cache] loaded {len(_CACHE)} records")


def _cache_save() -> None:
    os.makedirs(OUT, exist_ok=True)
    tmp = CACHE_F + ".tmp"
    json.dump(_CACHE, open(tmp, "w"))
    os.replace(tmp, CACHE_F)


def composite(rec: dict) -> float:
    """Selection signal. NDCG@10 is the metric; gold_recall densifies it."""
    return (rec.get("ndcg@10") or 0.0) + LAMBDA * (rec.get("gold_recall") or 0.0)


def _run_batch(strategy: str, qids: list[str], workers: int) -> dict[str, dict]:
    """Run qids under this STRATEGY across `workers` subprocesses. Cache-aware."""
    key = _sha(strategy)
    todo = [q for q in qids if f"{key}|{q}" not in _CACHE]
    if todo:
        tag = uuid.uuid4().hex[:8]
        sf = os.path.join(tempfile.gettempdir(), f"gepa2_strategy_{tag}.txt")
        open(sf, "w").write(strategy)
        shard_dir = os.path.join(tempfile.gettempdir(), f"gepa2_shards_{tag}")
        os.makedirs(shard_dir, exist_ok=True)
        procs = []
        for w in range(workers):
            part = todo[w::workers]
            if not part:
                continue
            idf = os.path.join(shard_dir, f"{w}.json")
            json.dump(part, open(idf, "w"))
            env = dict(os.environ)
            env.update(
                OBLIQ_STRATEGY_FILE=sf, PYTHONPATH="/home/extstu_00/rlmrepo",
                HF_HOME="/home/extstu_00/.cache/huggingface", PYTHONUNBUFFERED="1",
                EMBEDDING_SERVER_URL="http://localhost:8010", OBLIQ_INDEX="obliq-des-twitter",
                OBLIQ_TASK=TASK, OBLIQ_SUBSET=SUBSET,
                LOCAL_LLM_BASE_URL="http://localhost:8030/v1",
                LOCAL_LLM_MODEL="qwen3.8-27b-bf16", LOCAL_LM_TIMEOUT="1800",
                AZURE_OPENAI_REASONING_EFFORT="medium", SUB_LM_REASONING_EFFORT="medium",
                ORCHESTRATOR="0", LIGHTON_TOOL_DOCS="1", LIGHTON_TASK_TEXT="1",
                SEARCH_SNIPPET_TOKENS="512", OFFICIAL_TOP_K="20",
                RLM_TRAJ_DIR="logs/rlm_traj_gepa2",
            )
            procs.append(subprocess.Popen(
                [PYV, "-u", f"{R}/scripts/run_obliq_mine.py", "--ids-file", idf,
                 "--max-iterations", "20", "--max-depth", "1",
                 "--out", os.path.join(shard_dir, f"out{w}")],
                env=env, cwd=R, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        for p in procs:
            p.wait()
        for w in range(workers):
            f = os.path.join(shard_dir, f"out{w}.json")
            if not os.path.exists(f):
                continue
            try:
                for rec in json.load(open(f)):
                    _CACHE[f"{key}|{rec.get('query_id')}"] = rec
            except Exception as e:
                _log(f"[warn] unreadable shard {f}: {e}")
        _cache_save()
    return {q: _CACHE.get(f"{key}|{q}") for q in qids}


def _feedback(rec: dict, gold: set[str]) -> str:
    """Compact -- 50 of these go into one reflection prompt."""
    s = OD.score_answer(rec.get("predicted") or "", gold, SUBSET)
    ranked = s["ranked"]
    late = [d for d in ranked[10:] if d in gold]
    parts = [f"NDCG@10={s['ndcg@10']:.3f} goldR={rec.get('gold_recall',0):.2f} "
             f"({len(gold)} relevant, {s['n_parsed']} submitted, only top 10 score)"]
    if late:
        parts.append(f"ORDERING LOSS: {len(late)} relevant doc(s) retrieved but below rank 10 "
                     f"e.g. {late[:3]}")
    elif s["ndcg@10"] == 0.0 and (rec.get("gold_recall") or 0) == 0:
        parts.append("SEARCH LOSS: nothing relevant ever surfaced; the request names a latent "
                     "pattern, and documents echoing its wording rarely instantiate it")
    return " | ".join(parts)


def make_batch_evaluator(workers: int, null_ref: list[float]):
    st = {"batches": 0, "rollouts": 0, "best": -1.0, "t0": _t.time(), "seen": {}}

    def batch_evaluator(pairs, **kw):
        groups: dict[str, list[int]] = {}
        for i, (cand, _ex) in enumerate(pairs):
            text = cand["STRATEGY"] if isinstance(cand, dict) else str(cand)
            groups.setdefault(text, []).append(i)
        out: list = [None] * len(pairs)
        for text, idxs in groups.items():
            qids = [pairs[i][1]["qid"] for i in idxs]
            n_cached = sum(1 for q in qids if f"{_sha(text)}|{q}" in _CACHE)
            recs = _run_batch(text, qids, workers)
            ok = {q: r for q, r in recs.items()
                  if r and not r.get("error") and (r.get("predicted") or "")}
            # CORRECTION 3: neutral imputation for infra failures
            impute = statistics.mean([composite(r) for r in ok.values()]) if ok else 0.0
            nd, comp = [], []
            for i in idxs:
                q = pairs[i][1]["qid"]
                rec = ok.get(q)
                if rec is None:
                    out[i] = (impute, {"feedback": "RUN DID NOT COMPLETE (timeout/infra). "
                                                   "Not attributable to the strategy text.",
                                       "infra_failure": True})
                else:
                    out[i] = (composite(rec),
                              {"feedback": _feedback(rec, set(QRELS.get(q, set())))})
                    nd.append(rec.get("ndcg@10") or 0.0)
                    comp.append(composite(rec))
            st["batches"] += 1
            st["rollouts"] += len(idxs) - n_cached
            mc = statistics.mean(comp) if comp else 0.0
            mn = statistics.mean(nd) if nd else 0.0
            st["best"] = max(st["best"], mc)
            st["seen"][_sha(text)] = (mn, mc)
            infra = len(idxs) - len(ok)
            # CORRECTION 5: what would an UNCHANGED prompt score on this same fixed set?
            beat = sum(1 for x in null_ref if mn > x) / max(1, len(null_ref))
            _log(f"batch {st['batches']:>3} | cand {_sha(text)} | n={len(idxs)} "
                 f"cached={n_cached} | NDCG@10={mn:.4f} composite={mc:.4f} "
                 f"best_comp={st['best']:.4f} | vs NULL: beats {beat*100:.0f}% of "
                 f"same-prompt draws | infra {infra} | rollouts {st['rollouts']} | "
                 f"{(_t.time()-st['t0'])/60:.0f} min | SEED NDCG@10 {BASELINE:.4f}")
        return out

    return batch_evaluator


def _reflection_lm(prompt: str) -> str:
    from openai import OpenAI
    c = OpenAI(base_url="http://localhost:8030/v1", api_key="EMPTY", timeout=1800)
    r = c.chat.completions.create(
        model="qwen3.8-27b-bf16", messages=[{"role": "user", "content": prompt}],
        temperature=1.0, max_completion_tokens=8192,
        extra_body={"reasoning_effort": "medium"})
    return r.choices[0].message.content or ""


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-size", type=int, default=50)
    ap.add_argument("--max-metric-calls", type=int, default=420)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    _cache_load()

    # ---- seed STRATEGY = the 0.1302 configuration -------------------------------
    src = open(f"{R}/scripts/run_obliq_mine.py").read()
    seed_text = src.split("OBLIQ_STRATEGY = (", 1)[1].split("\n)", 1)[0]
    seed_text = "".join(l.strip().strip('"').replace('\\"', '"') for l in seed_text.splitlines())
    seed_key = _sha(seed_text)

    # ---- CORRECTION 4: pre-load the seed's results from the completed dev100 run --
    import glob
    pre = 0
    for f in sorted(glob.glob(f"{R}/logs/obliq_mine/dev100_fixed_s*.json")):
        for rec in json.load(open(f)):
            _CACHE.setdefault(f"{seed_key}|{rec['query_id']}", rec)
            pre += 1
    _cache_save()
    _log(f"[seed] STRATEGY {len(seed_text)} chars sha={seed_key}; "
         f"pre-loaded {pre} cached seed rollouts (cost 0)")

    # ---- CORRECTION 1: ONE fixed train set, used for every candidate -------------
    dev = [q for q in json.load(open(f"{R}/data/obliq_splits/twitter_dev.json")) if q in ROWS]
    rng = random.Random(a.seed)
    train = sorted(rng.sample(dev, a.train_size))
    holdout = sorted(set(dev) - set(train))
    json.dump({"train": train, "holdout": holdout}, open(f"{OUT}/split.json", "w"), indent=1)

    seed_recs = {q: _CACHE.get(f"{seed_key}|{q}") for q in train}
    seed_nd = [(r.get("ndcg@10") or 0.0) for r in seed_recs.values() if r]
    seed_cp = [composite(r) for r in seed_recs.values() if r]
    _log(f"[split] train={len(train)} (FIXED, paired across candidates) "
         f"holdout={len(holdout)} | seed on train: NDCG@10={statistics.mean(seed_nd):.4f} "
         f"composite={statistics.mean(seed_cp):.4f}")

    # ---- CORRECTION 5: the null distribution for THIS fixed train set ------------
    allv = [(r.get("ndcg@10") or 0.0) for f in sorted(glob.glob(
        f"{R}/logs/obliq_mine/dev100_fixed_s*.json")) for r in json.load(open(f))]
    rng2 = random.Random(1234)
    null_ref = [statistics.mean(rng2.sample(allv, a.train_size)) for _ in range(4000)]
    null_ref.sort()
    _log(f"[null] an UNCHANGED prompt on {a.train_size} random questions: "
         f"mean={statistics.mean(null_ref):.4f} p50={null_ref[2000]:.4f} "
         f"p95={null_ref[3800]:.4f} p99={null_ref[3960]:.4f} "
         f"-- a candidate must clear p95 to be interesting")

    import gepa.optimize_anything as OA
    res = OA.optimize_anything(
        seed_candidate={"STRATEGY": seed_text},
        dataset=[{"qid": q} for q in train],
        valset=[{"qid": q} for q in train],   # same fixed set: paired by construction
        batch_evaluator=make_batch_evaluator(a.workers, null_ref),
        config=OA.GEPAConfig(
            engine=OA.EngineConfig(
                max_metric_calls=a.max_metric_calls,
                candidate_selection_strategy="pareto",
                run_dir=OUT, raise_on_exception=False),
            tracking=OA.TrackingConfig(logger=OA.StdOutLogger()),
            reflection=OA.ReflectionConfig(
                reflection_lm=_reflection_lm,
                # CORRECTION 1: minibatch == |train| => every candidate sees identical questions
                reflection_minibatch_size=a.train_size),
        ),
    )
    best = getattr(res, "best_candidate", None) or {"STRATEGY": seed_text}
    json.dump(best, open(f"{OUT}/best_candidate.json", "w"), indent=1)
    _log(f"[done] best score {getattr(res,'best_score',None)} (seed NDCG@10 {BASELINE})")
    _log("[next] scripts/gepa_v2_validate.py runs seed vs best on the HELD-OUT questions")


if __name__ == "__main__":
    main()
