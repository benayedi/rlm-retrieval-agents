"""GEPA over the official RLM harness on OBLIQ Twitter -- WIDER MUTABLE SURFACE.

Refactored copy of original/RLM/scripts/gepa_wide.py (studies/prompt_optimisation/CHANGES_FROM_ORIGINAL.md).
Run: obliq_twitter/experiments/run4_gepa_wide.sh (GEPA-4 run 4, Qwen3.8-27B, 2026-10-04 to 10-05, logs/gepa_wide;
  started by scripts/overnight3.sh, restarted by night5.sh/night6.sh, finished by night8_gepa27_finish.sh).
Changed: as gepa_official_obliq.py; the seed TASK_SUFFIX file is read from ORIGINAL_REPO_ROOT (PO-I7).
Names: 'supervisor harness' / 'full-doc RLM' in the text below is the document REPL agent (agents/document_repl);
  'official RLM harness' / 'our harness' / run_obliq_mine.py is the snippet REPL agent (agents/snippet_repl,
  src/run_obliq.py).

WHY THIS RUN EXISTS. Every previous run let GEPA edit ONE block, STRATEGY: 970 of the 4,755
characters the agent actually reads, i.e. 20%. The paper does not do that -- Algorithm 1 line
8 is `j <- SELECTMODULE(\u03a6_k)` with a ROUND-ROBIN policy over every module in the program,
each rewritable in full.

The cost of our restriction was concrete. TASK_SUFFIX -- the answer contract, frozen -- ended
with "Submit more than 10 if you have them -- only the top 10 are scored, so extra candidates
cost nothing." That line is mechanically true and behaviourally harmful: the agent submits
22.6 doc_ids and 22% of the gold it found sits below rank 10, scoring nothing. We had
diagnosed that defect days earlier and then locked it OUTSIDE the search space, so no
candidate could ever repair it.

Here STRATEGY and TASK_SUFFIX are both mutable (2,071 chars, 44%). The four remaining blocks
stay frozen because they describe REPL mechanics, not task strategy -- a rewrite there could
invent tools that do not exist.


WHAT CHANGED FROM run 2, after actually reading the paper (ICLR 2026, Algorithm 1).

Run 2 set reflection_minibatch_size = |dataset| = 50 and valset = those same 50. That was
wrong, and it broke the algorithm's two-stage structure. The paper does:

    split D_train into D_feedback and D_pareto
    score the seed on ALL of D_pareto
    each iteration:  M <- minibatch of size b from D_feedback     (b = 3 in the paper)
                     mutate, accept ONLY if the minibatch score improves   <- CHEAP GATE
                     then score the accepted candidate on ALL of D_pareto  <- SELECTION
    "The majority of GEPA's rollout budget is spent on validation."

So the minibatch is an accept/reject gate, NOT the selection signal; selection runs on the
full D_pareto score matrix, per question (that is what makes pareto sampling per-task work).
Run 1's real defect was D_pareto = 12, against the paper's 300 -- 5x noisier -- plus a
200-rollout budget against the paper's 1,839-7,051. NOT the minibatch size, and NOT Qwen
critiquing itself (paper SE.2: "all modules relying on the same model" -- that IS their setup).

HONEST LIMIT: we use D_pareto = 50, not 300. Twitter has only 281 questions (100 dev / 181
held-out) and a rollout costs ~1335 s, so 300 is unaffordable. se of a 50-question mean is
~.029, so this run can confirm or rule out a LARGE effect and cannot certify a small one.
The null-model guard below prints, every batch, what an UNCHANGED prompt scores on the same
sample -- so "is this candidate beating noise?" stays answerable while it runs.


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

# Import wiring and paths (refactor, rules 2-3). Original: hard-coded locations of the original repository (R), the
# document REPL agent working directory (SUP) and the interpreter. Code now comes from this repository (computed from
# __file__), data and run outputs from configs/paths.env (rlm_shared.paths). The original logs/ directory is RUNS_DIR.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, os.pardir, os.pardir))
DOC_SRC = os.path.join(_REPO, "agents", "document_repl", "src")    # split_runner.py (was scripts/gepa_bcp.py), run_bm25.py
SPLITS = os.path.join(_REPO, "studies", "prompt_optimisation", "splits")   # original: <original repo>/data
sys.path.insert(0, os.path.join(_REPO, "agents", "shared"))
from rlm_shared.paths import path  # noqa: E402
RUNS = str(path("RUNS_DIR"))                                         # original: f"{R}/logs"
SNIPPET_SRC = os.path.join(_REPO, "agents", "snippet_repl", "src")
RUN_OBLIQ = os.path.join(SNIPPET_SRC, "run_obliq.py")               # original: f"{R}/scripts/run_obliq_mine.py"
from rlm_shared import obliq_data as OD  # noqa: E402  (original: from examples import obliq_data as OD)

_PORTS = ["http://localhost:8030/v1", "http://localhost:8031/v1"]
PYV = os.environ.get("PYTHON", "")  # original: the vllm conda env's python; checked before rollouts start
TASK, SUBSET = "descriptive", "twitter"
QRELS = OD.load_qrels(TASK, SUBSET)
ROWS = {r["_id"]: r for r in OD.load_queries(TASK, SUBSET)}
BASELINE = float(os.environ.get("GEPA_BASELINE", "0.1302"))  # display reference; per-model
LAMBDA = 0.5

OUT = os.environ.get("GEPA_OUT", f"{RUNS}/gepa_wide")
PROG = f"{OUT}/progress.txt"
CACHE_F = f"{OUT}/cache.json"
_CACHE: dict[str, dict] = {}
_SEED_COMPOSITE = [0.0]   # set once the seed is scored; used as the neutral fallback


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


def _run_batch(cand: dict, qids: list[str], workers: int) -> dict[str, dict]:
    """Run qids under this CANDIDATE (STRATEGY + TASK_SUFFIX). Cache-aware."""
    strategy = cand["STRATEGY"]; task = cand["TASK_SUFFIX"]
    key = _sha(strategy + task)
    todo = [q for q in qids if f"{key}|{q}" not in _CACHE]
    if todo:
        tag = uuid.uuid4().hex[:8]
        sf = os.path.join(tempfile.gettempdir(), f"gw_strategy_{tag}.txt")
        open(sf, "w").write(strategy)
        tf = os.path.join(tempfile.gettempdir(), f"gw_task_{tag}.txt")
        open(tf, "w").write(task)
        shard_dir = os.path.join(OUT, "shards", tag)
        os.makedirs(shard_dir, exist_ok=True)
        if not PYV:
            raise SystemExit("environment variable PYTHON is not set: interpreter of the rollouts")
        procs = []
        for w in range(workers):
            part = todo[w::workers]
            if not part:
                continue
            idf = os.path.join(shard_dir, f"{w}.json")
            json.dump(part, open(idf, "w"))
            env = dict(os.environ)
            env.update(
                OBLIQ_STRATEGY_FILE=sf, OBLIQ_TASK_FILE=tf, PYTHONPATH=str(path("RLMLIB_PATH")),
                HF_HOME=os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), PYTHONUNBUFFERED="1",
                EMBEDDING_SERVER_URL="http://localhost:8010", OBLIQ_INDEX="obliq-des-twitter",
                OBLIQ_TASK=TASK, OBLIQ_SUBSET=SUBSET,
                # Round-robin over BOTH vLLM instances (GPU0 :8030, GPU1 :8031).
                LOCAL_LLM_BASE_URL=_PORTS[w % len(_PORTS)],
                LOCAL_LLM_MODEL=os.environ.get("LOCAL_LLM_MODEL", "qwen3.8-27b-bf16"), LOCAL_LM_TIMEOUT="1800",
                AZURE_OPENAI_REASONING_EFFORT="medium", SUB_LM_REASONING_EFFORT="medium",
                ORCHESTRATOR="0", LIGHTON_TOOL_DOCS="1", LIGHTON_TASK_TEXT="1",
                SEARCH_SNIPPET_TOKENS="512", OFFICIAL_TOP_K="20",
                # per-run trajectory dir: two GEPA runs (different models) sharing one
                # dir would make any trace check report a blend of both.
                RLM_TRAJ_DIR=os.environ.get("GEPA_TRAJ_DIR", f"{RUNS}/rlm_traj_gepa2"),   # original default "logs/rlm_traj_gepa2"
            )
            procs.append(subprocess.Popen(
                [PYV, "-u", RUN_OBLIQ, "--ids-file", idf,
                 "--max-iterations", "20", "--max-depth", "1",
                 "--out", os.path.join(shard_dir, f"out{w}")],
                env=env, cwd=RUNS, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        # WALL-CLOCK CAP, SCALED TO SHARD SIZE.
        # First version used a flat 5400 s per batch and was ~7x too tight: with
        # D_pareto=100 over 16 workers a shard holds 6-7 questions needing 8,000-9,300 s.
        # It killed healthy work 48 times and drove three candidates to a fake 0.0000.
        # The budget is now PER QUESTION, multiplied by the shard's own length.
        _per_q = float(os.environ.get("GEPA_PER_QUESTION_CAP", "3600"))
        _t0 = _t.time()
        for w, p in enumerate(procs):
            _n_in_shard = max(1, len(todo[w::workers]))
            _deadline = _t0 + _per_q * _n_in_shard
            try:
                p.wait(timeout=max(120.0, _deadline - _t.time()))
            except Exception:
                p.kill()
                _log(f"[runaway] shard {w} ({_n_in_shard} q) killed at "
                     f"{_per_q*_n_in_shard:.0f}s cap")
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
            key = cand["STRATEGY"] + "\x00" + cand["TASK_SUFFIX"]
            groups.setdefault(key, []).append(i)
        out: list = [None] * len(pairs)
        for key, idxs in groups.items():
            strat, task = key.split("\x00", 1)
            cand_d = {"STRATEGY": strat, "TASK_SUFFIX": task}
            qids = [pairs[i][1]["qid"] for i in idxs]
            sha = _sha(strat + task)
            n_cached = sum(1 for q in qids if f"{sha}|{q}" in _CACHE)
            recs = _run_batch(cand_d, qids, workers)
            ok = {q: r for q, r in recs.items()
                  if r and not r.get("error") and (r.get("predicted") or "")}
            # CORRECTION 3: neutral imputation for infra failures
            # If NOTHING survived, every rollout was infrastructure failure, not a bad
            # prompt. Scoring 0.0 teaches GEPA that a slow prompt is a broken one -- the
            # exact "errors scored zero" poison this file was written to avoid. Fall back
            # to the SEED's score so the candidate is treated as uninformative, and say so.
            if ok:
                impute = statistics.mean([composite(r) for r in ok.values()])
            else:
                impute = _SEED_COMPOSITE[0]
                _log(f"[infra] cand {sha}: 0 of {len(idxs)} rollouts survived -- "
                     f"scoring it as NEUTRAL (seed composite {impute:.4f}), NOT zero")
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
            st["seen"][sha] = (mn, mc)
            infra = len(idxs) - len(ok)
            # CORRECTION 5: what would an UNCHANGED prompt score on this same fixed set?
            beat = sum(1 for x in null_ref if mn > x) / max(1, len(null_ref))
            _log(f"batch {st['batches']:>3} | cand {sha} | n={len(idxs)} "
                 f"cached={n_cached} | NDCG@10={mn:.4f} composite={mc:.4f} "
                 f"best_comp={st['best']:.4f} | vs NULL: beats {beat*100:.0f}% of "
                 f"same-prompt draws | infra {infra} | rollouts {st['rollouts']} | "
                 f"{(_t.time()-st['t0'])/60:.0f} min | SEED NDCG@10 {BASELINE:.4f}")
        return out

    return batch_evaluator


def _sanitise_reflection(raw: str) -> str:
    """Return ONE fenced block so GEPA's extractor cannot splice in the critic's monologue.

    THE BUG THIS FIXES. gepa/strategies/instruction_proposal.py:128 extracts the candidate as
    everything between the FIRST ``` and the LAST ```:
        start = lm_out.find("```") + 3 ;  end = lm_out.rfind("```")
    When the reflection LM writes a prompt, closes the fence, then reconsiders out loud and
    writes a second version, that span swallows both blocks AND the commentary between them.

    MEASURED on run 3: candidate 1 was 7,002 chars of which 4,918 (70%) were leaked monologue
    -- literally 'Wait, let me reconsider. The key issues: - Example 1: NDCG=0, so NOTHING in
    top 10 was relevant' sitting inside the agent's system prompt. Candidate 2: 4,971 of 7,556
    (66%). That bloats every turn the agent reads and, worse, bakes references to specific
    training questions into the prompt -- a direct overfitting path.

    We keep the LAST complete block: when the model reconsiders, its revised version is the
    one it meant. Falls back to the raw text if nothing is fenced, so a format change upstream
    degrades to current behaviour rather than returning an empty prompt.
    """
    import re as _re
    blocks = _re.findall(r"```[A-Za-z0-9_-]*\n(.*?)```", raw, _re.DOTALL)
    blocks = [b.strip() for b in blocks if b.strip()]
    if not blocks:
        return raw
    chosen = blocks[-1]
    if len(blocks) > 1:
        _log(f"[sanitise] reflection returned {len(blocks)} fenced blocks "
             f"({len(raw)} chars raw); kept the last ({len(chosen)} chars), "
             f"discarded {len(raw) - len(chosen)} chars of monologue")
    return "```\n" + chosen + "\n```"


def _reflection_lm(prompt: str) -> str:
    from openai import OpenAI
    c = OpenAI(base_url=_PORTS[0].replace("/v1", "/v1"), api_key="EMPTY", timeout=1800)
    r = c.chat.completions.create(
        model="qwen3.8-27b-bf16", messages=[{"role": "user", "content": prompt}],
        temperature=1.0, max_completion_tokens=8192,
        extra_body={"reasoning_effort": "medium"})
    return _sanitise_reflection(r.choices[0].message.content or "")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--valset", type=int, default=50)      # D_pareto -- the SELECTION set
    ap.add_argument("--minibatch", type=int, default=3)    # the cheap accept/reject GATE
    ap.add_argument("--max-metric-calls", type=int, default=400)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    _cache_load()

    # ---- seed STRATEGY = the 0.1302 configuration -------------------------------
    src = open(RUN_OBLIQ).read()
    seed_text = src.split("OBLIQ_STRATEGY = (", 1)[1].split("\n)", 1)[0]
    seed_text = "".join(l.strip().strip('"').replace('\\"', '"') for l in seed_text.splitlines())
    task_text = open(f"{path('ORIGINAL_REPO_ROOT')}/experiments/rank_ab/A_control.txt").read()   # seed TASK_SUFFIX; not in this repository (PO-I7)
    seed_key = _sha(seed_text + task_text)

    # ---- CORRECTION 4: pre-load the seed's results from the completed dev100 run --
    # SAME MODEL-SAFETY RULE as the D_pareto preload below: dev100_fixed_* is Qwen3.8-27B
    # data, and the cache key carries no model, so on a model swap this must point at the
    # new model's own dev100 seed run (GEPA_PRELOAD_DEV_GLOB) or be switched off. Leaving it
    # hardcoded would have fed 27B rollouts into a 35B run under the seed key -- invisible,
    # and fatal to every number downstream.
    import glob
    pre = 0
    _dg = os.environ.get("GEPA_PRELOAD_DEV_GLOB", f"{RUNS}/obliq_mine/dev100_fixed_s*.json")
    _dev_glob = sorted(glob.glob(_dg)) if _dg.strip() else []
    _log(f"[seed] dev preload glob {_dg!r} -> {len(_dev_glob)} file(s)")
    for f in _dev_glob:
        for rec in json.load(open(f)):
            _CACHE.setdefault(f"{seed_key}|{rec['query_id']}", rec)
            pre += 1
    _cache_save()
    # The seed was already run on all 133 held-out questions during the test133 arm, with
    # the IDENTICAL prompt (sha matches). Those shard JSONs are complete records -- including
    # `predicted`, which _feedback() needs -- so D_pareto costs the seed ZERO new rollouts.
    # Do NOT preload from the .log files: they carry only ndcg/goldR, and a record with an
    # empty `predicted` would hand the reflection LM a fabricated "0 docs submitted" failure.
    # These default records are Qwen3.8-27B results. The cache key is (prompt_sha, qid) and
    # carries NO model, so preloading them under a different model would silently mix two
    # models in one cache and every later comparison would be meaningless. On a model swap,
    # point GEPA_PRELOAD_GLOB at that model's OWN seed run instead (same prompt, same model),
    # or set it empty to pay for the seed from scratch.
    _pg = os.environ.get("GEPA_PRELOAD_GLOB", f"{RUNS}/test133_seed_s*.json")
    _pre_glob = sorted(glob.glob(_pg)) if _pg.strip() else []
    _log(f"[seed] preload glob {_pg!r} -> {len(_pre_glob)} file(s)")
    for f in _pre_glob:
        try:
            for rec in json.load(open(f)):
                if rec.get("predicted"):
                    _CACHE.setdefault(f"{seed_key}|{rec['query_id']}", rec); pre += 1
        except Exception as e:
            _log(f"[warn] could not preload {f}: {e}")
    _cache_save()
    _log(f"[seed] STRATEGY {len(seed_text)} chars sha={seed_key}; "
         f"pre-loaded {pre} cached seed rollouts (cost 0)")

    # ---- PAPER STRUCTURE: D_feedback (minibatch gate) vs D_pareto (selection) ----
    # D_feedback = the 100 dev questions -- minibatches of size b are drawn from here and
    # only gate accept/reject. D_pareto = a SEPARATE held-out sample; every accepted
    # candidate is scored on ALL of it, and that per-question matrix is what pareto
    # selection samples from. Run 1 had |D_pareto| = 12; the paper uses 300.
    dev = [q for q in json.load(open(f"{SPLITS}/obliq_splits/twitter_dev.json")) if q in ROWS]
    held = [q for q in json.load(open(f"{SPLITS}/obliq_splits/twitter_heldout.json")) if q in ROWS]
    rng = random.Random(a.seed)
    # REUSE the existing pareto set when one exists, sliced to the requested size, so the
    # seed's cached rollouts stay valid. Resampling would change the question set and throw
    # away 50 paid-for rollouts. 48 is chosen deliberately: with 16 workers a shard is
    # ceil(n/16) questions, so n=50 makes the slowest worker run FOUR waves while n=48 runs
    # exactly THREE -- a 25% cut in pareto wall time for se .0291 -> .0297.
    # PAPER-SHAPED SPLIT, sized by what the benchmark actually has.
    # Twitter is 281 questions total, so D_pareto can never reach the paper's 300. We take
    # 100 -- the largest selection set that still leaves a usable test set -- from the 133
    # questions that run 3 never touched. The remaining 33, PLUS run 3's old 48-question
    # pareto set (which this run never reads), form the 81-question TEST set.
    _t133 = sorted(set(held) - set(json.load(open(f"{RUNS}/gepa_official_v2/split.json"))["pareto"]))
    pareto = sorted(_t133)[: a.valset]
    _test = sorted(set(held) - set(pareto))
    json.dump({"test": _test}, open(f"{OUT}/test_split.json", "w"), indent=1)
    json.dump({"feedback": dev, "pareto": pareto},
              open(f"{OUT}/split.json", "w"), indent=1)

    _SEED_COMPOSITE[0] = statistics.mean(
        [composite(r) for r in (_CACHE.get(f"{seed_key}|{q}") for q in pareto) if r])
    seed_recs = {q: _CACHE.get(f"{seed_key}|{q}") for q in dev}
    seed_nd = [(r.get("ndcg@10") or 0.0) for r in seed_recs.values() if r]
    _log(f"[split] D_feedback={len(dev)} (minibatch b={a.minibatch} drawn from here) | "
         f"D_pareto={len(pareto)} held-out (SELECTION set, scored in full per candidate) | "
         f"seed on dev100: NDCG@10={statistics.mean(seed_nd):.4f}")
    _log(f"[budget] {a.max_metric_calls} metric calls. Each accepted candidate costs "
         f"{len(pareto)} pareto rollouts + {a.minibatch} gate rollouts, so expect roughly "
         f"{a.max_metric_calls//(len(pareto)+a.minibatch)} candidates.")

    # ---- null-model guard, sized to D_pareto -------------------------------------
    allv = [(r.get("ndcg@10") or 0.0) for f in sorted(glob.glob(
        f"{RUNS}/obliq_mine/dev100_fixed_s*.json")) for r in json.load(open(f))]
    rng2 = random.Random(1234)
    # BOOTSTRAP (with replacement). The previous version used rng.sample(), which draws
    # WITHOUT replacement -- and when |pool| == |pareto| that returns the same set every
    # time, giving mean==p50==p95 and a null bar that can never be cleared or failed.
    null_ref = [statistics.mean([rng2.choice(allv) for _ in range(len(pareto))])
                for _ in range(4000)]
    null_ref.sort()
    _log(f"[null] an UNCHANGED prompt scored on {len(pareto)} random questions: "
         f"mean={statistics.mean(null_ref):.4f} p50={null_ref[2000]:.4f} "
         f"p95={null_ref[3800]:.4f} p99={null_ref[3960]:.4f} "
         f"-- PRE-REGISTERED BAR: a candidate must clear p95 to be more than noise")

    import gepa.optimize_anything as OA
    res = OA.optimize_anything(
        seed_candidate={"STRATEGY": seed_text, "TASK_SUFFIX": task_text},
        dataset=[{"qid": q} for q in dev],         # D_feedback -> minibatch gate
        valset=[{"qid": q} for q in pareto],       # D_pareto  -> selection matrix
        batch_evaluator=make_batch_evaluator(a.workers, null_ref),
        config=OA.GEPAConfig(
            engine=OA.EngineConfig(
                max_metric_calls=a.max_metric_calls,
                candidate_selection_strategy="pareto",
                run_dir=OUT, raise_on_exception=False),
            tracking=OA.TrackingConfig(logger=OA.StdOutLogger()),
            reflection=OA.ReflectionConfig(
                reflection_lm=_reflection_lm,
                    # PAPER: b=3. This is the accept/reject GATE, not the selection signal.
                reflection_minibatch_size=a.minibatch),
        ),
    )
    best = getattr(res, "best_candidate", None) or {"STRATEGY": seed_text}
    json.dump(best, open(f"{OUT}/best_candidate.json", "w"), indent=1)
    _log(f"[done] best score {getattr(res,'best_score',None)} (seed NDCG@10 {BASELINE})")
    _log("[next] scripts/gepa_v2_validate.py runs seed vs best on the HELD-OUT questions")


if __name__ == "__main__":
    main()
