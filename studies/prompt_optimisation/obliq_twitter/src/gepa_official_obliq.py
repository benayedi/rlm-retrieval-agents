"""GEPA over YOUR official RLM harness on OBLIQ Twitter, via standalone gepa.optimize.

Refactored copy of original/RLM/scripts/gepa_official_obliq.py (studies/prompt_optimisation/CHANGES_FROM_ORIGINAL.md).
Run: obliq_twitter/experiments/run1_gepa_official.sh (GEPA-4 run 1, 2026-09-28, logs/gepa_official).
Changed: import wiring (rlm_shared.obliq_data, the snippet REPL agent's src/run_obliq.py and src/prompt_modules.py),
  the interpreter (PYTHON), RLMLIB_PATH, HF_HOME, outputs under RUNS_DIR (the rollouts' working directory is RUNS_DIR,
  original: the original repository root), splits from studies/prompt_optimisation/splits/obliq_splits/. The seed
  STRATEGY is still parsed from the agent's source text; tests/ checks that it is the 970-character fef4457968.
Names: 'supervisor harness' / 'full-doc RLM' in the text below is the document REPL agent (agents/document_repl);
  'official RLM harness' / 'our harness' / run_obliq_mine.py is the snippet REPL agent (agents/snippet_repl,
  src/run_obliq.py).

WHY standalone AND NOT dspy.GEPA. dspy.GEPA needs the program to BE a dspy.Module. This
harness is the rlm library driven by a script, so the native optimiser cannot see it. The
standalone API takes an arbitrary evaluator instead, which is what this file provides.

WHAT IS OPTIMISED: exactly one block. scripts/official_prompt_modules.py materialises the
prompt AFTER the eight monkeypatches have run and splits it into six named paragraphs, of
which only STRATEGY carries task knowledge; the other five are environment mechanics and stay
frozen. That is the whole point of the split -- GEPA_ASSESSMENT s4 warns that flattening to a
single blob "costs the module structure, so re-split it into named sections or every
reflection rewrites a 10k-character wall with no attribution".

BASELINE TO BEAT: 0.1302 NDCG@10 over the 100 dev questions, measured with the OBLIQ
STRATEGY already in place. The user's bar is .126, which this arm already clears; the target
here is the paper's .141.

PARALLELISM IS NOT OPTIONAL. A rollout costs ~930 s. The dspy.GEPA attempt ran 3 h
sequentially and never reached iteration 1 because the baseline valset evaluation alone
outlasted it. Every evaluation here fans out over WORKERS subprocesses.

ERRORS ARE DROPPED, NOT SCORED ZERO -- GEPA_ASSESSMENT s4's "the one I would worry about".
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
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

PYV = os.environ.get("PYTHON", "")  # original: the vllm conda env's python; checked before rollouts start
TASK, SUBSET = "descriptive", "twitter"
QRELS = OD.load_qrels(TASK, SUBSET)
ROWS = {r["_id"]: r for r in OD.load_queries(TASK, SUBSET)}
BASELINE = 0.1302


def _run_batch(strategy: str, qids: list[str], workers: int = 6) -> dict[str, dict]:
    """Run qids with this STRATEGY, fanned out over subprocesses. Returns qid -> record."""
    tag = uuid.uuid4().hex[:8]
    sf = os.path.join(tempfile.gettempdir(), f"gepa_strategy_{tag}.txt")
    open(sf, "w").write(strategy)
    shard_dir = os.path.join(tempfile.gettempdir(), f"gepa_shards_{tag}")
    os.makedirs(shard_dir, exist_ok=True)
    if not PYV:
        raise SystemExit("environment variable PYTHON is not set: interpreter of the rollouts")
    procs = []
    for w in range(workers):
        part = qids[w::workers]
        if not part:
            continue
        idf = os.path.join(shard_dir, f"{w}.json")
        json.dump(part, open(idf, "w"))
        env = dict(os.environ)
        env.update(
            OBLIQ_STRATEGY_FILE=sf, PYTHONPATH=str(path("RLMLIB_PATH")),
            HF_HOME=os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), PYTHONUNBUFFERED="1",
            EMBEDDING_SERVER_URL="http://localhost:8010", OBLIQ_INDEX="obliq-des-twitter",
            OBLIQ_TASK=TASK, OBLIQ_SUBSET=SUBSET,
            LOCAL_LLM_BASE_URL="http://localhost:8030/v1",
            LOCAL_LLM_MODEL="qwen3.8-27b-bf16", LOCAL_LM_TIMEOUT="1800",
            AZURE_OPENAI_REASONING_EFFORT="medium", SUB_LM_REASONING_EFFORT="medium",
            ORCHESTRATOR="0", LIGHTON_TOOL_DOCS="1", LIGHTON_TASK_TEXT="1",
            SEARCH_SNIPPET_TOKENS="512", OFFICIAL_TOP_K="20",
            RLM_TRAJ_DIR=f"{RUNS}/rlm_traj_gepa",   # original: "logs/rlm_traj_gepa" (cwd = original repo)
        )
        out = os.path.join(shard_dir, f"out{w}")
        procs.append(subprocess.Popen(
            [PYV, "-u", RUN_OBLIQ, "--ids-file", idf,
             "--max-iterations", "12", "--max-depth", "1", "--out", out],
            env=env, cwd=RUNS, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    for p in procs:
        p.wait()
    recs = {}
    for w in range(workers):
        f = os.path.join(shard_dir, f"out{w}.json")
        if not os.path.exists(f):
            continue
        try:
            for rec in json.load(open(f)):
                recs[str(rec.get("query_id"))] = rec
        except Exception:
            pass
    return recs


def _feedback(rec: dict, gold: set[str]) -> str:
    ans = rec.get("predicted") or ""
    s = OD.score_answer(ans, gold, SUBSET)
    ranked = s["ranked"]
    late = [d for d in ranked[10:] if d in gold]
    missed = sorted(gold - set(ranked))
    fb = [f"NDCG@10={s['ndcg@10']:.4f} over {len(gold)} relevant docs; "
          f"you submitted {s['n_parsed']} ids, only the first 10 score.",
          f"{rec.get('search_calls',0)} searches, {rec.get('get_document_calls',0)} documents opened."]
    if late:
        fb.append(f"IN HAND BUT RANKED TOO LOW: {late[:5]} sit below rank 10. "
                  f"The loss here is ORDERING, not search.")
    if missed:
        fb.append(f"NEVER SURFACED: {missed[:5]}.")
    if s["ndcg@10"] == 0.0:
        fb.append("Nothing relevant in the top 10. The request names a LATENT PATTERN: "
                  "documents echoing its wording usually do not instantiate it.")
    return " ".join(fb)


def make_batch_evaluator(workers: int):
    """GEPA's batch hook: it hands us ALL (candidate, example) pairs in ONE call, which is
    exactly what this task needs -- a rollout costs ~930 s, so anything evaluated one pair at
    a time never finishes. The dspy.GEPA attempt ran 3 h sequentially and never reached
    iteration 1. Here every call fans out over `workers` subprocesses.

    Returns one (score, side_info) per pair. side_info is the ASI GEPA reflects on.
    """
    import hashlib, time as _t
    _state = {"batches": 0, "rollouts": 0, "best": 0.0, "t0": _t.time()}
    PROG = f"{RUNS}/gepa_official/progress.txt"

    def _log(msg):
        os.makedirs(os.path.dirname(PROG), exist_ok=True)
        line = f"{_t.strftime('%F %T')}  {msg}"
        with open(PROG, "a") as fh:
            fh.write(line + "\n")
        print(line, flush=True)

    def batch_evaluator(pairs, **kw):
        # group by candidate so each distinct STRATEGY is one parallel sweep
        groups: dict[str, list[int]] = {}
        for i, (cand, _ex) in enumerate(pairs):
            text = cand["STRATEGY"] if isinstance(cand, dict) else str(cand)
            groups.setdefault(text, []).append(i)
        out: list = [None] * len(pairs)
        for text, idxs in groups.items():
            qids = [pairs[i][1]["qid"] for i in idxs]
            recs = _run_batch(text, qids, workers)
            for i in idxs:
                q = pairs[i][1]["qid"]
                rec = recs.get(q)
                gold = set(QRELS.get(q, set()))
                if rec is None or rec.get("error") or not (rec.get("predicted") or ""):
                    # infrastructure, NOT the strategy -- say so rather than teach a lesson
                    out[i] = (0.0, {"feedback": "RUN DID NOT COMPLETE (timeout/infra). "
                                                "NOT attributable to the strategy text.",
                                    "infra_failure": True})
                else:
                    out[i] = (rec.get("ndcg@10", 0.0), {"feedback": _feedback(rec, gold)})
            # PROGRESS. GEPA's own run_log.txt only gets written when an ITERATION completes,
            # which on this task is hours. Without this the run is opaque for most of its life
            # -- and an opaque run is exactly how the dspy attempt burned 3 h at iteration 0.
            vals = [out[i][0] for i in idxs if out[i] is not None]
            mean = sum(vals) / max(1, len(vals))
            infra = sum(1 for i in idxs if out[i] and out[i][1].get("infra_failure"))
            _state["batches"] += 1
            _state["rollouts"] += len(idxs)
            _state["best"] = max(_state["best"], mean)
            _log(f"batch {_state['batches']:>3} | cand {hashlib.sha1(text.encode()).hexdigest()[:8]} "
                 f"| n={len(idxs):<3} mean={mean:.4f} best_batch={_state['best']:.4f} "
                 f"| rollouts {_state['rollouts']}/{kw.get('budget','?')} "
                 f"| infra_fail {infra} | {(_t.time()-_state['t0'])/60:.0f} min elapsed "
                 f"| BASELINE {BASELINE:.4f}")
        return out
    return batch_evaluator


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default=f"{SPLITS}/obliq_splits/twitter_dev.json")
    ap.add_argument("--val", default=f"{SPLITS}/obliq_splits/twitter_heldout.json")
    ap.add_argument("--max-metric-calls", type=int, default=200)
    ap.add_argument("--minibatch", type=int, default=6)
    ap.add_argument("--val-size", type=int, default=12)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=f"{RUNS}/gepa_official")
    a = ap.parse_args()

    import importlib.util as ilu
    spec = ilu.spec_from_file_location("_opm", os.path.join(SNIPPET_SRC, "prompt_modules.py"))  # original: scripts/official_prompt_modules.py
    OPM = ilu.module_from_spec(spec); spec.loader.exec_module(OPM)

    # seed = the OBLIQ strategy currently in run_obliq_mine.py (the 0.1302 configuration)
    src = open(RUN_OBLIQ).read()
    seed_text = src.split('OBLIQ_STRATEGY = (', 1)[1].split('\n)', 1)[0]
    seed_text = "".join(l.strip().strip('"').replace('\\"', '"') for l in seed_text.splitlines())
    seed = {"STRATEGY": seed_text}
    print(f"[gepa] seed STRATEGY {len(seed_text)} chars; baseline {BASELINE:.4f}", flush=True)

    dev = [{"qid": q} for q in json.load(open(a.dev)) if q in ROWS]
    val = [{"qid": q} for q in json.load(open(a.val)) if q in ROWS][: a.val_size]
    print(f"[gepa] train={len(dev)} val={len(val)} budget={a.max_metric_calls} "
          f"workers={a.workers}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    import gepa.optimize_anything as OA
    res = OA.optimize_anything(
        seed_candidate=seed,
        dataset=dev,
        valset=val,
        batch_evaluator=make_batch_evaluator(a.workers),
        config=OA.GEPAConfig(
            engine=OA.EngineConfig(
                max_metric_calls=a.max_metric_calls,
                candidate_selection_strategy="pareto",
                run_dir=a.out,
                raise_on_exception=False,   # an infra failure must not kill the search
            ),
            tracking=OA.TrackingConfig(logger=OA.StdOutLogger()),
            reflection=OA.ReflectionConfig(
                reflection_lm=_reflection_lm,
                reflection_minibatch_size=a.minibatch,
            ),
        ),
    )

    best = getattr(res, "best_candidate", None) or seed
    json.dump(best, open(os.path.join(a.out, "best_candidate.json"), "w"), indent=1)
    print(f"[gepa] DONE. best score {getattr(res,'best_score',None)} (baseline {BASELINE})",
          flush=True)
    print(best.get("STRATEGY", "")[:1500])


def _reflection_lm(prompt: str) -> str:
    from openai import OpenAI
    c = OpenAI(base_url="http://localhost:8030/v1", api_key="EMPTY", timeout=1800)
    r = c.chat.completions.create(
        model="qwen3.8-27b-bf16",
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0, max_completion_tokens=8192,
        extra_body={"reasoning_effort": "medium"})
    return r.choices[0].message.content or ""


if __name__ == "__main__":
    main()
