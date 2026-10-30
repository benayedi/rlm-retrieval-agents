"""GEPA on BrowseComp-Plus, Full-Doc RLM harness (design B), Qwen3.8-27B -- Claude as the critic.

From original/RLM/scripts/gepa_bcp.py. Two ways this module is used:

  1. GEPA driver (its original purpose; no row in configs/ comes from a GEPA-optimised prompt):
         python src/split_runner.py [--max-metric-calls N --minibatch b --workers W] [--preload-seed|--dry-run|--smoke N]
     Needs `gepa` (optimize_anything) and, for the critic, the `claude` CLI (CLAUDE_BIN) or GEPA_REFLECTION=qwen.
  2. Split runner: `--seed-split <split>` runs the UNCHANGED seed prompt on every question of one split of
     GEPA_SPLIT (feedback / pareto / test200 / test_rest), grades it with grade_local.judge and exits. This is
     how the Qwen3.6 BrowseComp+ rows were produced: BCP-29 directly, BCP-27 and BCP-31 through bcp_qwen_dense.py and
     bcp_bm25_split.py, which import this file and call run_batch() on the seed.

Every rollout is `$PYTHON src/run_one.py ... BASE_ARGS` with PYTHONPATH=$RLMLIB_PATH (no startup hooks) and cwd
FULLDOC_ROOT. Results are cached in <GEPA_OUT>/cache.json under "<ckey>|<qid>"; ckey() hashes the four prompt blocks
and FINGERPRINT (model, BASE_ARGS, judge, lambda, Qwen switches). Neither contains a path, so the keys of the
original caches (e925f0f6c7, ee5ea32a1c, 11d0c5bc6b) are reproduced by this file; tests/ checks that.
Paths: GEPA_OUT is resolved against RUNS_DIR (original: against the original repository root); GEPA_SPLIT has no
default unless RLM_DATA_DIR is set (original default: <original repo>/data/bcp_gepa_splits/split.json).

WHAT IS OPTIMISED. The four prompt blocks the Full-Doc RLM harness exposes through its
GEPA_PROMPTS hook (src/prompts.py): SOLO_SYSTEM_PROMPT (2,124 ch), USER_PROLOGUE (999),
LEAN_CLAUSE (1,259), QUERY_TEMPLATE (687). All four are prompt text -- tools, caps, model and
retriever are fixed -- so unlike the Twitter runs every block is mutable, rotated round-robin
as in the paper (Algorithm 1, SELECTMODULE).

THE AGENT CONFIG IS THE 79.4% BASELINE, VERBATIM. BASE_ARGS below is the launch_argv of
<FULLDOC_ROOT>/runs/qwen27b_colbert8192_rest810/_provenance.json minus the run_experiment-only flags.
NOTE gepa_eval.py of the original working directory (not preserved) does NOT match it (max-timeout 1800 / lm-timeout 900 against the
baseline's 5400 / 1800): candidates would time out more than the seed did and lose for a
reason that is not in the prompt. That evaluator is not used here.

SPLITS (data/bcp_gepa_splits/split.json, made by scripts/bcp_gepa_split.py): stratified on the
seed's own outcome in the 830 run, proportional allocation, so every split's mean estimates
the full benchmark. D_feedback 200 (minibatch pool -- b=3 per iteration, so its size is
free), D_pareto 100 (selection), test200 (confirmation only).

OBJECTIVE. score = correct + LAMBDA * gold_recall, LAMBDA = 0.5. Accuracy is the reported
metric; gold recall densifies it. On Qwen recall converts to accuracy (830 run: accuracy
.124 at recall 0, .874 partial, .942 full), so a candidate cannot buy score with recall that
does not turn into answers -- the reason this was rejected on gpt-5 does not apply here.

THE CRITIC IS CLAUDE. `claude -p` (Claude Code headless, the user's own login -- no API key
on this box), model GEPA_REFLECTION_MODEL (default claude-opus-5-5), no tools, one turn.
Every call is logged with its cost to reflection_log.jsonl. If the CLI fails twice the step
falls back to local Qwen and says so in progress.txt -- set GEPA_REFLECTION=qwen to force it.

LESSONS FROM THE TWITTER RUNS, BUILT IN:
  * cache key = sha(blocks + config fingerprint). The Twitter key had neither model nor
    harness config in it, so a model swap would have mixed models silently.
  * infra failures (timeouts, crashes) are imputed with the candidate's mean, never zero.
  * every candidate is VALIDATED before any rollout: placeholders kept, str.format still
    works, answer-format markers present. A broken template would otherwise burn 100
    rollouts producing errors that get imputed as neutral.
  * the critic's reply is sanitised to one fenced block (gepa extracts first-``` to last-```).
  * the critic never sees gold answers -- only verdicts, gold docids it missed and its queries
    -- and is told not to write question-specific facts into the prompt.
  * null guard every batch: what the UNCHANGED prompt scores on a 100-question bootstrap.

--dry-run: no GPU. Serves the seed's cached 830-run records for every evaluation (so scores
are FAKE), but runs GEPA's real loop far enough to make ONE real Claude reflection call and
validate the proposal. It proves the wiring; it measures nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time as _t
import uuid
from concurrent.futures import ThreadPoolExecutor

# Import wiring. Original: R = the experiment repository (for examples.browsecomp_data, now shared/rlm_shared) and
# SUP = the Full-Doc RLM working directory (for src/ modules, data and earlier runs).
SRC = os.path.dirname(os.path.abspath(__file__))  # original: f"{SUP}/src"
sys.path.insert(0, os.path.join(SRC, os.pardir, os.pardir, os.pardir, "shared"))
sys.path.insert(0, SRC)
from rlm_shared.paths import path  # noqa: E402

RUNS = str(path("RUNS_DIR"))  # original: f"{R}/logs"
# Interpreter of the rollouts (original: /home/extstu_00/miniconda3/envs/vllm/bin/python). Checked in _rollout().
PYV = os.environ.get("PYTHON", "")
CLAUDE = os.environ.get("CLAUDE_BIN", "claude")  # original default: the CLI in the user's ~/.local/bin

SERVERS = [s.strip() for s in os.environ.get(
    "GEPA_BCP_SERVERS", "http://localhost:8030/v1").split(",") if s.strip()]
os.environ.setdefault("LOCAL_JUDGE_BASE_URL", SERVERS[0])
RETRIEVAL = os.environ.get("GEPA_BCP_RETRIEVAL", "http://127.0.0.1:8766")
BASELINE_MODEL = "qwen3.8-27b-bf16"   # the model of the 830-run cache (splits, preload, null guard)
ROOT_MODEL = os.environ.get("GEPA_BCP_MODEL", BASELINE_MODEL)
# the judge is the served agent model (grade_local reads LOCAL_JUDGE_MODEL at import)
os.environ.setdefault("LOCAL_JUDGE_MODEL", ROOT_MODEL)
FENCE_TOLERANT = os.environ.get("SUP_FENCE_TOLERANT", "") == "1"
MAX_BLOCKS = int(os.environ.get("SUP_MAX_BLOCKS", "0") or 0)   # src/fence_tolerant.py
PRESENCE_PENALTY = os.environ.get("SUP_PRESENCE_PENALTY", "")   # src/run_one.py _sampling()
BLOCKS = ["SOLO_SYSTEM_PROMPT", "USER_PROLOGUE", "LEAN_CLAUSE", "QUERY_TEMPLATE"]
LAMBDA = 0.5
SCORED = ("completed", "empty_output")   # statuses scored as agent outcomes, not infra
# GEPA_TIMEOUT_WRONG=1: harness timeouts are agent outcomes too (scored WRONG, not imputed).
# Measured 2026-10-09: run-2 prompts CAUSE timeouts (best candidate 8/100 vs 3 for the seed; all
# "What about X?" brainstorm loops after the 10 searches were spent). Imputing them rewarded that.
# Runaway kills / restarts ("killed") stay infrastructure. Not in FINGERPRINT: scoring, not rollouts.
TIMEOUT_WRONG = os.environ.get("GEPA_TIMEOUT_WRONG", "") == "1"


def scored(r) -> bool:
    if not r:
        return False
    if r.get("status") in SCORED:
        return True
    return TIMEOUT_WRONG and "Timeout" in str(r.get("error") or "")

# launch_argv of the 79.4% run (qwen27b_colbert8192_rest810), minus --name/--ids-file/
# --workers/--stagger (run_experiment only) and --base-url (set per worker).
BASE_ARGS = [
    "--design", "B", "--server", RETRIEVAL, "--backend", "openai",
    "--api-key-env", "ALIBABA_TOKEN", "--root-model", ROOT_MODEL, "--sub-model", ROOT_MODEL,
    "--reasoning-effort", "medium", "--sub-reasoning-effort", "medium",
    "--max-iterations", "30", "--max-searches", "10", "--seed-context", "--compaction",
    "--grounding-gate", "--lean-prologue", "--no-delegation", "--solo-prompt",
    "--no-orchestrator", "--compaction-threshold-pct", "0.85", "--max-concurrent-subcalls", "4",
    "--max-timeout", "5400", "--lm-timeout", "1800", "--save-trajectory",
]
HARD_CAP_S = 5400 + 900   # subprocess kill: harness timeout plus grace for the final write
FINGERPRINT = hashlib.sha1(json.dumps(
    {"model": ROOT_MODEL, "args": BASE_ARGS, "judge": os.environ["LOCAL_JUDGE_MODEL"],
     "lambda": LAMBDA, "fence_tolerant": FENCE_TOLERANT,
     **({"max_blocks": MAX_BLOCKS} if MAX_BLOCKS else {}),
     **({"presence_penalty": PRESENCE_PENALTY} if PRESENCE_PENALTY else {}),
     **({"rep_guard": "1"} if os.environ.get("SUP_REP_GUARD") == "1" else {})}).encode()).hexdigest()[:8]   # src/rep_guard.py

# ABSOLUTE: run_one.py runs with cwd=FULLDOC_ROOT, so a relative GEPA_OUT made rollouts land under
# the working directory while the driver looked elsewhere and recorded 119 finished rollouts
# as "killed" (2026-10-08, --seed-split feedback). A relative GEPA_OUT is resolved against RUNS_DIR.
OUT = os.path.abspath(os.path.join(RUNS, os.environ.get("GEPA_OUT", f"{RUNS}/gepa_bcp")))
PROG = f"{OUT}/progress.txt"
CACHE_F = f"{OUT}/cache.json"
SPLIT_F = os.environ.get("GEPA_SPLIT") or (
    os.path.join(os.environ["RLM_DATA_DIR"], "bcp_gepa_splits", "split.json") if os.environ.get("RLM_DATA_DIR") else "")


def _split_file() -> str:
    """GEPA_SPLIT, read where it is used (no machine-specific default)."""
    if not SPLIT_F:
        raise SystemExit("set GEPA_SPLIT to the split file (bcp_gepa_splits/split.json), or RLM_DATA_DIR")
    return SPLIT_F
_CACHE: dict[str, dict] = {}
_LOCK = threading.Lock()
_SEED_SCORE = [0.0]
_DRY = [False]
_SEED_KEY = [""]


def _log(msg: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    line = f"{_t.strftime('%F %T')}  {msg}"
    with open(PROG, "a") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def ckey(blocks: dict) -> str:
    s = json.dumps({b: blocks[b] for b in BLOCKS}, sort_keys=True) + FINGERPRINT
    return hashlib.sha1(s.encode()).hexdigest()[:10]


def _cache_load():
    global _CACHE
    if os.path.exists(CACHE_F):
        _CACHE = json.load(open(CACHE_F))
        _log(f"[cache] loaded {len(_CACHE)} records")


def _cache_save():
    os.makedirs(OUT, exist_ok=True)
    tmp = CACHE_F + ".tmp"
    with _LOCK:
        json.dump(_CACHE, open(tmp, "w"))
        os.replace(tmp, CACHE_F)


# ---------------------------------------------------------------- data ------------------
_ROWS: dict[str, dict] = {}


def rows() -> dict[str, dict]:
    if not _ROWS:
        from datasets import load_dataset
        for r in load_dataset("Tevatron/browsecomp-plus", split="test"):
            _ROWS[str(r["query_id"])] = r
    return _ROWS


def seed_blocks() -> dict[str, str]:
    import prompts as P   # src/prompts.py, GEPA_PROMPTS unset -> the baseline text
    assert not os.environ.get("GEPA_PROMPTS"), "unset GEPA_PROMPTS before reading the seed"
    return {b: getattr(P, b) for b in BLOCKS}


# ------------------------------------------------------------ validation ----------------
_FIELD = re.compile(r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)\}(?!\})")


_REQUIRED = {"SOLO_SYSTEM_PROMPT": "\n\n{custom_tools_section}", "QUERY_TEMPLATE": "\n\nQuestion: {Question}"}


def repair(cand: dict) -> tuple[dict, list[str]]:
    """Re-append a required placeholder the critic dropped (measured: the first Claude proposal
    in the dry run lost {custom_tools_section}). Appending is semantically safe for both: tool
    docs at the end of the system prompt, the question at the end of the user template. The
    REPAIRED text is what runs and what the cache key is computed from."""
    c, fixed = dict(cand), []
    for b, tail in _REQUIRED.items():
        ph = tail.strip().split(": ")[-1]
        if isinstance(c.get(b), str) and ph not in c[b]:
            c[b] = c[b].rstrip() + tail; fixed.append(f"{b}+{ph}")
    return c, fixed


def validate(cand: dict, seed: dict) -> str | None:
    """None if the candidate can run; otherwise a reason (which is a real prompt defect)."""
    for b in BLOCKS:
        v = cand.get(b)
        if not isinstance(v, str) or not v.strip():
            return f"{b} is empty"
        if set(_FIELD.findall(v)) != set(_FIELD.findall(seed[b])):
            return (f"{b} must keep exactly the placeholders "
                    f"{sorted(set(_FIELD.findall(seed[b])))}, got {sorted(set(_FIELD.findall(v)))}")
        if len(v) > 3 * len(seed[b]) + 1500:
            return f"{b} grew to {len(v)} chars (seed {len(seed[b])}); bloat is rejected"
    try:
        cand["QUERY_TEMPLATE"].format(Question="Q")
        cand["SOLO_SYSTEM_PROMPT"].format(custom_tools_section="")
    except Exception as e:
        return f"str.format fails ({type(e).__name__}: {e}); literal braces must be doubled"
    for marker in ("Exact Answer:", "Explanation:"):
        if marker not in cand["QUERY_TEMPLATE"]:
            return f"QUERY_TEMPLATE dropped the answer marker {marker!r} the grader parses"
    return None


# ------------------------------------------------------------- rollouts -----------------
def _grade(qid: str, raw: dict) -> dict:
    from rlm_shared.browsecomp_data import get_question, get_gold_answer, get_gold_doc_ids
    import grade_local
    row = rows()[qid]
    resp = (raw.get("result") or [{}])[0].get("output", "") or ""
    status = raw.get("status")
    gids = set(map(str, get_gold_doc_ids(row)))
    got = set(map(str, raw.get("server_docids") or raw.get("retrieved_docids") or []))
    ok = status == "completed" and bool(resp) and grade_local.judge(
        get_question(row), resp, get_gold_answer(row))
    return {"query_id": qid, "status": status, "error": raw.get("error"),
            "correct": bool(ok), "gold_recall": (len(gids & got) / len(gids)) if gids else 0.0,
            "missed_gold": sorted(gids - got)[:6],
            "searches": (raw.get("tool_call_counts") or {}).get("search", 0),
            "queries": [str(x.get("query", "") if isinstance(x, dict) else x)[:120] for x in (raw.get("search_log") or [])][:10],
            "wall": raw.get("wall_time_s"), "iters": raw.get("n_iterations"),
            "answer": resp[-400:]}


def _rollout(qid: str, pfile: str, server: str, out_dir: str) -> dict | None:
    out = os.path.join(out_dir, f"{qid}.json")
    if not PYV:
        raise SystemExit("environment variable PYTHON is not set: interpreter of the rollouts")
    # PYTHONPATH is the rlm library only, as in the original (no startup hooks in the rollouts).
    env = dict(os.environ, GEPA_PROMPTS=pfile, PYTHONPATH=str(path("RLMLIB_PATH")),
               HF_HOME=os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), ALIBABA_TOKEN="EMPTY")
    try:
        subprocess.run([PYV, f"{SRC}/run_one.py", "--query-id", qid, "--out", out,
                        "--base-url", server] + BASE_ARGS,
                       env=env, cwd=str(path("FULLDOC_ROOT")), capture_output=True, timeout=HARD_CAP_S)
    except subprocess.TimeoutExpired:
        _log(f"[runaway] q{qid} killed at {HARD_CAP_S}s")
        return None
    try:
        return json.load(open(out))
    except Exception:
        return None


def run_batch(cand: dict, qids: list[str], workers: int) -> dict[str, dict | None]:
    key = ckey(cand)
    # A "killed" record is a placeholder for a rollout that never finished (runaway cap, or a
    # restart killing in-flight workers). It is not a result: re-run it rather than reuse it.
    # (2026-10-08: 40 such placeholders from restarts were being skipped as "done" by
    # --seed-split and would have been read as seed FAILURES by bcp_enrich_split.py.)
    todo = [q for q in qids if (_CACHE.get(f"{key}|{q}") or {}).get("status") in (None, "killed")]
    if todo and _DRY[0]:
        # DRY RUN: any candidate is answered with the seed's cached records -- FAKE scores.
        for q in todo:
            _CACHE[f"{key}|{q}"] = dict(_CACHE.get(f"{_SEED_KEY[0]}|{q}") or {}, _fake=True)
        todo = []
    if todo:
        pfile = os.path.join(tempfile.gettempdir(), f"gepa_bcp_{key}_{uuid.uuid4().hex[:6]}.json")
        json.dump({b: cand[b] for b in BLOCKS}, open(pfile, "w"))
        out_dir = os.path.join(OUT, "rollouts", key)
        os.makedirs(out_dir, exist_ok=True)

        def one(i_q):
            i, q = i_q
            raw = _rollout(q, pfile, SERVERS[i % len(SERVERS)], out_dir)
            rec = _grade(q, raw) if raw else {"query_id": q, "status": "killed", "correct": False,
                                              "gold_recall": 0.0}
            with _LOCK:
                _CACHE[f"{key}|{q}"] = rec
            _cache_save()

        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(one, enumerate(todo)))
    return {q: _CACHE.get(f"{key}|{q}") for q in qids}


def score(rec: dict) -> float:
    return (1.0 if rec.get("correct") else 0.0) + LAMBDA * (rec.get("gold_recall") or 0.0)


def _ndcg10(ranked: list[str], rel: set[str]) -> float:
    import math
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked[:10]) if d in rel)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(10, len(rel))))
    return dcg / idcg if idcg else 0.0


def search_stats(key: str, qid: str) -> str | None:
    """Per-search quality of the agent's OWN queries, for the critic (NOT the score).

    Measured 2026-10-08 (scripts/bcp_search_ndcg.py): every Claude rewrite raised mean
    per-search nDCG@10 (+0.006..+0.031) yet cut searches (8.1 -> 7.2-7.7) and the best
    search per question (0.50 -> 0.44-0.48), so recall and accuracy did not move. Without
    these numbers the critic cannot see that tradeoff. Kept OUT of the selection score on
    purpose: mean-per-search nDCG rises when the agent searches less, which is what hurt.
    The first search is the harness's seed search (the question verbatim) and is excluded.
    """
    from rlm_shared.browsecomp_data import get_gold_doc_ids
    f = os.path.join(OUT, "rollouts", key, f"{qid}.json")
    try:
        log = (json.load(open(f)).get("search_log") or [])[1:]
    except Exception:
        return None
    if not log:
        return "Agent-written searches: 0 (only the seed search ran)"
    gold = set(map(str, get_gold_doc_ids(rows()[qid])))
    nd = [_ndcg10([str(x) for x in (s.get("docids") or [])], gold) for s in log]
    hit = sum(1 for s in log if gold & set(map(str, s.get("docids") or [])))
    return (f"Agent-written searches: {len(log)} of 9 available after the seed search | "
            f"per-search nDCG@10 vs gold: mean {sum(nd)/len(nd):.2f}, best {max(nd):.2f} | "
            f"{hit} of them returned at least one gold document")


def feedback(rec: dict, question: str, key: str | None = None) -> str:
    v = "CORRECT" if rec.get("correct") else "WRONG"
    if "Timeout" in str(rec.get("error") or ""):
        v = (f"WRONG -- TIMED OUT after {rec.get('searches', 0)} searches without submitting. "
             + ("Typical cause on this task: once all 10 searches are spent the agent keeps "
                "brainstorming candidate names ('What about X? What about Y?') instead of committing; "
                "the prompt should make it submit its best-supported answer when searches run out."
                if (rec.get("searches") or 0) >= 10 else ""))
    if rec.get("status") == "empty_output":
        v = ("WRONG -- EMPTY SUBMISSION: the agent set answer['ready']=True without setting "
             "answer['content'] (it printed its answer instead), so nothing was submitted")
    parts = [f"Question: {question[:600]}",
             f"Verdict: {v} | gold recall {rec.get('gold_recall', 0):.2f} | "
             f"{rec.get('searches', 0)} searches (cap 10) | {rec.get('iters')} iterations | "
             f"{(rec.get('wall') or 0):.0f}s"]
    if rec.get("missed_gold"):
        parts.append(f"Gold documents never retrieved: {len(rec['missed_gold'])} "
                     f"(ids {rec['missed_gold'][:4]})")
    ss = search_stats(key, str(rec.get("query_id"))) if key else None
    if ss:
        parts.append(ss)
    if rec.get("queries"):
        parts.append("Queries issued: " + " || ".join(rec["queries"]))
    if rec.get("answer"):
        parts.append("End of the agent's final response: " + rec["answer"][-300:].replace("\n", " "))
    return "\n".join(parts)


def make_evaluator(workers: int, seed: dict, null_ref: list[float]):
    st = {"batches": 0, "rollouts": 0, "t0": _t.time()}

    def evaluator(pairs, **kw):
        groups: dict[str, list[int]] = {}
        cands: dict[str, dict] = {}
        for i, (cand, _ex) in enumerate(pairs):
            k = ckey(cand); groups.setdefault(k, []).append(i); cands[k] = cand
        out: list = [None] * len(pairs)
        for k0, idxs in groups.items():
            cand, fixed = repair(cands[k0])
            k = ckey(cand)
            if fixed:
                _log(f"[repair] cand {k0} -> {k}: re-appended {fixed}")
            qids = [str(pairs[i][1]["qid"]) for i in idxs]
            bad = validate(cand, seed)
            if bad:
                _log(f"[invalid] cand {k}: {bad} -- scored 0 without rollouts")
                for i in idxs:
                    out[i] = (0.0, {"feedback": f"CANDIDATE REJECTED BEFORE RUNNING: {bad}"})
                continue
            n_cached = sum(1 for q in qids if f"{k}|{q}" in _CACHE)
            recs = run_batch(cand, qids, workers)
            # SCORED = the agent produced a verdict-able outcome. "empty_output" is the agent's own
            # failure (2026-10-07: it printed the answer, then set answer["ready"]=True without
            # answer["content"]) and must score WRONG -- imputing it would reward prompts that
            # cause it. Only timeouts/crashes/kills are infrastructure and get imputed.
            okr = {q: r for q, r in recs.items() if scored(r)}
            impute = statistics.mean([score(r) for r in okr.values()]) if okr else _SEED_SCORE[0]
            accs = []
            for i in idxs:
                q = str(pairs[i][1]["qid"]); r = okr.get(q)
                if r is None:
                    out[i] = (impute, {"feedback": "RUN DID NOT COMPLETE (timeout/infra). "
                                                   "Not attributable to the prompt.",
                                       "infra_failure": True})
                else:
                    from rlm_shared.browsecomp_data import get_question
                    out[i] = (score(r), {"feedback": feedback(r, get_question(rows()[q]), k)})
                    accs.append(1.0 if r.get("correct") else 0.0)
            st["batches"] += 1; st["rollouts"] += len(idxs) - n_cached
            acc = statistics.mean(accs) if accs else 0.0
            sc = statistics.mean(o[0] for o in (out[i] for i in idxs))
            beat = sum(1 for x in null_ref if sc > x) / max(1, len(null_ref))
            _log(f"batch {st['batches']:>3} | cand {k} | n={len(idxs)} cached={n_cached} | "
                 f"acc={acc:.3f} score={sc:.4f} | vs NULL(n=100): beats {beat*100:.0f}% | "
                 f"infra {len(idxs)-len(okr)} | rollouts {st['rollouts']} | "
                 f"{(_t.time()-st['t0'])/60:.0f} min{' | DRY-RUN (fake scores)' if _DRY[0] else ''}")
        return out

    return evaluator


# ------------------------------------------------------------- the critic ---------------
CRITIC_SYSTEM = (
    "You are the reflection step of GEPA, an evolutionary prompt optimiser. The system being "
    "optimised is a deep-research agent (" + ROOT_MODEL + " in a Python REPL with a search tool over "
    "a fixed 100k-document web corpus) answering BrowseComp-Plus questions. You will be shown "
    "ONE prompt component, the agent's results on a few questions and feedback. Propose an "
    "improved version of that component. Rules: (1) Write GENERAL strategy that transfers to "
    "unseen questions -- never copy entities, facts, answers, queries or document ids from the "
    "examples into the prompt. (2) Keep every {placeholder} exactly as it appears, and keep "
    "literal braces doubled ({{ }}). In particular SOLO_SYSTEM_PROMPT MUST still contain the "
    "literal text {custom_tools_section} (the tool documentation is inserted there) and "
    "QUERY_TEMPLATE MUST still contain {Question}. (3) Keep the 'Explanation:' / 'Exact Answer:' / "
    "'Confidence:' response format if the component contains it. (4) Stay concise: the agent "
    "re-reads this text every turn; do not more than double its length. (5) Do not make the agent "
    "search LESS: measured on this task, every rewrite that framed searches as scarce or told the "
    "agent to stop early cut its searches (8.1 -> ~7.4 per question) and lowered gold recall even "
    "though each query got sharper. Encourage spending the full search budget on DIFFERENT angles. "
    "(6) Reply with the "
    "complete new component inside exactly ONE ``` fenced block, nothing after it. "
    "Feedback metrics: gold recall = share of the question's gold documents the agent ever "
    "retrieved; per-search nDCG@10 = how well each of its own queries ranked gold documents "
    "(mean and best over its searches); the score you are optimised on is correct + 0.5 x "
    "gold recall, so query precision only helps if the gold documents end up retrieved.")


_FENCE_LINE = re.compile(r"\s*```[A-Za-z0-9_-]*\s*")


def _sanitise(raw: str) -> str:
    """Keep the LAST block delimited by fence-only LINES.

    The prompt blocks themselves contain inline fences ("write code in ```repl``` blocks": 6 in
    SOLO_SYSTEM_PROMPT, 2 in LEAN_CLAUSE, none on a line of their own). The Twitter sanitiser's
    regex split at those and truncated the first dry-run proposal mid-text, which then failed
    validation for a missing placeholder that Claude had in fact kept. Fence-only lines are the
    real delimiters; when a critic writes several versions, the last one is the one it meant.
    """
    lines = raw.splitlines()
    idx = [i for i, l in enumerate(lines) if _FENCE_LINE.fullmatch(l)]
    body = "\n".join(lines[idx[-2] + 1: idx[-1]]).strip() if len(idx) >= 2 else raw.strip()
    return "```\n" + body + "\n```"


def _claude(prompt: str) -> tuple[str, float]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CLAUDECODE", "CLAUDE_CODE_"))}
    model = os.environ.get("GEPA_REFLECTION_MODEL", "claude-opus-5-5")
    p = subprocess.run([CLAUDE, "-p", "--model", model, "--output-format", "json", "--tools", "",
                        "--no-session-persistence", "--system-prompt", CRITIC_SYSTEM],
                       input=prompt, capture_output=True, text=True, timeout=900, env=env,
                       cwd=tempfile.gettempdir())
    d = json.loads(p.stdout)
    if d.get("is_error") or not d.get("result"):
        raise RuntimeError(f"claude -p error: {str(d)[:300]}")
    return d["result"], float(d.get("total_cost_usd") or 0.0)


def _qwen(prompt: str) -> str:
    from openai import OpenAI
    c = OpenAI(base_url=SERVERS[0], api_key="EMPTY", timeout=1800)
    r = c.chat.completions.create(
        model=ROOT_MODEL, temperature=1.0, max_completion_tokens=8192,
        messages=[{"role": "system", "content": CRITIC_SYSTEM}, {"role": "user", "content": prompt}],
        extra_body={"reasoning_effort": "medium"})
    return r.choices[0].message.content or ""


def reflection_lm(prompt: str) -> str:
    who, cost, raw = "qwen", 0.0, ""
    if os.environ.get("GEPA_REFLECTION", "claude") == "claude":
        for attempt in (1, 2):
            try:
                raw, cost = _claude(prompt); who = "claude"; break
            except Exception as e:
                _log(f"[critic] claude attempt {attempt} failed: {str(e)[:200]}")
        if who != "claude":
            _log("[critic] FALLING BACK TO LOCAL QWEN for this reflection step")
    if who == "qwen":
        raw = _qwen(prompt)
    with open(f"{OUT}/reflection_log.jsonl", "a") as fh:
        fh.write(json.dumps({"t": _t.strftime('%F %T'), "critic": who, "cost_usd": cost,
                             "prompt_chars": len(prompt), "reply": raw}) + "\n")
    _log(f"[critic] {who} reflection: {len(prompt)} ch in, {len(raw)} ch out, ${cost:.3f}")
    return _sanitise(raw)


# ----------------------------------------------------------------- main -----------------
def preload_seed(seed_key: str) -> int:
    """Seed records from the 830 run (used by --dry-run and --preload-seed)."""
    import glob
    SUP = str(path("FULLDOC_ROOT"))  # earlier Qwen3.8-27B runs live in the working directory's runs/
    G = json.load(open(f"{SUP}/runs/_graded_cache.json"))
    raws = {}
    for d in ("qwen27b_colbert8192_first20", "qwen27b_colbert8192_rest810"):
        for f in glob.glob(f"{SUP}/runs/{d}/*.json"):
            if not os.path.basename(f).startswith("_"):
                raws[os.path.basename(f)[:-5]] = f
    n = 0
    for q, g in G.items():
        raw = json.load(open(raws[q])) if q in raws else {}
        rec = {"query_id": q, "status": g["status"], "correct": g["correct"],
               "gold_recall": g["gold_recall"], "searches": g.get("searches"),
               "iters": g.get("iters"), "wall": g.get("wall"),
               "queries": [str(x.get("query", "") if isinstance(x, dict) else x)[:120] for x in (raw.get("search_log") or [])][:10],
               "answer": ((raw.get("result") or [{}])[0].get("output", "") or "")[-400:],
               "_from": "830-run cache"}
        if raw and q in rows():
            from rlm_shared.browsecomp_data import get_gold_doc_ids
            got = set(map(str, raw.get("server_docids") or raw.get("retrieved_docids") or []))
            rec["missed_gold"] = sorted(set(map(str, get_gold_doc_ids(rows()[q]))) - got)[:6]
        _CACHE.setdefault(f"{seed_key}|{q}", rec); n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-metric-calls", type=int, default=800)
    ap.add_argument("--minibatch", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--preload-seed", action="store_true",
                    help="reuse the 830-run seed records instead of re-running the seed on D_pareto")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed-split", default="",
                    help="run the SEED on every question of this split (e.g. feedback), grade, report, exit")
    ap.add_argument("--smoke", type=int, default=0,
                    help="run the SEED on the first N D_pareto questions for real, grade, print, exit")
    a = ap.parse_args()
    _DRY[0] = a.dry_run
    if a.seed_split:
        # Seed outcomes on more questions, so a selection set can be enriched with this model's
        # OWN failures (plan B, 2026-10-08). Same cache, same key as the GEPA run -> reused free.
        _cache_load()
        seed = seed_blocks()
        qids = json.load(open(_split_file()))[a.seed_split]
        _log(f"[seed-split] seed on '{a.seed_split}' ({len(qids)} q, {sum(1 for q in qids if f'{ckey(seed)}|{q}' in _CACHE)} cached)")
        recs = run_batch(seed, qids, a.workers)
        ok = [r for r in recs.values() if r and r.get("status") in SCORED]
        _log(f"[seed-split] '{a.seed_split}': n={len(recs)} scored={len(ok)} acc="
             f"{statistics.mean(1.0 if (r and r.get('correct')) else 0.0 for r in recs.values()):.3f}")
        sys.exit(0)
    if a.smoke:
        # Real rollouts through the exact path GEPA uses. Catches the failure class that sank
        # the discordance re-run (every worker crashed on ModuleNotFoundError: rlm) in minutes
        # instead of after a night.
        global OUT, PROG, CACHE_F
        OUT = f"{RUNS}/gepa_bcp_smoke"; PROG = f"{OUT}/progress.txt"; CACHE_F = f"{OUT}/cache.json"
        seed = seed_blocks()
        qids = json.load(open(_split_file()))["pareto"][: a.smoke]
        recs = run_batch(seed, qids, min(a.workers, len(qids)))
        bad = [q for q, r in recs.items() if not r or r.get("status") != "completed"]
        for q, r in recs.items():
            _log(f"[smoke] q{q}: status={r and r.get('status')} correct={r and r.get('correct')} "
                 f"gold_recall={r and r.get('gold_recall')} searches={r and r.get('searches')} "
                 f"wall={r and r.get('wall')}")
        _log(f"[smoke] {'FAIL' if bad else 'OK'}: {len(qids) - len(bad)}/{len(qids)} completed")
        sys.exit(1 if bad else 0)
    if a.dry_run and "GEPA_OUT" not in os.environ:
        OUT = f"{RUNS}/gepa_bcp_dryrun"; PROG = f"{OUT}/progress.txt"; CACHE_F = f"{OUT}/cache.json"
    os.makedirs(OUT, exist_ok=True)
    _cache_load()

    seed = seed_blocks()
    sk = ckey(seed); _SEED_KEY[0] = sk
    sp = json.load(open(_split_file()))
    pareto, fb = sp["pareto"], sp["feedback"]
    _log(f"[setup] fingerprint {FINGERPRINT} | seed {sk} | blocks "
         f"{ {b: len(seed[b]) for b in BLOCKS} } | servers {SERVERS} | retrieval {RETRIEVAL} | "
         f"D_feedback {len(fb)} | D_pareto {len(pareto)} | b={a.minibatch} | "
         f"budget {a.max_metric_calls} | workers {a.workers} | "
         f"critic {os.environ.get('GEPA_REFLECTION', 'claude')}:"
         f"{os.environ.get('GEPA_REFLECTION_MODEL', 'claude-opus-5-5')}")
    same_model = ROOT_MODEL == BASELINE_MODEL
    if (a.dry_run or a.preload_seed) and not same_model:
        # The 830-run records are BASELINE_MODEL rollouts; filing them under another model's
        # seed key would be exactly the silent model-mixing the fingerprint exists to prevent.
        raise SystemExit(f"--preload-seed/--dry-run use {BASELINE_MODEL} records; refusing for {ROOT_MODEL}")
    if a.dry_run or a.preload_seed:
        _log(f"[seed] preloaded {preload_seed(sk)} records from the 830 run under {sk}")
    _cache_save()

    rng = random.Random(1234)
    if a.dry_run:
        G = json.load(open(f"{path('FULLDOC_ROOT')}/runs/_graded_cache.json"))
        allv = [(1.0 if g["correct"] else 0.0) + LAMBDA * (g["gold_recall"] or 0.0) for g in G.values()]
        src = "830-run bootstrap"
    else:
        # BASELINE FIRST, explicitly: the seed on all of D_pareto. For a new model this IS its
        # BrowseComp+ baseline; for the 27B it re-measures after the harness changes and gives
        # the run-to-run noise. GEPA then finds these 100 in the cache and pays nothing for them.
        _log(f"[seed] BASELINE: seed prompt on D_pareto ({len(pareto)} q) with {ROOT_MODEL}")
        recs = run_batch(seed, pareto, a.workers)
        okr = [r for r in recs.values() if r and r.get("status") in SCORED]
        accs = [1.0 if (r and r.get("correct")) else 0.0 for r in recs.values()]
        _log(f"[seed] BASELINE {ROOT_MODEL}: accuracy {statistics.mean(accs):.3f} "
             f"(n={len(accs)}, errors counted wrong) | completed {len(okr)} | gold_recall "
             f"{statistics.mean(r.get('gold_recall') or 0 for r in okr):.3f} | median wall "
             f"{statistics.median(r.get('wall') or 0 for r in okr):.0f}s")
        if len(okr) < 0.9 * len(pareto):
            raise SystemExit(f"[seed] only {len(okr)}/{len(pareto)} completed -- infrastructure, "
                             "not prompts. Fix before spending the GEPA budget.")
        allv = [score(r) for r in okr]
        src = "seed's own D_pareto scores"
    null_ref = sorted(statistics.mean(rng.choice(allv) for _ in range(len(pareto))) for _ in range(4000))
    seed_par = [score(_CACHE[f"{sk}|{q}"]) for q in pareto
                if f"{sk}|{q}" in _CACHE and _CACHE[f"{sk}|{q}"].get("status") in SCORED]
    _SEED_SCORE[0] = statistics.mean(seed_par) if seed_par else statistics.mean(allv)
    _log(f"[null] unchanged prompt, 100-q bootstrap of the {src}: mean {statistics.mean(null_ref):.4f} "
         f"p95 {null_ref[3800]:.4f} p99 {null_ref[3960]:.4f} (score = correct + {LAMBDA}*gold_recall)")

    import gepa.optimize_anything as OA
    res = OA.optimize_anything(
        seed_candidate=seed,
        dataset=[{"qid": q} for q in fb],
        valset=[{"qid": q} for q in pareto],
        batch_evaluator=make_evaluator(a.workers, seed, null_ref),
        config=OA.GEPAConfig(
            engine=OA.EngineConfig(max_metric_calls=a.max_metric_calls,
                                   candidate_selection_strategy="pareto",
                                   run_dir=OUT, raise_on_exception=False),
            tracking=OA.TrackingConfig(logger=OA.StdOutLogger()),
            reflection=OA.ReflectionConfig(reflection_lm=reflection_lm,
                                           reflection_minibatch_size=a.minibatch,
                                           module_selector="round_robin")))
    best = getattr(res, "best_candidate", None) or seed
    json.dump(best, open(f"{OUT}/best_candidate.json", "w"), indent=1)
    _log(f"[done] best {ckey(best)} (seed {sk}). Next: validate on D_pareto, then test200.")


if __name__ == "__main__":
    main()
