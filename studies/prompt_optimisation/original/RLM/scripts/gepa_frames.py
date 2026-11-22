"""GEPA on FRAMES, supervisor harness, Qwen3.6-35B-A3B, BM25 over the paper's June 2023 Wikipedia.

A THIN ADAPTER OVER scripts/gepa_bcp.py. The GEPA loop, evaluator, scoring (correct + 0.5 x gold
recall), candidate validation/repair, cache keying, critic plumbing (`claude -p`, sanitiser,
Qwen fallback) and infra-failure imputation are gepa_bcp.py's, imported unchanged. Replaced here:
  * data      FRAMES rows (data/frames/frames_rows.json), queries via QUERIES_FILE
  * judge     examples.frames_eval_rlm.judge_match -- the judge every FRAMES number was graded with
  * entry     scripts/frames_sup_run_one.py (facts-only BM25 retriever description) instead of
              suprlm/src/run_one.py; retrieval :8041, model servers :8030 (GPU 0) + :8031 (GPU 1)
  * seed      the four GEPA blocks as the 824-question seed run used them: USER_PROLOGUE carries
              the BM25 sentence (frames_sup_run_one.PROLOGUE_NEW), the others are verbatim
  * critic    CRITIC_SYSTEM rewritten for FRAMES (Wikipedia multi-hop QA, not BrowseComp+)
  * splits    data/frames/gepa_splits/split.json from the seed run's own outcomes, stratified;
              q0-99 are NEVER in test (the facts-only prompt was chosen on them)
  * preload   seed records from logs/frames/sup_q36_bm25f (the 824-question seed run)
The cache key adds a FRAMES tag to gepa_bcp's fingerprint, so no BrowseComp+ record can be reused.

  python scripts/gepa_frames.py --make-splits            # after the 824 seed run
  python scripts/gepa_frames.py --dry-run                # no GPU: fake scores, ONE real critic call
  python scripts/gepa_frames.py --smoke 2                # 2 real seed rollouts through GEPA's path
  python scripts/gepa_frames.py --preload-seed           # the run
  python scripts/gepa_frames.py --test best_candidate.json   # confirmation on the test split
"""
from __future__ import annotations

import argparse, collections, glob, hashlib, importlib.util, json, os, random, statistics, subprocess, sys

R = "/home/extstu_00/RLM"
SUP = "/home/extstu_00/suprlm"
MODEL = "qwen36-35b-a3b"
SEED_RUN = f"{R}/logs/frames/sup_q36_bm25f"
SPLIT_DIR = f"{R}/data/frames/gepa_splits"
ENTRY = f"{R}/scripts/frames_sup_run_one.py"

# --- environment BEFORE gepa_bcp is imported: it reads these at import (servers, fingerprint) --
os.environ.setdefault("GEPA_BCP_SERVERS", "http://localhost:8030/v1,http://localhost:8031/v1")
os.environ.setdefault("GEPA_BCP_RETRIEVAL", "http://127.0.0.1:8041")
os.environ.setdefault("GEPA_BCP_MODEL", MODEL)
os.environ.setdefault("GEPA_OUT", f"{R}/logs/gepa_frames")
os.environ.setdefault("GEPA_SPLIT", f"{SPLIT_DIR}/split.json")
os.environ.setdefault("GEPA_TIMEOUT_WRONG", "1")          # as the current BCP runs
os.environ.update(SUP_FENCE_TOLERANT="1", SUP_MAX_BLOCKS="5", SUP_PRESENCE_PENALTY="1.5",  # = run_frames_sup.sh
                  QUERIES_FILE=f"{R}/data/frames/frames_queries.jsonl",
                  LOCAL_LLM_BASE_URL="http://localhost:8030/v1", LOCAL_JUDGE_BASE_URL="http://localhost:8030/v1",
                  LOCAL_JUDGE_MODEL=MODEL, BM25_PROMPT="1")
# CPU threads per rollout. Unset, numpy/tokenizers size their pools to all 96 cores; 16 rollouts
# then oversubscribe the box (2026-10-15 ~14:05: 15-min load 121 on 96 cores while 16+ runaway
# rollouts ran, and the VS Code remote client dropped). Thread counts do not change results.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, "/home/extstu_00/rlmrepo")

_spec = importlib.util.spec_from_file_location("gepa_bcp", f"{R}/scripts/gepa_bcp.py")
G = importlib.util.module_from_spec(_spec)
sys.modules["gepa_bcp"] = G
_spec.loader.exec_module(G)

import examples.browsecomp_data as BD  # noqa: E402
from examples.frames_eval_rlm import judge_match  # noqa: E402

G.FINGERPRINT = hashlib.sha1((G.FINGERPRINT + "|frames|bm25f-wrapper").encode()).hexdigest()[:8]
G.BASELINE_MODEL = MODEL

# --- data -------------------------------------------------------------------------------------
_ROWS = {str(r["i"]): r for r in json.load(open(f"{R}/data/frames/frames_rows.json"))}
G.rows = lambda: _ROWS
# gepa_bcp's evaluator and search_stats call these BrowseComp accessors (which decrypt) at call time
BD.get_question = lambda r: r["question"]
BD.get_gold_answer = lambda r: r["answer"]
BD.get_gold_doc_ids = lambda r: set(map(str, r["gold_ids"]))


def _grade(qid: str, raw: dict) -> dict:
    row = _ROWS[qid]
    resp = (raw.get("result") or [{}])[0].get("output", "") or ""
    status = raw.get("status")
    gids = set(map(str, row["gold_ids"]))
    got = set(map(str, raw.get("server_docids") or raw.get("retrieved_docids") or []))
    ok = status == "completed" and bool(resp) and judge_match(row["question"], resp, row["answer"])
    return {"query_id": qid, "status": status, "error": raw.get("error"),
            "correct": bool(ok), "gold_recall": (len(gids & got) / len(gids)) if gids else 0.0,
            "missed_gold": sorted(gids - got)[:6],
            "searches": (raw.get("tool_call_counts") or {}).get("search", 0),
            "queries": [str(x.get("query", "") if isinstance(x, dict) else x)[:120] for x in (raw.get("search_log") or [])][:10],
            "wall": raw.get("wall_time_s"), "iters": raw.get("n_iterations"),
            "answer": resp[-400:]}


G._grade = _grade


def _rollout(qid: str, pfile: str, server: str, out_dir: str):
    out = os.path.join(out_dir, f"{qid}.json")
    env = dict(os.environ, GEPA_PROMPTS=pfile, PYTHONPATH="/home/extstu_00/rlmrepo",
               HF_HOME="/home/extstu_00/.cache/huggingface", ALIBABA_TOKEN="EMPTY")
    try:
        subprocess.run([G.PYV, ENTRY, "--query-id", qid, "--out", out, "--base-url", server] + G.BASE_ARGS,
                       env=env, cwd=SUP, capture_output=True, timeout=G.HARD_CAP_S)
    except subprocess.TimeoutExpired:
        G._log(f"[runaway] q{qid} killed at {G.HARD_CAP_S}s")
        return None
    try:
        return json.load(open(out))
    except Exception:
        return None


G._rollout = _rollout


# --- seed -------------------------------------------------------------------------------------
def seed_blocks() -> dict[str, str]:
    """The four blocks as the seed run saw them. Only USER_PROLOGUE differs from suprlm's text:
    frames_sup_run_one.py rewrites its dense-retrieval sentence. The tool description and
    SOLO_READ_SENTENCE are not GEPA blocks; the wrapper keeps patching those in every rollout."""
    sys.path.insert(0, f"{SUP}/src")
    import prompts as P
    assert not os.environ.get("GEPA_PROMPTS"), "unset GEPA_PROMPTS before reading the seed"
    src = open(ENTRY).read()
    ns: dict = {}
    exec(src[src.index("PROLOGUE_OLD = "): src.index("# USER_PROLOGUE is a GEPA block")], ns)
    b = {k: getattr(P, k) for k in G.BLOCKS}
    assert ns["PROLOGUE_OLD"] in b["USER_PROLOGUE"]
    b["USER_PROLOGUE"] = b["USER_PROLOGUE"].replace(ns["PROLOGUE_OLD"], ns["PROLOGUE_NEW"])
    return b


G.seed_blocks = seed_blocks


def preload_seed(seed_key: str) -> int:
    """Seed records from the 824-question seed run (same harness, prompt, model, judge)."""
    graded = {v["qid"]: v for v in json.load(open(f"{SEED_RUN}/graded.json")).values()}
    n = 0
    for f in glob.glob(f"{SEED_RUN}/[0-9][0-9][0-9].json"):
        q = str(int(os.path.basename(f)[:3]))
        try:
            raw = json.load(open(f))
        except Exception:
            continue
        rec = _grade(q, raw) if q not in graded else {
            **_grade_cached(q, raw), "correct": graded[q]["correct"]}
        rec["_from"] = "824 seed run"
        G._CACHE.setdefault(f"{seed_key}|{q}", rec); n += 1
    return n


def _grade_cached(q, raw):   # every field of _grade except the judge call (verdict from graded.json)
    row = _ROWS[q]; gids = set(map(str, row["gold_ids"]))
    got = set(map(str, raw.get("server_docids") or raw.get("retrieved_docids") or []))
    return {"query_id": q, "status": raw.get("status"), "error": raw.get("error"),
            "gold_recall": (len(gids & got) / len(gids)) if gids else 0.0,
            "missed_gold": sorted(gids - got)[:6],
            "searches": (raw.get("tool_call_counts") or {}).get("search", 0),
            "queries": [str(x.get("query", "") if isinstance(x, dict) else x)[:120] for x in (raw.get("search_log") or [])][:10],
            "wall": raw.get("wall_time_s"), "iters": raw.get("n_iterations"),
            "answer": ((raw.get("result") or [{}])[0].get("output", "") or "")[-400:]}


G.preload_seed = preload_seed

# --- critic -----------------------------------------------------------------------------------
G.CRITIC_SYSTEM = (
    "You are the reflection step of GEPA, an evolutionary prompt optimiser. The system being "
    "optimised is a research agent (" + MODEL + " in a Python REPL with a BM25 keyword search tool "
    "over 6.7 million English Wikipedia articles, June 2023) answering FRAMES questions: multi-hop "
    "factual questions that combine 2-11 Wikipedia articles and often need arithmetic, dates, "
    "counting or comparison over the facts found. Tables and infobox numbers are missing from this "
    "Wikipedia text. You will be shown ONE prompt component, the agent's results on a few "
    "questions and feedback. Propose an improved version of that component. Rules: (1) Write "
    "GENERAL strategy that transfers to unseen questions -- never copy entities, facts, answers, "
    "queries or document ids from the examples into the prompt. (2) Keep every {placeholder} "
    "exactly as it appears, and keep literal braces doubled ({{ }}). In particular "
    "SOLO_SYSTEM_PROMPT MUST still contain the literal text {custom_tools_section} (the tool "
    "documentation is inserted there) and QUERY_TEMPLATE MUST still contain {Question}. (3) Keep "
    "the 'Explanation:' / 'Exact Answer:' / 'Confidence:' response format if the component contains "
    "it. (4) Stay concise: the agent re-reads this text every turn; do not more than double its "
    "length. (5) Do not make the agent search LESS, and do not prescribe a query style: measured on "
    "this task, telling it to write short keyword queries made it repeat the same query verbatim "
    "(exact repeats 7% -> 12% of searches), run out of its 10 searches and turns more often, and did "
    "not raise gold recall. A repeated query returns the same documents; different angles find new "
    "ones. (6) Reply with the complete new component inside exactly ONE ``` fenced block, nothing "
    "after it. Feedback metrics: gold recall = share of the question's gold articles the agent ever "
    "retrieved; per-search nDCG@10 = how well each of its own queries ranked gold articles (mean and "
    "best over its searches); the score you are optimised on is correct + 0.5 x gold recall, so "
    "query precision only helps if the gold articles end up retrieved.")


# --- splits -----------------------------------------------------------------------------------
def make_splits(n_feedback=200, n_pareto=100, n_test=200, seed=1234, path=f"{SPLIT_DIR}/split.json",
                require_all=True) -> dict:
    """Stratified on the seed's own outcome (wrong / right-with-partial-recall / right-full-recall),
    proportional allocation, as data/bcp_gepa_splits. q0-99 go to feedback/pareto only."""
    graded = {v["qid"]: v for v in json.load(open(f"{SEED_RUN}/graded.json")).values()}
    if require_all:
        # A question the seed run never finished (hard-cap kill, no result file -- q26) is a seed
        # failure: stratum "wrong", as in every accuracy reported for this run.
        if len(graded) < 800:
            raise SystemExit(f"seed run has {len(graded)}/824 graded questions; wait for it (or --partial for a dry run)")
        for q in map(str, range(824)):
            graded.setdefault(q, {"qid": q, "correct": False, "gold_recall": 0.0, "_missing": True})
    qs = sorted(graded, key=int)
    stratum = lambda g: ("wrong" if not g["correct"] else "right_full" if g["gold_recall"] >= 0.999 else "right_partial")
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    for q in qs:
        by[stratum(graded[q])].append(q)
    for v in by.values():
        rng.shuffle(v)
    total = len(qs)
    want = {"feedback": n_feedback, "pareto": n_pareto, "test": min(n_test, total - n_feedback - n_pareto)}
    split = {k: [] for k in want}
    for s, members in by.items():
        early = [q for q in members if int(q) < 100]          # never test
        rest = [q for q in members if int(q) >= 100]
        quota = {k: round(want[k] * len(members) / total) for k in want}
        for k in ("test", "pareto", "feedback"):              # test drawn first, from rest only
            src = list(rest) if k == "test" else early + rest
            if k != "test":
                rng.shuffle(src)                              # q0-99 spread over pareto/feedback
            take = [q for q in src if not any(q in v for v in split.values())][: quota[k]]
            split[k] += take
            for q in take:
                (early if q in early else rest).remove(q)
    split["_meta"] = {"seed_run": SEED_RUN, "strata": {k: len(v) for k, v in by.items()},
                      "rule": "stratified on seed outcome, proportional; q0-99 never in test",
                      "seed_acc_by_split": {k: round(statistics.mean(graded[q]["correct"] for q in split[k]), 3)
                                            for k in want if split[k]}}
    assert not set(split["test"]) & {str(i) for i in range(100)}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(split, open(path, "w"), indent=1)
    G._log(f"[splits] {path}: { {k: len(split[k]) for k in want} } | {split['_meta']}")
    return split


# --- main -------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-splits", action="store_true")
    ap.add_argument("--partial", action="store_true", help="--make-splits from an unfinished seed run (dry runs only)")
    ap.add_argument("--test", default="", help="score this candidate JSON (and the seed) on the test split, exit")
    ap.add_argument("--smoke", type=int, default=0,
                    help="N real SEED rollouts through GEPA's rollout path (GEPA_PROMPTS set), own dir")
    a, rest = ap.parse_known_args()
    if a.smoke:   # gepa_bcp's --smoke would overwrite logs/gepa_bcp_smoke/cache.json
        G.OUT = f"{R}/logs/gepa_frames_smoke"; G.PROG = f"{G.OUT}/progress.txt"; G.CACHE_F = f"{G.OUT}/cache.json"
        seed = seed_blocks()
        sp = f"{SPLIT_DIR}/split.json" if os.path.exists(f"{SPLIT_DIR}/split.json") else f"{SPLIT_DIR}/split_partial.json"
        qids = json.load(open(sp))["pareto"][: a.smoke]
        recs = G.run_batch(seed, qids, len(qids))
        for q, r in recs.items():
            G._log(f"[smoke] q{q}: status={r and r.get('status')} correct={r and r.get('correct')} "
                   f"gold_recall={r and r.get('gold_recall')} searches={r and r.get('searches')} wall={r and r.get('wall')}")
        bad = [q for q, r in recs.items() if not r or r.get("status") != "completed"]
        G._log(f"[smoke] {'FAIL' if bad else 'OK'}: {len(qids) - len(bad)}/{len(qids)} completed")
        sys.exit(1 if bad else 0)
    if a.make_splits:
        p = f"{SPLIT_DIR}/split_partial.json" if a.partial else f"{SPLIT_DIR}/split.json"
        make_splits(path=p, require_all=not a.partial, n_feedback=60 if a.partial else 200,
                    n_pareto=30 if a.partial else 100, n_test=0 if a.partial else 200)
        return
    if a.test:
        G._cache_load()
        cand = json.load(open(a.test)); seed = seed_blocks()
        qids = json.load(open(G.SPLIT_F))["test"]
        G._SEED_KEY[0] = G.ckey(seed); preload_seed(G._SEED_KEY[0])
        for name, c in (("seed", seed), ("candidate", cand)):
            recs = G.run_batch(c, qids, 8 * len(G.SERVERS))
            acc = statistics.mean(1.0 if (r and r.get("correct")) else 0.0 for r in recs.values())
            G._log(f"[test] {name} {G.ckey(c)}: accuracy {acc:.3f} on {len(qids)} (errors counted wrong)")
        return
    if "--dry-run" in rest:
        # Partial splits (make them with --make-splits --partial) and a separate output dir.
        # Budget = |pareto| + 2 minibatches is enough for ONE real critic call, as the BCP dry run.
        if os.path.exists(f"{SPLIT_DIR}/split_partial.json") and not os.path.exists(f"{SPLIT_DIR}/split.json"):
            G.SPLIT_F = f"{SPLIT_DIR}/split_partial.json"
        G.OUT = f"{R}/logs/gepa_frames_dryrun"; G.PROG = f"{G.OUT}/progress.txt"; G.CACHE_F = f"{G.OUT}/cache.json"
        # gepa_bcp's dry-run null bootstrap reads the BrowseComp 830-run file; serve ours instead.
        graded = json.load(open(f"{SEED_RUN}/graded.json"))
        _real_load = json.load

        def _json_load(fh, *x, **k):   # serve the seed run's verdicts where gepa_bcp expects the 830-run file
            if getattr(fh, "name", "") == f"{SUP}/runs/_graded_cache.json":
                return {v["qid"]: v for v in graded.values()}
            return _real_load(fh, *x, **k)
        G.json.load = _json_load
    sys.argv = [sys.argv[0]] + rest
    G.main()


if __name__ == "__main__":
    main()
