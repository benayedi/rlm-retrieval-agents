"""GEPA on HoVer, replicating the paper's setup -- a sanity check of OUR GEPA machinery.

Refactored copy of original/RLM/scripts/hover_gepa.py (studies/prompt_optimisation/CHANGES_FROM_ORIGINAL.md).
Run: hover/experiments/run_hover_gepa.sh (GEPA-5), or python hover/src/hover_gepa.py [--baseline --test |
  --max-metric-calls N | --eval-best --test].
Changed: import wiring (split_runner.py, was gepa_bcp.py; hover_retriever.py next to this file), outputs under
  RUNS_DIR, data from HOVER_DATA_DIR. The reference copy of the GEPA paper's HoVer program named in the docstring
  (data/hover/gepa_artifact_hover_reference.py) is not part of this repository.
Names: 'supervisor harness' / 'full-doc RLM' in the text below is the document REPL agent (agents/document_repl);
  'official RLM harness' / 'our harness' / run_obliq_mine.py is the snippet REPL agent (agents/snippet_repl,
  src/run_obliq.py).

WHY. GEPA with Claude as critic found nothing on BrowseComp+ (run 1: 7 candidates, all within
noise of the seed) or OBLIQ-Twitter. Before blaming the tasks, run the same machinery where the
paper reports large gains (HoVer, Qwen3-8B: 35.33 -> 52.33; GPT-4.1-mini: 46.33 -> 51.67/56.67).
If we reproduce a gain of that size, our pipeline works; if not, something in it is wrong.

WHAT IS THE PAPER'S, VERBATIM (data/hover/gepa_artifact_hover_reference.py, gepa-artifact
cbefbc1): the program HoverMultiHop -- 4 ChainOfThought predictors (summarize1, create_query_hop2,
summarize2, create_query_hop3) with DSPy's default instructions as seeds; retrieval 7 + 7 + 10
over a bm25s index of the 2017 Wikipedia abstracts (k1 0.9, b 0.4, English stemmer); the metric
(binary: every gold title among the 24 retrieved); per-module feedback text; data splits
(train 150 / val 300 / test 300); minibatch 3; Pareto selection on val; round-robin modules;
budget 7,051 metric calls.

WHAT DIFFERS: task LM is Qwen3.6-35B-A3B (paper: Qwen3-8B / GPT-4.1-mini), thinking off,
temperature 0.6 / top_p 0.95 / top_k 20 (the paper's Qwen3-8B settings); prompts are formatted
like DSPy's ChatAdapter by hand rather than by DSPy; the critic is Claude (paper: the task LM).

Usage:  hover_gepa.py --baseline            seed on val (+ test with --test)
        hover_gepa.py [--max-metric-calls N] GEPA run
        hover_gepa.py --eval-best --test     score the best candidate on test
"""
from __future__ import annotations

import argparse, hashlib, json, os, re, statistics, sys, threading, time as _t
from concurrent.futures import ThreadPoolExecutor

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
sys.path.insert(0, DOC_SRC)   # split_runner.py (original: <original repo>/scripts/gepa_bcp.py)
sys.path.insert(0, _HERE)     # hover_retriever.py
os.environ.setdefault("GEPA_BCP_MODEL", "qwen36-35b-a3b")   # only used by the Qwen fallback critic
import split_runner as GB                  # (original: import gepa_bcp as GB) reuse the Claude critic + sanitiser + logging conventions
from hover_retriever import search_remote, normalize_text

OUT = os.environ.get("HOVER_OUT", f"{RUNS}/hover_gepa")
LM_URL = os.environ.get("HOVER_LM_URL", "http://localhost:8030/v1")
LM_MODEL = os.environ.get("HOVER_LM_MODEL", "qwen36-35b-a3b")
RETR = os.environ.get("HOVER_RETRIEVER", "http://127.0.0.1:8770")
MODULES = ["summarize1", "create_query_hop2", "summarize2", "create_query_hop3"]
SIG = {  # inputs -> output, as in the DSPy signature strings
    "summarize1": (["claim", "passages"], "summary"),
    "create_query_hop2": (["claim", "summary_1"], "query"),
    "summarize2": (["claim", "context", "passages"], "summary"),
    "create_query_hop3": (["claim", "summary_1", "summary_2"], "query"),
}
SEED = {m: f"Given the fields {', '.join('`'+f+'`' for f in SIG[m][0])}, produce the fields `{SIG[m][1]}`."
        for m in MODULES}
FINGERPRINT = hashlib.sha1(json.dumps({"lm": LM_MODEL, "k": [7, 7, 10], "think": False,
                                       "samp": [0.6, 0.95, 20]}).encode()).hexdigest()[:8]
_CACHE: dict = {}
_LOCK = threading.Lock()
_STATS = {"lm_calls": 0, "parse_fail": 0, "lm_error": 0, "empty_query": 0, "example_error": 0}


def _log(m):
    os.makedirs(OUT, exist_ok=True)
    line = f"{_t.strftime('%F %T')}  {m}"
    open(f"{OUT}/progress.txt", "a").write(line + "\n"); print(line, flush=True)


HOVER_CRITIC = (
    "You are the reflection step of GEPA, an evolutionary prompt optimiser. The system being optimised "
    "is a 3-hop retrieval program for claim verification over 5M Wikipedia abstracts (BM25 search): "
    "hop 1 searches the claim; summarize1 summarises those passages; create_query_hop2 writes the next "
    "search query; summarize2 summarises again; create_query_hop3 writes the last query. Success = ALL "
    "gold evidence pages retrieved across the hops. You will be shown ONE predictor's current "
    "instruction, its inputs/outputs on a few claims, and feedback naming the evidence each query found "
    "or missed. Propose an improved instruction for that predictor. Rules: write GENERAL strategy that "
    "transfers to unseen claims -- never copy entities, titles or facts from the examples; the search is "
    "lexical BM25, so queries that name the missing bridge entity (a person, work, place or title the "
    "claim refers to indirectly) retrieve its page. Reply with the complete new instruction inside "
    "exactly ONE ``` fenced block, nothing after it.")


def ckey(c): return hashlib.sha1((json.dumps({m: c[m] for m in MODULES}, sort_keys=True) + FINGERPRINT).encode()).hexdigest()[:10]


# ------------------------------------------------------------------ the program ---------------
def _fmt(v):
    if isinstance(v, list):
        return "\n".join(f"[{i+1}] «{x}»" for i, x in enumerate(v))
    return str(v)


def predict(module: str, instr: str, **inputs) -> str:
    """One ChainOfThought call, ChatAdapter-shaped: reasoning then the output field."""
    from openai import OpenAI
    ins, out = SIG[module]
    system = ("Your input fields are:\n" + "\n".join(f"{i+1}. `{f}`" for i, f in enumerate(ins)) +
              "\nYour output fields are:\n1. `reasoning`\n2. `" + out + "`\n"
              "All interactions will be structured in the following way, with the appropriate values filled in.\n\n" +
              "".join(f"[[ ## {f} ## ]]\n{{{f}}}\n\n" for f in ins) +
              f"[[ ## reasoning ## ]]\n{{reasoning}}\n\n[[ ## {out} ## ]]\n{{{out}}}\n\n[[ ## completed ## ]]\n"
              "In adhering to this structure, your objective is: \n        " + instr)
    user = "".join(f"[[ ## {f} ## ]]\n{_fmt(inputs[f])}\n\n" for f in ins) + (
        f"Respond with the corresponding output fields, starting with the field `[[ ## reasoning ## ]]`, "
        f"then `[[ ## {out} ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.")
    c = OpenAI(base_url=LM_URL, api_key="EMPTY", timeout=300)
    for attempt in range(3):
        try:
            r = c.chat.completions.create(
                model=LM_MODEL, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.6, top_p=0.95, max_tokens=2048,
                extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}})
            txt = r.choices[0].message.content or ""
            with _LOCK: _STATS["lm_calls"] += 1
            m = re.search(r"\[\[ ## " + out + r" ## \]\]\s*(.*?)(?:\[\[ ## completed ## \]\]|$)", txt, re.S)
            if m and m.group(1).strip():
                return m.group(1).strip()
            with _LOCK: _STATS["parse_fail"] += 1
            return txt.strip()[-500:]          # DSPy would fail; we degrade to the tail and count it
        except Exception:
            with _LOCK: _STATS["lm_error"] += 1
            _t.sleep(2 * (attempt + 1))
    return ""


def run_program(cand: dict, claim: str) -> dict:
    srch = lambda q, k: [f"{d['title']} | {d['text']}" for d in search_remote(q or claim, k, RETR)]
    h1 = srch(claim, 7)
    s1 = predict("summarize1", cand["summarize1"], claim=claim, passages=h1)
    q2 = predict("create_query_hop2", cand["create_query_hop2"], claim=claim, summary_1=s1)
    h2 = srch(q2, 7)
    s2 = predict("summarize2", cand["summarize2"], claim=claim, context=s1, passages=h2)
    q3 = predict("create_query_hop3", cand["create_query_hop3"], claim=claim, summary_1=s1, summary_2=s2)
    h3 = srch(q3, 10)
    if not q2.strip() or not q3.strip():
        with _LOCK: _STATS["empty_query"] += 1
    return {"hops": [h1, h2, h3], "summary_1": s1, "summary_2": s2, "q2": q2, "q3": q3}


def titles(docs): return {normalize_text(d.split(" | ")[0]) for d in docs}


def score_rec(ex: dict, tr: dict) -> dict:
    gold = {normalize_text(t) for t in ex["gold_titles"]}
    a1 = titles(tr["hops"][0]); a2 = a1 | titles(tr["hops"][1]); a3 = a2 | titles(tr["hops"][2])
    return {"id": ex["id"], "score": float(gold <= a3), "recall": len(gold & a3) / len(gold),
            "after": [sorted(gold & a1), sorted(gold & a2), sorted(gold & a3)], "missed": sorted(gold - a3),
            "q2": tr["q2"], "q3": tr["q3"], "summary_1": tr["summary_1"][:600], "summary_2": tr["summary_2"][:600]}


def feedback(ex: dict, rec: dict) -> str:
    """The paper's per-module feedback (query + summary modules), plus the trace the critic needs."""
    if rec["score"]:
        return ("Your queries and summaries are correct and useful: all evidence documents "
                "were retrieved.")
    gold = {normalize_text(t) for t in ex["gold_titles"]}
    a1, a2, a3 = (set(x) for x in rec["after"])
    h2_help = sorted((gold - a1) - (gold - a2)); h2_miss = sorted((gold - a1) & (gold - a2))
    h3_help = sorted((gold - a2) - (gold - a3)); h3_miss = sorted((gold - a2) & (gold - a3))
    return "\n".join([
        f"Claim: {ex['claim']}",
        f"Found after hop1 (claim search): {sorted(a1) or 'none'} | missed at the end: {rec['missed']}",
        f"Hop-2 query: {rec['q2']!r} -> helped retrieve {h2_help or 'nothing new'}; still missing {h2_miss or 'none'}",
        f"Hop-3 query: {rec['q3']!r} -> helped retrieve {h3_help or 'nothing new'}; still missing {h3_miss or 'none'}",
        f"summary_1 (truncated): {rec['summary_1'][:300]}",
        f"summary_2 (truncated): {rec['summary_2'][:300]}",
        "Think about how you can modify your query to make the connection between the provided summary "
        "and the missed evidence relevant to the claim; and how the summaries can surface the bridge "
        "entities (names, titles) that the next query needs."])


def run_batch(cand: dict, exs: list[dict], workers: int) -> dict:
    k = ckey(cand)
    # records flagged "error" (infrastructure, e.g. a reset retriever connection) are re-run, not reused
    todo = [e for e in exs if f"{k}|{e['id']}" not in _CACHE or _CACHE[f"{k}|{e['id']}"].get("error")]

    def save():
        with _LOCK:
            json.dump(_CACHE, open(f"{OUT}/cache.json.tmp", "w")); os.replace(f"{OUT}/cache.json.tmp", f"{OUT}/cache.json")

    done = [0]

    def one(e):
        # One bad example must not crash the batch (2026-10-08: an unlogged exception in one of
        # 24 threads killed the whole 300-claim baseline). Errors are logged with traceback and
        # recorded with score 0 + error flag -- infrastructure, so they are counted and alerted on.
        try:
            rec = score_rec(e, run_program(cand, e["claim"]))
        except Exception:
            import traceback
            tb = traceback.format_exc()
            open(f"{OUT}/errors.log", "a").write(f"{_t.strftime('%F %T')} {k} {e['id']}\n{tb}\n")
            with _LOCK: _STATS["example_error"] = _STATS.get("example_error", 0) + 1
            rec = {"id": e["id"], "score": 0.0, "recall": 0.0, "after": [[], [], []], "missed": [],
                   "q2": "", "q3": "", "summary_1": "", "summary_2": "", "error": tb.splitlines()[-1][:200]}
        with _LOCK:
            _CACHE[f"{k}|{e['id']}"] = rec; done[0] += 1
        if done[0] % 25 == 0:
            save()

    if todo:
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(one, todo))
        save()
    return {e["id"]: _CACHE[f"{k}|{e['id']}"] for e in exs}


def make_evaluator(workers, by_id):
    st = {"b": 0, "t0": _t.time()}

    def evaluator(pairs, **kw):
        groups, cands = {}, {}
        for i, (c, ex) in enumerate(pairs):
            k = ckey(c); groups.setdefault(k, []).append(i); cands[k] = c
        out = [None] * len(pairs)
        for k, idxs in groups.items():
            exs = [by_id[pairs[i][1]["id"]] for i in idxs]
            s0 = dict(_STATS)
            recs = run_batch(cands[k], exs, workers)
            for i in idxs:
                e = by_id[pairs[i][1]["id"]]; r = recs[e["id"]]
                out[i] = (r["score"], {"feedback": feedback(e, r)})
            st["b"] += 1
            d = {x: _STATS[x] - s0[x] for x in _STATS}
            _log(f"batch {st['b']:>3} | cand {k} | n={len(idxs)} | all-found {statistics.mean(recs[e['id']]['score'] for e in exs):.3f} "
                 f"recall {statistics.mean(recs[e['id']]['recall'] for e in exs):.3f} | lm_calls {d['lm_calls']} "
                 f"parse_fail {d['parse_fail']} lm_err {d['lm_error']} empty_q {d['empty_query']} ex_err {d['example_error']} | "
                 f"{(_t.time()-st['t0'])/60:.0f} min")
        return out
    return evaluator


def main():
    global _CACHE
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--eval-best", action="store_true")
    ap.add_argument("--max-metric-calls", type=int, default=7051)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0, help="baseline/eval on the first N of each split only")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    GB.OUT = OUT; GB.PROG = f"{OUT}/progress.txt"     # critic logs (reflection_log.jsonl) go here too
    GB.CRITIC_SYSTEM = HOVER_CRITIC                   # GB._claude reads it at call time
    if os.path.exists(f"{OUT}/cache.json"):
        _CACHE = json.load(open(f"{OUT}/cache.json"))
    data = json.load(open(f"{path('HOVER_DATA_DIR')}/splits.json"))
    by_id = {d["id"]: d for d in data}
    split = {s: [d for d in data if d["split"] == s] for s in ("train", "val", "test")}
    _log(f"[setup] lm {LM_MODEL} @ {LM_URL} | retriever {RETR} | fingerprint {FINGERPRINT} | "
         f"train {len(split['train'])} val {len(split['val'])} test {len(split['test'])} | seed key {ckey(SEED)}")

    if a.baseline or a.eval_best:
        cand = SEED
        if a.eval_best:
            cand = json.load(open(f"{OUT}/best_candidate.json"))
        for s in (["val"] + (["test"] if a.test else [])):
            recs = run_batch(cand, split[s][: a.limit or None], a.workers)
            v = list(recs.values())
            _log(f"[{'best' if a.eval_best else 'baseline'}] {s}: all-found {statistics.mean(r['score'] for r in v):.4f} "
                 f"recall {statistics.mean(r['recall'] for r in v):.4f} (n={len(v)}) | stats {_STATS}")
        return

    import gepa.optimize_anything as OA
    res = OA.optimize_anything(
        seed_candidate=SEED,
        dataset=[{"id": e["id"]} for e in split["train"]],
        valset=[{"id": e["id"]} for e in split["val"]],
        batch_evaluator=make_evaluator(a.workers, by_id),
        config=OA.GEPAConfig(
            engine=OA.EngineConfig(max_metric_calls=a.max_metric_calls, candidate_selection_strategy="pareto",
                                   run_dir=OUT, raise_on_exception=False),
            tracking=OA.TrackingConfig(logger=OA.StdOutLogger()),
            reflection=OA.ReflectionConfig(reflection_lm=GB.reflection_lm, reflection_minibatch_size=3,
                                           module_selector="round_robin", skip_perfect_score=True,
                                           perfect_score=1.0)))
    best = getattr(res, "best_candidate", None) or SEED
    json.dump(best, open(f"{OUT}/best_candidate.json", "w"), indent=1)
    _log(f"[done] best {ckey(best)} (seed {ckey(SEED)}). Next: --eval-best --test")


if __name__ == "__main__":
    main()
