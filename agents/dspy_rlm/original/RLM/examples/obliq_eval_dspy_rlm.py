"""
OBLIQ-Bench evaluation for the RAG RLM (dspy.RLM + FAISS).

Supports all three OBLIQ task types:
  tip-of-tongue  → congress
  analogues      → math, writing
  descriptive    → twitter, wildchat

Usage:
    # tip-of-tongue (congressional hearing clips)
    EMBEDDING_SERVER_URL=http://HOST:PORT \\
    python examples/obliq_eval_dspy_rlm.py \\
        --task tip-of-tongue --subset congress \\
        --start 0 --end 30 \\
        --out logs/obliq_tot_congress_q0_q30

    # analogues (math problems with same reasoning pattern)
    EMBEDDING_SERVER_URL=http://HOST:PORT \\
    python examples/obliq_eval_dspy_rlm.py \\
        --task analogues --subset math \\
        --start 0 --end 20 \\
        --out logs/obliq_ana_math_q0_q20

Grading:
  Primary   — retrieval recall: fraction of gold doc_ids that appeared
               in the agent's search trajectory (metrics.retrieved_doc_ids).
  Secondary — answer_has_gold: whether any gold doc_id appears verbatim
               in the agent's final answer string.

The embedding server must already be running and pointed at an index
built from the matching OBLIQ corpus (e.g. tip-of-tongue/congress/corpus).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import dspy
import requests as _requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import RemoteRetriever, RunMetrics, build_lm
from rag.signatures import (
    ObliqCongressSig,
    ObliqCongressV4Sig,
    ObliqCongressV5Sig,
    ObliqCongressV6Sig,
    ObliqCongressV7Sig,
    ObliqCongressV8Sig,
    ObliqMathSig,
    ObliqTwitterSig,
    ObliqTwitterV2Sig,
    ObliqTwitterBatch1Sig,
    ObliqMathRerankSig,
    ObliqMathOverviewSig,
    ObliqMathBatchSig,
    ObliqWildchatV5Sig,
    ObliqWildchatV6Sig,
    ObliqWildchatV8Sig,
    ObliqWildchatV9Sig,
    ObliqWritingSig,
    ReaderSignature,
)

load_dotenv()

OBLIQ_ROOT = Path(__file__).parent.parent.parent / "OBLIQ-Bench"

NDCG_KS = [1, 3, 5, 10, 20, 50]


def ndcg_at(retrieved_ordered: list[str], gold_ids: set[str], k: int) -> float:
    """NDCG@k where retrieved_ordered is deduplicated in first-retrieval order."""
    if not gold_ids:
        return 0.0
    dcg  = sum(1.0 / math.log2(i + 2) for i, d in enumerate(retrieved_ordered[:k]) if d in gold_ids)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_ids), k)))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at(retrieved_ordered: list[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    return sum(1 for d in retrieved_ordered[:k] if d in gold_ids) / len(gold_ids)



def extract_doc_ids_from_answer(answer: str, subset: str = "") -> list[str]:
    """Extract ordered doc_ids from the agent's ranked answer string.

    Handles all corpus ID formats:
      Math         : american-math-monthly___1998___10583  (triple underscore)
      Congress     : CHRG-116hhrg41317_p0001_09            (single underscore, CHRG- prefix)
      Twitter      : 2034254279364489632                   (pure numeric, 15-19 digits)
      Writing      : 4, 9, 13                              (small integers, 1-5 digits)
      Wildchat     : numeric or alphanumeric IDs
    """
    import re
    math_ids     = re.findall(r"[\w][\w\-]*(?:___[\w\.\-]+)+", answer)
    congress_ids = re.findall(r"CHRG-\w+(?:_\w+)*", answer)
    twitter_ids  = re.findall(r"\b\d{15,19}\b", answer)
    # Writing corpus uses plain small integers (1–10389). Only activate for
    # writing to avoid false matches in other subsets.
    writing_ids  = re.findall(r"\b\d{1,5}\b", answer) if subset == "writing" else []
    # Wildchat corpus uses integers 0–507729 (1–6 digits).
    wildchat_ids = re.findall(r"\b\d{1,6}\b", answer) if subset in ("wildchat", "wildchat-v6", "wildchat-v8", "wildchat-v9") else []
    return list(dict.fromkeys(math_ids + congress_ids + twitter_ids + writing_ids + wildchat_ids))

# Keyed by subset name (unique across all tasks)
SUBSET_SIGNATURES = {
    "congress":     ObliqCongressSig,
    "congress-v4":  ObliqCongressV4Sig,
    "congress-v5":  ObliqCongressV5Sig,
    "congress-v6":  ObliqCongressV6Sig,
    "congress-v7":  ObliqCongressV7Sig,
    "congress-v8":  ObliqCongressV8Sig,
    "math":         ObliqMathSig,
    "math-rerank":  ObliqMathRerankSig,
    "math-overview":ObliqMathOverviewSig,
    "math-batch":   ObliqMathBatchSig,
    "writing":      ObliqWritingSig,
    "twitter":      ObliqTwitterSig,
    "twitter-v2":   ObliqTwitterV2Sig,
    "twitter-b1":   ObliqTwitterBatch1Sig,
    "wildchat":     ObliqWildchatV5Sig,
    "wildchat-v6":  ObliqWildchatV6Sig,
    "wildchat-v8":  ObliqWildchatV8Sig,
    "wildchat-v9":  ObliqWildchatV9Sig,
}

# Maps subset key → actual data directory name (for ablation variants)
SUBSET_DATA = {
    "congress-v4":  "congress",
    "congress-v5":  "congress",
    "congress-v6":  "congress",
    "congress-v7":  "congress",
    "congress-v8":  "congress",
    "twitter-v2":   "twitter",
    "twitter-b1":   "twitter",
    "math-rerank":  "math",
    "math-overview":"math",
    "math-batch":   "math",
    "wildchat-v6":  "wildchat",
    "wildchat-v8":  "wildchat",
    "wildchat-v9":  "wildchat",
}

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_queries(task: str, subset: str) -> list[dict]:
    path = OBLIQ_ROOT / task / subset / "queries+qrels" / "queries.jsonl"
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_qrels(task: str, subset: str, pool: bool = False) -> dict[str, set[str]]:
    """Returns {query_id: {corpus_id, ...}}.

    pool=True loads qrels_pool.tsv (expanded judged pool) instead of qrels.tsv.
    Falls back to qrels.tsv if pool file not found.
    """
    fname = "qrels_pool.tsv" if pool else "qrels.tsv"
    path = OBLIQ_ROOT / task / subset / "queries+qrels" / fname
    if pool and not path.exists():
        path = OBLIQ_ROOT / task / subset / "queries+qrels" / "qrels.tsv"
        print(f"[warn] qrels_pool.tsv not found, falling back to qrels.tsv")
    qrels: dict[str, set[str]] = defaultdict(set)
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                qid, cid = parts[0], parts[1]
                qrels[qid].add(cid)
    return dict(qrels)


def load_excluded_ids(task: str, subset: str) -> dict[str, list[str]]:
    """Load per_query_excluded_ids.json (only present for analogues)."""
    path = OBLIQ_ROOT / task / subset / "queries+qrels" / "per_query_excluded_ids.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# RLM agent — thin wrapper that accepts a custom signature
# ─────────────────────────────────────────────────────────────────────────────

# DSPY_FORMAT_FIX=1 (2026-10-21, Qwen3.6): dspy.RLM wants every turn as two fields, `reasoning`
# and `code`. Qwen3.6 sometimes writes only `reasoning`; dspy then raises AdapterParseError, which
# ends the WHOLE question with score 0 (18/171 questions on the first Qwen3.6 run; Qwen3.8-27B 1/100).
# Same idea as the snippet RLM's Qwen3.6 fixes (docs/HANDOFF.md Part I §F): one sentence in the
# action instructions naming both fields, and a turn whose reply cannot be parsed is retried
# (2 more tries), then recorded as an error turn so the run continues. Unset = dspy unchanged.
if os.environ.get("DSPY_FORMAT_FIX") == "1":
    import dspy.predict.rlm as _drlm
    from dspy.utils.exceptions import AdapterParseError as _APE

    _drlm.ACTION_INSTRUCTIONS_TEMPLATE += (
        "\n\nEvery response must contain BOTH output fields: `reasoning` (your thinking) and `code` "
        "(the Python to run this step). A response without `code` is lost. Put candidate lists and "
        "rankings in `code`, and submit with SUBMIT(...) inside `code`, never in `reasoning`.")
    _orig_execute_iteration = _drlm.RLM._execute_iteration

    def _execute_iteration(self, *a, **kw):
        if not getattr(self, "_format_retry", False):
            gen = self.generate_action

            def _retrying(**inputs):
                for attempt in range(3):
                    try:
                        return gen(**inputs)
                    except _APE:
                        print(f"    [format] reply missing a field, attempt {attempt + 1}/3", flush=True)
                return dspy.Prediction(reasoning="", code=(
                    'print("[Error] Your last reply had no `code` field and was discarded. '
                    'Reply with both `reasoning` and `code`.")'))
            self.generate_action = _retrying
            self._format_retry = True
        return _orig_execute_iteration(self, *a, **kw)

    _drlm.RLM._execute_iteration = _execute_iteration
    print("[format] dspy reasoning/code cue + parse retry ON", flush=True)


def _build_sub_lm(metrics):
    """Build a fast cheap sub-LM (gpt-5.4-mini) for llm_query calls.
    Wraps the LM in a proxy class so __call__ interception works correctly
    (Python's call protocol uses type.__call__, not instance.__call__).
    """
    import os
    # Default to the SAME model as the agent. This used to fall back to
    # gpt-5.4-mini, which silently handed every llm_query judgement to a much
    # weaker model — confounding "structure hurts" with "we delegated to mini"
    # across the whole v4-v7 line of work. Set AZURE_OPENAI_JUDGE_MODEL to
    # override (e.g. gpt-5.4-mini for cheap grading).
    judge = os.environ.get("AZURE_OPENAI_JUDGE_MODEL") \
            or os.environ.get("AZURE_OPENAI_MODEL", "gpt-5.4-mini")
    # LOCAL_SUB_LM=1 with LOCAL_LLM_BASE_URL: llm_query runs on the same local model as the agent
    # (the 2026-09-27 Qwen3.8 runs still sent it to Azure gpt-5.4-mini). Unset = unchanged.
    if os.environ.get("LOCAL_SUB_LM") == "1" and os.environ.get("LOCAL_LLM_BASE_URL"):
        base_lm = build_lm()
    else:
        # SUB_LM_REASONING_EFFORT (2026-10-27): pass an explicit effort to llm_query, as the snippet harness does.
        # Reasoning tokens count against max_tokens, so the 4k cap rises to 16k and the timeout to 300 s; unset =
        # unchanged (no effort sent, 4k, 60 s).
        _sub_effort = os.environ.get("SUB_LM_REASONING_EFFORT", "")
        _sub_extra = {"reasoning_effort": _sub_effort} if _sub_effort else {}
        base_lm = dspy.LM(
            model=f"azure/{judge}",
            api_base=os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/"),
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            temperature=1.0,
            max_tokens=16000 if _sub_effort else 4000,
            cache=False,
            timeout=300 if _sub_effort else 60,
            **_sub_extra,
        )

    class _CountedLM(type(base_lm)):
        def __call__(self, *args, **kwargs):
            metrics.llm_query_calls += 1
            print(f"    [llm_query #{metrics.llm_query_calls}]")
            return super().__call__(*args, **kwargs)

    base_lm.__class__ = _CountedLM
    return base_lm


def _build_agent_with_sig(signature, retriever, max_iterations, max_search_calls,
                          verbose, default_top_k, metrics, include_get_document=True,
                          include_rlm_query=False):
    """Build dspy.RLM with the given signature (reuses agent tool logic)."""
    import requests as req

    def search_index(query: str, top_k: int = default_top_k) -> list[dict]:
        """Search the OBLIQ corpus for relevant documents.

        Returns list of {score: float, doc_id: str, text: str (≤2000 chars)}.
        Search for specific named entities, titles, or proper nouns.
        Tip: scores below 0.30 are usually off-topic — try a different query.
        """
        if metrics.search_calls >= max_search_calls:
            return [{"score": 0.0, "doc_id": "", "text": (
                f"[SEARCH DISABLED — all {max_search_calls} calls used. "
                "Call SUBMIT(answer=...) immediately."
            )}]

        if query in metrics._query_cache:
            return metrics._query_cache[query]

        t0 = time.perf_counter()
        # Retry a mid-question search rather than killing the whole run: a server
        # restart or a slow batch used to abort here, discarding every question
        # already completed. Only give up if it stays down for ~5 minutes.
        results = None
        for _attempt in range(6):
            try:
                results = retriever.search_index(query, top_k=top_k)
                break
            except (req.exceptions.ConnectionError, req.exceptions.Timeout) as e:
                if _attempt == 5:
                    print(f"\n[ABORT] Embedding server unreachable after 6 attempts "
                          f"over ~5min: {e}")
                    sys.exit(1)
                print(f"    [warn] search failed ({type(e).__name__}); "
                      f"retry {_attempt+1}/5 in 60s")
                time.sleep(60)
        elapsed = time.perf_counter() - t0

        metrics.retrieved_doc_ids.extend(r["doc_id"] for r in results)
        snippet_len = int(os.environ.get("SEARCH_SNIPPET_LEN", "500"))
        for r in results:
            if "text" in r and len(r["text"]) > snippet_len:
                r["text"] = r["text"][:snippet_len] + "…"
        metrics.retrieval_latency_s += elapsed
        metrics.search_calls += 1
        metrics._query_cache[query] = results

        gold_hits = sum(1 for r in results if r["doc_id"] in metrics._gold_ids)
        hit_str = f"  ★{gold_hits}" if gold_hits else ""
        remaining = max_search_calls - metrics.search_calls
        print(f"    [search {metrics.search_calls}/{max_search_calls}  {remaining} remaining] "
              f"{query[:60]!r}  → {len(results)} docs{hit_str}  ({elapsed*1000:.0f}ms)")
        return results

    def get_document(doc_id: str) -> dict:
        """Fetch the full text of a document by doc_id."""
        t0 = time.perf_counter()
        result = retriever.get_document(doc_id)
        elapsed = time.perf_counter() - t0
        metrics.retrieval_latency_s += elapsed
        metrics.get_document_calls += 1
        print(f"    [getdoc #{metrics.get_document_calls}] {doc_id}  ({elapsed*1000:.0f}ms)")
        return result

    tools = [search_index, get_document] if include_get_document else [search_index]

    if include_rlm_query:
        _reader_sig = ReaderSignature
        _inner_max_iters = 12

        def rlm_query(prompt: str) -> str:
            """Spawn a sub-RLM with get_document access to read and verify documents.

            Pass a prompt describing the task and the candidate doc_ids to verify.
            The sub-RLM fetches each document, reads it, and returns
            a JSON object mapping doc_id to relevance score (0.0-1.0).
            """
            reader = dspy.RLM(
                signature=_reader_sig,
                tools=[get_document],
                max_iters=_inner_max_iters,   # dspy >=3.3 renamed max_iterations
                verbose=verbose,
                sub_lm=_build_sub_lm(metrics),
            )
            result = reader(question=prompt)
            metrics.rlm_query_calls += 1
            print(f"    [rlm_query #{metrics.rlm_query_calls} done]")
            return getattr(result, "answer", str(result))

        tools.append(rlm_query)

    rlm = dspy.RLM(
        signature=signature,
        tools=tools,
        max_iters=max_iterations,   # dspy >=3.3 renamed max_iterations
        verbose=verbose,
        sub_lm=_build_sub_lm(metrics),
    )
    return rlm



def run_obliq_question(
    retriever,
    question: str,
    signature,
    gold_ids: set[str],
    max_iterations: int = 12,
    max_search_calls: int = 8,
    verbose: bool = False,
    default_top_k: int = 20,
    include_get_document: bool = True,
    include_rlm_query: bool = False,
) -> tuple[str, RunMetrics]:
    metrics = RunMetrics()
    metrics._gold_ids = gold_ids

    rlm = _build_agent_with_sig(
        signature=signature,
        retriever=retriever,
        max_iterations=max_iterations,
        max_search_calls=max_search_calls,
        verbose=verbose,
        default_top_k=default_top_k,
        metrics=metrics,
        include_get_document=include_get_document,
        include_rlm_query=include_rlm_query,
    )

    from dspy.utils.usage_tracker import track_usage

    # Profiling: dspy.RLM is a black box that alternates Azure API calls with Deno
    # REPL execution, so total latency alone cannot say which dominates. Wrap the
    # LM's __call__ to accumulate time blocked on the API. Purely additive — the
    # wrapper is restored in `finally`, and a failure to install it degrades to the
    # old behaviour (lm_latency_s stays 0) rather than killing the run.
    _lm = getattr(rlm, "lm", None) or dspy.settings.lm
    _orig_call = None
    if _lm is not None:
        _orig_call = type(_lm).__call__

        # Depth guard: patching the CLASS also catches the sub-LM used by
        # llm_query, and dspy retries by re-entering __call__. Without this, a
        # nested call is billed twice and lm_latency_s can exceed wall-clock.
        # Only the outermost call is charged, which is the true blocked time.
        _depth = [0]

        def _timed_call(self, *a, **kw):
            if _depth[0]:
                return _orig_call(self, *a, **kw)
            _depth[0] = 1
            _t = time.perf_counter()
            try:
                return _orig_call(self, *a, **kw)
            finally:
                _depth[0] = 0
                _d = time.perf_counter() - _t
                metrics.lm_latency_s += _d
                metrics.lm_calls += 1
                metrics.lm_call_latencies.append(round(_d, 3))

        try:
            type(_lm).__call__ = _timed_call
        except Exception:
            _orig_call = None

    with track_usage() as tracker:
        t0 = time.time()
        try:
            result = rlm(question=question)
        finally:
            metrics.latency_s = time.time() - t0
            if _orig_call is not None:
                type(_lm).__call__ = _orig_call

    for usage in tracker.get_total_tokens().values():
        metrics.input_tokens += usage.get("prompt_tokens", 0)
        metrics.output_tokens += usage.get("completion_tokens", 0)

    trajectory = getattr(result, "trajectory", None) or {}
    if isinstance(trajectory, dict):
        metrics.iterations = len([k for k in trajectory if k.startswith("iteration_")])
    elif hasattr(trajectory, "__len__"):
        metrics.iterations = len(trajectory)

    # Keep the transcript, not just its length: this is the only record of what the
    # agent actually wrote and saw, which is what a prompt-strategy result rests on.
    try:
        if isinstance(trajectory, dict):
            metrics.trajectory = [{"key": k, "value": v} for k, v in trajectory.items()]
        else:
            metrics.trajectory = list(trajectory)
    except Exception:
        metrics.trajectory = []

    answer = getattr(result, "answer", str(result))
    return str(answer), metrics


# ─────────────────────────────────────────────────────────────────────────────
# Per-question runner
# ─────────────────────────────────────────────────────────────────────────────

def run_one(
    retriever,
    query: dict,
    q_num: int,
    gold_ids: set[str],
    signature,
    max_iterations: int = 12,
    max_search_calls: int = 8,
    verbose: bool = False,
    default_top_k: int = 20,
    timeout: int = 0,
    subset: str = "",
    excluded_ids: set[str] | None = None,
    include_get_document: bool = True,
    include_rlm_query: bool = False,
) -> dict:
    question = query["text"]
    q_id = query["_id"]

    print(f"\n[Q{q_num:03d}|{q_id}] {question[:100]}…")
    if gold_ids:
        preview = list(gold_ids)[:3]
        print(f"  [gold docs: {preview}{'...' if len(gold_ids) > 3 else ''}]")

    # Wrap retriever to filter same-source snippets (writing/math analogues tasks)
    if excluded_ids:
        class _FilteredRetriever:
            def search_index(self, query, top_k=10):
                results = retriever.search_index(query, top_k=top_k)
                return [r for r in results if str(r.get("doc_id", "")) not in excluded_ids]
            def get_document(self, doc_id):
                return retriever.get_document(doc_id)
        active_retriever = _FilteredRetriever()
    else:
        active_retriever = retriever

    if timeout > 0:
        import threading
        result_box = {}
        def _run():
            try:
                result_box["ok"] = run_obliq_question(
                    retriever=active_retriever,
                    question=question,
                    signature=signature,
                    gold_ids=gold_ids,
                    max_iterations=max_iterations,
                    max_search_calls=max_search_calls,
                    verbose=verbose,
                    default_top_k=default_top_k,
                    include_get_document=include_get_document,
                    include_rlm_query=include_rlm_query,
                )
            except Exception as e:
                result_box["err"] = e
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            print(f"  [TIMEOUT] timed out after {timeout}s")
            predicted = "TIMEOUT"
            metrics = RunMetrics(latency_s=float(timeout))
        elif "err" in result_box:
            import traceback
            traceback.print_exc()
            print(f"  ERROR: {result_box['err']}")
            predicted = "ERROR"
            metrics = RunMetrics(latency_s=0.0)
        else:
            predicted, metrics = result_box["ok"]
    else:
        try:
            predicted, metrics = run_obliq_question(
                retriever=active_retriever,
                question=question,
                signature=signature,
                gold_ids=gold_ids,
                max_iterations=max_iterations,
                max_search_calls=max_search_calls,
                verbose=verbose,
                default_top_k=default_top_k,
                include_get_document=include_get_document,
                include_rlm_query=include_rlm_query,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ERROR: {e}")
            predicted = "ERROR"
            metrics = RunMetrics(latency_s=0.0)

    # Deduplicate while preserving first-retrieval order (needed for NDCG)
    retrieved_ordered = list(dict.fromkeys(metrics.retrieved_doc_ids))
    gold_recall = metrics.gold_recall(list(gold_ids))

    # Trajectory-based metrics (first-retrieval order — not a ranked list)
    ndcg     = {k: ndcg_at(retrieved_ordered, gold_ids, k) for k in NDCG_KS}
    recall_k = {k: recall_at(retrieved_ordered, gold_ids, k) for k in NDCG_KS}

    # Answer-based metrics (agent's explicit ranking — comparable to paper baselines)
    answer_doc_ids = extract_doc_ids_from_answer(predicted, subset=subset) if predicted not in ("TIMEOUT", "ERROR") else []
    answer_ndcg     = {k: ndcg_at(answer_doc_ids, gold_ids, k) for k in NDCG_KS}
    answer_recall_k = {k: recall_at(answer_doc_ids, gold_ids, k) for k in NDCG_KS}

    # Tail-fill: agent's ranked answer + all remaining trajectory docs (mirrors paper Multi-Hop)
    answer_set = set(answer_doc_ids)
    tail_doc_ids = answer_doc_ids + [d for d in retrieved_ordered if d not in answer_set]
    tail_ndcg     = {k: ndcg_at(tail_doc_ids, gold_ids, k) for k in NDCG_KS}
    tail_recall_k = {k: recall_at(tail_doc_ids, gold_ids, k) for k in NDCG_KS}

    sym = "★" if gold_recall > 0 else "○"
    print(
        f"  [{sym}] recall={gold_recall:.2f}"
        f"  NDCG@10 traj={ndcg[10]:.3f} ans={answer_ndcg[10]:.3f} tail={tail_ndcg[10]:.3f}"
        f"  iters={metrics.iterations}  search={metrics.search_calls}"
        f"  getdoc={metrics.get_document_calls}  llmq={metrics.llm_query_calls}"
        f"  docs={metrics.unique_docs_retrieved}"
        f"  lat={metrics.latency_s:.1f}s  tok={metrics.input_tokens}/{metrics.output_tokens}"
        f"  [lm={metrics.lm_latency_s:.0f}s retr={metrics.retrieval_latency_s:.1f}s "
        f"repl={max(0.0, metrics.latency_s - metrics.lm_latency_s - metrics.retrieval_latency_s):.0f}s]"
    )
    if predicted not in ("TIMEOUT", "ERROR"):
        print(f"  [answer doc_ids: {len(answer_doc_ids)}] {predicted[:120]!r}")

    return {
        "q_num": q_num,
        "q_id": q_id,
        "question": question,
        "gold_ids": sorted(gold_ids),
        "predicted": predicted,
        "gold_recall": gold_recall,
        "ndcg": ndcg,
        "recall_k": recall_k,
        "answer_doc_ids": answer_doc_ids,
        "answer_ndcg": answer_ndcg,
        "answer_recall_k": answer_recall_k,
        "tail_doc_ids": tail_doc_ids,
        "tail_ndcg": tail_ndcg,
        "tail_recall_k": tail_recall_k,
        "retrieved_doc_ids": retrieved_ordered,
        # REPL transcript: the code written each turn and what it printed. Set
        # OBLIQ_LOG_TRAJECTORY=0 to omit it if the JSONs get unwieldy.
        "trajectory": (metrics.trajectory
                       if os.environ.get("OBLIQ_LOG_TRAJECTORY", "1") != "0" else []),
        "latency_s": round(metrics.latency_s, 2),
        "retrieval_latency_s": round(metrics.retrieval_latency_s, 2),
        "lm_latency_s": round(metrics.lm_latency_s, 2),
        "lm_calls": metrics.lm_calls,
        "lm_call_latencies": metrics.lm_call_latencies,
        "repl_latency_s": round(
            max(0.0, metrics.latency_s - metrics.lm_latency_s - metrics.retrieval_latency_s), 2),
        "input_tokens": metrics.input_tokens,
        "output_tokens": metrics.output_tokens,
        "iterations": metrics.iterations,
        "search_calls": metrics.search_calls,
        "get_document_calls": metrics.get_document_calls,
        "llm_query_calls": metrics.llm_query_calls,
        "rlm_query_calls": metrics.rlm_query_calls,
        "unique_docs": metrics.unique_docs_retrieved,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_eval(args) -> None:
    task   = args.task
    subset = args.subset

    lm = build_lm()
    dspy.configure(lm=lm)

    signature = SUBSET_SIGNATURES[subset]
    # OBLIQ_RELATIVE_SCORES=1 (2026-10-27, Gemini runs): no fixed score cutoffs, the model judges relevance itself.
    from examples.obliq_score_advice import ENABLED as _REL_SCORES, relative as _relative
    if _REL_SCORES:
        signature = signature.with_instructions(_relative(signature.instructions))
        print("[prompt] score cutoffs replaced by relative-relevance advice (OBLIQ_RELATIVE_SCORES=1)")
    data_subset = SUBSET_DATA.get(subset, subset)
    use_pool = getattr(args, "pool", False)
    print(f"Task    : {task}/{subset}  (data: {data_subset})")
    print(f"Sig     : {signature.__name__}")
    print(f"QRels   : {'pooled (qrels_pool.tsv)' if use_pool else 'gold (qrels.tsv)'}")

    queries = load_queries(task, data_subset)
    qrels   = load_qrels(task, data_subset, pool=use_pool)
    all_excluded = load_excluded_ids(task, data_subset)

    retriever = RemoteRetriever(server_url=os.environ["EMBEDDING_SERVER_URL"])

    if args.indices:
        index_list = [int(x.strip()) for x in args.indices.split(",")]
        rows = [(i, queries[i]) for i in index_list]
    else:
        end = args.end if args.end > 0 else len(queries)
        rows = [(i, queries[i]) for i in range(args.start, min(end, len(queries)))]

    print(f"Running {len(rows)} questions (total available: {len(queries)})")
    print(f"Server  : {os.environ['EMBEDDING_SERVER_URL']}")

    results: list[dict] = []
    for run_i, (q_idx, query) in enumerate(rows):
        # Health-check server before each question
        server_url = os.environ["EMBEDDING_SERVER_URL"].rstrip("/")
        # Wait out a restarting or still-loading server rather than aborting on the
        # first refused connection: the congress index takes ~2min to load 213k docs,
        # and a run launched during that window used to die instantly.
        for _attempt in range(6):
            try:
                _requests.get(f"{server_url}/health", timeout=15).raise_for_status()
                break
            except Exception as e:
                if _attempt == 5:
                    print(f"\n[ABORT] Server unreachable before Q{q_idx+1} "
                          f"after 6 attempts over ~5min: {e}")
                    sys.exit(1)
                print(f"  [warn] /health failed ({e}); retry {_attempt+1}/5 in 60s")
                time.sleep(60)

        gold_ids = qrels.get(query["_id"], set())
        excl = set(str(x) for x in all_excluded.get(query["_id"], []))
        r = run_one(
            retriever=retriever,
            query=query,
            q_num=q_idx + 1,
            gold_ids=gold_ids,
            signature=signature,
            max_iterations=args.max_iters,
            max_search_calls=args.max_search,
            verbose=args.verbose,
            default_top_k=args.top_k,
            timeout=args.timeout,
            subset=subset,
            excluded_ids=excl if excl else None,
            include_get_document=(subset not in ("wildchat", "wildchat-v6")),
            include_rlm_query=(subset in ("wildchat", "wildchat-v6")),
            # wildchat-v8 is free-form with get_document — handled by default (_REPL_TOOLS)
        )
        results.append(r)

        # Save checkpoint after every question so crashes don't lose progress
        ckpt_path = f"{args.out}_checkpoint.json"
        with open(ckpt_path, "w") as _f:
            json.dump(results, _f)

        if (run_i + 1) % 5 == 0:
            n_hit = sum(1 for x in results if x["gold_recall"] > 0)
            avg_recall = sum(x["gold_recall"] for x in results) / len(results)
            avg_ndcg10 = sum(x["ndcg"][10] for x in results) / len(results)
            print(f"\n  --- Checkpoint {len(results)}: hit={n_hit}/{len(results)}  avg_recall={avg_recall:.3f}  avg_NDCG@10={avg_ndcg10:.3f} ---\n")

    _save_results(results, args)


def _save_results(results: list[dict], args) -> None:
    n = len(results)
    if n == 0:
        print("No results to save.")
        return

    avg = lambda key: sum(r[key] for r in results) / n
    n_hit = sum(1 for r in results if r["gold_recall"] > 0)
    avg_recall = avg("gold_recall")
    avg_ndcg            = {k: sum(r["ndcg"][k] for r in results) / n for k in NDCG_KS}
    avg_recall_k        = {k: sum(r["recall_k"][k] for r in results) / n for k in NDCG_KS}
    avg_answer_ndcg     = {k: sum(r["answer_ndcg"][k] for r in results) / n for k in NDCG_KS}
    avg_answer_recall   = {k: sum(r["answer_recall_k"][k] for r in results) / n for k in NDCG_KS}
    avg_tail_ndcg       = {k: sum(r["tail_ndcg"][k] for r in results) / n for k in NDCG_KS}
    avg_tail_recall     = {k: sum(r["tail_recall_k"][k] for r in results) / n for k in NDCG_KS}

    out = args.out
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    json_path = f"{out}.json"
    with open(json_path, "w") as f:
        json.dump({
            "task": args.task,
            "subset": args.subset,
            "model": os.environ.get("AZURE_OPENAI_MODEL", ""),
            # Recorded so a results table can be reconstructed from the JSON alone;
            # previously effort was only inferable from the output filename.
            "reasoning_effort": os.environ.get("AZURE_OPENAI_REASONING_EFFORT", "") or "(model default)",
            "judge_model": os.environ.get("AZURE_OPENAI_JUDGE_MODEL", "gpt-5.4-mini"),
            # Resolved from the subset: `signature` is not in this function's scope.
            "signature": SUBSET_SIGNATURES[args.subset].__name__ if args.subset in SUBSET_SIGNATURES else "?",
            "relative_scores": os.environ.get("OBLIQ_RELATIVE_SCORES") == "1",
            "sub_lm_reasoning_effort": os.environ.get("SUB_LM_REASONING_EFFORT", "") or "(model default)",
            "date": datetime.now().isoformat(),
            "n": n,
            "n_hit": n_hit,
            "avg_recall": round(avg_recall, 4),
            "avg_ndcg":          {str(k): round(v, 4) for k, v in avg_ndcg.items()},
            "avg_recall_k":      {str(k): round(v, 4) for k, v in avg_recall_k.items()},
            "avg_answer_ndcg":     {str(k): round(v, 4) for k, v in avg_answer_ndcg.items()},
            "avg_answer_recall":   {str(k): round(v, 4) for k, v in avg_answer_recall.items()},
            "avg_tail_ndcg":       {str(k): round(v, 4) for k, v in avg_tail_ndcg.items()},
            "avg_tail_recall":     {str(k): round(v, 4) for k, v in avg_tail_recall.items()},
            "avg_latency_s": round(avg("latency_s"), 2),
            "avg_search_calls": round(avg("search_calls"), 2),
            "avg_iterations": round(avg("iterations"), 2),
            "results": results,
        }, f, indent=2)
    print(f"\nJSON → {json_path}")

    txt_path = f"{out}.txt"
    lines = [
        "=" * 72,
        f"OBLIQ-Bench Evaluation — RAG RLM (dspy.RLM + FAISS)",
        f"Task: {args.task}/{args.subset}   N={n}",
        "=" * 72,
        "",
        f"Model         : {os.environ.get('AZURE_OPENAI_MODEL', '?')}",
        f"QRels         : {'pooled (qrels_pool.tsv)' if getattr(args, 'pool', False) else 'gold (qrels.tsv)'}",
        f"Date          : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "─" * 72,
        "SUMMARY",
        "─" * 72,
        f"Questions run     : {n}",
        f"Gold doc retrieved: {n_hit}/{n} ({100*n_hit/n:.1f}%)  ← hit rate",
        f"Avg gold recall   : {avg_recall:.4f}  (full trajectory — all searched docs)",
        "",
        f"  Trajectory order (first-retrieval, NOT ranked — lower bound on NDCG):",
        f"  {'K':<6}  {'NDCG@K':>8}  {'Recall@K':>9}",
        f"  {'─'*6}  {'─'*8}  {'─'*9}",
    ] + [
        f"  {k:<6}  {avg_ndcg[k]:>8.4f}  {avg_recall_k[k]:>9.4f}"
        for k in NDCG_KS
    ] + [
        "",
        f"  Answer-ranked (agent's explicit ranking):",
        f"  {'K':<6}  {'NDCG@K':>8}  {'Recall@K':>9}",
        f"  {'─'*6}  {'─'*8}  {'─'*9}",
    ] + [
        f"  {k:<6}  {avg_answer_ndcg[k]:>8.4f}  {avg_answer_recall[k]:>9.4f}"
        for k in NDCG_KS
    ] + [
        "",
        f"  Tail-fill (ans + trajectory tail — mirrors paper Multi-Hop):",
        f"  {'K':<6}  {'NDCG@K':>8}  {'Recall@K':>9}",
        f"  {'─'*6}  {'─'*8}  {'─'*9}",
    ] + [
        f"  {k:<6}  {avg_tail_ndcg[k]:>8.4f}  {avg_tail_recall[k]:>9.4f}"
        for k in NDCG_KS
    ] + [
        "",
        f"Avg latency       : {avg('latency_s'):.1f}s",
        f"Avg retr latency  : {avg('retrieval_latency_s'):.1f}s",
        f"Avg tokens in/out : {avg('input_tokens'):.0f} / {avg('output_tokens'):.0f}",
        f"Avg iterations    : {avg('iterations'):.1f}",
        f"Avg search calls  : {avg('search_calls'):.1f}",
        f"Avg getdoc calls  : {avg('get_document_calls'):.1f}",
        f"Avg unique docs   : {avg('unique_docs'):.1f}",
        "",
        "─" * 72,
        "PER-QUESTION RESULTS",
        "─" * 72,
    ]
    for r in results:
        sym = "★" if r["gold_recall"] > 0 else "○"
        lines += [
            "",
            f"[{sym}Q{r['q_num']:03d}|{r['q_id']}] {r['question'][:80]}",
            f"  recall={r['gold_recall']:.3f}"
            f"  traj NDCG@10={r['ndcg'][10]:.3f}  ans NDCG@10={r['answer_ndcg'][10]:.3f}"
            f"  iters={r['iterations']}  search={r['search_calls']}"
            f"  getdoc={r['get_document_calls']}  lat={r['latency_s']:.1f}s"
            f"  tok={r['input_tokens']}/{r['output_tokens']}",
            f"  answer [{len(r['answer_doc_ids'])} ids]: {r['predicted'][:120]}",
        ]

    Path(txt_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"TXT  → {txt_path}")

    # Print summary to stdout
    print(f"\n{'='*60}")
    print(f"OBLIQ {args.task}/{args.subset} — {n} questions")
    print(f"  Gold retrieved : {n_hit}/{n} ({100*n_hit/n:.1f}%)")
    print(f"  Avg recall     : {avg_recall:.4f}")
    print(f"  NDCG@10 (traj)   : {avg_ndcg[10]:.4f}  (first-retrieval order)")
    print(f"  NDCG@10 (ans)    : {avg_answer_ndcg[10]:.4f}  (agent-ranked)")
    print(f"  NDCG@50 (traj)   : {avg_ndcg[50]:.4f}")
    print(f"  NDCG@50 (ans)    : {avg_answer_ndcg[50]:.4f}")
    print(f"  Recall@50 (ans)  : {avg_answer_recall[50]:.4f}")
    print(f"  NDCG@10 (tail)   : {avg_tail_ndcg[10]:.4f}  (ans + trajectory tail)")
    print(f"  NDCG@50 (tail)   : {avg_tail_ndcg[50]:.4f}")
    print(f"  Recall@50 (tail) : {avg_tail_recall[50]:.4f}")
    print(f"  Avg latency      : {avg('latency_s'):.1f}s")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OBLIQ-Bench RAG RLM evaluation")
    parser.add_argument("--task",    required=True,
                        choices=["tip-of-tongue", "analogues", "descriptive"],
                        help="OBLIQ task type")
    parser.add_argument("--subset",  required=True,
                        choices=["congress", "congress-v4", "congress-v5", "congress-v6", "congress-v7", "congress-v8", "math", "math-rerank", "math-overview", "math-batch", "writing", "twitter", "twitter-v2", "twitter-b1", "wildchat", "wildchat-v6", "wildchat-v7", "wildchat-v8", "wildchat-v9"],
                        help="OBLIQ dataset subset")
    parser.add_argument("--start",   type=int, default=0,
                        help="First query index (0-based, default 0)")
    parser.add_argument("--end",     type=int, default=0,
                        help="End query index exclusive (0 = all, default 0)")
    parser.add_argument("--indices", type=str, default="",
                        help="Comma-separated 0-based query indices (overrides --start/--end)")
    parser.add_argument("--out",     default="logs/obliq_run",
                        help="Output file prefix (no extension)")
    parser.add_argument("--max-iters",  type=int, default=12,
                        help="Max REPL iterations per question (default 16)")
    parser.add_argument("--max-search", type=int, default=8,
                        help="Max search_index calls per question (default 8)")
    parser.add_argument("--top-k",   type=int, default=20,
                        help="Default top_k for search_index (default 20)")
    parser.add_argument("--timeout", type=int,
                        default=int(os.environ.get("OBLIQ_TIMEOUT", "0")),
                        help="Per-question timeout in seconds (0=no limit)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable dspy.RLM verbose logging")
    parser.add_argument("--pool", action="store_true",
                        help="Use qrels_pool.tsv (pooled judgments) instead of qrels.tsv")
    parser.add_argument("--model", type=str, default="",
                        help="Override AZURE_OPENAI_MODEL (e.g. ismail-gpt5-2)")
    args = parser.parse_args()

    if args.model:
        os.environ["AZURE_OPENAI_MODEL"] = args.model

    run_eval(args)
