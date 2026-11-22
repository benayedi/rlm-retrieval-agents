"""
BrowseComp+ evaluation for the RAG RLM (dspy.RLM + FAISS).

Usage:
    python examples/browsecomp_eval_dspy_rlm.py --start 0 --end 50 --out logs/run_q0_q50
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import requests as _requests

import dspy
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import RemoteRetriever, RunMetrics, build_lm
from rag.agent import run_question
from examples.browsecomp_data import (  # noqa: E402
    decrypt, get_question, get_gold_answer, get_gold_doc_ids, get_evidence_doc_ids,
    judge_match, heuristic_match,
)
from rag.signatures import (BrowseCompSignature, BrowseCompMinimalSignature,
                            BrowseCompMinimalScoringSignature,
                            BrowseCompOverviewSignature,
                            BrowseCompBatchSignature,
                            BrowseCompFreeBatchSignature,
                            BrowseCompReadFirstSignature,
                            BrowseCompSubtractiveSignature,
                            BrowseCompCitationSignature,
                            BrowseCompReadInContextSignature,
                            BrowseCompMinimalCommitSignature,
                            BrowseCompSearchOnlySignature,
                            browsecomp_minimal_budget_sig)

_SIGNATURES = {
    "v1": BrowseCompSignature,
    "v2": BrowseCompMinimalSignature,
    # v3 is built per-run: its prompt states the real --max-iters budget, so it
    # cannot be a module constant the way v1/v2 are. Resolved in main().
    "v3": None,
    "v4": BrowseCompMinimalScoringSignature,
    "v5": BrowseCompOverviewSignature,
    "v6": BrowseCompBatchSignature,
    "v7": BrowseCompFreeBatchSignature,
    "v8": BrowseCompReadFirstSignature,
    "v9": BrowseCompSubtractiveSignature,
    "v10": BrowseCompCitationSignature,
    "v15": __import__("rag.signatures", fromlist=["x"]).BrowseCompNoPrintSignature,
    "v16": __import__("rag.signatures", fromlist=["x"]).BrowseCompTitleTriageSignature,
    "v17": __import__("rag.signatures", fromlist=["x"]).BrowseCompNoRulesSignature,
    "v20": __import__("rag.signatures", fromlist=["x"]).BrowseCompDecompositionSignature,
    "v21": __import__("rag.signatures", fromlist=["x"]).BrowseCompSnippetPromptSignature,
    "v11": BrowseCompReadInContextSignature,
    "v13": BrowseCompMinimalCommitSignature,
    "v14": BrowseCompSearchOnlySignature,
}
for _k, _s in _SIGNATURES.items():
    if _s is not None:
        _s._sig_key = _k

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# BrowseComp+ decryption  (fixed CANARY key, XOR + base64)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Dataset access and grading now live in examples/browsecomp_data.py, which both
# harnesses import. Nothing there depends on dspy or rag/, so the official-RLM
# harness no longer loads this module (and the whole dspy stack) to decrypt a string.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Module-level worker for process-based timeout (must be picklable)
# ─────────────────────────────────────────────────────────────────────────────

def _question_worker(result_queue, server_url, question, max_depth, max_iterations,
                     max_search_calls, verbose, default_top_k, gold_ids, sig_key):
    try:
        import dspy
        from rag import RemoteRetriever, build_lm
        from rag.agent import run_question as _rq
        from rag.signatures import BrowseCompSignature, BrowseCompMinimalSignature
        from rag.signatures import browsecomp_minimal_budget_sig as _bud
        from rag.signatures import BrowseCompMinimalScoringSignature as _v4
        from rag.signatures import BrowseCompOverviewSignature as _v5
        _SIGS = {"v1": BrowseCompSignature, "v2": BrowseCompMinimalSignature, "v4": _v4, "v5": _v5,
                 "v7": __import__("rag.signatures", fromlist=["x"]).BrowseCompFreeBatchSignature,
                 "v8": __import__("rag.signatures", fromlist=["x"]).BrowseCompReadFirstSignature,
                 "v9": __import__("rag.signatures", fromlist=["x"]).BrowseCompSubtractiveSignature,
                 "v10": __import__("rag.signatures", fromlist=["x"]).BrowseCompCitationSignature,
                 "v11": __import__("rag.signatures", fromlist=["x"]).BrowseCompReadInContextSignature,
                 "v6": __import__("rag.signatures", fromlist=["x"]).BrowseCompBatchSignature,
                 "v3": _bud(max_iterations, max_search_calls)}
        lm = build_lm()
        dspy.configure(lm=lm)
        retriever = RemoteRetriever(server_url=server_url)
        pred, m = _rq(
            retriever=retriever, question=question, max_depth=max_depth,
            max_iterations=max_iterations, max_search_calls=max_search_calls,
            verbose=verbose, default_top_k=default_top_k,
            gold_ids=gold_ids, signature=_SIGS.get(sig_key),
        )
        result_queue.put(("ok", pred, m))
    except Exception as e:
        import traceback
        traceback.print_exc()
        result_queue.put(("error", str(e), None))


# ─────────────────────────────────────────────────────────────────────────────
# Single question runner
# ─────────────────────────────────────────────────────────────────────────────

def run_one(
    retriever: FaissRetriever,
    row: dict,
    q_idx: int,
    max_iterations: int = 25,
    max_search_calls: int = 50,
    max_depth: int = 5,
    verbose: bool = False,
    default_top_k: int = 10,
    timeout: int = 0,
    signature=None,
) -> dict:
    question = get_question(row)
    gold = get_gold_answer(row)
    gold_ids = get_gold_doc_ids(row)
    evidence_ids = get_evidence_doc_ids(row)

    print(f"\n[Q{q_idx+1:03d}] {question[:100]}…")
    if gold_ids:
        print(f"  [gold docs: {list(gold_ids)[:3]}{'...' if len(gold_ids) > 3 else ''}]")

    if timeout > 0:
        q: multiprocessing.Queue = multiprocessing.Queue()
        server_url = os.environ["EMBEDDING_SERVER_URL"]
        sig_key = getattr(signature, "_sig_key", "v1")
        p = multiprocessing.Process(
            target=_question_worker,
            args=(q, server_url, question, max_depth, max_iterations,
                  max_search_calls, verbose, default_top_k, gold_ids, sig_key),
        )
        p.start()
        # Read the result BEFORE joining. A worker that puts more than the pipe buffer
        # (64 KB) on the queue cannot exit until the parent reads it, so join-then-get
        # deadlocks until the timeout and a finished answer is recorded as TIMEOUT
        # (found 2026-10-31 in the DSPy ablation, 60K output limit -> large results).
        import queue as _queue
        _t0, msg = time.time(), None
        while time.time() - _t0 < timeout:
            try:
                msg = q.get(timeout=5)
                break
            except _queue.Empty:
                if not p.is_alive():
                    try:
                        msg = q.get(timeout=5)
                    except _queue.Empty:
                        pass
                    break
        p.join(30)
        if p.is_alive():
            p.kill()
            p.join()
        if msg is None and time.time() - _t0 >= timeout:
            print(f"  [TIMEOUT] question exceeded {timeout}s — skipping")
            metrics = RunMetrics(latency_s=float(timeout))
            predicted = "TIMEOUT"
        elif msg is None:
            print(f"  ERROR: worker exited without a result (exitcode {p.exitcode})")
            metrics = RunMetrics(latency_s=time.time() - _t0)
            predicted = "ERROR"
        else:
            status, *rest = msg
            if status == "ok":
                predicted, metrics = rest
            else:
                print(f"  ERROR: {rest[0]}")
                metrics = RunMetrics(latency_s=0.0)
                predicted = "ERROR"
    else:
        try:
            predicted, metrics = run_question(
                retriever=retriever,
                question=question,
                max_depth=max_depth,
                max_iterations=max_iterations,
                max_search_calls=max_search_calls,
                verbose=verbose,
                default_top_k=default_top_k,
                gold_ids=gold_ids,
                signature=signature,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ERROR: {e}")
            metrics = RunMetrics(latency_s=0.0)
            predicted = "ERROR"

    correct = judge_match(question, predicted, gold)
    gold_recall = metrics.gold_recall(list(gold_ids))
    evid_recall = metrics.gold_recall(list(evidence_ids))

    symbol = "✓" if correct else "✗"
    print(
        f"  [{symbol}] Pred={predicted[:80]!r}  Gold={gold!r}\n"
        f"      iters={metrics.iterations}  search={metrics.search_calls}"
        f"  getdoc={metrics.get_document_calls}  deleg={metrics.delegation_calls}"
        f"  docs={metrics.unique_docs_retrieved}  lat={metrics.latency_s:.1f}s"
        f"  retr={metrics.retrieval_latency_s:.1f}s"
        f"  llmq={metrics.llm_query_calls}"
        f"  [lm={metrics.lm_latency_s:.0f}s repl="
        f"{max(0.0, metrics.latency_s - metrics.lm_latency_s - metrics.retrieval_latency_s):.0f}s"
        f" cached={metrics.cached_input_tokens}]"
        f"  tok={metrics.input_tokens}/{metrics.output_tokens}"
        f"  gold_recall={gold_recall:.2f}  evid_recall={evid_recall:.2f}"
    )

    return {
        "q_idx": q_idx,
        "question": question,
        "gold": gold,
        "predicted": predicted,
        "correct": correct,
        "latency_s": metrics.latency_s,
        "retrieval_latency_s": metrics.retrieval_latency_s,
        "llm_query_calls": metrics.llm_query_calls,
        "lm_latency_s": round(metrics.lm_latency_s, 2),
        "lm_calls": metrics.lm_calls,
        "lm_call_latencies": metrics.lm_call_latencies,
        "cached_input_tokens": metrics.cached_input_tokens,
        "repl_latency_s": round(
            max(0.0, metrics.latency_s - metrics.lm_latency_s - metrics.retrieval_latency_s), 2),
        "input_tokens": metrics.input_tokens,
        "output_tokens": metrics.output_tokens,
        "iterations": metrics.iterations,
        "search_calls": metrics.search_calls,
        "get_document_calls": metrics.get_document_calls,
        "delegation_calls": metrics.delegation_calls,
        "unique_docs": metrics.unique_docs_retrieved,
        "gold_recall": gold_recall,
        "evid_recall": evid_recall,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_eval(
    start: int = 0,
    end: int = 50,
    out_prefix: str = "logs/run",
    max_iterations: int = 25,
    max_search_calls: int = 50,
    max_depth: int = 5,
    verbose: bool = False,
    default_top_k: int = 10,
    args=None,
    signature=None,
) -> None:
    from datasets import load_dataset

    lm = build_lm()
    dspy.configure(lm=lm)

    print("Loading BrowseComp+ dataset…")
    dataset = load_dataset("Tevatron/browsecomp-plus", split="test")

    retriever = RemoteRetriever(server_url=os.environ["EMBEDDING_SERVER_URL"])

    results: list[dict] = []
    all_rows = list(dataset)
    if hasattr(args, "indices") and args.indices:
        index_list = [int(x.strip()) for x in args.indices.split(",")]
        rows = [(idx, all_rows[idx]) for idx in index_list]
    else:
        rows = [(start + i, all_rows[start + i]) for i in range(end - start)]

    for i, (q_idx, row) in enumerate(rows):
        # Wait out a transient blip before giving up: a single timed-out /health
        # used to abort the whole run, which cost the q1_q830 medium run 52
        # questions (~4h) with nothing saved.
        server_url = os.environ["EMBEDDING_SERVER_URL"].rstrip("/")
        for _attempt in range(6):
            try:
                _requests.get(f"{server_url}/health", timeout=15).raise_for_status()
                break
            except Exception as e:
                if _attempt == 5:
                    print(f"\n[ABORT] Embedding server unreachable before Q{q_idx+1} "
                          f"after 6 attempts over ~5min: {e}")
                    sys.exit(1)
                print(f"  [warn] /health failed ({e}); retry {_attempt+1}/5 in 60s")
                time.sleep(60)

        r = run_one(
            retriever=retriever,
            row=row,
            q_idx=q_idx,
            max_iterations=max_iterations,
            max_search_calls=max_search_calls,
            max_depth=max_depth,
            verbose=verbose,
            default_top_k=default_top_k,
            timeout=args.timeout if hasattr(args, "timeout") else 0,
            signature=signature,
        )
        results.append(r)

        if (i + 1) % 5 == 0:
            n_correct = sum(x["correct"] for x in results)
            print(f"\n  --- Running total: {n_correct}/{len(results)} correct ---\n")
            # Incremental checkpoint. Previously the JSON was written only after the
            # final question, so any abort discarded the entire run.
            _ckpt = f"{out_prefix}_checkpoint.json"
            Path(_ckpt).parent.mkdir(parents=True, exist_ok=True)
            with open(_ckpt, "w") as _f:
                json.dump(results, _f, indent=2)

    # Save JSON
    json_path = f"{out_prefix}.json"
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON saved → {json_path}")

    txt_path = f"{out_prefix}.txt"
    _write_txt(results, txt_path, start, end)
    print(f"TXT  saved → {txt_path}")

    if "baseline" in out_prefix:
        import subprocess
        summary_script = Path(__file__).parent.parent / "scripts" / "update_baseline_summary.py"
        subprocess.run([sys.executable, str(summary_script)], check=False)


def _write_txt(results: list[dict], path: str, start: int, end: int) -> None:
    n = len(results)
    n_correct = sum(r["correct"] for r in results)
    avg = lambda key: sum(r[key] for r in results) / n if n else 0

    gold_recalls = [r["gold_recall"] for r in results if r.get("gold_recall") is not None]
    evid_recalls = [r["evid_recall"] for r in results if r.get("evid_recall") is not None]

    lines = [
        "=" * 72,
        "BrowseComp+ Evaluation — RAG RLM (dspy.RLM + FAISS + Delegation)",
        f"Q{start+1}–Q{end}  ({n} questions)",
        "=" * 72,
        "",
        f"Model         : {os.environ.get('AZURE_OPENAI_MODEL', 'gpt-5.4-mini')} (Azure)",
        f"Date          : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Max iters     : 25   Max depth: 5",
        "",
        "─" * 72,
        "SUMMARY",
        "─" * 72,
        f"Accuracy          : {n_correct}/{n} ({100*n_correct/n:.1f}%)" if n else "Accuracy          : 0/0 (N/A)",
        f"Avg latency       : {avg('latency_s'):.1f}s",
        f"Avg retrieval lat : {avg('retrieval_latency_s'):.1f}s",
        f"Avg tokens in/out : {avg('input_tokens'):.0f} / {avg('output_tokens'):.0f}",
        f"Avg iterations    : {avg('iterations'):.1f}",
        f"Avg search calls  : {avg('search_calls'):.1f}",
        f"Avg getdoc calls  : {avg('get_document_calls'):.1f}",
        f"Avg delegations   : {avg('delegation_calls'):.1f}",
        f"Avg unique docs   : {avg('unique_docs'):.1f}",
        "",
        "Retrieval Recall:",
        f"  Gold recall     : {sum(gold_recalls)/len(gold_recalls):.3f}" if gold_recalls else "  Gold recall     : N/A",
        f"  Evidence recall : {sum(evid_recalls)/len(evid_recalls):.3f}" if evid_recalls else "  Evidence recall : N/A",
        "",
        "─" * 72,
        "PER-QUESTION RESULTS",
        "─" * 72,
    ]

    for r in results:
        sym = "✓" if r["correct"] else "✗"
        lines += [
            "",
            f"[Q{r['q_idx']+1:03d}] {sym}  {r['question'][:80]}",
            f"       Answer    : {r['predicted'][:100]}",
            f"       Gold      : {r['gold']}",
            f"       Iters={r['iterations']}  Search={r['search_calls']}"
            f"  GetDoc={r['get_document_calls']}  Deleg={r['delegation_calls']}"
            f"  Docs={r['unique_docs']}  Lat={r['latency_s']:.1f}s"
            f"  Retr={r.get('retrieval_latency_s', 0):.1f}s"
            f"  Tok={r['input_tokens']}/{r['output_tokens']}",
            f"       Gold recall={r['gold_recall']:.2f}  Evid recall={r['evid_recall']:.2f}",
        ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BrowseComp+ RAG RLM evaluation")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=50)
    parser.add_argument("--indices", type=str, default="",
                        help="Comma-separated 0-based question indices to run (overrides --start/--end)")
    parser.add_argument("--out", default="logs/run", help="Output file prefix (no extension)")
    parser.add_argument("--max-iters", type=int, default=16)
    parser.add_argument("--max-search", type=int, default=35,
                        help="Max search_index calls per question before forcing answer (default: 35)")
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--timeout", type=int,
                        default=int(os.environ.get("BROWSECOMP_TIMEOUT", 0)),
                        help="Per-question timeout in seconds (0 = no limit, default from BROWSECOMP_TIMEOUT env)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Default number of results returned by search_index (default: 10)")
    parser.add_argument("--sig", choices=["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v13", "v14", "v15", "v16", "v17", "v20", "v21"], default="v1",
                        help="Signature version: v1=structured (default), v2=minimal free-form")
    args = parser.parse_args()

    # v3's prompt quotes the real budget, so it must be built after parsing.
    sig = _SIGNATURES[args.sig]
    if sig is None:
        sig = browsecomp_minimal_budget_sig(args.max_iters, args.max_search)

    run_eval(
        start=args.start,
        end=args.end,
        out_prefix=args.out,
        max_iterations=args.max_iters,
        max_search_calls=args.max_search,
        max_depth=args.max_depth,
        verbose=args.verbose,
        default_top_k=args.top_k,
        args=args,
        signature=sig,
    )
