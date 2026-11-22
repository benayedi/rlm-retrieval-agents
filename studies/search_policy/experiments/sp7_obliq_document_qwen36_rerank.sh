#!/usr/bin/env bash
# SP-7: OBLIQ-Twitter (281), document REPL agent, Qwen3.6-35B-A3B, Qwen3-Embedding-8B dense search (through
# agents/document_repl/src/obliq_adapter.py on :8767) + novelty filter + Qwen3-Reranker-4B (depth 100, 512-token posts,
# a post-matching instruction) on :8774. Control: TWQ-2 (the same agent on the adapter :8767 directly).
# RECONSTRUCTED from one command line of 2026-10-12 20:55 in the Claude Code session transcript 8ba31816:
#   obliq_adapter.py --port 8767   (its defaults: --upstream http://localhost:8010 --index obliq-des-twitter)
#   QUERIES_FILE=<FULLDOC_ROOT>/data/obliq_twitter/queries.jsonl RERANK_INSTR="<below>" UPSTREAM=http://127.0.0.1:8767
#   CUDA_VISIBLE_DEVICES=1 POLICY=novelty+rerank DEPTH=100 MAX_DOC_TOK=512 RERANK_TOK_BUDGET=8000 RERANK_MEM_FRACTION=0.13
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ... bcp_rerank_proxy.py --port 8774
# This proxy ran before the 2026-10-13 09:17 backup (original/RLM/logs/restart_1013/bcp_rerank_proxy.v3.py), whose
# sup-wire path is the same code as src/rerank_proxy.py (CHANGES SP-I1). The agent side is
# agents/document_repl/experiments/obliq_twitter_qwen36.sh ARM=rr. Prerequisites: dense search on :8010, vLLM on :8030.
#   experiments/sp7_obliq_document_qwen36_rerank.sh
set -u
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
: "${FULLDOC_ROOT:?set FULLDOC_ROOT}"
need_up http://localhost:8010/health "dense search (:8010)"
if ! curl -s -m5 -o /dev/null 127.0.0.1:8767/health; then
  nohup "$PYTHON" -u "$REPO/agents/document_repl/src/obliq_adapter.py" --port 8767 > "$RUNS_DIR/obliq_rr_adapter8767.log" 2>&1 < /dev/null &
  for _i in $(seq 1 30); do grep -q READY "$RUNS_DIR/obliq_rr_adapter8767.log" 2>/dev/null && break; sleep 2; done
fi
INSTR="Given a request for social media posts that implicitly express a described stance or pattern, judge whether this post matches the request"
QUERIES_FILE=$FULLDOC_ROOT/data/obliq_twitter/queries.jsonl RERANK_INSTR="$INSTR" UPSTREAM=http://127.0.0.1:8767 \
  CUDA_VISIBLE_DEVICES=1 POLICY=novelty+rerank DEPTH=100 MAX_DOC_TOK=512 RERANK_TOK_BUDGET=8000 RERANK_MEM_FRACTION=0.13 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True start_proxy 8774 "$RUNS_DIR/obliq_rr_proxy8774.log"
ARM=rr "$REPO/agents/document_repl/experiments/obliq_twitter_qwen36.sh"
say "proxy stats: $(curl -s -m5 127.0.0.1:8774/stats)"
