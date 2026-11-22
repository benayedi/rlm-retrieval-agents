#!/usr/bin/env bash
# SP-6: OBLIQ-Twitter (281), snippet REPL agent, Qwen3.6-35B-A3B, Qwen3-Embedding-8B dense search + novelty filter +
# Qwen3-Reranker-4B (depth 100, 512-token posts, a post-matching instruction), proxy in the snippet REPL agent's wire
# format (WIRE=mine) on :8775. Control: TWQ-1 (the same agent on plain dense search :8010).
# RECONSTRUCTED proxy starts, from command lines in Claude Code session transcripts:
#   dev100  (8ba31816, 2026-10-13 09:18): WIRE=mine QUERIES_FILE=<FULLDOC_ROOT>/data/obliq_twitter/queries.jsonl
#           RERANK_INSTR="<below>" UPSTREAM=http://127.0.0.1:8010 CUDA_VISIBLE_DEVICES=1 POLICY=novelty+rerank DEPTH=100
#           MAX_DOC_TOK=512 RERANK_TOK_BUDGET=8000 RERANK_MEM_FRACTION=0.15 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   rest181 (1a0da3c7, 2026-10-21 12:34): the same except RERANK_MEM_FRACTION=0.20 and no PYTORCH_CUDA_ALLOC_CONF
# The agent side is agents/snippet_repl/experiments/obliq_twitter_qwen36.sh ARM=rr (its header gives the agent lines).
# Prerequisites: the dense search server on :8010 (benchmarks/retrieval multi_search_server.py, index
# obliq-des-twitter, --nprobe 64), vLLM qwen36-35b-a3b on :8030 (rest181 also :8031).
#   experiments/sp6_obliq_snippet_qwen36_rerank.sh dev100|rest181
set -u
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
SLICE=${1:?slice: dev100 or rest181}
: "${FULLDOC_ROOT:?set FULLDOC_ROOT (holds data/obliq_twitter/queries.jsonl)}"
need_up http://localhost:8010/health "dense search (:8010)"
INSTR="Given a request for social media posts that implicitly express a described stance or pattern, judge whether this post matches the request"
case "$SLICE" in
  dev100)  WIRE=mine QUERIES_FILE=$FULLDOC_ROOT/data/obliq_twitter/queries.jsonl RERANK_INSTR="$INSTR" \
             UPSTREAM=http://127.0.0.1:8010 CUDA_VISIBLE_DEVICES=1 POLICY=novelty+rerank DEPTH=100 MAX_DOC_TOK=512 \
             RERANK_TOK_BUDGET=8000 RERANK_MEM_FRACTION=0.15 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
             start_proxy 8775 "$RUNS_DIR/obliq_mine_rr_proxy8775.log" ;;
  rest181) CUDA_VISIBLE_DEVICES=1 POLICY=novelty+rerank WIRE=mine UPSTREAM=http://127.0.0.1:8010 DEPTH=100 MAX_DOC_TOK=512 \
             RERANK_MEM_FRACTION=0.20 RERANK_TOK_BUDGET=8000 QUERIES_FILE=$FULLDOC_ROOT/data/obliq_twitter/queries.jsonl \
             RERANK_INSTR="$INSTR" start_proxy 8775 "$RUNS_DIR/obliq_q36_next_proxy8775.log" ;;
  *) say "unknown slice $SLICE"; exit 2 ;;
esac
ARM=rr SLICE=$SLICE "$REPO/agents/snippet_repl/experiments/obliq_twitter_qwen36.sh"
say "proxy stats: $(curl -s -m5 127.0.0.1:8775/stats)"
