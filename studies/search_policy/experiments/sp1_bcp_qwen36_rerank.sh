#!/usr/bin/env bash
# SP-1: BrowseComp-Plus, document REPL agent, Qwen3.6-35B-A3B, Reason-ModernColBERT + novelty filter + Qwen3-Reranker-4B
# (POLICY=novelty+rerank, depth 30), all 830 questions in five slices. Control: BCP-29 (same agent on ColBERT :8766).
# RECONSTRUCTED: no launcher file existed. Each slice was started by hand; the proxy and agent command lines are in the
# Claude Code session transcript 8ba31816 (proxy and agent of each slice, times below), the pareto/feedback resume of
# 2026-10-13 in <original repo>/logs/obliq_rr/overnight.sh, the proxy restart in logs/restart_1013/
# restart_bcp_rr_servers.sh (original/RLM/logs/). Proxy variables per slice, copied from those lines:
#   test200   2026-10-10 11:19  :8768  CUDA_VISIBLE_DEVICES=0 POLICY=novelty+rerank DEPTH=30 RERANK_BATCH=6
#                                     RERANK_MEM_FRACTION=0.15   agent 11:28, 20 workers
#   test_rest 2026-10-11 08:47  :8772  CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#                                     POLICY=novelty+rerank DEPTH=30 RERANK_TOK_BUDGET=8000 RERANK_MEM_FRACTION=0.15
#                                     agent 12 workers (with the BCP-29 test_rest control on :8766 at the same time)
#   pareto, feedback  2026-10-12 20:29 on the same :8772 proxy, 10 workers; resumed 10-13 06:28 and 10-14 12:22, 12 workers
#   rest300   2026-10-14 14:47 on :8772 (pareto + feedback merged; split_rest300.json), 24 workers
# The test200 proxy (2026-10-10) predates cross-request batching and is not preserved (CHANGES SP-I1). It processed one
# search at a time; this launcher sets RERANK_XBATCH=0, the one-search-at-a-time path the later file keeps, for that
# slice. That variable did not exist then and is the only addition to the original command lines.
# Prerequisites (not started here): vLLM serving qwen36-35b-a3b on :8030 (2026-10-10: serve_qwen36_moe.sh with
# GEN_OVERRIDE='{"max_new_tokens": 16384}' MAX_LEN=262144 UTIL=0.72 GPU=0; the reranker shares GPU 0) and the
# Reason-ModernColBERT server on :8766 (agents/document_repl/src/retrieval_server_colbert.py, GPU 1).
#   experiments/sp1_bcp_qwen36_rerank.sh test200|test_rest|pareto|feedback|rest300
set -u
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
SLICE=${1:?slice: test200 test_rest pareto feedback rest300}
need_up http://127.0.0.1:8766/health "Reason-ModernColBERT (:8766)"
need_up http://localhost:8030/v1/models "vLLM qwen36-35b-a3b (:8030)"
AGENT=$REPO/agents/document_repl/experiments/bcp_qwen36_split.sh
case "$SLICE" in
  test200)
    CUDA_VISIBLE_DEVICES=0 POLICY=novelty+rerank DEPTH=30 RERANK_BATCH=6 RERANK_MEM_FRACTION=0.15 RERANK_XBATCH=0 \
      start_proxy 8768 "$RUNS_DIR/bcp_policy_proxy8768.log"
    RETRIEVAL=http://127.0.0.1:8768 OUT=bcp_policy_e2e SPLIT=test200 W=20 "$AGENT" ;;
  test_rest|pareto|feedback|rest300)
    CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True POLICY=novelty+rerank DEPTH=30 \
      RERANK_TOK_BUDGET=8000 RERANK_MEM_FRACTION=0.15 start_proxy 8772 "$RUNS_DIR/bcp_policy_proxy8772.log"
    case "$SLICE" in
      test_rest) RETRIEVAL=http://127.0.0.1:8772 OUT=bcp_testrest_rr SPLIT=test_rest W=12 "$AGENT" ;;
      pareto|feedback) RETRIEVAL=http://127.0.0.1:8772 OUT=bcp_${SLICE}_rr SPLIT=$SLICE W=${W:-10} "$AGENT" ;;
      rest300) RETRIEVAL=http://127.0.0.1:8772 OUT=bcp_rest300_rr SPLIT=rest300 W=24 "$AGENT" ;;
    esac ;;
  *) say "unknown slice $SLICE"; exit 2 ;;
esac
say "proxy stats: $(curl -s -m5 127.0.0.1:8768/stats 2>/dev/null)$(curl -s -m5 127.0.0.1:8772/stats 2>/dev/null)"
