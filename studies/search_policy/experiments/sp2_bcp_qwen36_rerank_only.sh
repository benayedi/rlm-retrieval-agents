#!/usr/bin/env bash
# SP-2: BrowseComp-Plus test200, document REPL agent, Qwen3.6-35B-A3B, Reason-ModernColBERT + Qwen3-Reranker-4B only
# (POLICY=rerank, no novelty filter, depth 30). Control: BCP-29 on test200.
# RECONSTRUCTED from one command line of 2026-10-10 18:16 in the Claude Code session transcript 8ba31816:
#   CUDA_VISIBLE_DEVICES=0 POLICY=rerank DEPTH=30 RERANK_BATCH=6 RERANK_MEM_FRACTION=0.15 ... bcp_rerank_proxy.py --port 8769
#   then the agent on test200 with GEPA_BCP_RETRIEVAL=http://127.0.0.1:8769 GEPA_OUT=logs/bcp_policy_e2e_rerankonly, 20 workers.
# Like SP-1 test200, it ran on the unpreserved pre-batching proxy; RERANK_XBATCH=0 (one search at a time) is added.
# Prerequisites: vLLM qwen36-35b-a3b on :8030, Reason-ModernColBERT on :8766 (see sp1_bcp_qwen36_rerank.sh).
#   experiments/sp2_bcp_qwen36_rerank_only.sh
set -u
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
need_up http://127.0.0.1:8766/health "Reason-ModernColBERT (:8766)"
need_up http://localhost:8030/v1/models "vLLM qwen36-35b-a3b (:8030)"
CUDA_VISIBLE_DEVICES=0 POLICY=rerank DEPTH=30 RERANK_BATCH=6 RERANK_MEM_FRACTION=0.15 RERANK_XBATCH=0 \
  start_proxy 8769 "$RUNS_DIR/bcp_policy_proxy8769_rerankonly.log"
RETRIEVAL=http://127.0.0.1:8769 OUT=bcp_policy_e2e_rerankonly SPLIT=test200 W=20 \
  "$REPO/agents/document_repl/experiments/bcp_qwen36_split.sh"
say "proxy stats: $(curl -s -m5 127.0.0.1:8769/stats)"
