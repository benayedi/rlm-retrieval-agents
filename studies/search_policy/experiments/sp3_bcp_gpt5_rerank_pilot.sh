#!/usr/bin/env bash
# SP-3: BrowseComp-Plus pilot, 100 questions (50 BCP-23 got wrong + 50 it got right), document REPL agent with GPT-5
# (effort high), Reason-ModernColBERT 8192 + novelty filter + Qwen3-Reranker-4B. Reference: BCP-23's stored answers.
# The agent side is the original wrapper (original/RLM/scripts/bcp_gpt5_sup.sh of agents/document_repl, published as
# agents/document_repl/experiments/bcp_gpt5_colbert.sh), whose header names this use: SERVER=:8772 = the rerank proxy.
# RECONSTRUCTED proxy start: the command line of 2026-10-26 22:35 in the Claude Code session transcript 2a1318b5 (the
# second start; the first, without RERANK_MEM_FRACTION, ran out of GPU memory, logs/bcp_gpt5_rr/proxy8772_oom.log):
#   CUDA_VISIBLE_DEVICES=0 POLICY=novelty+rerank UPSTREAM=http://127.0.0.1:8766 RERANK_MEM_FRACTION=0.5 RERANK_TOK_BUDGET=8000
#   ... bcp_rerank_proxy.py --port 8772
# and the agent: SERVER=http://127.0.0.1:8772 IDS_FILE=data/bcp_gpt5_rr/pilot100_ids.txt OUT=logs/bcp_gpt5_rr/rr W=6
# (restarted at 22:49 with W=4; logs/bcp_gpt5_rr/run_rr.log). Prerequisite: Reason-ModernColBERT 8192 on :8766, started
# then on GPU 1 with COLBERT_INDEX_NAME=bc_rmcb_dl8192 COLBERT_DOC_LEN=8192 COLBERT_SEARCH_BATCH=2048 COLBERT_LOW_MEMORY=1.
# Grading: analysis/grade_gpt5_pilot.py; the reported verdicts are the gpt-4.1 re-judge (pilot_rr.json).
#   [IDS_FILE=<pilot100_ids.txt>] [W=6] experiments/sp3_bcp_gpt5_rerank_pilot.sh
set -u
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
need_up http://127.0.0.1:8766/health "Reason-ModernColBERT 8192 (:8766)"
CUDA_VISIBLE_DEVICES=0 POLICY=novelty+rerank UPSTREAM=http://127.0.0.1:8766 RERANK_MEM_FRACTION=0.5 RERANK_TOK_BUDGET=8000 \
  start_proxy 8772 "$RUNS_DIR/bcp_gpt5_rr_proxy8772.log"
IDS_FILE=${IDS_FILE:-$STUDY/experiments/inputs/pilot100_ids.txt}   # original: <RLM_DATA_DIR>/bcp_gpt5_rr/pilot100_ids.txt (byte-identical copy)
SERVER=http://127.0.0.1:8772 IDS_FILE=$IDS_FILE OUT=$RUNS_DIR/bcp_gpt5_rr/rr W=${W:-6} \
  "$REPO/agents/document_repl/experiments/bcp_gpt5_colbert.sh"
say "proxy stats: $(curl -s -m5 127.0.0.1:8772/stats)"
