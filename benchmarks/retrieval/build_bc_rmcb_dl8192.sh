#!/bin/bash
# Refactored copy of scripts/build_bc_rmcb_dl8192.sh of the original repository (agents/shared/CHANGES_FROM_ORIGINAL.md).
# Changed: the working directory (the checkout holding scripts/retrieval_eval.py, which is NOT part of this repository)
# is ${RETRIEVAL_EVAL_ROOT}, and the interpreter of the original "embed_server" conda env is ${PYLATE_PYTHON}; both
# come from configs/paths.env, sourced below. Needs a GPU with >30 GB free. Run: bash benchmarks/retrieval/build_bc_rmcb_dl8192.sh
# reindex_rmcb_8192.py (next to this file) is the resumable replacement written after this script.
#
# BUILD Reason-ModernColBERT at 8192 — the model's NATIVE document window.
#
# LINEAGE. bc_reason_mcb was built at doc-len 512 to match an earlier LateOn run, and every
# result in this project used it. bc_rmcb_dl4096 removed most of that handicap. This is the
# last step: the model card specifies 8192 and the document REPL agent's branch measured 512 -> 8192 as
# 82.00% -> 87.00% on 200 paired questions with HIS scaffold.
#
# WHAT WE ALREADY KNOW, so nobody repeats it:
#   * On OUR scaffold 4096 did NOT beat 512 (q0-99: 79/100 vs 83/100), but the breakage rate
#     (11.6%) sat inside this benchmark's own run-to-run noise (~13.5% predicted), so that is
#     "no detectable effect", NOT "4096 is worse".
#   * Of the 21 questions still failing at 4096, ELEVEN have a gold document longer than 8192
#     tokens (up to 174,413). 8192 cannot reach those either. The expected gain is bounded.
#
# THE TWO FAILURE MODES THAT HAVE ACTUALLY BITTEN, both recorded rather than guessed:
#   1. PLAID construction holds a chunk's embeddings on the GPU. The 4096 build died after 2h
#      when only 7.7 GB of the card was free; the document REPL agent's 8192 attempt OOM'd trying to
#      allocate 18.18 GiB. Hence the >30 GB gate below.
#   2. nvidia-smi is BROKEN on this host (kernel module 595.71.05 vs NVML 595.91), so the
#      4096 script's gate cannot run. It is replaced with a torch mem_get_info check, which
#      goes through the CUDA runtime and works.
#
# --batch 4 (the 4096 build used 8): each sequence is twice as long, so halve the batch.
set -u
# Refactor: configs/paths.env supplies RETRIEVAL_EVAL_ROOT and PYLATE_PYTHON (was: cd <home>/master and an absolute
# conda-env interpreter path).
_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
set -a; . "${_REPO}/configs/paths.env"; set +a
cd "${RETRIEVAL_EVAL_ROOT:?set RETRIEVAL_EVAL_ROOT in configs/paths.env}"
: "${PYLATE_PYTHON:?set PYLATE_PYTHON in configs/paths.env}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say() { printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a logs/bc_rmcb_dl8192.log; }

need=70   # GB disk; 512 is 6 GB, 4096 is 30 GB, so 8192 should land near 44 GB
free=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
[ "$free" -gt "$need" ] || { say "ABORT: only ${free}G disk free, want >${need}G"; exit 1; }

# nvidia-smi is unusable here; ask the CUDA runtime instead.
gpufree=$("${PYLATE_PYTHON}" -c \
  "import torch;f,_=torch.cuda.mem_get_info(0);print(int(f/1048576))" 2>/dev/null | tail -1)
[ -n "$gpufree" ] && [ "$gpufree" -gt 30000 ] || {
  say "ABORT: only ${gpufree:-0}MiB GPU free, need >30000 for PLAID construction."
  say "       This is exactly what killed the 4096 build after 2 hours."; exit 1; }
say "disk free ${free}G, GPU free ${gpufree}MiB"
say "building bc_rmcb_dl8192 — Reason-ModernColBERT, doc_len 8192, query_len 256, 100,195 docs"
say "expect ~3-4h encoding plus PLAID construction; the 4096 encode took 8,660s"

"${PYLATE_PYTHON}" scripts/retrieval_eval.py \
  --task browsecomp --retriever pylate --model lightonai/Reason-ModernColBERT \
  --k 100 --doc-len 8192 --query-len 256 --batch 4 --qbatch 4 \
  --plaid-batch 4096 --n-full 512 --trust-remote-code \
  --index-name bc_rmcb_dl8192 --out logs/bc_rmcb_dl8192_retrieval.json \
  > logs/bc_rmcb_dl8192_build.log 2>&1
rc=$?
if [ $rc -ne 0 ] || [ ! -s logs/bc_rmcb_dl8192_retrieval.json ]; then
  say "FAILED (exit $rc) — see logs/bc_rmcb_dl8192_build.log"; tail -5 logs/bc_rmcb_dl8192_build.log | tee -a logs/bc_rmcb_dl8192.log; exit 1
fi
say "index size: $(du -sh .pylate_idx/bc_rmcb_dl8192 2>/dev/null | cut -f1)"
say "RETRIEVAL-ONLY GATE (830 queries, doc_len 8192):"
grep -aE "^  (NDCG|Recall)@" logs/bc_rmcb_dl8192_build.log | tail -6 | tee -a logs/bc_rmcb_dl8192.log
say "compare: doc_len 512 baseline Recall@10 .1971, Recall@50 .3196, NDCG@10 .1545"
say "NOTE the document REPL agent work measured first-stage recall going DOWN at 8192 (R@10 13.2% vs 18.8%)"
say "     while end-to-end accuracy went UP — so a lower gate number here is not disqualifying."
say "done"
