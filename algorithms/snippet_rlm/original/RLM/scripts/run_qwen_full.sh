#!/bin/bash
# OFFICIAL RLM — run 32.  THE FULL DATASET (q0-829) on Qwen3-Embedding-8B.
#
# ONE VARIABLE vs runs 24/25/29/30/31: EMBEDDING_SERVER_URL 8024 -> 8001.
#   8024 = Reason-ModernColBERT, multi-vector late interaction (what LightOn used for 87.59)
#   8001 = Qwen3-Embedding-8B, single-vector dense, FAISS IVF-SQ8, nprobe 64, max-length 8192
#
# WHY THE FULL 830 AND NOT 100: run 23 measured this swap on q0-99 at .7500 vs .8300,
# gold_recall 0.73 vs 0.82, p=0.057 — the largest effect in the project, but borderline at
# n=100 and measured on the block that turned out to be the HARDEST of the four
# (q0-99 = .8300 against .8700/.8600/.8700 elsewhere). At 830 questions the retriever
# comparison stops being borderline.
#
# IT IS ALSO A NUMBER LIGHTON NEVER PUBLISHED. Their blog names Qwen3-Embed-8B as a competitor
# and reports no BrowseComp-Plus accuracy for it — only that Reason-ModernColBERT beats it.
# This measures it directly with an identical agent, which makes it an original contribution
# rather than a reproduction.
#
# EVERYTHING ELSE FROZEN: addendum OFF, sub-LM gpt-5 MEDIUM, LightOn task text and tool docs,
# dense-retriever line, no score sentence, "every turn must execute code", snippet 512 tokens,
# top_k 5 default, 60K ceiling, no Unknown escape.
#
# NOTE the prompt stays honest: "the index is a dense semantic retriever ... quotation marks
# have NO effect" is true of Qwen3-Embed-8B as well. Its scores ARE real cosine similarities,
# unlike the 8024 proxy's 1/rank — but we say nothing about scores, so nothing misleads.
#
# ~10 h at 8 shards. Resumable: per-question files, workers skip what already exists.
set -u
#
# Completes BrowseComp-Plus end to end. q0-399 are already done under this exact configuration
# (runs 24, 25, 29, 30): 343/400 = .8575, sd 1.9pp across blocks, 34.8M input tokens.
#
# CONFIG, unchanged and frozen: addendum OFF, sub-LM gpt-5 at MEDIUM, LightOn task text and
# tool docs, dense-retriever line, no score sentence, "every turn must execute code",
# snippet 512 tokens, top_k 5 default, 60K ceiling, no Unknown escape,
# Reason-ModernColBERT on 8024. NOTHING will be tuned to these questions.
#
# EXPECTED: ~3.5 h wall-clock at 8 shards, ~37M additional input tokens (scaling the observed
# 87k/question), bringing the full-dataset total to roughly 72M.
#
# Per-question result files mean this is resumable: the guard below skips anything already
# written, so it can be re-launched after an interruption without losing work.
set -u
cd /home/extstu_00/master
say() { printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a logs/rlmoffqwenfull.log; }
curl -s -m10 http://localhost:8001/health | grep -q ok || { say "ABORT: 8001 not healthy"; exit 1; }
say "8001 (Qwen3-Embedding-8B) verified"
set -a; . /home/extstu_00/master/.env 2>/dev/null; set +a
export PYTHONUNBUFFERED=1
export EMBEDDING_SERVER_URL=http://localhost:8001   # <- THE SINGLE VARIABLE
export AZURE_OPENAI_REASONING_EFFORT=high SUB_LM_REASONING_EFFORT=medium
export AZURE_OPENAI_API_VERSION=2025-04-01-preview
export LIGHTON_TOOL_DOCS=1 LIGHTON_TASK_TEXT=1
export SEARCH_SNIPPET_TOKENS=512 OFFICIAL_TOP_K=5
export ORCHESTRATOR=0
export RLM_TRAJ_DIR=logs/rlm_traj32
unset MAX_TOP_K SEARCH_SNIPPET_LEN SUB_LM_MODEL
W=${W:-8}
for w in $(seq 0 $((W-1))); do
(
  for q in $(seq $w $W 829); do
    [ -f logs/rlmfix32_q$q.json ] && continue
    ./.venv/bin/python examples/browsecomp_eval_official_rlm.py --start $q --end $((q+1)) \
      --max-iterations 100 --max-depth 1 --out logs/rlmfix32_q$q > logs/rlmfix32_q$q.log 2>&1
  done
) &
done
say "launched q0-829 (FULL dataset) on Qwen3-Embedding-8B across $W workers"
