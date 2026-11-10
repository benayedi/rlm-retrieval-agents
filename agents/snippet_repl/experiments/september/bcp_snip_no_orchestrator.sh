#!/bin/bash
# PUBLICATION COPY of scripts/run_no_orchestrator.sh: the launcher of run rlmfix24 (part of row BCP-11, GPT-5,
# Reason-ModernColBERT on :8024) for run 19's 17 failures only; the question list was /tmp/noorch.txt and is now the
# file named by $NOORCH_QIDS_FILE (whitespace-separated question indices; its contents are not preserved: UNKNOWN).
# The other 83 rlmfix24 questions (q0-99) and runs rlmfix25/29/30/31 were launched by scripts that are not in the
# repository (see ../../configs/BCP-11.env).
# Changes (see ../../CHANGES_FROM_ORIGINAL.md): working directory $RUNS_DIR (was <original machine>/master, a checkout
# that no longer exists), so logs/... below is $RUNS_DIR/logs/...; interpreter $PYTHON (was ./.venv/bin/python with rlm
# installed editable); PYTHONPATH = rlm + shared + dspy_rlm/src and NO startup hook, as the original had none (so the
# reasoning_effort below was not sent to Azure, see issues); the harness is src/harness.py (was
# examples/browsecomp_eval_official_rlm.py). The original's .env may have set further variables: UNKNOWN.
# CODE VERSION: these runs were made by an EARLIER harness than src/harness.py (their records lack the child_* fields
# that the September commit 7685538 already writes); with every switch below at its run value the prompt/tool path
# of 7685538 and of src/ are identical, but the exact file that ran is not preserved.
#
# Original header:
# OFFICIAL RLM — run 24.  ORCHESTRATOR ADDENDUM OFF, on run 19's 17 failures.
#
# ONE VARIABLE vs run 19: orchestrator=False. The system prompt drops from 7,035 to 3,779
# characters — the addendum is 3,256 chars, 46% of it, and the largest single block.
#
# IT IS ENTIRELY THE AUTHORS' TEXT (rlm/utils/prompts.py, orchestrator=True is their default):
# "act as an orchestrator, not a solver"; decompose and sketch the concrete sequence of turns;
# push every long-context operation into llm_query; sub-LLMs have no REPL; the two-axis
# sub-call budget (~100K chars/prompt, ~20 prompts/batch); reserve your own tokens for
# high-level decisions. It has been ON in all 23 runs and has NEVER been ablated.
#
# WHY IT MIGHT HELP: hosted MCP scores .8500 — the best arm measured here — with NO
# orchestration guidance at all, a four-sentence task prompt and nine-word tool descriptions.
# Across this session, instructions that tell the agent HOW TO WORK have measured as zero
# (change-the-entity: query overlap 0.273 vs 0.294, i.e. backwards; the 0.30 score threshold:
# rank-4/5 fetches 17% vs 18%; "open the document": get_document stayed at 0), while facts
# that correct a false belief did move behaviour. The addendum is 3,256 characters of the
# first kind.
#
# THE SET AND ITS BIAS: run 19's 17 failures, chosen BECAUSE run 19 failed them. Any win beats
# zero and none of it generalises — seven of them (q6, 20, 41, 56, 60, 66, 70) have never been
# solved by ANY arm, so the realistic ceiling here is about 10. If several convert, the ablation
# earns a full 100; if none do, it does not.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
mkdir -p "${RUNS_DIR:?}/logs"; cd "$RUNS_DIR"
say() { printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a logs/rlmoffnoorch.log; }
curl -s -m10 http://localhost:8024/health | grep -q 'bc_reason_mcb' || { say "ABORT: 8024 wrong index"; exit 1; }
set -a; . "${AZ_ENV:?set AZ_ENV in configs/paths.env (file with AZ_EP and AZ_KEY)}"; set +a
# (refactor) the original sourced <original machine>/master/.env, which defined the Azure variables directly:
export AZURE_OPENAI_API_KEY="$AZ_KEY" AZURE_OPENAI_ENDPOINT="$AZ_EP/" AZURE_OPENAI_MODEL=${AZURE_DEPLOYMENT_GPT5:?}
export PYTHONUNBUFFERED=1 EMBEDDING_SERVER_URL=http://localhost:8024 PYTHONPATH=$(snip_pythonpath "")
export AZURE_OPENAI_REASONING_EFFORT=high SUB_LM_REASONING_EFFORT=medium
export AZURE_OPENAI_API_VERSION=2025-04-01-preview
export LIGHTON_TOOL_DOCS=1 LIGHTON_TASK_TEXT=1
export SEARCH_SNIPPET_TOKENS=512 OFFICIAL_TOP_K=5
export ORCHESTRATOR=0                 # <- THE SINGLE VARIABLE
export RLM_TRAJ_DIR=logs/rlm_traj24
unset MAX_TOP_K SEARCH_SNIPPET_LEN
Q=($(cat "${NOORCH_QIDS_FILE:?file with the question indices to run (was /tmp/noorch.txt)}"))
W=${W:-5}
for w in $(seq 0 $((W-1))); do
(
  for idx in $(seq $w $W $((${#Q[@]}-1))); do
    q=${Q[$idx]}
    "$PYTHON" "$SRC/harness.py" --start $q --end $((q+1)) \
      --max-iterations 100 --max-depth 1 --out logs/rlmfix24_q$q > logs/rlmfix24_q$q.log 2>&1
  done
) &
done
say "launched ${#Q[@]} run-19 failures with orchestrator=False"
