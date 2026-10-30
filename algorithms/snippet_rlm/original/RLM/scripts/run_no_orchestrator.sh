#!/bin/bash
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
cd /home/extstu_00/master
say() { printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a logs/rlmoffnoorch.log; }
curl -s -m10 http://localhost:8024/health | grep -q 'bc_reason_mcb' || { say "ABORT: 8024 wrong index"; exit 1; }
set -a; . /home/extstu_00/master/.env 2>/dev/null; set +a
export PYTHONUNBUFFERED=1 EMBEDDING_SERVER_URL=http://localhost:8024
export AZURE_OPENAI_REASONING_EFFORT=high SUB_LM_REASONING_EFFORT=medium
export AZURE_OPENAI_API_VERSION=2025-04-01-preview
export LIGHTON_TOOL_DOCS=1 LIGHTON_TASK_TEXT=1
export SEARCH_SNIPPET_TOKENS=512 OFFICIAL_TOP_K=5
export ORCHESTRATOR=0                 # <- THE SINGLE VARIABLE
export RLM_TRAJ_DIR=logs/rlm_traj24
unset MAX_TOP_K SEARCH_SNIPPET_LEN
Q=($(cat /tmp/noorch.txt))
W=${W:-5}
for w in $(seq 0 $((W-1))); do
(
  for idx in $(seq $w $W $((${#Q[@]}-1))); do
    q=${Q[$idx]}
    ./.venv/bin/python examples/browsecomp_eval_official_rlm.py --start $q --end $((q+1)) \
      --max-iterations 100 --max-depth 1 --out logs/rlmfix24_q$q > logs/rlmfix24_q$q.log 2>&1
  done
) &
done
say "launched ${#Q[@]} run-19 failures with orchestrator=False"
