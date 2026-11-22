#!/usr/bin/env bash
# The document REPL agent with Qwen3.6-35B-A3B on OBLIQ-Twitter (281 questions), design B, fixed k 30, search cap 8,
# 12 turns. TWQ-2 = ARM ctrl (plain dense search through src/obliq_adapter.py on :8767); SP-7 = ARM rr (the novelty +
# reranker proxy on :8774, started by studies/search_policy/experiments/sp7_obliq_document_qwen36_rerank.sh).
#
# RECONSTRUCTED: no launcher file existed. The two arms were started together on 2026-10-12 20:55 by one command line in
# the Claude Code session transcript 8ba31816:
#   cd <FULLDOC_ROOT>; for arm in "ctrl 8767" "rr 8774"; do set -- $arm; env PYTHONPATH=<rlm> HF_HOME=<hf cache>
#     ALIBABA_TOKEN=EMPTY QUERIES_FILE=<FULLDOC_ROOT>/data/obliq_twitter/queries.jsonl
#     GEPA_PROMPTS=<FULLDOC_ROOT>/data/obliq_twitter/prompts_obliq.json SUP_FENCE_TOLERANT=1 SUP_MAX_BLOCKS=5
#     SUP_PRESENCE_PENALTY=1.5 setsid nohup <python> -u src/run_experiment.py --name obliq_q36_$1 ... --server
#     http://127.0.0.1:$2 ...
# The run_experiment.py arguments below are the launch_argv recorded in <FULLDOC_ROOT>/runs/obliq_q36_{ctrl,rr}/
# _provenance.json (identical in both arms except --name and --server). The records' prompt_id 00a9b02c72d6 is the
# one prompts.py computes with GEPA_PROMPTS = that prompts_obliq.json, read here from its byte-identical copy
# configs/obliq/prompts_obliq.json (configs/obliq/README.md). The question ids are <LOGS_ROOT>/obliq_rr/ids281.json
# (the same 281 ids as configs' all281.json, in another order). Results: $RUNS_DIR/obliq_q36_<arm>.
#   ARM=ctrl|rr [IDS_FILE=<ids.json>] experiments/obliq_twitter_qwen36.sh
set -u
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
ARM=${ARM:?set ARM=ctrl or rr}
case "$ARM" in ctrl) PORT=8767 ;; rr) PORT=8774 ;; *) echo "ARM must be ctrl or rr" >&2; exit 2 ;; esac
IDS_FILE=${IDS_FILE:-$ALG/configs/obliq_ids/ids281.json}   # original: <LOGS_ROOT>/obliq_rr/ids281.json (byte-identical copy)
curl -s -m10 -o /dev/null http://127.0.0.1:$PORT/health || { echo "ABORT: retrieval :$PORT down" >&2; exit 1; }
cd "$FULLDOC_ROOT"                           # original: cd <FULLDOC_ROOT> (the document REPL agent working directory)
env PYTHONPATH=$RLMLIB_PATH HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} ALIBABA_TOKEN=EMPTY \
  QUERIES_FILE=$FULLDOC_ROOT/data/obliq_twitter/queries.jsonl GEPA_PROMPTS=$ALG/configs/obliq/prompts_obliq.json \
  SUP_FENCE_TOLERANT=1 SUP_MAX_BLOCKS=5 SUP_PRESENCE_PENALTY=1.5 \
  "$PYTHON" -u "$SRC/run_experiment.py" --name obliq_q36_$ARM --ids-file "$IDS_FILE" --workers 10 --stagger 0.3 \
    --design B --server http://127.0.0.1:$PORT --backend openai --base-url http://localhost:8030/v1 \
    --api-key-env ALIBABA_TOKEN --root-model qwen36-35b-a3b --sub-model qwen36-35b-a3b --reasoning-effort medium \
    --sub-reasoning-effort medium --default-k 30 --fixed-k --max-searches 8 --max-iterations 12 --seed-context \
    --compaction --lean-prologue --no-delegation --solo-prompt --no-orchestrator --compaction-threshold-pct 0.85 \
    --max-concurrent-subcalls 4 --max-timeout 3600 --lm-timeout 1800 --save-trajectory
