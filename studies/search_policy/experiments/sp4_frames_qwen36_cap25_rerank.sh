#!/usr/bin/env bash
# SP-4: FRAMES, all 824 questions, document REPL agent, Qwen3.6-35B-A3B, search cap 25 / 50 turns, BM25 + novelty filter
# + Qwen3-Reranker-4B (BM25 top 30). Control: FRC-1 (the same agent and caps on plain BM25).
# From original/RLM/scripts/frames_s25_rr_chain.sh (2026-10-21). Arguments and environment unchanged; locations from
# configs/paths.env; the agent wrapper is agents/document_repl/experiments/frames_qwen.sh (was scripts/run_frames_sup.sh);
# outputs and logs go to $RUNS_DIR/frames/ (original logs/frames/); the id files are $FRAMES_DATA_DIR/gepa_splits/
# all824_shuf_gpu{0,1}.txt (original data/frames/gepa_splits/). Kept as in the original: the wait for
# scripts/frames_title_chain.sh (a run of the original repository; returns at once when it is not running), and the final
# comparison `ARM=rr frames_title_compare.py --final`, a script of the original repository that is not part of this
# repository: it runs only when FRAMES_TITLE_COMPARE names it. frames_qwen.sh also needs FRAMES_GRADE_SUP (the interim
# Qwen3.6 grader of the original repository). Prerequisites: the FRAMES BM25 server on :8041 (--protocol sup), vLLM
# qwen36-35b-a3b on :8030 and :8031.
#   experiments/sp4_frames_qwen36_cap25_rerank.sh
set -u
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$REPO"                                   # original: cd <original repo>
PYV=$PYTHON                                  # original: the vllm conda env's python
: "${FRAMES_DATA_DIR:?set FRAMES_DATA_DIR in configs/paths.env}" "${RLMLIB_PATH:?set RLMLIB_PATH}"
export PYTHONPATH=$RLMLIB_PATH OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
L=$RUNS_DIR/frames; mkdir -p "$L"            # original: logs/frames
say() { printf '%s  [s25-rr] %s\n' "$(date '+%F %T')" "$*"; }
say "waiting for the title run"
while ps -eo args | awk '/^(\/bin\/)?bash scripts\/frames_title_chain[.]sh/ && !/awk/' | grep -q .; do sleep 120; done
if ! curl -s -m5 127.0.0.1:8042/stats >/dev/null; then
  CUDA_VISIBLE_DEVICES=1 HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} POLICY=novelty+rerank UPSTREAM=http://127.0.0.1:8041 \
    QUERIES_FILE=$FRAMES_DATA_DIR/frames_queries.jsonl RERANK_MEM_FRACTION=0.20 RERANK_TOK_BUDGET=8000 \
    setsid nohup $PYV -u "$PROXY" --port 8042 > $L/rerank_proxy_8042_s25.log 2>&1 < /dev/null &
  for i in $(seq 1 60); do grep -aq READY $L/rerank_proxy_8042_s25.log 2>/dev/null && break; sleep 5; done
fi
grep -ao "READY.*" $L/rerank_proxy_8042_s25.log || { say "ABORT: proxy not ready"; exit 1; }
run() {  # gpu-port ids-file out-dir
  SERVER=http://127.0.0.1:8042 BM25_PROMPT=1 MAX_SEARCHES=25 MAX_ITERS=50 BASE_URL=http://localhost:$1/v1 W=8 IDS_FILE=$2 OUT=$3 \
    "$REPO/agents/document_repl/experiments/frames_qwen.sh"
}
say "start: 412 q on GPU 0, 412 q on GPU 1"
FR_IDS=$REPO/agents/document_repl/configs/frames_ids   # original: <FRAMES_DATA_DIR>/gepa_splits (byte-identical copies)
run 8030 $FR_IDS/all824_shuf_gpu0.txt $L/sup_q36_s25_rr_g0 > $L/s25_rr_g0.log 2>&1 & P0=$!
sleep 120
run 8031 $FR_IDS/all824_shuf_gpu1.txt $L/sup_q36_s25_rr_g1 > $L/s25_rr_g1.log 2>&1 & P1=$!
wait $P0 $P1
say "proxy stats: $(curl -s 127.0.0.1:8042/stats)"
if [ -n "${FRAMES_TITLE_COMPARE:-}" ]; then ARM=rr $PYV "$FRAMES_TITLE_COMPARE" --final   # original: ARM=rr $PYV scripts/frames_title_compare.py --final
else say "FRAMES_TITLE_COMPARE not set: final comparison skipped (see analysis/ and results/follow_up/)"; fi
say "done"
