#!/usr/bin/env bash
# 2026-10-21, queued after scripts/frames_title_chain.sh: FRAMES full-doc RLM, 25 searches / 50 turns, ONE change vs
# logs/frames/sup_q36_bm25f_s25*: retrieval through the novelty + Qwen3-Reranker-4B proxy (:8042, BM25 top-30, the
# settings of the 10-search test logs/frames/sup_q36_bm25f_rr). No title clause. All 824, same shuffled split.
cd /home/extstu_00/RLM
PYV=/home/extstu_00/miniconda3/envs/vllm/bin/python
export PYTHONPATH=/home/extstu_00/rlmrepo OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
say() { printf '%s  [s25-rr] %s\n' "$(date '+%F %T')" "$*"; }
say "waiting for the title run"
while ps -eo args | awk '/^(\/bin\/)?bash scripts\/frames_title_chain[.]sh/ && !/awk/' | grep -q .; do sleep 120; done
if ! curl -s -m5 127.0.0.1:8042/stats >/dev/null; then
  CUDA_VISIBLE_DEVICES=1 HF_HOME=/home/extstu_00/.cache/huggingface POLICY=novelty+rerank UPSTREAM=http://127.0.0.1:8041 \
    QUERIES_FILE=/home/extstu_00/RLM/data/frames/frames_queries.jsonl RERANK_MEM_FRACTION=0.20 RERANK_TOK_BUDGET=8000 \
    setsid nohup $PYV -u scripts/bcp_rerank_proxy.py --port 8042 > logs/frames/rerank_proxy_8042_s25.log 2>&1 < /dev/null &
  for i in $(seq 1 60); do grep -aq READY logs/frames/rerank_proxy_8042_s25.log 2>/dev/null && break; sleep 5; done
fi
grep -ao "READY.*" logs/frames/rerank_proxy_8042_s25.log || { say "ABORT: proxy not ready"; exit 1; }
run() {  # gpu-port ids-file out-dir
  SERVER=http://127.0.0.1:8042 BM25_PROMPT=1 MAX_SEARCHES=25 MAX_ITERS=50 BASE_URL=http://localhost:$1/v1 W=8 IDS_FILE=$2 OUT=$3 scripts/run_frames_sup.sh
}
say "start: 412 q on GPU 0, 412 q on GPU 1"
run 8030 data/frames/gepa_splits/all824_shuf_gpu0.txt logs/frames/sup_q36_s25_rr_g0 > logs/frames/s25_rr_g0.log 2>&1 & P0=$!
sleep 120
run 8031 data/frames/gepa_splits/all824_shuf_gpu1.txt logs/frames/sup_q36_s25_rr_g1 > logs/frames/s25_rr_g1.log 2>&1 & P1=$!
wait $P0 $P1
say "proxy stats: $(curl -s 127.0.0.1:8042/stats)"
ARM=rr $PYV scripts/frames_title_compare.py --final
say "done"
