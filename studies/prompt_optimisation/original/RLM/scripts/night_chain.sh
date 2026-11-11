#!/usr/bin/env bash
# UNATTENDED: validate revert -> embedding to CPU -> dual vLLM -> probe -> GEPA.
# Every phase logs to logs/night/chain.log and aborts loudly rather than continuing broken.
set -uo pipefail
cd /home/extstu_00/RLM
PYV=/home/extstu_00/miniconda3/envs/vllm/bin/python
L=logs/night; mkdir -p $L
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L/chain.log; }
die(){ say "ABORT: $*"; exit 1; }
export PYTHONPATH=/home/extstu_00/rlmrepo
export HF_HOME=/home/extstu_00/.cache/huggingface

# ---------- PHASE A: the smoke must have passed ----------
say "PHASE A: waiting for the Qwen revert smoke"
while pgrep -f 'python.*run_obliq_mine[.]py --ids-file' >/dev/null 2>&1; do sleep 30; done
[ -f logs/qwen_revert_smoke.json ] || die "smoke produced no json"
$PYV - <<'PY' || exit 1
import json,sys
d=json.load(open('logs/qwen_revert_smoke.json'))
err=sum(1 for r in d if r.get('error'))
it=[r.get('iterations') or 0 for r in d]
print(f"  smoke: n={len(d)} errors={err} iters={it}")
# The gpt-oss failure mode was: every question errors in 1-3 s with iters=0. Gate on that.
if err == len(d): sys.exit("every smoke question errored -- harness is broken on Qwen too")
if max(it) == 0: sys.exit("no question ran a single iteration -- agent loop is not executing")
print("  smoke PASSED")
PY
say "PHASE A passed"

# ---------- PHASE B: embedding server GPU -> CPU, same port 8010 ----------
# Measured: CPU 1.351 s/query vs GPU 0.204 s. At 9.2 searches/rollout that is +10.6 s on a
# 1335 s rollout = 0.8%. It frees 14.7 GiB on GPU1, taking the second vLLM's KV pool from
# 180,988 to 421,068 tokens -- 2.3 -> 5.4 concurrent 77k questions. Worth 0.8%.
say "PHASE B: moving embedding server to CPU on :8010"
for p in $(ps -eo pid,cmd | grep '[m]ulti_search_server.py' | awk '{print $1}'); do kill $p 2>/dev/null; done
sleep 20
SNAP=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-Embedding-8B/snapshots/*/ | head -1)
CUDA_VISIBLE_DEVICES="" setsid nohup $PYV -u scripts/multi_search_server.py \
  --model-path "$SNAP" --index obliq-des-twitter=/home/extstu_00/indexes/obliq-des-twitter \
  --default-index obliq-des-twitter --nprobe 64 --port 8010 --device cpu \
  > $L/embed8010_cpu.log 2>&1 < /dev/null &
for i in $(seq 1 90); do curl -s -m 3 http://localhost:8010/health >/dev/null 2>&1 && break; sleep 10; done
curl -s -m 5 http://localhost:8010/health >/dev/null || die "CPU embedding server never came up"
say "PHASE B done — GPU1 freed"

# ---------- PHASE C: two vLLM instances, NO MTP ----------
# MTP was measured to COST 76,693 KV tokens (277,701 with vs 354,394 without). KV capacity is
# our binding constraint -- we were running 4.6 concurrent questions on a 64 KB/token model --
# so a 22% KV cut to speed single-stream decode is the wrong trade for a batch workload.
say "PHASE C: GPU0 :8030 WITH MTP, GPU1 :8031 WITHOUT -- the probe decides"
for p in $(ps -eo pid,cmd | grep '[v]llm serve' | awk '{print $1}'); do kill $p 2>/dev/null; done
sleep 40
GPU=0 PORT=8030 MAX_NUM_SEQS=64 USE_MTP=1 setsid nohup ./scripts/serve_qwen27b_bf16_tuned.sh \
  > $L/vllm_gpu0.log 2>&1 < /dev/null &
sleep 30
GPU=1 PORT=8031 MAX_NUM_SEQS=64 USE_MTP=0 setsid nohup ./scripts/serve_qwen27b_bf16_tuned.sh \
  > $L/vllm_gpu1.log 2>&1 < /dev/null &
for i in $(seq 1 120); do
  a=$(curl -s -m 3 http://localhost:8030/v1/models >/dev/null 2>&1 && echo 1 || echo 0)
  b=$(curl -s -m 3 http://localhost:8031/v1/models >/dev/null 2>&1 && echo 1 || echo 0)
  [ "$a$b" = "11" ] && break; sleep 10
done
[ "$(curl -s -m 5 http://localhost:8030/v1/models >/dev/null 2>&1 && echo 1)" = "1" ] || die ":8030 down"
[ "$(curl -s -m 5 http://localhost:8031/v1/models >/dev/null 2>&1 && echo 1)" = "1" ] || die ":8031 down"
for P in 8030 8031; do
  say "  :$P KV = $(curl -s -m 5 http://localhost:$P/metrics | grep -oE 'kv_cache_size_tokens="[0-9]*"' | head -1)"
done
say "PHASE C done — two instances live"

# ---------- PHASE D: throughput probe across both ports ----------
say "PHASE D: throughput probe"
$PYV -u scripts/dual_probe.py 2>&1 | tee $L/probe.log
BEST_W=$(grep -oE '^BEST_W=[0-9]+' $L/probe.log | tail -1 | cut -d= -f2)
[ -n "$BEST_W" ] || BEST_W=12
say "PHASE D done — chosen W=$BEST_W"

# ---------- PHASE E: GEPA, paper-shaped ----------
say "PHASE E: GEPA (valset=50 pareto, minibatch=3, budget 400, W=$BEST_W across 2 ports)"
$PYV -u scripts/gepa_night.py --workers "$BEST_W" --valset 50 --minibatch 3 \
  --max-metric-calls 400 2>&1 | tee $L/gepa.log
say "NIGHT CHAIN COMPLETE"
