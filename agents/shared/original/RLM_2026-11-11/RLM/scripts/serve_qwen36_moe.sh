#!/usr/bin/env bash
# Serve Qwen3.6-35B-A3B. 35B total but only 3B ACTIVE per token (256 experts, 8 routed),
# against Qwen3.8-27B's fully-dense 27B. Decode is memory-bandwidth bound, so ACTIVE
# parameters set the token rate -- this is the same reason gpt-oss-20b measured ~7x faster.
#
# WHY THIS MODEL AND NOT gpt-oss-20b. gpt-oss was 7x faster but its OpenAI "harmony" channel
# format broke the agent loop: with the real 3,806-char system prompt it put everything on the
# analysis channel and returned message.content = None on 8 of 8 calls, so the rlm library's
# find_code_blocks() got None. Qwen3.6 is the SAME FAMILY as our current model, so the chat
# template and ```repl``` convention are unchanged -- none of that breakage should recur.
#
# FP8 ON AMPERE: the A100 has no FP8 tensor cores, but vLLM runs FP8 weight-only through
# Marlin (exactly how it ran gpt-oss's MXFP4 here). Memory and bandwidth savings survive;
# compute stays bf16. If the FP8 repo is unavailable the chain falls back to bf16, which needs
# ~70 GB and leaves little KV -- the launcher reports which it used.
set -euo pipefail
ENV_BIN=/home/extstu_00/miniconda3/envs/vllm/bin
MODEL=${MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}
PORT=${PORT:-8030}
MAX_LEN=${MAX_LEN:-262144}   # native; 131072 overflowed on BrowseComp+ (2026-10-07)
UTIL=${UTIL:-0.90}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
export PATH="$ENV_BIN:$PATH"
export HF_HOME=/home/extstu_00/.cache/huggingface
export CUDA_VISIBLE_DEVICES=${GPU:-0}
# GEN_OVERRIDE e.g. '{"max_new_tokens": 16384}': default per-call output cap. Qwen3.6 can loop
# inside one turn until it fills the whole context (BrowseComp+ smoke, 2026-10-07).
echo "[serve] $MODEL port=$PORT max-len=$MAX_LEN max-num-seqs=$MAX_NUM_SEQS"
exec "$ENV_BIN/vllm" serve "$MODEL" \
    --served-model-name qwen36-35b-a3b \
    --port "$PORT" --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$UTIL" --max-num-seqs "$MAX_NUM_SEQS" \
    ${GEN_OVERRIDE:+--override-generation-config "$GEN_OVERRIDE"}
