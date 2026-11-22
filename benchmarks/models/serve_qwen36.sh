#!/usr/bin/env bash
# Serve Qwen3.6-35B-A3B-FP8 with vLLM for every Qwen3.6 run (snippet and document REPL agents, the search-policy and
# GEPA studies). From agents/shared/original/RLM_2026-11-11/RLM/scripts/serve_qwen36_moe.sh; arguments and defaults
# unchanged, only locations come from configs/paths.env (vLLM binary, HF cache).
#
# Two of the five settings Qwen3.6 needed (GEPA_BROWSECOMP_PLAN.md section 2) live here, on the server, not in the agent:
#   - native context 262,144 tokens (MAX_LEN; 131,072 overflowed on BrowseComp-Plus, 2026-10-07)
#   - 16,384-token output cap per call: ONLY when GEN_OVERRIDE='{"max_new_tokens": 16384}' is set (the script's default
#     leaves it off). The runs had it on: the saved vLLM logs show the non-default args
#       10-07 (GEPA run 1):        max_model_len 262144, override_generation_config {max_new_tokens: 16384}, gpu_memory_utilization 0.9, max_num_seqs 64
#       10-13 (BC+ search policy): the same with gpu_memory_utilization 0.7
#       10-27 (results-table runs): the same with gpu_memory_utilization 0.7
#     (logs/gepa_bcp/vllm8030.log, logs/restart_1013/vllm8030.log, logs/run5d/vllm_8030.log of the original repository).
# The other three (tolerant code fences, at most 5 code blocks per turn, presence penalty 1.5) are agent settings:
# SUP_FENCE_TOLERANT=1 SUP_MAX_BLOCKS=5 SUP_PRESENCE_PENALTY=1.5 in agents/document_repl/configs/*.env.
# To reproduce the runs:
#   UTIL=0.7 GEN_OVERRIDE='{"max_new_tokens": 16384}' GPU=0 PORT=8030 benchmarks/models/serve_qwen36.sh
#
# Original header:
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
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [ ! -f "$REPO_ROOT/configs/paths.env" ]; then
  echo "missing $REPO_ROOT/configs/paths.env: copy configs/paths.example.env and fill it in" >&2; exit 1
fi
set -a; . "$REPO_ROOT/configs/paths.env"; set +a
ENV_BIN=$(dirname "${VLLM_BIN:?set VLLM_BIN (the vllm executable) in configs/paths.env}")   # original: the vllm conda env's bin/
MODEL=${MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}
PORT=${PORT:-8030}
MAX_LEN=${MAX_LEN:-262144}   # native; 131072 overflowed on BrowseComp+ (2026-10-07)
UTIL=${UTIL:-0.90}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
export PATH="$ENV_BIN:$PATH"
export HF_HOME=${HF_HOME:?set HF_HOME in configs/paths.env}   # original: the user's ~/.cache/huggingface
export CUDA_VISIBLE_DEVICES=${GPU:-0}
# GEN_OVERRIDE e.g. '{"max_new_tokens": 16384}': default per-call output cap. Qwen3.6 can loop
# inside one turn until it fills the whole context (BrowseComp+ smoke, 2026-10-07).
echo "[serve] $MODEL port=$PORT max-len=$MAX_LEN max-num-seqs=$MAX_NUM_SEQS"
exec "$ENV_BIN/vllm" serve "$MODEL" \
    --served-model-name qwen36-35b-a3b \
    --port "$PORT" --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$UTIL" --max-num-seqs "$MAX_NUM_SEQS" \
    ${GEN_OVERRIDE:+--override-generation-config "$GEN_OVERRIDE"}
