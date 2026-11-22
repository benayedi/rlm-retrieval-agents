#!/usr/bin/env bash
# Overnight 10-12/13: watch OBLIQ ctrl vs rerank arms; when both finish, resume the paused BC+ pareto/feedback reranker runs.
cd /home/extstu_00/RLM
PY=/home/extstu_00/miniconda3/envs/vllm/bin/python
B=0
while true; do
  echo "--- $(date '+%m-%d %H:%M')"
  for a in ctrl rr; do
    d=/home/extstu_00/suprlm/runs/obliq_q36_$a
    n=$(ls $d/q*.json 2>/dev/null | wc -l)
    st=$($PY -c "import json,glob,collections;print(dict(collections.Counter(json.load(open(f)).get('status') for f in glob.glob('$d/q*.json'))))" 2>/dev/null)
    echo "obliq $a: $n/281 $st"
  done
  curl -s -m 5 localhost:8774/stats; echo
  timeout 120 $PY scripts/obliq_rr_compare.py 2>&1 | grep -E "^== |NDCG@10|delta|status" | head -12
  if [ $B = 0 ] && grep -q "'test200': n=200" logs/bcp_test200_d30g/run.log; then
    B=1; echo "== BC+ ARM B DONE: d30g(ctrl) vs d100g(treat)"
    SPLIT=test200 CTRL_CACHE=logs/bcp_test200_d30g/cache.json TREAT_CACHE=logs/bcp_test200_d100g/cache.json timeout 300 $PY scripts/bcp_policy_compare.py 2>&1 | tail -n 6
  fi
  if ! pgrep -f "run_experiment[.]py --name obliq_q36_" >/dev/null; then
    echo "== OBLIQ BOTH ARMS FINISHED"; timeout 300 $PY scripts/obliq_rr_compare.py
    for s in pareto feedback; do
      env -u GEPA_PROMPTS GEPA_BCP_MODEL=qwen36-35b-a3b SUP_FENCE_TOLERANT=1 SUP_MAX_BLOCKS=5 SUP_PRESENCE_PENALTY=1.5 \
        GEPA_BCP_SERVERS=http://localhost:8030/v1 GEPA_BCP_RETRIEVAL=http://127.0.0.1:8772 GEPA_OUT=logs/bcp_${s}_rr \
        GEPA_SPLIT=data/bcp_gepa_splits/split.json HF_HOME=/home/extstu_00/.cache/huggingface \
        setsid nohup $PY -u scripts/gepa_bcp.py --seed-split $s --workers 12 >> logs/bcp_${s}_rr/run.log 2>&1 < /dev/null &
    done
    echo "== resumed BC+ pareto/feedback reranker runs"; exit 0
  fi
  sleep 1800
done
