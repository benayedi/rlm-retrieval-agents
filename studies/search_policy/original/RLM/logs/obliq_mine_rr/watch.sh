#!/usr/bin/env bash
# 10-13: watch our-harness OBLIQ ctrl vs rerank arms every 30 min; final comparison when both finish.
cd /home/extstu_00/RLM; PY=/home/extstu_00/miniconda3/envs/vllm/bin/python
while pgrep -f "run_obliq_mine[.]py" >/dev/null; do
  sleep 1800; echo "--- $(date '+%m-%d %H:%M')"
  curl -s -m 5 localhost:8775/stats; echo
  grep -h -i -E "error|traceback" logs/obliq_mine_rr/*_dev100_s*.log | sort | uniq -c | head -5
  timeout 120 $PY scripts/obliq_mine_rr_compare.py
done
echo "== BOTH ARMS FINISHED $(date '+%m-%d %H:%M')"; $PY scripts/obliq_mine_rr_compare.py
