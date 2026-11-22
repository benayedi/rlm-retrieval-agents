#!/usr/bin/env bash
# 10-14: watch the single BC+ remaining-reranker run (rest300, 24 workers); trace health + full-830 comparison every 30 min.
cd /home/extstu_00/RLM; PY=/home/extstu_00/miniconda3/envs/vllm/bin/python
while true; do
  echo "--- $(date '+%m-%d %H:%M')"
  echo "rest300: $(GEPA_OUT=logs/bcp_rest300_rr timeout 120 $PY scripts/bcp_trace_health.py --since 1791989276 2>&1 | tail -n 3)"
  curl -s -m5 localhost:8772/stats; echo
  timeout 120 $PY scripts/bcp_full830_compare.py 2>&1 | tail -n 5
  if ! pgrep -f "gepa_bcp[.]py --seed-split" >/dev/null; then echo "== FINISHED"; timeout 120 $PY scripts/bcp_full830_compare.py; exit 0; fi
  sleep 1800
done
