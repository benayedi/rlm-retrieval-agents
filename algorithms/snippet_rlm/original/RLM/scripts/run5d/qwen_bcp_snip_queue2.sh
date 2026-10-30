#!/usr/bin/env bash
# Qwen3.6 BrowseComp+ snippet runs one after another (BM25 -> dense -> ColBERT), 8 workers each, 2026-10-28 relaunch of
# qwen_bcp_snip_queue.sh through bcp_snip2.sh: repl-fence cue + prose-final submission (scripts/run_bcp_snippet.py) and
# 50 turns. The first attempt (no cue, 100 turns) is kept in logs/run5d/full/bcp_snip_qwen_bm25_nocue (62 results in
# 13.5 h, 32% of questions killed at 2 h). Fresh directories: every question is re-run under the new configuration.
cd /home/extstu_00/RLM
for r in bm25:http://localhost:8044 dense:http://localhost:8001 colbert:http://localhost:8024; do
  scripts/run5d/bcp_snip2.sh qwen ${r#*:} logs/run5d/full/bcp_snip_qwen_${r%%:*} 8 > logs/run5d/full/bcp_snip_qwen_${r%%:*}.out 2>&1
done
echo "$(date '+%F %T') QWEN SNIPPET QUEUE DONE"
