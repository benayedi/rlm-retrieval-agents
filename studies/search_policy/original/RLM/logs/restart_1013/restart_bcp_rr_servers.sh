# Restart ColBERT :8766 and reranker proxy :8772 for the BC+ full-830 resume (stopped 10-14 ~20:15 UTC)
cd /home/extstu_00/RLM
CUDA_VISIBLE_DEVICES=1 nohup /home/extstu_00/miniconda3/envs/pylate/bin/python -u /home/extstu_00/suprlm/src/retrieval_server_colbert.py --port 8766 > logs/colbert8766_restart.log 2>&1 &
# wait for ColBERT to load before starting the proxy
CUDA_VISIBLE_DEVICES=1 POLICY=novelty+rerank DEPTH=30 RERANK_TOK_BUDGET=8000 RERANK_MEM_FRACTION=0.15 \
  nohup /home/extstu_00/miniconda3/envs/vllm/bin/python -u scripts/bcp_rerank_proxy.py --port 8772 >> logs/bcp_policy_proxy8772_full830.log 2>&1 &
