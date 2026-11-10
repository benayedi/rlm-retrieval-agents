"""BrowseComp-Plus (830 q), full-document RLM with Qwen3.6, dense Qwen3-Embedding-8B retrieval (2026-10-27).

Everything is the Qwen3.6 full-doc rows' setup (scripts/gepa_bcp.py: seed prompt, design B, 30 turns, max 10 searches,
compaction, grounding gate, solo prompt, fence-tolerant + max 5 blocks + presence penalty 1.5, judge grade_local on
Qwen3.6 as an interim grade), so the only change against the ColBERT row (0.5795) and the BM25 row (0.2241) is the
retriever: the supervisor's own exact dense server suprlm/src/retrieval_server.py on :8765 (Qwen3-Embedding-8B, exact
cosine over the Tevatron embeddings), the server behind his GPT-5 dense row (0.8313). Both model servers (:8030 GPU 0,
:8031 GPU 1) are used alternately. Resumable through gepa_bcp's cache. Graded afterwards with gpt-4.1.
  python scripts/run5d/bcp_fd_qwen_dense.py [workers]
"""
import hashlib, importlib.util, json, os, sys

R = "/home/extstu_00/RLM"
OUT = f"{R}/logs/run5d/bcp_fd_qwen_dense"
os.environ.update(GEPA_BCP_SERVERS="http://localhost:8030/v1,http://localhost:8031/v1",
                  GEPA_BCP_RETRIEVAL="http://127.0.0.1:8765", GEPA_BCP_MODEL="qwen36-35b-a3b", GEPA_OUT=OUT,
                  LOCAL_JUDGE_MODEL="qwen36-35b-a3b", SUP_FENCE_TOLERANT="1", SUP_MAX_BLOCKS="5",
                  SUP_PRESENCE_PENALTY="1.5", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
for k in ("QUERIES_FILE", "GEPA_PROMPTS", "BM25_CORPUS"):
    os.environ.pop(k, None)
sys.path.insert(0, "/home/extstu_00/rlmrepo")
_s = importlib.util.spec_from_file_location("gepa_bcp", f"{R}/scripts/gepa_bcp.py")
G = importlib.util.module_from_spec(_s); sys.modules["gepa_bcp"] = G; _s.loader.exec_module(G)
G.FINGERPRINT = hashlib.sha1((G.FINGERPRINT + "|dense-exact-8765").encode()).hexdigest()[:8]

if __name__ == "__main__":
    W = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    ids = [str(json.loads(l)["query_id"]) for l in open("/home/extstu_00/suprlm/data/prepared/queries.jsonl")]
    if os.environ.get("BCP_IDS_LIMIT"):
        ids = ids[:int(os.environ["BCP_IDS_LIMIT"])]
    G._cache_load()
    seed = G.seed_blocks()
    G._log(f"[dense] seed {G.ckey(seed)} fingerprint {G.FINGERPRINT} on {len(ids)} ids via {G.RETRIEVAL}, model {G.SERVERS}")
    G.run_batch(seed, ids, W)
