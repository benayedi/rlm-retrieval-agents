"""BrowseComp-Plus, Full-Doc RLM harness, Qwen3.6, BM25 (DEFAULT Lucene k1=0.9 b=0.4, no reranker) on test200.

Row BCP-31 (test200 here; the rest of the 830 with BCP_IDS / BCP_OUT / BCP_SERVER, cache key 11d0c5bc6b).
From original/RLM/scripts/bcp_bm25_test.py.
    python src/bcp_bm25_split.py            # run (resumable via cache) + paired comparison
    python src/bcp_bm25_split.py --report   # comparison only
    BCP_IDS=<ids.txt> BCP_OUT=$RUNS_DIR/bcp_bm25_rest_gpu0 BCP_SERVER=http://localhost:8030/v1 python src/bcp_bm25_split.py

Paired against the ColBERT control: gepa_bcp2/cache.json key ee5ea32a1c (seed prompt, Qwen3.6,
fingerprint 29587b4e = fence-tolerant + max 5 blocks + presence penalty 1.5, retrieval :8766).
Everything is split_runner.py's (rollout args, grading, judge = grade_local on Qwen3.6), except:
  * retrieval  :8043 = a BM25 search server over BCP_BM25_INDEX_DIR speaking this harness's wire protocol
                 (frames_search_server.py --protocol sup --dataset bcp in the original repository; not part of
                 this algorithm)
  * entry      src/run_bm25.py with BM25_CORPUS=bcp: the dense-retriever text in the
               tool description and USER_PROLOGUE replaced by facts-only BM25 text (as on FRAMES)
  * order      test200 shuffled, so partial results are representative
The comparison step runs bcp_policy_compare.py of the original repository (not part of this algorithm), named by
BCP_POLICY_COMPARE.
"""
import hashlib, importlib.util, json, os, random, subprocess, sys

# Import wiring: shared/rlm_shared (configs/paths.example.env variables), found relative to this file.
SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SRC, os.pardir, os.pardir, os.pardir, "shared"))
from rlm_shared.paths import path  # noqa: E402

ENTRY = f"{SRC}/run_bm25.py"  # original: scripts/frames_sup_run_one.py
# BCP_OUT / BCP_SERVER / BCP_IDS (file of query ids) run another slice (the rest of the 830) on
# another GPU; default = test200 on :8030 into $RUNS_DIR/bcp_bm25_test200. Same fingerprint, so the cache
# keys match across slices.
OUT = os.environ.get("BCP_OUT", f"{path('RUNS_DIR')}/bcp_bm25_test200")  # original default: <original repo>/logs/...
os.environ.update(GEPA_BCP_SERVERS=os.environ.get("BCP_SERVER", "http://localhost:8030/v1"), GEPA_BCP_RETRIEVAL="http://127.0.0.1:8043",
                  GEPA_BCP_MODEL="qwen36-35b-a3b", GEPA_OUT=OUT, LOCAL_JUDGE_MODEL="qwen36-35b-a3b",
                  SUP_FENCE_TOLERANT="1", SUP_MAX_BLOCKS="5", SUP_PRESENCE_PENALTY="1.5",
                  BM25_CORPUS="bcp", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
os.environ.pop("QUERIES_FILE", None)          # BrowseComp-Plus queries (run_one.py default)
sys.path.insert(0, str(path("RLMLIB_PATH")))
_s = importlib.util.spec_from_file_location("split_runner", f"{SRC}/split_runner.py")  # original: scripts/gepa_bcp.py
G = importlib.util.module_from_spec(_s); sys.modules["split_runner"] = G; _s.loader.exec_module(G)
G.FINGERPRINT = hashlib.sha1((G.FINGERPRINT + "|bcp-bm25-default|bm25f-wrapper").encode()).hexdigest()[:8]


def _rollout(qid, pfile, server, out_dir):
    out = os.path.join(out_dir, f"{qid}.json")
    if not G.PYV:
        raise SystemExit("environment variable PYTHON is not set: interpreter of the rollouts")
    env = dict(os.environ, GEPA_PROMPTS=pfile, PYTHONPATH=str(path("RLMLIB_PATH")),
               HF_HOME=os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), ALIBABA_TOKEN="EMPTY")
    try:
        subprocess.run([G.PYV, ENTRY, "--query-id", qid, "--out", out, "--base-url", server] + G.BASE_ARGS,
                       env=env, cwd=str(path("FULLDOC_ROOT")), capture_output=True, timeout=G.HARD_CAP_S)
    except subprocess.TimeoutExpired:
        G._log(f"[runaway] q{qid} killed at {G.HARD_CAP_S}s"); return None
    try:
        return json.load(open(out))
    except Exception:
        return None


G._rollout = _rollout

if __name__ == "__main__":
    # original: <original repo>/data/bcp_gepa_splits/split.json (always this file, never GEPA_SPLIT)
    _data = os.environ.get("RLM_DATA_DIR") or sys.exit("set RLM_DATA_DIR (holds bcp_gepa_splits/split.json)")
    test = json.load(open(os.path.join(_data, "bcp_gepa_splits", "split.json")))["test200"]
    if os.environ.get("BCP_IDS"):
        ids = [l.strip() for l in open(os.environ["BCP_IDS"]) if l.strip()]
        G._cache_load(); seed = G.seed_blocks()
        G._log(f"[bm25] seed {G.ckey(seed)} fingerprint {G.FINGERPRINT} on {len(ids)} ids from {os.environ['BCP_IDS']} "
               f"via {G.RETRIEVAL}, model {G.SERVERS}")
        G.run_batch(seed, ids, 8)
        sys.exit(0)
    if "--report" not in sys.argv:
        G._cache_load()
        seed = G.seed_blocks()
        order = list(test); random.Random(0).shuffle(order)
        G._log(f"[bm25] seed {G.ckey(seed)} fingerprint {G.FINGERPRINT} on test200 via {G.RETRIEVAL}")
        G.run_batch(seed, order, 8)
    # original: <original repo>/scripts/bcp_policy_compare.py (not part of this algorithm)
    _cmp = os.environ.get("BCP_POLICY_COMPARE") or sys.exit("set BCP_POLICY_COMPARE to bcp_policy_compare.py")
    subprocess.run([sys.executable, _cmp],
                   env=dict(os.environ, SPLIT="test200", TREAT_CACHE=f"{OUT}/cache.json"))
