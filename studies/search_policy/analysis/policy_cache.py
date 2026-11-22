"""Search-policy study: cache ColBERT and BM25 top-100 lists for a sample of the logged agent searches.

From original/RLM/scripts/bcp_policy_cache.py; changed: import wiring, locations from configs/paths.env, this
paragraph. Needs the ColBERT server of the document REPL agent on :8766 (unchanged URL) and the bm25s index of
benchmarks/retrieval/build_bm25_index.py (bm25s defaults, not the k1 0.9 / b 0.4 index; KNOWN_ISSUES K25): its
directory is BM25_INDEX_DIR (not yet in configs/paths.example.env; original: ~/bm25_index). Reads and appends
<RLM_DATA_DIR>/bcp_policy/{replay,lists}.jsonl.
  BM25_INDEX_DIR=<dir> [PER_Q=4] python analysis/policy_cache.py

Original docstring follows.

Cache deep candidate lists for a question-stratified sample of logged agent queries.

For each sampled unique (qid, query): ColBERT top-100 (the production retriever, :8766, ids mode)
and BM25 top-100 (bm25s index over the same 100,195 docs, ~/bm25_index, in-process). Search
policies are then scored OFFLINE by rearranging these lists -- milliseconds per policy instead of
a 15-minute agent rollout. ColBERT costs ~2.4 s/query (serial PLAID search, any k), so the sample
is capped (default 4 queries per question). Resumable: appends, skips keys already cached.

Output: data/bcp_policy/lists.jsonl  {qid, split, query, colbert:[ids], bm25:[ids], gold, evidence}
"""
import json, os, random, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

# Import wiring: agents/shared/rlm_shared is located from this file; locations come from configs/paths.env
# (rlm_shared.paths.path). Original: absolute paths of the machine the experiments ran on.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir, "agents", "shared"))
from rlm_shared.paths import path  # noqa: E402
REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir))
POLICY_DIR = os.path.join(str(path("RLM_DATA_DIR")), "bcp_policy")  # original: <original repo>/data/bcp_policy
OUT = f"{POLICY_DIR}/lists.jsonl"


def _bm25_dir() -> str:
    """BM25_INDEX_DIR, read where it is used (original: the hard-coded ~/bm25_index of the original machine)."""
    d = os.environ.get("BM25_INDEX_DIR")
    if not d:
        raise SystemExit("set BM25_INDEX_DIR to the bm25s index directory (benchmarks/retrieval/build_bm25_index.py)")
    return d
PER_Q = int(os.environ.get("PER_Q", "4"))
DEPTH = 100


def colbert(q):
    req = urllib.request.Request("http://127.0.0.1:8766/search",
                                 data=json.dumps({"query": q, "k": DEPTH, "mode": "ids", "untagged_ok": True}).encode(),
                                 headers={"Content-Type": "application/json"})
    for a in range(4):
        try:
            return [str(h["docid"]) for h in json.load(urllib.request.urlopen(req, timeout=600))["results"]]
        except Exception:
            time.sleep(5 * (a + 1))
    return None


def main():
    import bm25s, Stemmer
    recs = [json.loads(l) for l in open(f"{POLICY_DIR}/replay.jsonl")]
    by_q = {}
    for r in recs:
        if r["dup"]:
            continue
        by_q.setdefault(r["qid"], []).append(r)
    rng = random.Random(7)
    sample = []
    for q, rs in sorted(by_q.items(), key=lambda kv: int(kv[0])):
        rng.shuffle(rs); sample += rs[:PER_Q]
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            d = json.loads(l); done.add((d["qid"], d["query"]))
    todo = [r for r in sample if (r["qid"], r["query"]) not in done]
    print(f"sample {len(sample)} queries from {len(by_q)} questions; {len(done)} cached; {len(todo)} to do", flush=True)
    bm = bm25s.BM25.load(_bm25_dir(), load_corpus=False)
    ids = json.load(open(os.path.join(_bm25_dir(), "docids.json")))
    stem = Stemmer.Stemmer("english")
    t0 = time.time(); n = 0
    with open(OUT, "a") as fh, ThreadPoolExecutor(4) as ex:
        for r, cb in zip(todo, ex.map(lambda r: colbert(r["query"]), todo)):
            if cb is None:
                continue
            tok = bm25s.tokenize([r["query"]], stopwords="en", stemmer=stem, show_progress=False)
            res, _ = bm.retrieve(tok, k=DEPTH, show_progress=False)
            fh.write(json.dumps({"qid": r["qid"], "split": r["split"], "query": r["query"], "source": r["source"],
                                 "seed_search": r["seed_search"], "agent_k": r["k"], "colbert": cb,
                                 "bm25": [ids[int(i)] for i in res[0]], "gold": r["gold"], "evidence": r["evidence"]}) + "\n")
            fh.flush(); n += 1
            if n % 100 == 0:
                print(f"{n}/{len(todo)} cached, {n/(time.time()-t0):.2f} q/s", flush=True)
    print("done", n, flush=True)


if __name__ == "__main__":
    main()
