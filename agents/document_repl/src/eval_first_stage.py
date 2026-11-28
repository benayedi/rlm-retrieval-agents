"""First-stage retrieval with the raw question as one query: evidence recall@k and hit@k on all 830 questions.

Port of the original research repository's src/eval_first_stage.py (see original/FULLDOC/src/apply_design_g_patch.py,
whose search-description figures "top-5 15%, top-30 33%, top-50 39%" with hit@k beside them are documented as coming
from its output, runs/first_stage_eval.json; that output file was not preserved). The measurement is unchanged:

  * each question, unchanged, is sent as one query; dense rankings come from the live dense server
    (retrieval_server.py, Qwen3-Embedding-8B, exact cosine) in "ids" mode; BM25 runs in-process; "hybrid" is
    reciprocal rank fusion of the two (as in the original, which compared all three);
  * recall@k = |evidence ∩ top-k| / |evidence|, averaged over questions with evidence labels;
    hit@k = share of those questions with at least one evidence document in the top k.

Added in the port: paths from configs/paths.example.env (FULLDOC_ROOT, RUNS_DIR) instead of hard-coded ones, and,
with --dev-ids, the same numbers reported separately for the development questions and for the others.

    python agents/document_repl/src/eval_first_stage.py [--port 8765] [--dev-ids FILE] [--out FILE]

--dev-ids: a CSV with a query_id column (e.g. a 150-question per-question result file of the replication package)
or a text file with one id per line. No model API is called; no question text is written to the output.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.request

import bm25s
import Stemmer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, os.pardir, os.pardir, "shared"))
from hybrid import rrf  # noqa: E402
from rlm_shared.paths import path  # noqa: E402

KS = [5, 10, 20, 30, 50]
DEPTH = 150  # fuse deeper lists than the largest k (as in the original)


def post(port: int, payload: dict) -> list:
    r = urllib.request.Request(f"http://127.0.0.1:{port}/search", data=json.dumps(payload).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=120) as x:
        return json.load(x)["results"]


def read_ids(fn: str) -> set[str]:
    if fn.endswith(".csv"):
        with open(fn, newline="") as f:
            return {str(row["query_id"]).strip() for row in csv.DictReader(f)}
    return {line.strip() for line in open(fn) if line.strip()}


def score(ranks: dict, rows: list) -> dict:
    n = len(rows)
    rec = {k: 0.0 for k in KS}
    hit = {k: 0 for k in KS}
    for qid, rel in rows:
        got = ranks.get(qid, [])
        for k in KS:
            inter = set(got[:k]) & rel
            rec[k] += len(inter) / len(rel)
            hit[k] += bool(inter)
    return ({"n": n} | {f"recall@{k}": 100 * rec[k] / n for k in KS}
            | {f"hit@{k}": 100 * hit[k] / n for k in KS})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--dev-ids", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    root = str(path("FULLDOC_ROOT"))
    out_fn = a.out or os.path.join(str(path("RUNS_DIR")), "first_stage_eval.json")
    qs = [json.loads(line) for line in open(f"{root}/data/prepared/queries.jsonl", encoding="utf-8")]
    ev = json.load(open(f"{root}/data/prepared/qrel_evidence.json"))
    idx = bm25s.BM25.load(f"{root}/data/indexes/bm25s", mmap=True)
    docids = json.load(open(f"{root}/data/indexes/bm25s/docids.json"))
    st = Stemmer.Stemmer("english")

    systems = {"dense": {}, "bm25": {}, "hybrid": {}, "hybrid_0.7d": {}}
    t0 = time.time()
    for i, q in enumerate(qs):
        qid = str(q["query_id"])
        dense = [r["docid"] for r in post(a.port, {"query": q["query"], "k": DEPTH, "mode": "ids",
                                                    "untagged_ok": True})]
        qt = bm25s.tokenize([q["query"]], stopwords="en", stemmer=st, show_progress=False)
        res, _ = idx.retrieve(qt, k=DEPTH, show_progress=False)
        bm = [docids[int(j)] for j in res[0]]
        systems["dense"][qid] = dense
        systems["bm25"][qid] = bm
        systems["hybrid"][qid] = [d for d, _ in rrf([dense, bm])]
        systems["hybrid_0.7d"][qid] = [d for d, _ in rrf([dense, bm], weights=[0.7, 0.3])]
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(qs)}  {time.time() - t0:.0f}s", flush=True)

    rows = [(str(qid), set(r)) for qid, r in ev.items() if r]
    subsets = {"all": rows}
    if a.dev_ids:
        dev = read_ids(a.dev_ids)
        subsets["development"] = [r for r in rows if r[0] in dev]
        subsets["held_out"] = [r for r in rows if r[0] not in dev]
        print(f"development ids: {len(dev)}; with evidence labels: {len(subsets['development'])}; "
              f"others: {len(subsets['held_out'])}")

    out = {"questions": len(qs), "questions_with_evidence": len(rows), "depth": DEPTH, "results": {}}
    for sub, srows in subsets.items():
        print(f"\n[{sub}] evidence recall@k / hit@k over {len(srows)} questions (single search, raw question)")
        print(f"{'system':12s}" + "".join(f"{'R@%d' % k:>8s}" for k in KS) + "".join(f"{'hit@%d' % k:>8s}" for k in KS))
        out["results"][sub] = {}
        for name, ranks in systems.items():
            s = score(ranks, srows)
            out["results"][sub][name] = s
            print(f"{name:12s}" + "".join(f"{s[f'recall@{k}']:7.2f}%" for k in KS)
                  + "".join(f"{s[f'hit@{k}']:7.1f}%" for k in KS))

    os.makedirs(os.path.dirname(os.path.abspath(out_fn)), exist_ok=True)
    json.dump(out, open(out_fn, "w"), indent=2)
    print(f"\nwrote {out_fn}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
