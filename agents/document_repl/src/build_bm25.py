"""Build a BM25 index over the 100,195-doc corpus for hybrid retrieval.

From original/FULLDOC/src/build_bm25.py.   python src/build_bm25.py   (CPU; reads and writes under FULLDOC_ROOT).
Not the index of the BM25 rows: those used the default-parameter index in BCP_BM25_INDEX_DIR (k1 0.9, b 0.4)
served by a separate search server on :8043.

Why: the Qwen3-Embedding-8B index embedded only the first 4,096 tokens of each
document, and 36% of the corpus is longer than that — evidence in a page's tail
is invisible to dense retrieval. BM25 over the full text sees all of it, and
Meng et al. (arXiv 2602.21456) show whole-document BM25 needs k1≈10, b≈1 on this
corpus (default Lucene k1=0.9, b=0.4 scores 0.259 vs 0.506 tuned). Sail
Research's 90.72% leaderboard row uses exactly "Hybrid BM25 + Qwen3-Embed-8B".

CPU only. Output: data/indexes/bm25s/  (bm25s native format) + docids.json.
"""

import json
import os
import sys
import time

import bm25s
import Stemmer

# Import wiring: agents/shared/rlm_shared (configs/paths.example.env variables), found relative to this file.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "shared"))
from rlm_shared.paths import path  # noqa: E402

ROOT = str(path("FULLDOC_ROOT"))  # original: /home/extstu_00/suprlm
CORPUS = f"{ROOT}/data/prepared/corpus.jsonl"
OUT = f"{ROOT}/data/indexes/bm25s"


def main():
    t0 = time.time()
    docids, texts = [], []
    with open(CORPUS, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            docids.append(d["docid"])
            texts.append(d["text"])
    print(f"loaded {len(texts)} docs ({time.time() - t0:.0f}s)", flush=True)

    stemmer = Stemmer.Stemmer("english")
    tokens = bm25s.tokenize(texts, stopwords="en", stemmer=stemmer, show_progress=True)
    print(f"tokenized ({time.time() - t0:.0f}s)", flush=True)

    # Whole-document BM25 on long web pages: length normalisation dominates with
    # default parameters. Meng et al. tuned k1=10, b=1 for this corpus.
    retriever = bm25s.BM25(k1=10.0, b=1.0)
    retriever.index(tokens, show_progress=True)
    print(f"indexed ({time.time() - t0:.0f}s)", flush=True)

    os.makedirs(OUT, exist_ok=True)
    retriever.save(OUT)
    json.dump(docids, open(f"{OUT}/docids.json", "w"))
    print(f"saved -> {OUT}  total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
