"""BM25 over the BrowseComp-Plus corpus with DEFAULT Lucene parameters (k1=0.9, b=0.4), as in the
BrowseComp-Plus paper's BM25 baseline (Pyserini defaults). Same tokenisation as the FRAMES index
(scripts/frames_build_bm25.py: bm25s, method lucene, English stemmer + stopwords).
(suprlm/data/indexes/bm25s is the TUNED k1=10, b=1 index -- not used here.)
Output data/bcp_bm25_default/: bm25_index/, docids.json, offsets.npy (byte offset of each line of
suprlm/data/prepared/corpus.jsonl, same order as the index)."""
import json, time
import bm25s, Stemmer, numpy as np
CORPUS = "/home/extstu_00/suprlm/data/prepared/corpus.jsonl"
OUT = "/home/extstu_00/RLM/data/bcp_bm25_default"
t0 = time.time(); docids, texts, offs = [], [], []
with open(CORPUS, "rb") as f:
    while True:
        o = f.tell(); line = f.readline()
        if not line: break
        d = json.loads(line); docids.append(str(d["docid"])); texts.append(d["text"]); offs.append(o)
print(f"loaded {len(texts)} docs ({time.time()-t0:.0f}s)", flush=True)
tok = bm25s.tokenize(texts, stopwords="en", stemmer=Stemmer.Stemmer("english"), show_progress=False)
print(f"tokenized ({time.time()-t0:.0f}s)", flush=True)
r = bm25s.BM25(k1=0.9, b=0.4, method="lucene"); r.index(tok, show_progress=False)
r.save(f"{OUT}/bm25_index"); json.dump(docids, open(f"{OUT}/docids.json", "w")); np.save(f"{OUT}/offsets.npy", np.array(offs, dtype=np.int64))
print(f"saved {OUT} ({time.time()-t0:.0f}s)", flush=True)
