"""Build bm25s index over 2017 Wikipedia abstracts, following the DSPy Multi-Hop Search tutorial."""
import json, time, sys, os
import bm25s, Stemmer

SRC = "/home/extstu_00/RLM/data/hover/raw/wiki.abstracts.2017.jsonl"
OUT = "/home/extstu_00/RLM/data/hover"
t0 = time.time()
corpus = []
with open(SRC) as f:
    for line in f:
        d = json.loads(line)
        corpus.append(f"{d['title']} | {' '.join(d['text'])}")
print(f"loaded {len(corpus)} docs in {time.time()-t0:.1f}s", flush=True)
with open(f"{OUT}/corpus.jsonl", "w") as f:
    for c in corpus:
        f.write(json.dumps(c) + "\n")
t1 = time.time()
stemmer = Stemmer.Stemmer("english")
tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)
print(f"tokenized in {time.time()-t1:.1f}s", flush=True)
t2 = time.time()
retriever = bm25s.BM25(k1=0.9, b=0.4)
retriever.index(tokens)
print(f"indexed in {time.time()-t2:.1f}s", flush=True)
retriever.save(f"{OUT}/bm25_index")
print(f"total build {time.time()-t0:.1f}s (tokenize+index {time.time()-t1:.1f}s)", flush=True)
