"""Build bm25s index over 2017 Wikipedia abstracts, following the DSPy Multi-Hop Search tutorial.

Refactored copy of original/RLM/scripts/hover_build_index.py (studies/prompt_optimisation/CHANGES_FROM_ORIGINAL.md).
Run: python hover/src/build_index.py (once; writes corpus.jsonl and bm25_index/ into HOVER_DATA_DIR).
Changed: the data directory (HOVER_DATA_DIR).
Names: 'supervisor harness' / 'full-doc RLM' in the text below is the document REPL agent (agents/document_repl);
  'official RLM harness' / 'our harness' / run_obliq_mine.py is the snippet REPL agent (agents/snippet_repl,
  src/run_obliq.py).
"""
import json, time, sys, os
import bm25s, Stemmer

# Import wiring and paths (refactor, rules 2-3). Original: hard-coded locations of the original repository (R), the
# document REPL agent working directory (SUP) and the interpreter. Code now comes from this repository (computed from
# __file__), data and run outputs from configs/paths.env (rlm_shared.paths). The original logs/ directory is RUNS_DIR.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, os.pardir, os.pardir))
DOC_SRC = os.path.join(_REPO, "agents", "document_repl", "src")    # split_runner.py (was scripts/gepa_bcp.py), run_bm25.py
SPLITS = os.path.join(_REPO, "studies", "prompt_optimisation", "splits")   # original: <original repo>/data
sys.path.insert(0, os.path.join(_REPO, "agents", "shared"))
from rlm_shared.paths import path  # noqa: E402

SRC = f"{path('HOVER_DATA_DIR')}/raw/wiki.abstracts.2017.jsonl"   # original: <original repo>/data/hover/raw/...
OUT = str(path("HOVER_DATA_DIR"))
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
