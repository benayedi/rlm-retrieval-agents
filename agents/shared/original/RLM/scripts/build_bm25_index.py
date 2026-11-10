# -*- coding: utf-8 -*-
"""Build a BM25 index over the BrowseComp-Plus corpus.

WHY. 62 of the 830 questions finish with gold recall exactly 0.00 under Reason-ModernColBERT:
the gold document is never surfaced at all. That is a different failure from the marginal
recall gains this project has measured before (doc_len 4096: +.027 recall, accuracy null), and
it is the one case where a lexical arm could plausibly help — BM25 catches rare exact tokens
(names, identifiers, years) that a dense or late-interaction model smears away.

This builds the index only. Nothing here touches the agent.
"""
import time, os, sys
import bm25s, Stemmer
from datasets import load_dataset

OUT = os.path.expanduser('~/bm25_index')
t0 = time.time()
ds = load_dataset("Tevatron/browsecomp-plus-corpus", split="train")
docids = ds['docid']
print(f"corpus: {len(docids):,} docs  ({time.time()-t0:.1f}s)", flush=True)

stemmer = Stemmer.Stemmer('english')
t1 = time.time()
tokens = bm25s.tokenize(ds['text'], stopwords='en', stemmer=stemmer, show_progress=False)
print(f"tokenized in {time.time()-t1:.1f}s", flush=True)

t2 = time.time()
retriever = bm25s.BM25()
retriever.index(tokens, show_progress=False)
retriever.save(OUT, corpus=None)
import json
json.dump(list(docids), open(os.path.join(OUT, 'docids.json'), 'w'))  # ds['docid'] is a Column, not a list
print(f"indexed + saved in {time.time()-t2:.1f}s -> {OUT}", flush=True)
print(f"TOTAL {time.time()-t0:.1f}s", flush=True)
