"""BM25 retriever over 2017 Wikipedia abstracts for HoVer (matches gepa-artifact / DSPy multi-hop tutorial).

Refactored copy of original/RLM/scripts/hover_retriever.py (studies/prompt_optimisation/CHANGES_FROM_ORIGINAL.md).
Run: server: python hover/src/hover_retriever.py --serve --port 8770 (HOVER_DATA_DIR holds corpus.jsonl and
  bm25_index/ from build_index.py).
Changed: the data directory (HOVER_DATA_DIR).
Names: 'supervisor harness' / 'full-doc RLM' in the text below is the document REPL agent (agents/document_repl);
  'official RLM harness' / 'our harness' / run_obliq_mine.py is the snippet REPL agent (agents/snippet_repl,
  src/run_obliq.py).

Library:   from hover_retriever import search, recall_gold, all_found
           search(query, k) -> [{"title": ..., "text": ...}, ...] in rank order
Server:    python hover_retriever.py --serve --port 8770
           POST /search {"query": str, "k": int} -> {"results": [...]}
Client:    search_remote(query, k, url="http://127.0.0.1:8770")

Index: bm25s BM25(k1=0.9, b=0.4), PyStemmer english, stopwords "en", docs = "title | ' '.join(text)".
Titles are compared with dspy.evaluate.normalize_text, exactly as the GEPA HoVer metric does.
"""
import json
import os
import re
import string
import sys
import threading
import unicodedata

# Import wiring and paths (refactor, rules 2-3). Original: hard-coded locations of the original repository (R), the
# document REPL agent working directory (SUP) and the interpreter. Code now comes from this repository (computed from
# __file__), data and run outputs from configs/paths.env (rlm_shared.paths). The original logs/ directory is RUNS_DIR.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, os.pardir, os.pardir))
DOC_SRC = os.path.join(_REPO, "agents", "document_repl", "src")    # split_runner.py (was scripts/gepa_bcp.py), run_bm25.py
SPLITS = os.path.join(_REPO, "studies", "prompt_optimisation", "splits")   # original: <original repo>/data
sys.path.insert(0, os.path.join(_REPO, "agents", "shared"))
from rlm_shared.paths import path  # noqa: E402

DATA = str(path("HOVER_DATA_DIR"))   # original: <original repo>/data/hover
INDEX_DIR = f"{DATA}/bm25_index"
# "numba" (~10-35 ms/query) or "numpy" (bm25s default here, ~300-470 ms/query because of argpartition over 5.2M
# scores). Both give identical score lists; they can differ only in the order of exactly tied scores.
BACKEND = os.environ.get("HOVER_BM25_BACKEND", "numba")
CORPUS = f"{DATA}/corpus.jsonl"  # one JSON string per line: "title | text", line i == bm25 doc id i

_lock = threading.Lock()
_state = {}


def _load():
    if _state:
        return _state
    with _lock:
        if not _state:
            import bm25s
            import Stemmer

            retriever = bm25s.BM25.load(INDEX_DIR, mmap=False)
            with open(CORPUS) as f:
                corpus = [json.loads(line) for line in f]
            stemmer = Stemmer.Stemmer("english")
            if BACKEND == "numba":  # warm up JIT so the first real query is not ~3 s
                retriever.retrieve(bm25s.tokenize("warm up", stemmer=stemmer, show_progress=False), k=1,
                                   n_threads=1, show_progress=False, backend_selection="numba")
            _state.update(retriever=retriever, corpus=corpus, stemmer=stemmer, bm25s=bm25s)
    return _state


def _split(doc: str) -> dict:
    title, _, text = doc.partition(" | ")
    return {"title": title, "text": text}


def search(query: str, k: int) -> list[dict]:
    s = _load()
    tokens = s["bm25s"].tokenize(query, stopwords="en", stemmer=s["stemmer"], show_progress=False)
    results, scores = s["retriever"].retrieve(tokens, k=k, n_threads=1, show_progress=False, backend_selection=BACKEND)
    # dict de-dup on full doc string, as in gepa-artifact's search()
    run = {s["corpus"][int(d)]: float(sc) for d, sc in zip(results[0], scores[0])}
    return [_split(doc) for doc in list(run)[:k]]


def search_remote(query: str, k: int, url: str = "http://127.0.0.1:8770", timeout: float = 60) -> list[dict]:
    import urllib.request

    req = urllib.request.Request(
        url.rstrip("/") + "/search",
        data=json.dumps({"query": query, "k": k}).encode(),
        headers={"Content-Type": "application/json"},
    )
    # Retry transient socket errors: the stdlib server's accept backlog is small, and with 32
    # concurrent clients a few connections were reset (2026-10-10, HoVer baseline) -- each one
    # would otherwise score an example 0 for an infrastructure reason.
    import time
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())["results"]
        except (ConnectionError, OSError) as e:
            if attempt == 4:
                raise
            time.sleep(0.5 * (2 ** attempt))


# ---- metric (dspy.evaluate.normalize_text copy) ----
_PUNC = set(string.punctuation)


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFD", s).lower()
    s = "".join(ch for ch in s if ch not in _PUNC)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def recall_gold(retrieved_titles, gold_titles) -> float:
    gold = {normalize_text(t) for t in gold_titles}
    found = {normalize_text(t) for t in retrieved_titles}
    return len(gold & found) / len(gold) if gold else 0.0


def all_found(retrieved_titles, gold_titles) -> bool:
    """GEPA HoVer metric (discrete_retrieval_eval): gold titles subset of retrieved titles."""
    gold = {normalize_text(t) for t in gold_titles}
    return gold.issubset({normalize_text(t) for t in retrieved_titles})


# ---- HTTP server ----
def serve(port: int, host: str = "127.0.0.1"):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    _load()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ok": True, "n_docs": len(_state["corpus"])})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/search":
                return self._send(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                self._send(200, {"results": search(str(req["query"]), int(req.get("k", 7)))})
            except Exception as e:  # noqa: BLE001
                self._send(400, {"error": repr(e)})

    srv = ThreadingHTTPServer((host, port), H)
    srv.daemon_threads = True
    print(f"hover_retriever serving on http://{host}:{port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("query", nargs="?")
    ap.add_argument("-k", type=int, default=7)
    a = ap.parse_args()
    if a.serve:
        serve(a.port, a.host)
    elif a.query:
        for r in search(a.query, a.k):
            print(r["title"], "|", r["text"][:150])
