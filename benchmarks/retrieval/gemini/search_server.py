"""Refactored copy of scripts/gemini/search_server.py of the original repository (unchanged code; documentation only,
see shared/CHANGES_FROM_ORIGINAL.md). Needs Vertex AI credentials (vertex.py: GCP_VERTEX_ENV). Run:
  ${GEMINI_PYTHON} benchmarks/retrieval/gemini/search_server.py --port 8020 --index-root ~/indexes/gemini \
      --default-index obliq-ana-math         # 8021 obliq-des-twitter, 8022 obliq-tot-congress in the OBLIQ rows
Original docstring follows (interpreter path replaced by ${GEMINI_PYTHON}).

Gemini Embedding 2 search server over the indexes built by build_index.py (2026-10-27).

Wire-compatible with scripts/multi_search_server.py (/health, /indexes, /search, /document, /encode; an optional
"index" field picks the index, otherwise --default-index), so every harness and front-end that talks to the Qwen
server (e.g. scripts/obliq_math_search.py with --upstream) works unchanged. Queries are embedded on Vertex AI as
they are (--query-format raw, the default: this reproduces the OBLIQ paper's Gemini-only numbers on Math and Twitter)
or as "task: search result | query: {query}" (--query-format google; task_type is not supported by this model), and
cached; exact search, FAISS IndexFlatIP on L2-normalised vectors, kept on CPU.

  ${GEMINI_PYTHON} benchmarks/retrieval/gemini/search_server.py --port 8020 \
      --index-root ~/indexes/gemini --default-index obliq-ana-math
"""
import argparse, json, os, sys, threading
from functools import lru_cache

import faiss
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vertex  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8020)
ap.add_argument("--index-root", default=os.path.expanduser("~/indexes/gemini"))
ap.add_argument("--default-index", default=None)
ap.add_argument("--query-format", choices=["raw", "google"], default="raw",
                help="raw = the query as is (OBLIQ paper's setup, default); google = 'task: search result | query: ...'")
args = ap.parse_args()

INDEX_DIRS = {n: os.path.join(args.index_root, n) for n in sorted(os.listdir(args.index_root))
              if os.path.exists(os.path.join(args.index_root, n, "emb.npy"))}
DEFAULT_INDEX = args.default_index or next(iter(INDEX_DIRS))
CLIENT = vertex.client()
_loaded, _lock = {}, threading.Lock()


def get_index(name):
    with _lock:
        if name not in _loaded:
            d = INDEX_DIRS[name]
            meta = json.load(open(f"{d}/meta.json"))
            emb = np.load(f"{d}/emb.npy")
            fi = faiss.IndexFlatIP(emb.shape[1])
            fi.add(emb)
            corpus = {}
            for line in open(meta["corpus"]):
                if line.strip():
                    x = json.loads(line)
                    corpus[str(x["_id"])] = x.get("text") or ""
            _loaded[name] = {"faiss": fi, "ids": json.load(open(f"{d}/ids.json")), "corpus": corpus}
            print(f"[load] {name}: {fi.ntotal} vectors", flush=True)
        return _loaded[name]


@lru_cache(maxsize=20000)
def _embed(text):
    v = np.asarray(CLIENT.models.embed_content(model="gemini-embedding-2", contents=text).embeddings[0].values,
                   dtype=np.float32)
    return v / np.linalg.norm(v)


def embed_query(q):
    for attempt in range(6):
        try:
            return _embed(("task: search result | query: " + q) if args.query_format == "google" else q)
        except Exception:  # noqa: BLE001  transient quota / 5xx
            if attempt == 5:
                raise
            import time
            time.sleep(2 ** attempt)


app = FastAPI(title="OBLIQ Gemini Embedding 2 Search Server")


class EncodeRequest(BaseModel):
    inputs: list[str]


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    index: str | None = None


class DocumentRequest(BaseModel):
    doc_id: str
    index: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "model": "gemini-embedding-2", "indexes": list(INDEX_DIRS),
            "loaded": list(_loaded), "default": DEFAULT_INDEX}


@app.get("/indexes")
def list_indexes():
    return {"indexes": [{"name": n, "dir": d, "loaded": n in _loaded,
                         "vectors": _loaded[n]["faiss"].ntotal if n in _loaded else None}
                        for n, d in INDEX_DIRS.items()]}


@app.post("/encode")
def encode(req: EncodeRequest):
    return {"embeddings": [embed_query(t).tolist() for t in req.inputs]}


@app.post("/search")
def search(req: SearchRequest):
    name = req.index or DEFAULT_INDEX
    if name not in INDEX_DIRS:
        return {"error": f"unknown index '{name}'", "available": list(INDEX_DIRS), "results": []}
    store = get_index(name)
    scores, idx = store["faiss"].search(embed_query(req.query)[None, :], req.top_k)
    return {"results": [{"score": round(float(s), 4), "doc_id": store["ids"][i],
                         "text": store["corpus"].get(store["ids"][i], "")[:2000]}
                        for s, i in zip(scores[0], idx[0]) if i >= 0]}


@app.post("/document")
def get_document(req: DocumentRequest):
    name = req.index or DEFAULT_INDEX
    if name not in INDEX_DIRS:
        return {"error": f"unknown index '{name}'", "doc_id": req.doc_id, "text": ""}
    text = get_index(name)["corpus"].get(req.doc_id)
    if text is None:
        return {"error": f"doc_id '{req.doc_id}' not found", "doc_id": req.doc_id, "text": ""}
    return {"doc_id": req.doc_id, "text": text}


if __name__ == "__main__":
    print(f"indexes: {list(INDEX_DIRS)}  default: {DEFAULT_INDEX}  port {args.port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
