"""
Refactored copy of scripts/fusion_proxy.py of the original repository (unchanged code; documentation only, see
agents/shared/CHANGES_FROM_ORIGINAL.md). Serves the Reason-ModernColBERT rows of the snippet REPL agent on BrowseComp-Plus as
`--mv-only` over the PLAID index bc_reason_mcb (doc length 512), port 8024 in those runs; see benchmarks/README.md.
Run: ${PYLATE_PYTHON} benchmarks/retrieval/fusion_proxy.py <flags as below>. Usage text below names the original paths.

Fusion proxy — puts RRF(dense, multi-vector) behind the SAME /search wire format.

Why a proxy rather than a patch to the search servers: the agent reaches retrieval
through rag/tools.py RemoteRetriever, which posts to $EMBEDDING_SERVER_URL/search.
Pointing that env var at this process therefore turns fusion on with NO change to
the eval loops, the signatures, the agent, or either existing server — so a fusion
run and a dense run differ in exactly one variable. /document, /encode and /health
pass straight through, so get_document keeps working unchanged.

The multi-vector arm runs on CPU by default. Measured 0.95 s/query on the 214k-doc
congress PLAID index, against ~8 searches per question on a ~104 s question, so it
costs a few seconds per question and leaves the GPU entirely to the two embedding
servers (15 GB + 17 GB of a 40 GB card). Putting PLAID on the GPU is what caused
the OOMs recorded in the Retrieval sheet's pitfalls table.

k and the multi-vector weight are NOT guesses: they come from the held-out sweep in
Retrieval sheet section 9. congress/math want k=60 w=1.0; browsecomp wants k=10
w=1.0 (at k=60 fusion LOSES recall to dense there); twitter should not fuse at all.

Usage:
    python scripts/fusion_proxy.py \\
        --dense-url http://localhost:8010 --dense-index obliq-tot-congress \\
        --mv-model lightonai/LateOn --mv-index cong_lightonai_LateOn \\
        --rrf-k 60 --mv-weight 1.0 --query-len 512 --port 8020

    # then, for the run:
    EMBEDDING_SERVER_URL=http://localhost:8020 ./.venv/bin/python examples/obliq_eval_dspy_rlm.py ...
"""

import argparse
import json
import os
import threading
import time
from collections import defaultdict

import requests
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

parser = argparse.ArgumentParser()
parser.add_argument("--dense-url", required=True, help="the existing embedding server")
parser.add_argument("--dense-index", default="", help="index name to forward (multi-index servers)")
parser.add_argument("--mv", action="append", default=[], metavar="MODEL:INDEX",
                    help="multi-vector arm, REPEATABLE. math's +24% is THREE-way "
                         "(dense + Reason-MCB + LateOn-reg); two-way is only +10%.")
parser.add_argument("--mv-model", default="", help="single-arm shorthand (legacy)")
parser.add_argument("--mv-index", default="", help="single-arm shorthand (legacy)")
parser.add_argument("--index-folder", default=".pylate_idx")
parser.add_argument("--rrf-k", type=int, default=60, help="RRF constant; see sheet section 9")
parser.add_argument("--mv-weight", type=float, default=1.0)
parser.add_argument("--query-len", type=int, default=512)
parser.add_argument("--depth", type=int, default=50,
                    help="how deep each arm is read before fusing; the fused list is "
                         "then cut to the caller's top_k")
parser.add_argument("--device", default="cpu", help="cpu (default) keeps the GPU for the servers")
parser.add_argument("--plaid-batch", type=int, default=4096)
parser.add_argument("--n-full", type=int, default=512)
parser.add_argument("--trust-remote-code", action="store_true", default=True)
parser.add_argument("--bm25-index", default="",
                    help="bm25s index directory (needs docids.json). Adds a LEXICAL arm to the "
                         "fusion. Off by default, so omitting it reproduces earlier runs exactly.")
parser.add_argument("--bm25-weight", type=float, default=1.0,
                    help="RRF weight for the BM25 arm")
parser.add_argument("--port", type=int, default=8020)
parser.add_argument("--passthrough", action="store_true",
                    help="disable fusion entirely and just forward — for A/B sanity checks")
parser.add_argument("--chunk-parent", action="store_true",
                    help="the multi-vector index is CHUNKED: its ids look like "
                         "'CHRG-110hhrg36731_p0000_00##c3' while qrels and the dense "
                         "server use the parent id 'CHRG-110hhrg36731_p0000_00'. Strip the "
                         "'##cN' suffix and collapse chunks to their parent, keeping each "
                         "parent at its BEST chunk rank. Without this every returned id is "
                         "unscoreable and the run silently reports 0 for every question.")
parser.add_argument("--mv-only", action="store_true",
                    help="serve the MULTI-VECTOR arm alone: no dense retrieval, no fusion. "
                         "The dense server is still contacted for /document text lookups, "
                         "because PLAID returns ids only. This is the one retrieval "
                         "condition the agent has never run in — every previous run used "
                         "dense, or dense fused with a multi-vector arm.")
args = parser.parse_args()
if args.mv_model and args.mv_index:
    args.mv.append(f"{args.mv_model}:{args.mv_index}")
if not args.passthrough and not args.mv:
    raise SystemExit("need at least one --mv MODEL:INDEX (or --mv-model/--mv-index)")
MV_SPECS = [tuple(spec.split(":", 1)) for spec in args.mv]

DENSE = args.dense_url.rstrip("/")
app = FastAPI()
_lock = threading.Lock()          # PLAID search is not thread-safe; serialise it
_stats = defaultdict(float)
_arms = []   # [(model_name, index_name, model, retriever)]


def _load_mv():
    """Load each ColBERT model and attach it to an EXISTING PLAID index. Never builds."""
    from pylate import indexes, models, retrieve
    folder = args.index_folder
    for model_name, index_name in MV_SPECS:
        if not os.path.isdir(os.path.join(folder, index_name)):
            raise SystemExit(f"no PLAID index '{index_name}' under {folder}/ — build it first; "
                             f"this proxy only reads existing indexes")
        kw = {"query_length": args.query_len, "device": args.device}
        if args.trust_remote_code:
            kw["trust_remote_code"] = True
        print(f"[proxy] loading {model_name} on {args.device} (query_length={args.query_len})",
              flush=True)
        m = models.ColBERT(model_name_or_path=model_name, **kw)
        ix = indexes.PLAID(index_folder=folder, index_name=index_name, override=False,
                           batch_size=args.plaid_batch, n_full_scores=args.n_full,
                           n_ivf_probe=8, device=args.device)
        _arms.append((model_name, index_name, m, retrieve.ColBERT(index=ix)))
        print(f"[proxy] arm ready: {model_name} on {index_name}", flush=True)


# ── LEXICAL ARM ───────────────────────────────────────────────────────────────────────────
# WHY BM25 IS HERE. Replaying the agent's own 3,084 queries (logs/bm25_agentqueries.json),
# Reason-ModernColBERT reached a gold document on 0 of 58 questions and BM25 on 21. On the
# same slice RRF(mcb, bm25) scored 20/58 — better than ModernColBERT on 20 questions and
# worse on ZERO. On raw questions (logs/bm25_goldzero.json) fusion beats either arm alone,
# 0.175 mean recall against 0.114 (mcb) and 0.097 (bm25). BM25 catches the rare exact tokens
# — names, identifiers, years — that a late-interaction model smears away.
#
# DEPTH MATTERS MORE THAN USUAL. Fusion@100 scores 0.1747 but Fusion@5 only 0.0121: almost
# all the gain sits below rank 5, and this harness calls with top_k=5. --depth is what fuses
# deep before cutting to the caller's top_k, so a small --depth throws the effect away.
_BM25 = None
_BM25_DOCIDS = None
_BM25_STEMMER = None
if args.bm25_index:
    import bm25s as _bm25s
    import Stemmer as _Stemmer
    _bi = os.path.expanduser(args.bm25_index)
    print(f"[proxy] loading BM25 index from {_bi} ...", flush=True)
    _BM25 = _bm25s.BM25.load(_bi, mmap=True)
    _BM25_DOCIDS = json.load(open(os.path.join(_bi, "docids.json")))
    _BM25_STEMMER = _Stemmer.Stemmer("english")
    print(f"[proxy] BM25 arm ready: {len(_BM25_DOCIDS):,} documents", flush=True)


def _bm25_search(query: str, depth: int) -> list[str]:
    """Ranked docids from the lexical arm."""
    import bm25s as _b
    q = _b.tokenize([query], stopwords="en", stemmer=_BM25_STEMMER, show_progress=False)
    res, _sc = _BM25.retrieve(q, k=min(depth, len(_BM25_DOCIDS)), show_progress=False)
    return [_BM25_DOCIDS[int(i)] for i in res[0]]


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    index: str | None = None


class DocumentRequest(BaseModel):
    doc_id: str
    index: str | None = None


def _dense_search(query: str, top_k: int, index: str | None):
    payload = {"query": query, "top_k": top_k}
    idx = index or args.dense_index
    if idx:
        payload["index"] = idx
    r = requests.post(f"{DENSE}/search", json=payload, timeout=120)
    r.raise_for_status()
    body = r.json()
    if "error" in body:
        raise RuntimeError(f"dense server: {body['error']}")
    return body["results"]


def _collapse_chunks(ids):
    """Chunk ids -> parent ids, each parent kept at its best (first) chunk rank.

    A chunked index returns several chunks of the same document, often consecutively, so
    the raw top-10 can be three documents. Collapsing first and cutting afterwards is what
    makes top_k mean the same thing it does for an unchunked arm.
    """
    seen, out = set(), []
    for d in ids:
        parent = d.split("##", 1)[0]
        if parent not in seen:
            seen.add(parent)
            out.append(parent)
    return out


def _mv_search(query: str, top_k: int):
    """One ranked list per multi-vector arm. Serialised: PLAID search is not thread-safe."""
    out = []
    with _lock:
        for _mn, _ix, m, r in _arms:
            # Chunks collapse, so ask for more than top_k or the collapsed list comes back
            # short. 4x is comfortably above the worst observed chunks-per-document.
            k = top_k * 4 if args.chunk_parent else top_k
            emb = m.encode([query], is_query=True, show_progress_bar=False)
            ids = [h["id"] for h in r.retrieve(queries_embeddings=emb, k=k)[0]]
            if args.chunk_parent:
                ids = _collapse_chunks(ids)[:top_k]
            out.append(ids)
    return out


# multi_search_server.py /search returns corpus text[:2000]; match it exactly so a
# fused result never carries more text than a dense one.
TEXT_CAP = int(os.environ.get("PROXY_TEXT_CAP", "2000"))


def _fetch_text(doc_id: str, index: str | None):
    """Text for a document only the multi-vector arm returned.

    The dense response already carries text for its own hits; PLAID returns ids only.
    Without this the agent would print empty snippets for exactly the documents fusion
    was added to contribute, which would look like a retrieval failure rather than a
    plumbing one.

    TRUNCATED TO MATCH THE DENSE SERVER. /search caps text at TEXT_CAP chars while
    /document returns the whole document, so hydrating from /document un-capped made
    fused results carry far more text than dense ones — one BrowseComp+ document came
    back at 146,820 chars against the 2,000-char cap, for 4-7 of every 10 results.
    That inflated the agent's context until questions blew their timeout: 17 of 25
    BrowseComp+ questions were censored at 600s, which read as a collapse in accuracy
    rather than as a plumbing bug. Fusion must change WHICH documents the agent sees,
    never how much text each one carries.
    """
    payload = {"doc_id": doc_id}
    idx = index or args.dense_index
    if idx:
        payload["index"] = idx
    try:
        r = requests.post(f"{DENSE}/document", json=payload, timeout=60)
        r.raise_for_status()
        return (r.json().get("text", "") or "")[:TEXT_CAP]
    except Exception:
        return ""


@app.post("/search")
def search(req: SearchRequest):
    t0 = time.time()
    # read both arms deeper than the caller asked, then cut the FUSED list to top_k:
    # fusing two top-10s and returning 10 discards documents only one arm found, which
    # is what made RRF lose to plain pooling in the congress replay (sheet section 8).
    depth = max(args.depth, req.top_k)
    if args.mv_only:
        # No dense arm. Multi-vector alone, or RRF(multi-vector, BM25) when --bm25-index
        # is given — which is the arm the offline probes measured.
        mv_lists = _mv_search(req.query, depth)
        if _BM25 is not None:
            _sc = defaultdict(float)
            for mv_ids in mv_lists:
                for rank, d in enumerate(mv_ids, 1):
                    _sc[d] += args.mv_weight / (args.rrf_k + rank)
            for rank, d in enumerate(_bm25_search(req.query, depth), 1):
                _sc[d] += args.bm25_weight / (args.rrf_k + rank)
            ids = sorted(_sc, key=lambda d: -_sc[d])[:req.top_k]
        else:
            ids = mv_lists[0][:req.top_k]
        _stats["n"] += 1; _stats["t"] += time.time() - t0
        _stats["from_mv_only"] += len(ids)
        return {"results": [{"score": round(1.0 / (1 + r), 6), "doc_id": d,
                             "text": _fetch_text(d, req.index)}
                            for r, d in enumerate(ids)]}
    dense_hits = _dense_search(req.query, depth, req.index)
    if args.passthrough:
        _stats["n"] += 1
        return {"results": dense_hits[:req.top_k]}

    dense_ids = [h["doc_id"] for h in dense_hits]
    try:
        mv_lists = _mv_search(req.query, depth)
    except Exception as e:
        # A broken multi-vector arm must not silently become "dense-only results" —
        # the run would look like a fusion run and score like a dense one.
        print(f"[proxy] MULTI-VECTOR ARM FAILED: {e}", flush=True)
        raise

    scores = defaultdict(float)
    for rank, d in enumerate(dense_ids, 1):
        scores[d] += 1.0 / (args.rrf_k + rank)
    for mv_ids in mv_lists:
        for rank, d in enumerate(mv_ids, 1):
            scores[d] += args.mv_weight / (args.rrf_k + rank)
    if _BM25 is not None:
        for rank, d in enumerate(_bm25_search(req.query, depth), 1):
            scores[d] += args.bm25_weight / (args.rrf_k + rank)
    fused = sorted(scores, key=lambda d: -scores[d])[:req.top_k]

    text = {h["doc_id"]: h.get("text", "") for h in dense_hits}
    results = [{"score": round(scores[d], 6), "doc_id": d,
                "text": text.get(d) if d in text else _fetch_text(d, req.index)}
               for d in fused]

    _stats["n"] += 1
    _stats["t"] += time.time() - t0
    _stats["from_mv_only"] += sum(1 for d in fused if d not in set(dense_ids))
    _stats["arms"] = len(_arms)
    return {"results": results}


@app.post("/document")
def document(req: DocumentRequest):
    payload = {"doc_id": req.doc_id}
    idx = req.index or args.dense_index
    if idx:
        payload["index"] = idx
    r = requests.post(f"{DENSE}/document", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


@app.post("/encode")
def encode(body: dict):
    r = requests.post(f"{DENSE}/encode", json=body, timeout=120)
    r.raise_for_status()
    return r.json()


@app.get("/health")
def health():
    try:
        up = requests.get(f"{DENSE}/health", timeout=10).ok
    except Exception:
        up = False
    n = _stats["n"] or 1
    return {"status": "ok" if (up and (_arms or args.passthrough)) else "degraded",
            "dense": {"url": DENSE, "up": up, "index": args.dense_index or None},
            "fusion": ("OFF (passthrough)" if args.passthrough else
                       {"MULTI-VECTOR ONLY — no dense arm":
                        [f"{mn}:{ix}" for mn, ix, _, _ in _arms]} if args.mv_only else
                       {"arms": [f"{mn}:{ix}" for mn, ix, _, _ in _arms],
                        "rrf_k": args.rrf_k, "mv_weight": args.mv_weight,
                        "query_len": args.query_len, "depth": args.depth,
                        "device": args.device}),
            "chunk_parent": args.chunk_parent,
            "searches": int(_stats["n"]),
            "mean_search_s": round(_stats["t"] / n, 3),
            "mean_docs_from_mv_only": round(_stats["from_mv_only"] / n, 2)}


if __name__ == "__main__":
    if not args.passthrough:
        _load_mv()
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
