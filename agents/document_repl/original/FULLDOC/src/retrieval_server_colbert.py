"""Retrieval server backed by the multi-vector index (Reason-ModernColBERT, LightOn's prebuilt PLAID).

Speaks exactly the same HTTP API as src/retrieval_server.py, so the agent, the tools and the harness
are unchanged and only the retriever differs:

    POST /search        {"query", "k", "mode": "ids"|"snippet"|"full", "tag", "untagged_ok"}
    POST /tag_counts    {"tag"}     -> the per-run counters the judge trusts
    POST /tag_docids    {"tag"}     -> the per-run set of docids actually served
    GET  /stats

Run inside the pylate environment: .venv-colbert/bin/python src/retrieval_server_colbert.py --port 8766
"""
import argparse, os, json, os, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDX_FOLDER = f"{ROOT}/data/indexes/lighton/pylate"
IDX_NAME = os.environ.get("COLBERT_INDEX_NAME", "Reason-ModernColBERT")
MODEL = "lightonai/Reason-ModernColBERT"
MAX_K = 1000
DEFAULT_SNIPPET_TOKENS = 512
# ── LEXICAL ARM (off unless COLBERT_BM25_DIR is set) ──────────────────────────────────────
# WHY. On the agent's own replayed queries (logs/bm25_agentqueries.json) Reason-ModernColBERT
# reached a gold document on 0 of 58 questions and BM25 on 21; RRF(mcb, bm25) scored 20/58 --
# better than ColBERT alone on 20 questions and worse on ZERO. In the Qwen 830 run, 133
# questions (16%) finished with gold recall exactly 0 and scored .120 accuracy, against .877
# once any gold document is present. Retrieval coverage, not reasoning, is the binding
# constraint, and a lexical arm is the one lever measured to move it.
#
# FUSE DEEP, CUT LATE. Fusion@100 scores 0.1747 mean recall but Fusion@5 only 0.0121 -- the
# gain lives below rank 5 and this harness calls with k as low as 5. Both arms are read to
# COLBERT_FUSE_DEPTH and the FUSED list is cut to the caller's k.
#
# EQUAL WEIGHTS. RRF weighted toward the multi-vector arm scored 0.183 against plain RRF's
# 0.271 on the same slice, so both weights default to 1.0.
#
# UNSET COLBERT_BM25_DIR AND THIS FILE BEHAVES EXACTLY AS IT DID FOR THE 830-QUESTION RUN.
BM25_DIR = os.environ.get("COLBERT_BM25_DIR", "")
FUSE_DEPTH = int(os.environ.get("COLBERT_FUSE_DEPTH", "100"))
BM25_WEIGHT = float(os.environ.get("COLBERT_BM25_WEIGHT", "1.0"))
MV_WEIGHT = float(os.environ.get("COLBERT_MV_WEIGHT", "1.0"))
RRF_K = int(os.environ.get("COLBERT_RRF_K", "60"))

RETRIEVER = None


class ColbertRetriever:
    def __init__(self, device="cuda", query_length=512):
        from pylate import indexes, models, retrieve
        import torch

        t0 = time.time()
        self.corpus = {}
        with open(f"{ROOT}/data/prepared/corpus.jsonl", encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                self.corpus[o["docid"]] = o["text"]
        print(f"corpus: {len(self.corpus)} documents in {time.time()-t0:.0f}s", flush=True)

        # query_length and document_length MUST be passed explicitly. LightOn's own searcher
        # (github.com/lightonai/BrowseComp-Plus, searcher/searchers/pylate_searcher.py) constructs
        # the model with both, and every run command in their README passes --query-length 512.
        # Passing neither makes pylate inherit query_length=128 from config_sentence_transformers.json,
        # which silently truncates any query longer than 128 tokens. pylate does not mask-pad here
        # (a 9-token query encodes to 9 tokens at either setting), so this is purely a truncation cap:
        # 3.2% of the agent's own queries and 31% of the raw benchmark questions were being cut.
        _doclen = int(os.environ.get("COLBERT_DOC_LEN", "512"))
        self.model = models.ColBERT(model_name_or_path=MODEL, device=device,
                                    query_length=query_length, document_length=_doclen)
        self.query_length = query_length
        # The window the INDEX actually saw, which the tool description must state honestly.
        # The downloaded LightOn index used 512 (mean 424 embeddings/doc). Our rebuilt index uses
        # the model's native 8192 (mean 3091 embeddings/doc, median 2127 tokens per document).
        self.doc_len = _doclen
        # PLAID fully scores n_full_scores candidates per query. That cost scales with
        # embeddings-per-document, so the 8192 index (3091/doc) needs ~7x what the 512 index
        # (424/doc) did and asks for 41GiB on a 40GB card at the default 8192. Tunable here.
        _nfs = int(os.environ.get("COLBERT_N_FULL_SCORES", "8192"))
        _probe = int(os.environ.get("COLBERT_N_IVF_PROBE", "8"))
        _lowmem = os.environ.get("COLBERT_LOW_MEMORY", "") == "1"
        # THE knob that matters for a large index: fast_plaid's search batch_size defaults to
        # 1<<18 = 262144 candidate embeddings per batch. That working set scales with
        # embeddings-per-document, so the 8192 index (3091/doc vs 424/doc) asks for 41GiB on a
        # 40GB card. n_full_scores and n_ivf_probe do NOT control it; batch_size does.
        _bs = int(os.environ.get("COLBERT_SEARCH_BATCH", str(1 << 18)))
        self.index = indexes.PLAID(index_folder=IDX_FOLDER, index_name=IDX_NAME, override=False,
                                   device=device, n_full_scores=_nfs, n_ivf_probe=_probe,
                                   low_memory=_lowmem, batch_size=_bs)
        self.retriever = retrieve.ColBERT(index=self.index)
        print(f"index={IDX_NAME!r} loaded on {device}; document_length={self.doc_len}", flush=True)

        # A HuggingFace fast tokenizer is a Rust object and is not thread-safe: calling it from
        # several ThreadingHTTPServer worker threads raises, which surfaced as HTTP 500 and cost
        # the agent 31% of its searches in the first multi-vector run. One tokenizer per thread.
        from transformers import AutoTokenizer
        self._tls = threading.local()
        self._tok_name = MODEL
        self._span = {}
        self._lock = threading.Lock()
        self._tag_lock = threading.Lock()
        self.tag_counts = {}
        self.tag_docids = {}
        self.n_search = self.n_get_doc = 0

        # Lexical arm. mmap keeps the ~740 MB index off the heap.
        self.bm25 = None
        self.bm25_docids = None
        self._stemmer = None
        if BM25_DIR and os.path.exists(os.path.join(BM25_DIR, "docids.json")):
            import bm25s
            import Stemmer
            t4 = time.time()
            self.bm25 = bm25s.BM25.load(BM25_DIR, mmap=True)
            self.bm25_docids = json.load(open(os.path.join(BM25_DIR, "docids.json")))
            self._stemmer = Stemmer.Stemmer("english")
            print(f"[server] BM25 arm: {len(self.bm25_docids):,} docs "
                  f"({time.time() - t4:.1f}s) fuse_depth={FUSE_DEPTH} rrf_k={RRF_K}", flush=True)
        elif BM25_DIR:
            raise SystemExit(f"COLBERT_BM25_DIR={BM25_DIR} has no docids.json")

    @property
    def tok(self):
        t = getattr(self._tls, "tok", None)
        if t is None:
            from transformers import AutoTokenizer
            t = AutoTokenizer.from_pretrained(self._tok_name)
            self._tls.tok = t
        return t

    def embedded_char_span(self, docid, text):
        """How much of the document the index actually saw, in characters."""
        c = self._span.get(docid)
        if c is None:
            # pylate tokenizes documents at max_seq_length = document_length - 1 and then inserts
            # the [D] marker at position 1, so the real content window is two tokens short of
            # document_length, not one. Verified against the index's own stored token counts.
            enc = self.tok(text, add_special_tokens=True, truncation=True,
                           max_length=self.doc_len - 1, return_offsets_mapping=True)
            offs = [o for o in enc["offset_mapping"] if o != (0, 0)]
            c = offs[-1][1] if offs else min(len(text), 1200)
            self._span[docid] = c
        return c

    def _truncate(self, text, max_tokens):
        enc = self.tok(text, add_special_tokens=False, truncation=True,
                       max_length=max_tokens, return_offsets_mapping=True)
        offs = [o for o in enc["offset_mapping"] if o != (0, 0)]
        end = offs[-1][1] if offs else len(text)
        return text[:end], end

    def search(self, query, k=5, mode="snippet", snippet_max_tokens=DEFAULT_SNIPPET_TOKENS, tag=None):
        k = max(1, min(int(k), MAX_K))
        # With the lexical arm on, BOTH arms are read to FUSE_DEPTH and the fused list is cut
        # to k. Reading each arm to k and fusing two top-k lists would discard exactly the
        # documents only one arm found, which is the point of fusing at all.
        depth = max(k, FUSE_DEPTH) if self.bm25 is not None else k
        with self._lock:
            emb = self.model.encode([query], batch_size=1, is_query=True, show_progress_bar=False)
            hits = self.retriever.retrieve(queries_embeddings=emb, k=depth)[0]
        if self.bm25 is not None:
            import bm25s as _b
            import sys as _sys
            _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from hybrid import rrf as _rrf
            mv_ids = [h["id"] for h in hits]
            q_tok = _b.tokenize([query], stopwords="en", stemmer=self._stemmer,
                                show_progress=False)
            res, _sc = self.bm25.retrieve(q_tok, k=min(depth, len(self.bm25_docids)),
                                          show_progress=False)
            bm_ids = [self.bm25_docids[int(i)] for i in res[0]]
            fused = _rrf([mv_ids, bm_ids], k=RRF_K, weights=[MV_WEIGHT, BM25_WEIGHT])[:k]
            # Rebuilt so downstream formatting is unchanged. NOTE the score is now an RRF
            # score, not a MaxSim score -- the two are not comparable across runs.
            hits = [{"id": d, "score": sc} for d, sc in fused]
        self.n_search += 1
        docids = [h["id"] for h in hits]
        if tag:
            with self._tag_lock:
                c = self.tag_counts.setdefault(tag, {"search": 0, "k_sum": 0, "hits": 0})
                c["search"] += 1
                c["k_sum"] += k
                c["hits"] += len(hits)
                self.tag_docids.setdefault(tag, set()).update(docids)
        out = []
        for h in hits:
            docid = h["id"]
            text = self.corpus.get(docid, "")
            score = round(float(h["score"]), 6)
            if mode == "ids":
                out.append({"docid": docid, "score": score})
                continue
            if mode == "full":
                emb_end = self.embedded_char_span(docid, text)
                _, snip_chars = self._truncate(text, snippet_max_tokens)
                out.append({"docid": docid, "score": score, "text": text, "n_chars": len(text),
                            "embedded_span": [0, emb_end], "snippet_span": [0, snip_chars]})
            else:
                snippet, _ = self._truncate(text, snippet_max_tokens)
                out.append({"docid": docid, "score": score, "snippet": snippet})
        return out


    def stats(self):
        return {"retriever": ("RRF(Reason-ModernColBERT, BM25)" if self.bm25 is not None
                              else "Reason-ModernColBERT (multi-vector, PLAID)"),
                "fuse_depth": FUSE_DEPTH if self.bm25 is not None else None,
                "n_search": self.n_search,
                "n_docs": len(self.corpus),
                "document_length_tokens": self.doc_len,
                "query_length_tokens": self.query_length,
                "uptime_s": round(time.time() - T0, 1)}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/stats":
            self._send(200, RETRIEVER.stats())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            self._send(400, {"error": f"bad request: {e}"})
            return
        try:
            if self.path == "/search" and not req.get("tag") and not req.get("untagged_ok"):
                self._send(400, {"error": "tag required"})
                return
            if self.path == "/search":
                res = RETRIEVER.search(query=req["query"], k=req.get("k", 5), mode=req.get("mode", "snippet"),
                                       snippet_max_tokens=req.get("snippet_max_tokens", DEFAULT_SNIPPET_TOKENS),
                                       tag=req.get("tag"))
                self._send(200, {"results": res})
            elif self.path == "/tag_counts":
                self._send(200, {"counts": RETRIEVER.tag_counts.get(req.get("tag"))})
            elif self.path == "/tag_docids":
                ids = RETRIEVER.tag_docids.get(req.get("tag"))
                self._send(200, {"docids": sorted(ids) if ids is not None else None})
            else:
                self._send(404, {"error": "not found"})
        except Exception as e:
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


T0 = time.time()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--query-length", type=int, default=512,
                    help="Query truncation cap in tokens. 512 matches LightOn's documented runs; "
                         "the model config's 128 is what pylate uses if this is not passed.")
    a = ap.parse_args()
    RETRIEVER = ColbertRetriever(device=a.device, query_length=a.query_length)
    print(f"READY after ~{time.time()-T0:.0f}s on port {a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
