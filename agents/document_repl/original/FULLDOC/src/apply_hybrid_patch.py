"""Staged patch: add BM25 / hybrid first stage and a `passages` mode to the server.

Apply only when no evaluation sweep is in flight, then restart the server.
Adds request fields (all optional, defaults preserve today's behaviour):
    retriever: "dense" (default) | "bm25" | "hybrid"
    mode:      ... | "passages"
    n_passages, per_doc, rescore_top
"""

P = "/home/extstu_00/suprlm/src/retrieval_server.py"
s = open(P).read()
assert "def search(" in s and "hybrid" not in s, "already patched or unexpected file"

# --- imports -----------------------------------------------------------------
s = s.replace(
    "from encoder import Qwen3Encoder  # noqa: E402",
    "from encoder import Qwen3Encoder  # noqa: E402\n"
    "from hybrid import bm25_preselect, group_passages, rrf, windows_for  # noqa: E402\n"
    "import bm25s  # noqa: E402\n"
    "import Stemmer  # noqa: E402",
)
s = s.replace(
    'SNIPPET_TOKENIZER = f"{ROOT}/data/models/Qwen3-0.6B"',
    'SNIPPET_TOKENIZER = f"{ROOT}/data/models/Qwen3-0.6B"\nBM25_DIR = f"{ROOT}/data/indexes/bm25s"',
)

# --- load BM25 at startup (optional) -----------------------------------------
s = s.replace(
    "        self.snippet_tok = AutoTokenizer.from_pretrained(SNIPPET_TOKENIZER)",
    """        self.snippet_tok = AutoTokenizer.from_pretrained(SNIPPET_TOKENIZER)

        # Optional BM25 first stage (whole documents, k1=10 b=1 per Meng et al.).
        self.bm25 = None
        self.bm25_docids = None
        self._stemmer = Stemmer.Stemmer("english")
        if os.path.exists(os.path.join(BM25_DIR, "docids.json")):
            t3 = time.time()
            self.bm25 = bm25s.BM25.load(BM25_DIR, mmap=True)
            self.bm25_docids = json.load(open(os.path.join(BM25_DIR, "docids.json")))
            print(f"[server] bm25 index loaded ({time.time() - t3:.1f}s)", flush=True)
        else:
            print("[server] no bm25 index; retriever=bm25/hybrid unavailable", flush=True)""",
)

# --- first-stage helpers -----------------------------------------------------
s = s.replace(
    "    def _truncate(self, text: str, max_tokens: int | None) -> tuple[str, int]:",
    """    def _dense_rank(self, q: torch.Tensor, k: int) -> list[str]:
        qd = q.to(self.index.device, self.index.dtype)
        scores = torch.mv(self.index, qd).float()
        top = torch.topk(scores, k)
        return [self.lookup[i] for i in top.indices.tolist()]

    def _bm25_rank(self, query: str, k: int) -> list[str]:
        if self.bm25 is None:
            return []
        q_tok = bm25s.tokenize([query], stopwords="en", stemmer=self._stemmer, show_progress=False)
        res, _ = self.bm25.retrieve(q_tok, k=k, show_progress=False)
        return [self.bm25_docids[int(i)] for i in res[0]]

    def first_stage(self, query: str, q: torch.Tensor, k: int, retriever: str) -> list[str]:
        \"\"\"Ranked docids from dense, bm25, or RRF-fused hybrid.\"\"\"
        if retriever == "bm25":
            return self._bm25_rank(query, k)
        if retriever == "hybrid" and self.bm25 is not None:
            # Fuse deeper lists than k so RRF has room to promote agreement.
            fused = rrf([self._dense_rank(q, 3 * k), self._bm25_rank(query, 3 * k)])
            return [d for d, _ in fused[:k]]
        return self._dense_rank(q, k)

    def passages(self, query: str, q: torch.Tensor, docids: list[str],
                 n_passages: int = 15, per_doc: int = 3, rescore_top: int = 64) -> list[dict]:
        \"\"\"Window the docs, BM25-preselect windows, dense-rescore survivors, group.\"\"\"
        wins = []
        for d in docids:
            wins.extend(windows_for(d, self.docid_to_text.get(d, "")))
        if not wins:
            return []
        keep = bm25_preselect(query, wins, top=rescore_top)
        cand = [wins[i] for i in keep]
        with self._enc_lock:
            reps = self.encoder.encode_passages([w.text for w in cand])  # (n, 4096) unit-norm
        scores = (reps @ q.float()).tolist()
        grouped = group_passages(cand, scores, per_doc=per_doc)
        # trim to n_passages total, keeping doc order
        out, used = [], 0
        for g in grouped:
            room = n_passages - used
            if room <= 0:
                break
            g["passages"] = g["passages"][:room]
            used += len(g["passages"])
            out.append(g)
        return out

    def _truncate(self, text: str, max_tokens: int | None) -> tuple[str, int]:""",
)

# --- search(): retriever + passages mode -------------------------------------
s = s.replace(
    """        snippet_max_tokens: int | None = DEFAULT_SNIPPET_TOKENS,
        tag: str | None = None,
    ) -> list[dict]:
        k = max(1, min(int(k), MAX_K))
        q = self.embed_query(query)
        qd = q.to(self.index.device, self.index.dtype)
        scores = torch.mv(self.index, qd).float()
        top = torch.topk(scores, k)
        self.n_search += 1""",
    """        snippet_max_tokens: int | None = DEFAULT_SNIPPET_TOKENS,
        tag: str | None = None,
        retriever: str = "dense",
        n_passages: int = 15,
        per_doc: int = 3,
        rescore_top: int = 64,
    ) -> list[dict]:
        k = max(1, min(int(k), MAX_K))
        q = self.embed_query(query)
        ranked = self.first_stage(query, q, k, retriever)
        # Keep the dense score for display where available; bm25/hybrid hits
        # outside the dense top-k get the cosine to the query computed directly.
        qd = q.to(self.index.device, self.index.dtype)
        scores_all = torch.mv(self.index, qd).float()
        row = {d: i for i, d in enumerate(self.lookup)} if not hasattr(self, "_row") else self._row
        self._row = row
        top_vals = [float(scores_all[row[d]]) if d in row else 0.0 for d in ranked]
        top_idx = [row.get(d, -1) for d in ranked]
        class _T: pass
        top = _T(); top.values = torch.tensor(top_vals); top.indices = torch.tensor(top_idx)
        self.n_search += 1""",
)
# passages mode: return grouped passages + full text (for DOCS) + head
s = s.replace(
    """        if mode == "ids":
            # Lightweight path for retrieval-only evaluation: no text payload.""",
    """        if mode == "passages":
            groups = self.passages(query, q, ranked, n_passages=n_passages,
                                   per_doc=per_doc, rescore_top=rescore_top)
            for g in groups:
                text = self.docid_to_text.get(g["docid"], "")
                g["text"] = text
                g["n_chars"] = len(text)
                g["head"] = text[:300]
            return groups

        if mode == "ids":
            # Lightweight path for retrieval-only evaluation: no text payload.""",
)
# HTTP plumbing
s = s.replace(
    """                    tag=req.get("tag"),
                )
                self._send(200, {"results": res})""",
    """                    tag=req.get("tag"),
                    retriever=req.get("retriever", "dense"),
                    n_passages=int(req.get("n_passages", 15)),
                    per_doc=int(req.get("per_doc", 3)),
                    rescore_top=int(req.get("rescore_top", 64)),
                )
                self._send(200, {"results": res})""",
)
open(P, "w").write(s)
import ast
ast.parse(s)
print("retrieval_server.py patched: bm25/hybrid first stage + passages mode (parse OK)")
