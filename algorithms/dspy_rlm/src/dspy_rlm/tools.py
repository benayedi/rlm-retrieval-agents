"""Remote retrieval tools — delegates FAISS search and doc fetch to the GPU server.

Publication copy of rag/tools.py (package renamed rag -> dspy_rlm; code unchanged). RemoteRetriever is the HTTP
client behind the agent's search_index/get_document tools: POST {EMBEDDING_SERVER_URL}/search and /document.
Not run directly; imported by dspy_rlm.agent and by the drivers src/run_browsecomp.py and src/run_obliq.py.
"""

from __future__ import annotations

from typing import Any

import os
import requests


QUERY_INSTRUCTION = (
    "Instruct: Given a web search query, retrieve relevant passages "
    "that answer the query\nQuery: "
)


class RemoteRetriever:
    """HTTP client for the GPU embed+search server (/search, /document)."""

    def __init__(self, server_url: str, timeout: int | None = None, index: str | None = None) -> None:
        self.base = server_url.rstrip("/")
        # 60s is ample for a GPU index (0.09 s/search) but NOT for a large CPU-served
        # PLAID one: the 30 GB doc_len-4096 BrowseComp+ index averages ~5 s/search with a
        # long tail, and a single search exceeding 60 s aborted a whole shard 10 questions
        # in — the run looks like a dead process, with the ABORT line the only clue and the
        # proxy still healthy. Overridable so a slow index does not silently cost a shard.
        self.timeout = timeout if timeout is not None else int(
            os.environ.get("RETRIEVER_TIMEOUT", "60"))
        # Which index to query when the server hosts several (multi_search_server.py).
        # Omitted -> the server's default index, so single-index servers are unaffected.
        self.index = index or os.environ.get("OBLIQ_INDEX") or None

    def search_index(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {"query": query, "top_k": top_k}
        if self.index:
            payload["index"] = self.index
        r = requests.post(
            f"{self.base}/search",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["results"]

    def get_document(self, doc_id: str) -> dict[str, Any]:
        r = requests.post(
            f"{self.base}/document",
            json=({"doc_id": doc_id, "index": self.index} if self.index else {"doc_id": doc_id}),
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()


# Keep old name as alias so any external references don't break
FaissRetriever = RemoteRetriever


# Refactor note: dead code. Nothing in this algorithm calls make_embedding_fn; it is kept and exported unchanged.
def make_embedding_fn(server_url: str):
    """Embedding function backed by the GPU server's /encode endpoint."""
    def embed(text: str) -> list[float]:
        r = requests.post(
            f"{server_url.rstrip('/')}/encode",
            json={"inputs": [text]},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["embeddings"][0]
    return embed
