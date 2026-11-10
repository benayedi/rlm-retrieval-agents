"""RemoteRetriever: request payloads, URLs and timeouts, with requests.post monkeypatched (no network)."""
from __future__ import annotations

import pytest

import dspy_rlm
from dspy_rlm import tools as T


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


@pytest.fixture
def calls(monkeypatch):
    log = []

    def fake_post(url, json=None, timeout=None):
        log.append({"url": url, "json": json, "timeout": timeout})
        if url.endswith("/search"):
            return _Resp({"results": [{"score": 0.5, "doc_id": "d1", "text": "t"}]})
        if url.endswith("/document"):
            return _Resp({"doc_id": json["doc_id"], "text": "full"})
        if url.endswith("/encode"):
            return _Resp({"embeddings": [[0.1, 0.2]]})
        raise AssertionError(url)

    monkeypatch.setattr(T.requests, "post", fake_post)
    monkeypatch.delenv("OBLIQ_INDEX", raising=False)
    monkeypatch.delenv("RETRIEVER_TIMEOUT", raising=False)
    return log


def test_public_names():
    assert dspy_rlm.FaissRetriever is dspy_rlm.RemoteRetriever is T.RemoteRetriever
    assert set(dspy_rlm.__all__) == {"RemoteRetriever", "FaissRetriever", "make_embedding_fn", "RunMetrics",
                                     "build_lm", "build_rlm_agent", "run_question"}


def test_search_payload_without_index(calls):
    r = T.RemoteRetriever("http://srv:8010/")
    out = r.search_index("who wrote x", top_k=7)
    assert out == [{"score": 0.5, "doc_id": "d1", "text": "t"}]
    assert calls == [{"url": "http://srv:8010/search", "json": {"query": "who wrote x", "top_k": 7}, "timeout": 60}]


def test_search_default_top_k(calls):
    T.RemoteRetriever("http://srv").search_index("q")
    assert calls[0]["json"] == {"query": "q", "top_k": 10}


def test_document_payload_without_index(calls):
    out = T.RemoteRetriever("http://srv").get_document("abc")
    assert out == {"doc_id": "abc", "text": "full"}
    assert calls == [{"url": "http://srv/document", "json": {"doc_id": "abc"}, "timeout": 60}]


def test_index_argument_is_sent(calls):
    r = T.RemoteRetriever("http://srv", index="obliq-des-twitter")
    r.search_index("q", top_k=3)
    r.get_document("7")
    assert calls[0]["json"] == {"query": "q", "top_k": 3, "index": "obliq-des-twitter"}
    assert calls[1]["json"] == {"doc_id": "7", "index": "obliq-des-twitter"}


def test_index_from_env_and_argument_wins(calls, monkeypatch):
    monkeypatch.setenv("OBLIQ_INDEX", "env-index")
    assert T.RemoteRetriever("http://srv").index == "env-index"
    assert T.RemoteRetriever("http://srv", index="arg-index").index == "arg-index"
    monkeypatch.setenv("OBLIQ_INDEX", "")
    assert T.RemoteRetriever("http://srv").index is None


def test_timeout_env_and_argument(calls, monkeypatch):
    monkeypatch.setenv("RETRIEVER_TIMEOUT", "600")
    r = T.RemoteRetriever("http://srv")
    assert r.timeout == 600
    r.search_index("q")
    assert calls[-1]["timeout"] == 600
    assert T.RemoteRetriever("http://srv", timeout=5).timeout == 5


def test_payload_matches_original(original_rag, calls, monkeypatch):
    """The same calls produce byte-identical requests from the original rag.tools."""
    import rag.tools as OT
    monkeypatch.setattr(OT.requests, "post", T.requests.post)   # same fake (requests is one shared module)
    for ret_cls in (T.RemoteRetriever, OT.RemoteRetriever):
        for idx in (None, "i1"):
            r = ret_cls("http://srv/", index=idx)
            r.search_index("q", top_k=4)
            r.get_document("d")
    half = len(calls) // 2
    assert calls[:half] == calls[half:]


def test_make_embedding_fn_payload(calls):
    emb = T.make_embedding_fn("http://srv/")("hello")
    assert emb == [0.1, 0.2]
    assert calls == [{"url": "http://srv/encode", "json": {"inputs": ["hello"]}, "timeout": 60}]
