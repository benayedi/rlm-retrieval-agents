"""ToolBox (src/tools.py): search cap, refusal counter, k override, DOCS merging, against a fake retrieval server."""
from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

import pytest

import tools


class FakeServer:
    """Stands in for the retrieval server: records every POST and answers from a script or a default."""

    def __init__(self):
        self.requests: list[tuple[str, dict]] = []
        self.script: list[list[dict]] = []

    def urlopen(self, req, timeout=None):
        path = req.full_url.split("127.0.0.1:8765", 1)[-1]
        payload = json.loads(req.data.decode())
        self.requests.append((path, payload))
        if path == "/search":
            if self.script:
                results = self.script.pop(0)
            else:
                k = payload["k"]
                results = [{"docid": f"{payload['query']}-{i}", "score": 1.0 - i / 100,
                            "text": f"text of {payload['query']}-{i}"} for i in range(k)]
            body = {"results": results}
        elif path == "/tag_counts":
            body = {"counts": {"search": sum(1 for p, _ in self.requests if p == "/search")}}
        else:
            body = {"docids": None}
        return _Resp(json.dumps(body).encode())

    @property
    def searches(self):
        return [p for path, p in self.requests if path == "/search"]


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def server(monkeypatch):
    s = FakeServer()
    monkeypatch.setattr(urllib.request, "urlopen", s.urlopen)
    return s


def box(**kw):
    kw.setdefault("tag", "q1-abcd")
    return tools.ToolBox(base_url="http://127.0.0.1:8765", **kw)


# ---------------------------------------------------------------- cap and refusals
def test_search_cap_of_ten_and_refusal_counter(server, capsys):
    tb = box(design="B", max_searches=10)
    out = [tb.search(f"query {i}") for i in range(12)]
    assert tb.tool_call_counts["search"] == 10
    assert tb.search_refused == 2
    assert len(server.searches) == 10          # refused calls never reach the server
    assert out[10] == [] and out[11] == []      # an empty list, not an error record
    assert "[search refused] all 10 search calls are used" in capsys.readouterr().out


def test_cap_is_enforced_client_side_only(server):
    """The cap lives in the ToolBox; nothing is sent to the server that would let it refuse."""
    tb = box(design="B", max_searches=10)
    tb.search("q")
    assert "max_searches" not in server.searches[0] and "cap" not in server.searches[0]


def test_search_many_counts_each_query_against_the_cap(server):
    tb = box(design="B", max_searches=3)
    res = tb.search_many(["a", "b", "c", "d"])
    assert [len(r) for r in res] == [5, 5, 5, 0]
    assert tb.tool_call_counts["search"] == 3 and tb.search_refused == 1


def test_no_cap_when_zero(server):
    tb = box(design="B", max_searches=0)
    for i in range(15):
        tb.search(f"q{i}")
    assert tb.tool_call_counts["search"] == 15 and tb.search_refused == 0


def test_empty_query_is_counted_but_not_sent(server):
    tb = box(design="B", max_searches=10)
    assert tb.search("   ") == []
    assert tb.tool_call_counts["search"] == 1 and server.searches == []
    assert tb.errors == ["search called with empty query"]


# ---------------------------------------------------------------- k
def test_design_b_lets_the_model_choose_k_up_to_max_k(server):
    assert tools.MAX_K == 50
    tb = box(design="B", default_k=5)
    assert tb.allow_k_override is True
    for asked, sent in ((None, 5), (30, 30), (100, 50), (0, 1), (-3, 1), ("7", 7), ("x", 5)):
        tb.search("q", asked)
        assert server.searches[-1]["k"] == sent, asked
    assert "k is yours to choose per call, from 1 to 50" in tb.descriptions()["search"]


def test_fixed_k_ignores_the_models_k(server):
    tb = box(design="B", default_k=20, allow_k_override=False)   # run_one --fixed-k passes False
    tb.search("q", 50)
    assert server.searches[-1]["k"] == 20
    assert "search(query: str) -> list[dict]" in tb.descriptions()["search"]


def test_design_a_keeps_default_k(server):
    tb = box(design="A", default_k=5)
    tb.search("q", 30)
    assert server.searches[-1]["k"] == 5 and server.searches[-1]["mode"] == "snippet"


def test_request_fields_of_design_b(server):
    tb = box(design="B")
    tb.search("q")
    p = server.searches[-1]
    assert p["mode"] == "full" and p["tag"] == "q1-abcd"
    assert p["retriever"] == "dense"   # sent for BM25 and ColBERT servers too; the servers ignore it


# ---------------------------------------------------------------- DOCS
def test_docs_merge_keeps_full_text_and_dedups_docids(server):
    tb = box(design="B")
    server.script = [
        [{"docid": "d1", "score": 0.9, "text": "FULL ONE"}, {"docid": "d2", "score": 0.8, "text": "FULL TWO"}],
        [{"docid": "d2", "score": 0.7, "snippet": "snip two"}, {"docid": "d3", "score": 0.6, "text": "FULL THREE"}],
    ]
    tb.search("first")
    tb.search("second")
    assert tb.retrieved_docids == ["d1", "d2", "d3"]
    d2 = dict.__getitem__(tb.DOCS, "d2")
    assert d2["text"] == "FULL TWO" and d2["snippet"] == "snip two" and d2["score"] == 0.7
    assert [e["docids"] for e in tb.SEARCH_LOG] == [["d1", "d2"], ["d2", "d3"]]
    assert [e["k"] for e in tb.SEARCH_LOG] == [5, 5]


def test_unread_docs_tracks_access(server):
    tb = box(design="B")
    server.script = [[{"docid": "d1", "text": "a"}, {"docid": "d2", "text": "b"}]]
    tb.search("q")
    assert {d["docid"] for d in tb.unread_docs()} == {"d1", "d2"}
    tb.DOCS["d1"]
    assert [d["docid"] for d in tb.unread_docs()] == ["d2"]


def test_grep_docs_reads_docs_only(server):
    tb = box(design="B")
    server.script = [[{"docid": "d1", "text": "born in 1907 in Riga"}]]
    tb.search("q")
    n = len(server.requests)
    hits = tb.grep_docs(r"\b1907\b", window=5)
    assert hits[0]["docid"] == "d1" and len(server.requests) == n


def test_server_counts_unreachable_server_is_none(monkeypatch):
    def refuse(*a, **k):
        raise urllib.error.URLError("refused")
    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    tb = box(design="B")
    assert tb.server_counts() is None and tb.server_docids() is None


def test_custom_tools_surface():
    tb = box(design="B", max_searches=10)
    ct = tb.custom_tools()
    assert sorted(ct) == ["DOCS", "SEARCH_LOG", "grep_docs", "search", "search_many", "unread_docs"]
    assert "read_docs" not in ct     # advertised in the description but not installed (see CHANGES_FROM_ORIGINAL.md)
    assert "read_docs(docs, question)" in ct["search"]["description"]
