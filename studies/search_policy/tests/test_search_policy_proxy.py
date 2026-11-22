"""src/rerank_proxy.py with a mocked upstream retriever and a mocked reranker (no network, no GPU, no model)."""
from __future__ import annotations

import contextlib
import copy
import json
import threading

import pytest

from search_policy_testlib import SRC, load_file


def proxy(monkeypatch, upstream_results=None, scores=None, **env):
    """Load the proxy with env, replace upstream() and rerank_scores(). Returns (module, calls)."""
    for k in ("POLICY", "DEPTH", "WIRE", "QUERIES_FILE", "UPSTREAM", "RERANK_XBATCH"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    m = load_file(SRC / "rerank_proxy.py")
    calls = {"upstream": [], "rerank": []}

    def fake_upstream(path, payload):
        calls["upstream"].append((path, copy.deepcopy(payload)))
        return {"results": copy.deepcopy(upstream_results or [])}

    def fake_scores(qtext, texts):
        calls["rerank"].append((qtext, list(texts)))
        return [scores[t] for t in texts]

    m.upstream = fake_upstream
    m.rerank_scores = fake_scores
    return m, calls


def docs(*ids):
    return [{"docid": i, "score": 1.0 / (n + 1), "text": f"text-{i}"} for n, i in enumerate(ids)]


def test_passthrough_is_byte_identical(monkeypatch):
    res = docs("d1", "d2", "d3")
    m, calls = proxy(monkeypatch, res, POLICY="passthrough")
    out = m.search({"query": "q", "k": 2, "mode": "snippet", "tag": "7-abc"})
    assert json.dumps(out["results"]) == json.dumps(res[:2])
    path, payload = calls["upstream"][0]
    assert path == "/search" and payload["k"] == 2 and payload["mode"] == "snippet"
    assert "tag" not in payload and payload["untagged_ok"] is True
    assert calls["rerank"] == []


def test_upstream_asked_for_depth_in_full_mode(monkeypatch):
    m, calls = proxy(monkeypatch, docs("a", "b"), POLICY="novelty", DEPTH=30)
    m.search({"query": "q", "k": 5, "mode": "snippet", "tag": "7-x"})
    payload = calls["upstream"][0][1]
    assert payload["k"] == 30 and payload["mode"] == "full" and payload["untagged_ok"] is True


def test_novelty_drops_seen_and_falls_back(monkeypatch):
    m, _ = proxy(monkeypatch, docs("a", "b", "c", "d"), POLICY="novelty", DEPTH=4)
    first = m.search({"query": "q1", "k": 2, "mode": "full", "tag": "7-t"})["results"]
    assert [r["docid"] for r in first] == ["a", "b"]
    second = m.search({"query": "q2", "k": 2, "mode": "full", "tag": "7-t"})["results"]
    assert [r["docid"] for r in second] == ["c", "d"]          # a, b already returned to this tag
    assert m.STATS["novelty_filtered"] == 2
    third = m.search({"query": "q3", "k": 2, "mode": "full", "tag": "7-t"})["results"]
    assert [r["docid"] for r in third] == ["a", "b"]           # fewer than k unseen: falls back to the full list
    assert m.STATS["fallback_seen"] == 1
    other = m.search({"query": "q1", "k": 2, "mode": "full", "tag": "8-t"})["results"]
    assert [r["docid"] for r in other] == ["a", "b"]           # novelty is per tag (per run)


def test_rerank_orders_by_score_against_question_and_query(monkeypatch):
    res = docs("a", "b", "c")
    sc = {"text-a": 0.1, "text-b": 0.9, "text-c": 0.5}
    m, calls = proxy(monkeypatch, res, sc, POLICY="rerank", DEPTH=3)
    m.QUESTIONS["7"] = "THE QUESTION"
    out = m.search({"query": "the query", "k": 2, "mode": "full", "tag": "7-u"})["results"]
    assert [r["docid"] for r in out] == ["b", "c"]
    assert [r["score"] for r in out] == [0.9, 0.5]
    assert calls["rerank"][0][0] == "THE QUESTION\nCurrent search: the query"
    assert m.STATS["reranked"] == 1


def test_rerank_skipped_when_question_unknown(monkeypatch):
    m, calls = proxy(monkeypatch, docs("a", "b", "c"), {}, POLICY="novelty+rerank", DEPTH=3)
    out = m.search({"query": "q", "k": 2, "mode": "full", "tag": "99-u"})["results"]
    assert [r["docid"] for r in out] == ["a", "b"] and calls["rerank"] == []   # documented as SP-I2


def test_snippet_reshaping_when_mode_not_full(monkeypatch):
    long = [{"docid": "a", "score": 1.0, "text": "x" * 5000}]
    m, _ = proxy(monkeypatch, long, POLICY="novelty", DEPTH=1)
    out = m.search({"query": "q", "k": 1, "mode": "snippet", "snippet_max_tokens": 10, "tag": "7-s"})["results"]
    assert out == [{"docid": "a", "score": 1.0, "snippet": "x" * 40}]


def test_per_tag_counts(monkeypatch):
    m, _ = proxy(monkeypatch, docs("a", "b", "c"), POLICY="novelty", DEPTH=3)
    m.search({"query": "q", "k": 2, "mode": "full", "tag": "7-c"})
    m.search({"query": "q2", "k": 1, "mode": "full", "tag": "7-c"})
    t = m.TAGS["7-c"]
    assert (t["search"], t["k_sum"], t["hits"]) == (2, 3, 3) and t["docids"] == {"a", "b", "c"}
    assert m.STATS["searches"] == 2


def test_wire_mine_translation(monkeypatch):
    up = [{"doc_id": 11, "text": "t11", "score": 0.3}, {"doc_id": 12, "text": "t12", "score": 0.2}]
    m, calls = proxy(monkeypatch, up, {"t11": 0.1, "t12": 0.8}, POLICY="novelty+rerank", WIRE="mine", DEPTH=100)
    m.QUESTIONS["q0001"] = "request"
    out = m.search_mine({"query": "posts", "top_k": 1, "tag": "q0001-z", "index": "obliq-des-twitter"})
    path, payload = calls["upstream"][0]
    assert path == "/search" and payload == {"query": "posts", "top_k": 100, "index": "obliq-des-twitter"}
    assert out["results"] == [{"doc_id": "12", "text": "t12", "score": 0.8}]
    assert "docid" not in out["results"][0]


def test_wire_mine_passthrough_top_k(monkeypatch):
    m, calls = proxy(monkeypatch, [{"doc_id": 1, "text": "t", "score": 1.0}], POLICY="passthrough", WIRE="mine")
    m.search_mine({"query": "p", "top_k": 7})
    assert calls["upstream"][0][1] == {"query": "p", "top_k": 7}


class _Tok:
    """Stand-in tokenizer: one id per character."""
    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [ord(c) for c in text]
        return {"input_ids": ids[:max_length] if truncation and max_length else ids}


class _Torch:
    @staticmethod
    def no_grad():
        return contextlib.nullcontext()


@pytest.mark.parametrize("xbatch", ["1", "0"])
def test_rerank_scores_keeps_order_across_batches(monkeypatch, xbatch):
    monkeypatch.setenv("RERANK_XBATCH", xbatch)
    monkeypatch.setenv("RERANK_TOK_BUDGET", "60")
    monkeypatch.setenv("RERANK_BATCH", "2")
    m = load_file(SRC / "rerank_proxy.py")
    m.M.update(tok=_Tok(), torch=_Torch(), pre=[1], suf=[2])
    seen = []

    def fake_forward(seqs):           # score = sequence length, so the order of results is checkable
        seen.append(len(seqs))
        return [float(len(s)) for s in seqs]

    m._forward = fake_forward
    if xbatch == "1":
        threading.Thread(target=m._worker, daemon=True).start()
    texts = ["aaaa", "a", "aaaaaaa", "aa"]
    out = m.rerank_scores("q", texts)
    body = len(f"<Instruct>: {m.INSTR}\n<Query>: q\n<Document>: ")
    assert out == [float(1 + body + len(t) + 1) for t in texts]
    assert sum(seen) == len(texts)
