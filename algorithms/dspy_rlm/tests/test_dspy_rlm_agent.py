"""build_rlm_agent / search_index / get_document / run_question with a fake retriever and a fake dspy.RLM.

No LM is called: dspy.RLM is replaced by a recorder (and, in one test, constructed for real, which makes no call).
"""
from __future__ import annotations

import time

import dspy
import pytest
import requests

from dspy_rlm import agent as A
from dspy_rlm import signatures as S


class FakeRetriever:
    def __init__(self, n_docs=3, text_len=800):
        self.search_log = []
        self.doc_log = []
        self.n_docs, self.text_len = n_docs, text_len

    def search_index(self, query, top_k=10):
        self.search_log.append((query, top_k))
        return [{"score": 0.9 - i / 10, "doc_id": f"{query}-{i}", "text": "x" * self.text_len}
                for i in range(min(top_k, self.n_docs))]

    def get_document(self, doc_id):
        self.doc_log.append(doc_id)
        return {"doc_id": doc_id, "text": "body of " + doc_id}


class FakeRLM:
    """Records the constructor kwargs; calling it returns a canned prediction."""
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeRLM.instances.append(self)

    def __call__(self, question):
        self.question = question
        return dspy.Prediction(answer="Paris", trajectory={"iteration_1": "a", "iteration_2": "b", "other": 0})


@pytest.fixture
def fake_rlm(monkeypatch, azure_env):
    FakeRLM.instances = []
    monkeypatch.setattr(dspy, "RLM", FakeRLM)
    return FakeRLM


def _tools(rlm):
    return {f.__name__: f for f in rlm.kwargs["tools"]}


# ── wiring ────────────────────────────────────────────────────────────────────────────────────────────────

def test_build_rlm_agent_wires_max_iters_tools_signature(fake_rlm):
    rlm, metrics = A.build_rlm_agent(FakeRetriever(), max_iterations=9, verbose=True)
    kw = rlm.kwargs
    assert kw["max_iters"] == 9
    assert kw["verbose"] is True
    assert kw["signature"] is S.BrowseCompSignature        # default signature
    assert [f.__name__ for f in kw["tools"]] == ["search_index", "get_document"]
    assert "max_output_chars" not in kw                     # dspy default unless RLM_MAX_OUTPUT_CHARS
    assert isinstance(metrics, A.RunMetrics)


def test_build_rlm_agent_signature_and_metrics_passthrough(fake_rlm):
    m = A.RunMetrics()
    rlm, metrics = A.build_rlm_agent(FakeRetriever(), signature=S.BrowseCompMinimalSignature, metrics=m)
    assert rlm.kwargs["signature"] is S.BrowseCompMinimalSignature
    assert metrics is m
    assert rlm.kwargs["max_iters"] == 16                    # function default


def test_sub_lm_defaults_and_overrides(fake_rlm, monkeypatch):
    rlm, metrics = A.build_rlm_agent(FakeRetriever())
    sub = rlm.kwargs["sub_lm"]
    assert sub.model == "azure/gpt-5.4-mini"
    assert sub.kwargs["max_tokens"] == 8000 or sub.kwargs.get("max_completion_tokens") == 8000
    assert type(sub).__name__ == "_CountedLM" and isinstance(sub, dspy.LM)
    monkeypatch.setenv("AZURE_OPENAI_JUDGE_MODEL", "judge-dep")
    assert A.build_rlm_agent(FakeRetriever())[0].kwargs["sub_lm"].model == "azure/judge-dep"
    monkeypatch.setenv("SUB_LM_MODEL", "reader-dep")
    assert A.build_rlm_agent(FakeRetriever())[0].kwargs["sub_lm"].model == "azure/reader-dep"


def test_sub_lm_needs_azure_even_for_local_root_lm(fake_rlm, monkeypatch):
    """Documented issue: LOCAL_LLM_BASE_URL switches only the root LM; the sub-LM still reads Azure settings."""
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://localhost:8030/v1")
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT")
    assert A.build_lm().model.startswith("openai/")
    with pytest.raises(KeyError):
        A.build_rlm_agent(FakeRetriever())


def test_disable_get_document_and_max_output_chars(fake_rlm, monkeypatch):
    monkeypatch.setenv("DISABLE_GET_DOCUMENT", "1")
    monkeypatch.setenv("RLM_MAX_OUTPUT_CHARS", "50000")
    rlm, _ = A.build_rlm_agent(FakeRetriever())
    assert [f.__name__ for f in rlm.kwargs["tools"]] == ["search_index"]
    assert rlm.kwargs["max_output_chars"] == 50000


def test_real_dspy_rlm_construction(azure_env):
    """Constructing the real dspy.RLM makes no LM call; check what it ends up holding."""
    rlm, _ = A.build_rlm_agent(FakeRetriever(), max_iterations=7)
    assert isinstance(rlm, dspy.RLM)
    assert rlm.max_iters == 7
    assert set(rlm.tools) == {"search_index", "get_document"}
    assert rlm.tools["search_index"].func.__name__ == "search_index"


def test_tool_docstrings(fake_rlm, monkeypatch):
    t = _tools(A.build_rlm_agent(FakeRetriever())[0])
    assert "Documents up to 6,000 characters are printed in full" in t["get_document"].__doc__
    assert t["search_index"].__doc__.startswith("Search the BrowseComp+ corpus")
    monkeypatch.setenv("GETDOC_INLINE_CHARS", "0")
    assert "The text is NOT printed" in _tools(A.build_rlm_agent(FakeRetriever())[0])["get_document"].__doc__
    monkeypatch.setenv("MINIMAL_TOOL_DOCS", "1")
    t = _tools(A.build_rlm_agent(FakeRetriever(), default_top_k=5)[0])
    assert t["get_document"].__doc__ == "Retrieve a full document by its doc_id."
    assert "Returns top-5 hits" in t["search_index"].__doc__


# ── search_index ──────────────────────────────────────────────────────────────────────────────────────────

def test_search_truncates_counts_and_caches(fake_rlm):
    ret = FakeRetriever(text_len=800)
    rlm, m = A.build_rlm_agent(ret, default_top_k=2)
    search = _tools(rlm)["search_index"]
    out = search("alpha")
    assert ret.search_log == [("alpha", 2)]
    assert len(out) == 2 and all(r["text"] == "x" * 500 + "…" for r in out)
    assert m.search_calls == 1 and m.retrieved_doc_ids == ["alpha-0", "alpha-1"]
    assert search("alpha") is out                         # cached: no new request, no new count
    assert ret.search_log == [("alpha", 2)] and m.search_calls == 1


def test_hard_max_search_calls_stub(fake_rlm):
    ret = FakeRetriever()
    rlm, m = A.build_rlm_agent(ret, max_search_calls=2)
    search = _tools(rlm)["search_index"]
    search("q1"); search("q2")
    out = search("q3")
    assert len(ret.search_log) == 2                       # the retriever is not called once the budget is spent
    assert m.search_calls == 2
    assert out == [{"score": 0.0, "doc_id": "", "text": (
        "[SEARCH DISABLED — you have used all 2 search calls. "
        "You must call SUBMIT(answer=...) immediately with your best answer. "
        "Do not call search_index again.")}]
    assert search("q1")[0]["doc_id"] == ""                # the stub comes before the cache lookup


def test_per_turn_cap(fake_rlm, monkeypatch):
    monkeypatch.setenv("MAX_SEARCH_PER_TURN", "2")
    ret = FakeRetriever()
    rlm, m = A.build_rlm_agent(ret, max_search_calls=35)
    search = _tools(rlm)["search_index"]
    search("a"); search("b")
    refused = search("c")                                 # third search within 5 s = same code block
    assert refused == [{"score": 0.0, "doc_id": "", "text": (
        "[SEARCH-PER-TURN LIMIT (2) REACHED IN THIS CODE BLOCK. Print and read "
        "the results you already have, reason about them, then search again in your "
        "NEXT code block. Do not loop over long lists of queries.")}]
    assert [q for q, _ in ret.search_log] == ["a", "b"] and m.search_calls == 2
    m._last_search_t = time.perf_counter() - 6.0          # > 5 s gap = a new turn: counter resets
    assert search("c")[0]["doc_id"] == "c-0"
    assert m.search_calls == 3 and m._turn_search_n == 1


def test_per_turn_cap_counts_cached_and_runs_before_budget(fake_rlm, monkeypatch):
    monkeypatch.setenv("MAX_SEARCH_PER_TURN", "1")
    rlm, m = A.build_rlm_agent(FakeRetriever(), max_search_calls=1)
    search = _tools(rlm)["search_index"]
    search("a")
    assert "SEARCH-PER-TURN LIMIT (1)" in search("a")[0]["text"]   # even a cached repeat counts
    m._last_search_t = time.perf_counter() - 6.0
    assert "SEARCH DISABLED" in search("b")[0]["text"]            # then the hard budget


def test_max_top_k_clamp(fake_rlm, monkeypatch):
    monkeypatch.setenv("MAX_TOP_K", "5")
    ret = FakeRetriever(n_docs=50)
    search = _tools(A.build_rlm_agent(ret)[0])["search_index"]
    search("q", top_k=20); search("r", top_k=3)
    assert ret.search_log == [("q", 5), ("r", 3)]


def test_search_snippet_len(fake_rlm, monkeypatch):
    monkeypatch.setenv("SEARCH_SNIPPET_LEN", "10")
    search = _tools(A.build_rlm_agent(FakeRetriever(text_len=10))[0])["search_index"]
    assert search("q")[0]["text"] == "x" * 10               # not longer than the cap: unchanged


def test_unreachable_server_exits(fake_rlm):
    class Down(FakeRetriever):
        def search_index(self, query, top_k=10):
            raise requests.exceptions.ConnectionError("down")
    search = _tools(A.build_rlm_agent(Down())[0])["search_index"]
    with pytest.raises(SystemExit):
        search("q")


def test_get_document_inline_print(fake_rlm, capsys):
    ret = FakeRetriever()
    rlm, m = A.build_rlm_agent(ret)
    out = _tools(rlm)["get_document"]("d9")
    assert out == {"doc_id": "d9", "text": "body of d9"}
    assert m.get_document_calls == 1 and m.fetched_docs == {"d9": "body of d9"} and m.retrieved_doc_ids == ["d9"]
    assert "--- BEGIN d9 ---\nbody of d9\n--- END d9 ---" in capsys.readouterr().out


# ── parity with the original module ──────────────────────────────────────────────────────────────────────

def test_search_sequence_matches_original(original_rag, fake_rlm, monkeypatch):
    monkeypatch.setenv("MAX_SEARCH_PER_TURN", "3")
    outs = []
    for mod in (A, original_rag.agent):
        ret = FakeRetriever(n_docs=4, text_len=700)
        rlm, m = mod.build_rlm_agent(ret, max_search_calls=4, default_top_k=3)
        tools = {f.__name__: f for f in rlm.kwargs["tools"]}
        seq = [tools["search_index"](q) for q in ("a", "b", "a", "c", "d")]
        m._last_search_t = time.perf_counter() - 6.0
        seq += [tools["search_index"](q) for q in ("e", "f")]
        seq.append(tools["get_document"]("a-0"))
        outs.append((seq, ret.search_log, m.search_calls, m.retrieved_doc_ids,
                     [f.__doc__ for f in rlm.kwargs["tools"]], rlm.kwargs["max_iters"],
                     rlm.kwargs["sub_lm"].model))
    assert outs[0] == outs[1]


# ── run_question ──────────────────────────────────────────────────────────────────────────────────────────

def test_run_question_with_fake_rlm(fake_rlm):
    answer, m = A.run_question(FakeRetriever(), "Capital of France?", max_iterations=5, gold_ids={"g"})
    assert answer == "Paris"
    assert m.iterations == 2                               # keys starting with iteration_
    assert m._gold_ids == {"g"}
    assert m.latency_s >= 0.0
    assert FakeRLM.instances[-1].kwargs["max_iters"] == 5
    assert FakeRLM.instances[-1].question == "Capital of France?"


def test_run_metrics_helpers():
    m = A.RunMetrics(input_tokens=3, output_tokens=4, retrieved_doc_ids=["a", "b", "a"])
    assert m.total_tokens == 7 and m.unique_docs_retrieved == 2
    assert m.gold_recall(["a", "z"]) == 0.5 and m.gold_recall([]) == 0.0
