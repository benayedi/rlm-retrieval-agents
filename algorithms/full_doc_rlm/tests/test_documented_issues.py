"""Behaviour recorded under "Issues found, not fixed" in CHANGES_FROM_ORIGINAL.md, pinned so it cannot drift."""
from __future__ import annotations

import sys

import pytest

from conftest import SRC, load_file, needs_rlm


@needs_rlm
def test_no_delegation_is_ignored_by_the_stock_library():
    from rlm.environments.local_repl import LocalREPL
    env = LocalREPL(no_delegation=True)
    try:
        for name in ("llm_query", "llm_query_batched", "rlm_query", "rlm_query_batched"):
            assert callable(env.globals[name]), name
        assert env.kwargs.get("no_delegation") is True      # stored by BaseEnv and never read
    finally:
        env.cleanup()


def test_prompt_id_does_not_hash_the_solo_prompt_or_lean_clause(clean_env):
    p = load_file("_src_prompts_issue", SRC / "prompts.py")
    before = p.prologue_id()
    p.SOLO_SYSTEM_PROMPT = p.SOLO_SYSTEM_PROMPT + " changed"
    p.LEAN_CLAUSE = p.LEAN_CLAUSE + " changed"
    p.NO_DELEGATION_CLAUSE = p.NO_DELEGATION_CLAUSE + " changed"
    assert p.prologue_id() == before


def test_obliq_adapter_snippet_mode_returns_full_text(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["obliq_adapter.py"])
    mod = load_file("_src_obliq_adapter", SRC / "obliq_adapter.py")
    long_text = "word " * 5000

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return {"results": [{"doc_id": 7, "score": 0.5, "text": long_text}]}

    monkeypatch.setattr(mod.requests, "post", lambda *a, **k: R())
    hits = mod._search("q", 5, "snippet", "tag-1")
    assert hits == [{"docid": "7", "score": 0.5, "snippet": long_text}]   # no truncation to snippet_max_tokens
    assert mod._tag_counts["tag-1"] == {"search": 1, "k_sum": 5, "hits": 1}
