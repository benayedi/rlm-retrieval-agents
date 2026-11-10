"""The refactored src/ must show the model exactly what the original showed, and compute exactly what it computed.

Each scenario imports one entry point twice in separate interpreters -- the original file from original/RLM/ and the
refactored file from src/ -- under the same environment, and compares: the system prompt (both module copies), the
orchestrator addendum, the task text, the tool specs, the full first message list, the turn-0 user message, the
search-cap refusal, code-block parsing on synthetic turns, the format gate, the record run_one writes (fake agent and
fake retriever), the kwargs passed to RLM(...) and the retriever requests.
"""
from __future__ import annotations

import pytest

from snippet_repl_testlib import LIGHTON, MATH, SCENARIOS, needs_rlm

@needs_rlm
@pytest.mark.parametrize("name", list(SCENARIOS))
def test_src_equals_original(probe, name):
    entry, env = SCENARIOS[name]
    a = probe("orig", entry, env)
    b = probe("src", entry, env)
    assert sorted(a) == sorted(b)
    diff = [k for k in a if a[k] != b[k]]
    assert not diff, f"{name}: differs in {diff}"


@needs_rlm
def test_scenarios_exercise_the_switches(probe):
    """Guard against a vacuous comparison: the switches really change what the model sees."""
    base = probe("src", "harness", {})
    run5d = probe("src", "harness", dict(LIGHTON))
    assert base["task_suffix"] != run5d["task_suffix"]
    assert base["tool_specs"] != run5d["tool_specs"]
    assert len(base["messages"][0]["content"]) > len(run5d["messages"][0]["content"])  # addendum dropped
    qwen = probe("src", "bcp", SCENARIOS["bcp-qwen (BCP-15/17/19)"][1])
    assert "hard budget of 25 search_index calls" in qwen["system_prompt_core"]
    assert "The fence label matters" in qwen["system_prompt_core"]
    assert qwen["search_at_huge_count"][0]["text"].startswith("ERROR: search budget exhausted (25 of 25 calls used)")
    assert qwen["rlm_kwargs"]["backend_kwargs"]["timeout"] == 1800.0
    assert qwen["rlm_kwargs"]["sampling_args"] == {"reasoning_effort": "medium", "max_tokens": 8192,
                                                   "presence_penalty": 1.5}
    tw = probe("src", "obliq", SCENARIOS["obliq-twitter (TWI-9)"][1])
    assert tw["max_rejects"] == 3          # the BrowseComp+ format gate stays ON in the TWI-9 setting
    assert all("tag" in p["json"] for p in tw["posts"] if p["url"] == "search")
    math = probe("src", "obliq", SCENARIOS["obliq-math (MAT-9 files + other switches)"][1])
    assert math["max_rejects"] == 0
    assert math["task_suffix"] != tw["task_suffix"]
    assert any(line.startswith("[task] answer contract overridden from") for line in math["import_stdout"])
