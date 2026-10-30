"""run_one.main() record building, with the RLM replaced by a fake agent that drives the real rlm logger.

Checks n_iterations = len(trajectory iterations) including the forced final-answer call that rlm makes after the
turn cap (rlm.core.rlm.RLM._default_answer, called here for real), and records the documented issues:
served_models is None with the stock library, and a partial answer is discarded on timeout.
"""
from __future__ import annotations

import json
import sys
import types
import urllib.error
import urllib.request

import pytest

from conftest import SRC, load_file, needs_rlm

pytestmark = needs_rlm


@pytest.fixture
def run_one(clean_env, monkeypatch):
    mod = load_file("_src_run_one", SRC / "run_one.py")
    monkeypatch.setattr(mod, "load_query", lambda qid: "What is the answer?")
    monkeypatch.setattr(mod, "harden_sandbox", lambda hosts: None)   # it patches process-wide resolver/builtins

    def refuse(*a, **k):
        raise urllib.error.URLError("no server in tests")
    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    return mod


class FakeAgent:
    """Logs `turns` iterations like RLM.completion does, then either hits the cap or raises a timeout."""

    def __init__(self, logger, turns, timeout=False):
        self.logger, self.turns, self.timeout = logger, turns, timeout

    def completion(self, payload, root_prompt=None):
        from rlm.core.rlm import RLM
        from rlm.core.types import RLMIteration, RLMMetadata
        from rlm.utils.exceptions import TimeoutExceededError

        self.logger.log_metadata(RLMMetadata(root_model="fake", max_depth=1, max_iterations=30, backend="fake",
                                             backend_kwargs={}, environment_type="local", environment_kwargs={}))
        for i in range(self.turns):
            self.logger.log(RLMIteration(prompt="p", response=f"turn {i}", code_blocks=[]))
        if self.timeout:
            raise TimeoutExceededError(elapsed=99.0, timeout=10.0, partial_answer="Exact Answer: partial")
        handler = types.SimpleNamespace(completion=lambda prompt: "Exact Answer: forced")
        stub = types.SimpleNamespace(logger=self.logger)
        answer = RLM._default_answer(stub, [{"role": "user", "content": "q"}], handler)   # the library's own code
        usage = types.SimpleNamespace(to_dict=lambda: {"model_usage_summaries": {}})
        return types.SimpleNamespace(response=answer, usage_summary=usage)


def _run(mod, monkeypatch, tmp_path, agent_kw, extra=()):
    out = tmp_path / "1.json"
    monkeypatch.setattr(mod, "build_rlm", lambda toolbox, args, logger: FakeAgent(logger, **agent_kw))
    monkeypatch.setattr(sys, "argv", ["run_one.py", "--query-id", "1", "--out", str(out), "--design", "B",
                                      "--max-searches", "10", "--save-trajectory", *extra])
    mod.main()
    return json.loads(out.read_text())


def test_n_iterations_counts_the_forced_final_answer(run_one, monkeypatch, tmp_path):
    rec = _run(run_one, monkeypatch, tmp_path, {"turns": 30})
    assert rec["status"] == "completed"
    assert rec["result"][0]["output"] == "Exact Answer: forced"
    assert len(rec["trajectory"]["iterations"]) == 31
    assert rec["n_iterations"] == 31        # 30 turns + the forced call: "31" means the turn cap was hit


def test_n_iterations_without_cap(run_one, monkeypatch, tmp_path):
    rec = _run(run_one, monkeypatch, tmp_path, {"turns": 4})
    assert rec["n_iterations"] == len(rec["trajectory"]["iterations"]) == 5


def test_config_fields(run_one, monkeypatch, tmp_path):
    rec = _run(run_one, monkeypatch, tmp_path, {"turns": 1})
    cfg = rec["config"]
    assert cfg["k_settable_by_model"] is True and cfg["fixed_k"] is None and cfg["default_k"] == 5
    assert cfg["hard_search_cap"] == 10 and cfg["max_iterations"] == 30
    assert cfg["prompt_id"] == "94f31e91f6b7"
    rec = _run(run_one, monkeypatch, tmp_path, {"turns": 1}, extra=("--fixed-k", "--default-k", "20"))
    assert rec["config"]["k_settable_by_model"] is False and rec["config"]["fixed_k"] == 20


def test_served_models_is_none_with_the_stock_library(run_one, monkeypatch, tmp_path):
    rec = _run(run_one, monkeypatch, tmp_path, {"turns": 1})
    assert rec["served_models"] is None     # rlm @ 854e688 has no SERVED_MODELS (issue, not fixed)


def test_partial_answer_is_discarded_on_timeout(run_one, monkeypatch, tmp_path):
    rec = _run(run_one, monkeypatch, tmp_path, {"turns": 3, "timeout": True})
    assert rec["status"] == "error" and rec["error"].startswith("TimeoutExceededError")
    assert rec["result"][0]["output"] == ""     # e.partial_answer is never read (issue, not fixed)
    assert rec["n_iterations"] == 3


def test_record_counts_are_client_side_when_server_is_down(run_one, monkeypatch, tmp_path):
    rec = _run(run_one, monkeypatch, tmp_path, {"turns": 1})
    assert rec["tool_call_counts"] == {"search": 0}
    assert rec["server_counts"] is None and rec["server_docids"] is None
    assert "error" in rec["retriever_stats"]
