"""The 2026-10-31 additions (DSA-3 / DSA-5 ablation) in src/: signature v21 and the opt-in switches DSPY_QUIET_GETDOC and
DSPY_MAX_LLM_CALLS, with their defaults unchanged. Offline: dspy.RLM is the recorder of test_dspy_rlm_agent.py.
"""
from __future__ import annotations

import ast
from pathlib import Path

from dspy_rlm import agent as A
from dspy_rlm import signatures as S

from test_dspy_rlm_agent import FakeRetriever, _tools, fake_rlm  # noqa: F401  (fixture re-export)

ALGO = Path(__file__).resolve().parents[1]
ORIG2 = ALGO / "original" / "RLM_2026-11-11" / "RLM"


def test_v21_instructions_identical_to_second_snapshot():
    tree = ast.parse((ORIG2 / "rag" / "signatures.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BrowseCompSnippetPromptSignature")
    assert S.BrowseCompSnippetPromptSignature.instructions == ast.get_docstring(cls, clean=False)
    assert S.BrowseCompSnippetPromptSignature._sig_key == "v21"
    assert set(S.BrowseCompSnippetPromptSignature.input_fields) == {"question"}
    assert set(S.BrowseCompSnippetPromptSignature.output_fields) == {"answer"}


def test_quiet_getdoc_default_unchanged(fake_rlm, capsys, monkeypatch):
    monkeypatch.delenv("DSPY_QUIET_GETDOC", raising=False)
    rlm, _ = A.build_rlm_agent(FakeRetriever())
    _tools(rlm)["get_document"]("d9")
    assert "chars, INLINE)" in capsys.readouterr().out


def test_quiet_getdoc_on(fake_rlm, capsys, monkeypatch):
    monkeypatch.setenv("DSPY_QUIET_GETDOC", "1")
    rlm, _ = A.build_rlm_agent(FakeRetriever())
    _tools(rlm)["get_document"]("d9")
    out = capsys.readouterr().out
    assert "INLINE" not in out and "chars)" in out


def test_max_llm_calls_switch(fake_rlm, monkeypatch):
    monkeypatch.delenv("DSPY_MAX_LLM_CALLS", raising=False)
    rlm, _ = A.build_rlm_agent(FakeRetriever())
    assert "max_llm_calls" not in rlm.kwargs                       # dspy.RLM's own default (50) applies
    monkeypatch.setenv("DSPY_MAX_LLM_CALLS", "100000")
    rlm, _ = A.build_rlm_agent(FakeRetriever())
    assert rlm.kwargs["max_llm_calls"] == 100000


def test_dsa_launchers_run_the_second_snapshot():
    for arm, top_k in (("3", "10"), ("5", "5")):
        sh = (ALGO / "experiments" / f"dsa_arm{arm}.sh").read_text(encoding="utf-8")
        assert 'SNAP="$ALGO_DIR/original/RLM_2026-11-11/RLM"' in sh
        assert f'"$SNAP/examples/browsecomp_eval_dspy_rlm.py" --indices "$1" --sig v21 --top-k {top_k} --max-iters 100' in sh
        orig = (ORIG2 / "scripts" / "run5d" / f"dspy_ablation_arm{arm}.sh").read_text(encoding="utf-8")
        for line in orig.splitlines():   # every exported setting of the original, except the deployment name
            if line.startswith(("export EMBEDDING", "export MINIMAL", "unset ")):
                assert line in sh, line
        assert "/home/" not in sh and "/tmp/claude" not in sh and "ismail" not in sh
