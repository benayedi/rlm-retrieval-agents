"""Shared fixtures for the DSPy.RLM tests: import paths, fake Azure credentials, the original `rag` package.

Everything here is offline. Azure credentials are dummies on an .invalid host; no test calls an LM or the network
(requests.post / requests.get are monkeypatched wherever a retriever is exercised).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ALGO_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ALGO_DIR.parents[1]
SRC_DIR = ALGO_DIR / "src"
ORIGINAL_DIR = ALGO_DIR / "original" / "RLM"

for p in (REPO_ROOT / "agents" / "shared", SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


@pytest.fixture
def azure_env(monkeypatch):
    """Dummy Azure settings: build_rlm_agent reads them unconditionally to build the sub-LM (no call is made)."""
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://dummy.invalid/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "dummy-key")
    for name in ("MAX_TOP_K", "MAX_SEARCH_PER_TURN", "SEARCH_SNIPPET_TOKENS", "SEARCH_SNIPPET_LEN",
                 "GETDOC_INLINE_CHARS", "MINIMAL_TOOL_DOCS", "DISABLE_GET_DOCUMENT", "RLM_MAX_OUTPUT_CHARS",
                 "SUB_LM_MODEL", "AZURE_OPENAI_JUDGE_MODEL", "AZURE_OPENAI_MODEL", "SUB_LM_REASONING_EFFORT",
                 "AZURE_OPENAI_REASONING_EFFORT", "LOCAL_LLM_BASE_URL", "OBLIQ_INDEX", "RETRIEVER_TIMEOUT",
                 "BROWSECOMP_COMMIT_TURN"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="session")
def original_rag():
    """The original package, imported under its own name `rag` from original/RLM (read-only).

    Any `rag` modules another test suite may have loaded are set aside and restored afterwards, and original/RLM is
    on sys.path only while importing, so this cannot leak into other agents' tests.
    """
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "rag" or k.startswith("rag.")}
    sys.path.insert(0, str(ORIGINAL_DIR))
    dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True          # never write __pycache__ into original/ (it is checksummed)
    try:
        import rag
        import rag.agent
        import rag.signatures
        import rag.tools
    finally:
        sys.path.remove(str(ORIGINAL_DIR))
        sys.dont_write_bytecode = dont_write
    assert Path(rag.__file__).resolve().parent == ORIGINAL_DIR / "rag"
    yield rag
    for k in [k for k in sys.modules if k == "rag" or k.startswith("rag.")]:
        del sys.modules[k]
    sys.modules.update(saved)
