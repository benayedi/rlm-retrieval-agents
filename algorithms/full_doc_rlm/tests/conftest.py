"""Shared helpers for the Full-Doc RLM tests. Mocks only: no network, no model, no retrieval server."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

ALG = Path(__file__).resolve().parents[1]
SRC = ALG / "src"
ORIG = ALG / "original"
ORIG_SRC = ORIG / "FULLDOC" / "src"
ORIG_SCRIPTS = ORIG / "RLM" / "scripts"
REPO = ALG.parents[1]
SHARED = REPO / "shared"

# src/ modules import each other by bare name (prompts, tools, ...), as in the original working directory.
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_file(name: str, path: Path):
    """Import a file under a private module name (used for the original copies, which share module names)."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rlm_importable() -> bool:
    try:
        import rlm  # noqa: F401
        return True
    except Exception:
        return False


needs_rlm = pytest.mark.skipif(not rlm_importable(),
                               reason="the rlm library (RLMLIB_PATH, alexzhang13/rlm @ 854e688) is not importable")


@pytest.fixture
def clean_env(monkeypatch):
    """Unset every variable that switches harness behaviour at import time."""
    for v in ("GEPA_PROMPTS", "SUP_FENCE_TOLERANT", "SUP_MAX_BLOCKS", "SUP_REP_GUARD", "SUP_PRESENCE_PENALTY",
              "BM25_CORPUS", "FRAMES_TITLE_SEARCH", "QUERIES_FILE"):
        monkeypatch.delenv(v, raising=False)
    return monkeypatch


def subprocess_env(**extra) -> dict:
    """A minimal environment for helper subprocesses; machine paths point at a dummy directory."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("GEPA_", "SUP_", "BM25_", "BCP_", "LOCAL_JUDGE", "FRAMES_"))}
    env.update(extra)
    return env
