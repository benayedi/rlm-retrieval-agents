"""Helpers for the search-policy study tests (imported by name; conftest.py only re-exports the fixture).
Mocks only: no network, no GPU, no model, no LLM call."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
SRC = STUDY / "src"
ANALYSIS = STUDY / "analysis"
EXPERIMENTS = STUDY / "experiments"
ORIG = STUDY / "original" / "RLM"
SHARED = REPO / "agents" / "shared"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

_n = [0]


def load_file(path: Path, name: str | None = None):
    """Import a file under a fresh private module name (module-level code runs again each time)."""
    _n[0] += 1
    spec = importlib.util.spec_from_file_location(name or f"_sp_test_mod_{_n[0]}", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """configs/paths.env stand-in: every location variable points into tmp_path."""
    d = {}
    for name in ("RUNS_DIR", "LOGS_ROOT", "RLM_DATA_DIR", "FULLDOC_ROOT", "FRAMES_DATA_DIR", "OBLIQ_BENCH_ROOT"):
        p = tmp_path / name.lower()
        p.mkdir()
        monkeypatch.setenv(name, str(p))
        d[name] = p
    return d


def write_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))
    return p
