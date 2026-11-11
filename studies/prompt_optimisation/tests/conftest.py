"""pytest fixtures of the prompt-optimisation tests; constants are in po_testlib.py."""
import pytest

from .po_testlib import *  # noqa: F401,F403
from .po_testlib import REPO, STUDY  # noqa: F401


@pytest.fixture
def synth_env(tmp_path, monkeypatch):
    """Point every path variable at an empty temporary tree (no original data is read)."""
    env = {}
    for name in ("RUNS_DIR", "FULLDOC_ROOT", "FRAMES_DATA_DIR", "HOVER_DATA_DIR", "OBLIQ_BENCH_ROOT",
                 "ORIGINAL_REPO_ROOT", "RLM_DATA_DIR"):
        d = tmp_path / name.lower()
        d.mkdir()
        env[name] = str(d)
        monkeypatch.setenv(name, str(d))
    env.setdefault("RLMLIB_PATH", os.environ.get("RLMLIB_PATH", str(tmp_path / "rlm")))
    monkeypatch.setenv("RLMLIB_PATH", env["RLMLIB_PATH"])
    return env
