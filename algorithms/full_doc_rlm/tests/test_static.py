"""py_compile of src/, bash -n of the launchers, the row configs parse, launcher flags equal the originals."""
from __future__ import annotations

import py_compile
import re
import subprocess
from pathlib import Path

import pytest

from conftest import ALG, ORIG_SCRIPTS, ORIG_SRC, SRC

ROWS = ["BCP-21", "BCP-23", "BCP-25", "BCP-27", "BCP-29", "BCP-31", "FR-11", "FR-13", "MAT-11", "TWI-11", "CON-11"]
REQUIRED = ["ROW_ID", "BENCHMARK", "ENTRY", "ROOT_MODEL", "REASONING_EFFORT", "EFFORT_SENT", "SERVER",
            "MAX_SEARCHES", "MAX_ITERATIONS", "RUN_ARGS", "ORIGINAL_OUTPUT"]
SH = sorted((ALG / "experiments").rglob("*.sh"))


@pytest.mark.parametrize("f", sorted(SRC.glob("*.py")), ids=lambda p: p.name)
def test_py_compile(f, tmp_path):
    py_compile.compile(str(f), cfile=str(tmp_path / "x.pyc"), doraise=True)


@pytest.mark.parametrize("f", SH, ids=lambda p: str(p.relative_to(ALG)))
def test_bash_syntax(f):
    r = subprocess.run(["bash", "-n", str(f)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_no_machine_paths_in_code():
    for f in list(SRC.glob("*.py")) + SH:
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if "/home/" in line or "/tmp/claude" in line:
                assert "original" in line and "/tmp/claude" not in line, f"{f.name}:{i}: {line}"


def _source(env_file: Path) -> dict:
    script = ('set -eu; RLM_DATA_DIR=/data/rlm; FULLDOC_ROOT=/data/fulldoc; set -a; . "$1"; set +a; '
              'env -0')
    r = subprocess.run(["bash", "-c", script, "x", str(env_file)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return dict(kv.split("=", 1) for kv in r.stdout.split("\0") if "=" in kv)


@pytest.mark.parametrize("row", ROWS)
def test_config_parses(row):
    f = ALG / "configs" / f"{row}.env"
    first = f.read_text().splitlines()[0]
    assert first.startswith(f"# {row} |")
    env = _source(f)
    for k in REQUIRED:
        assert k in env, (row, k)
    assert env["ROW_ID"] == row
    args = env["RUN_ARGS"]
    assert "--design B" in args and "--max-iterations 30" in args
    if env["MAX_SEARCHES"] != "UNKNOWN":
        assert f"--max-searches {env['MAX_SEARCHES']}" in args
    assert f"--reasoning-effort {env['REASONING_EFFORT']}" in args


def test_config_set_is_complete():
    assert sorted(p.stem for p in (ALG / "configs").glob("*.env")) == sorted(ROWS)


def _flags(path: Path) -> list[str]:
    text = "\n".join(l for l in path.read_text().splitlines() if not l.lstrip().startswith("#"))
    return sorted(set(re.findall(r"(?<![\w-])--[a-z][a-z0-9-]+", text)))


PAIRS = [
    ("bcp_gpt5_colbert.sh", ORIG_SCRIPTS / "bcp_gpt5_sup.sh"),
    ("bcp_gpt5_bm25.sh", ORIG_SCRIPTS / "bcp_gpt5_bm25.sh"),
    ("frames_qwen.sh", ORIG_SCRIPTS / "run_frames_sup.sh"),
    ("frames_gpt5.sh", ORIG_SCRIPTS / "run5d" / "frames_fd_gpt5.sh"),
    ("bcp_gpt5_final830/final830_run.sh", ORIG_SRC / "final830_run.sh"),
    ("bcp_gpt5_final830/final830_complete.sh", ORIG_SRC / "final830_complete.sh"),
    ("bcp_gpt5_final830/launch_high400.sh", ORIG_SRC / "launch_high400.sh"),
    ("bcp_gpt5_final830/launch_high100b.sh", ORIG_SRC / "launch_high100b.sh"),
    ("bcp_gpt5_final830/launch_high100c.sh", ORIG_SRC / "launch_high100c.sh"),
    ("bcp_gpt5_final830/launch_high_rest230.sh", ORIG_SRC / "launch_high_rest230.sh"),
]


@pytest.mark.parametrize("new,old", PAIRS, ids=[p[0] for p in PAIRS])
def test_launcher_flags_unchanged(new, old):
    assert _flags(ALG / "experiments" / new) == _flags(old)
