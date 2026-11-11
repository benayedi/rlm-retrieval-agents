"""Static checks of the refactored code: it compiles, says where it comes from, holds no machine paths, and keeps
every prompt-sized string of the original unchanged (REFACTOR_RULES: no prompt, critic or feedback text changes)."""
import ast
import re
import subprocess
from pathlib import Path

import pytest

from .po_testlib import ORIG, PY_MAP, STUDY

SRC_FILES = sorted(p for d in ("src", "obliq_twitter/src", "hover/src") for p in (STUDY / d).glob("*.py"))
SH_FILES = sorted(p for d in ("experiments", "obliq_twitter/experiments", "hover/experiments")
                  for p in (STUDY / d).glob("*.sh"))


def test_every_original_python_file_has_a_copy():
    origs = {str(p.relative_to(ORIG)) for p in (ORIG / "scripts").glob("*.py")}
    assert origs == set(PY_MAP), origs ^ set(PY_MAP)
    assert {str(p.relative_to(STUDY)) for p in SRC_FILES} == set(PY_MAP.values())


@pytest.mark.parametrize("p", SRC_FILES, ids=lambda p: p.name)
def test_compiles_and_names_its_original(p):
    tree = ast.parse(p.read_text(), filename=str(p))
    doc = ast.get_docstring(tree) or ""
    orig = next(o for o, n in PY_MAP.items() if STUDY / n == p)
    assert f"original/RLM/{orig}" in doc


@pytest.mark.parametrize("p", SRC_FILES + SH_FILES, ids=lambda p: p.name)
def test_no_machine_paths(p):
    t = p.read_text()
    assert "/home/" not in t and "/tmp/claude" not in t and "miniconda" not in t


@pytest.mark.parametrize("p", SH_FILES, ids=lambda p: p.name)
def test_launchers_parse(p):
    assert subprocess.run(["bash", "-n", str(p)]).returncode == 0


def _strings(path):
    out = set()
    for node in ast.walk(ast.parse(Path(path).read_text())):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) >= 40:
            out.add(node.value)
        elif isinstance(node, ast.JoinedStr):   # f-strings: compare their literal parts
            for v in node.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str) and len(v.value) >= 40:
                    out.add(v.value)
    return out


@pytest.mark.parametrize("orig,new", sorted(PY_MAP.items()))
def test_long_strings_unchanged(orig, new):
    """Every string of 40+ characters in the original (prompt text, critic text, feedback, log lines) is still there,
    byte for byte; only module docstrings (documentation) and machine paths may differ."""
    o, n = ORIG / orig, STUDY / new
    odoc = ast.get_docstring(ast.parse(o.read_text()), clean=False) or ""
    missing = [s for s in _strings(o) - _strings(n) if s != odoc and "/home/" not in s
               and not s.startswith(("/data/", "/logs/", "/scripts/", "/experiments/"))]   # path fragments of f-strings
    assert not missing, [m[:80] for m in missing]


@pytest.mark.parametrize("orig,new", sorted(PY_MAP.items()))
def test_only_wiring_lines_changed(orig, new):
    """Each removed original line is a path, import or interpreter line (rules 2-3), the docstring, or one of the
    listed lines of CHANGES_FROM_ORIGINAL.md."""
    o = (ORIG / orig).read_text().splitlines()
    n = set((STUDY / new).read_text().splitlines())
    allowed = re.compile(r"/home/extstu_00|\{R\}|\{SUP\}|\bR\b = |^import |sys\.path\.insert|from examples|"
                         r"import gepa_bcp|cwd=(R|SUP)\b|RLM_TRAJ_DIR=|^\s*procs = \[\]|^\s*for w in range\(workers\):|"
                         r"^\s*part = qids|^\s*env = dict\(os\.environ, GEPA_PROMPTS|^\"\"\"|\"\"\"$")
    removed = [l for l in o if l not in n and l.strip() and not allowed.search(l)]
    assert not removed, removed[:5]
