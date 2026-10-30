"""Static checks: every .py compiles, every launcher parses, and the agent package is code-identical to rag/.

Code identity: the AST of each src/dspy_rlm module equals the AST of the original rag/ module once the module
docstring is removed (comments are not in the AST). So the only differences in those four files are documentation.
"""
from __future__ import annotations

import ast
import py_compile
import shutil
import subprocess
from pathlib import Path

import pytest

ALGO = Path(__file__).resolve().parents[1]
SRC = ALGO / "src"
ORIG = ALGO / "original" / "RLM"


@pytest.mark.parametrize("path", sorted(SRC.rglob("*.py")), ids=lambda p: str(p.relative_to(ALGO)))
def test_py_compile(path, tmp_path):
    py_compile.compile(str(path), cfile=str(tmp_path / "x.pyc"), doraise=True)


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
@pytest.mark.parametrize("path", sorted((ALGO / "experiments").glob("*.sh")), ids=lambda p: p.name)
def test_bash_n(path):
    r = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def _ast_without_module_docstring(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = tree.body
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    tree.body = body
    return ast.dump(tree, include_attributes=False)


@pytest.mark.parametrize("name", ["__init__.py", "agent.py", "signatures.py", "tools.py"])
def test_package_code_identical_to_original(name):
    assert _ast_without_module_docstring(SRC / "dspy_rlm" / name) == \
        _ast_without_module_docstring(ORIG / "rag" / name)


def _normalised_driver_ast(path: Path, is_original: bool) -> str:
    return _normalised_from_source(path.read_text(encoding="utf-8"), is_original)


def test_run_browsecomp_differs_only_in_import_wiring():
    assert _normalised_driver_ast(SRC / "run_browsecomp.py", False) == \
        _normalised_driver_ast(ORIG / "examples" / "browsecomp_eval_dspy_rlm.py", True)


def test_run_obliq_differs_only_in_wiring_and_obliq_root():
    new = (SRC / "run_obliq.py").read_text(encoding="utf-8")
    old = (ORIG / "examples" / "obliq_eval_dspy_rlm.py").read_text(encoding="utf-8")
    # Undo the two allowed changes on the original side, textually: OBLIQ_ROOT -> _obliq_root() and the
    # constant's definition -> the helper + its import.
    old = old.replace('OBLIQ_ROOT = Path(__file__).parent.parent.parent / "OBLIQ-Bench"\n', "") \
             .replace("OBLIQ_ROOT /", "_obliq_root() /")
    # Third allowed change (rule 4, documentation): the --model help text no longer names a personal Azure
    # deployment. Compare the two files with that one help string blanked on both sides.
    import re
    help_re = r'help="Override AZURE_OPENAI_MODEL \([^)]*\)"'
    old, new = re.sub(help_re, 'help=""', old), re.sub(help_re, 'help=""', new)
    new_tree = ast.parse(new)
    new_tree.body = [n for n in new_tree.body
                     if not (isinstance(n, ast.FunctionDef) and n.name == "_obliq_root")
                     and not (isinstance(n, ast.ImportFrom) and n.module == "rlm_shared.paths")]
    assert _normalised_from_source(ast.unparse(new_tree), False) == _normalised_from_source(old, True)


def _normalised_from_source(src: str, is_original: bool) -> str:
    """Driver AST without the module docstring and sys.path.insert lines, with `rag`/`examples` imports renamed."""
    if is_original:
        src = src.replace('"rag.signatures"', '"dspy_rlm.signatures"').replace("from rag", "from dspy_rlm") \
                 .replace("from examples.", "from rlm_shared.")
    tree = ast.parse(src)
    keep = []
    for i, node in enumerate(tree.body):
        if i == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) \
                and ast.unparse(node.value.func) == "sys.path.insert":
            continue
        keep.append(node)
    tree.body = keep
    return ast.dump(tree, include_attributes=False)
