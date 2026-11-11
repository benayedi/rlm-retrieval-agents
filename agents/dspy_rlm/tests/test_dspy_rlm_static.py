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
ORIG2 = ALGO / "original" / "RLM_2026-11-11" / "RLM"   # second snapshot: the files the DSA-3/5 ablation ran


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


@pytest.mark.parametrize("name", ["__init__.py", "tools.py"])
def test_package_code_identical_to_original(name):
    # unchanged between the two snapshots
    assert _ast_without_module_docstring(SRC / "dspy_rlm" / name) == \
        _ast_without_module_docstring(ORIG / "rag" / name) == _ast_without_module_docstring(ORIG2 / "rag" / name)


@pytest.mark.parametrize("name", ["agent.py", "signatures.py"])
def test_package_code_identical_to_second_snapshot(name):
    # agent.py / signatures.py gained the opt-in switches DSPY_QUIET_GETDOC, DSPY_MAX_LLM_CALLS and signature v21
    # on 2026-10-31 (defaults unchanged); src/ carries exactly those additions.
    assert _ast_without_module_docstring(SRC / "dspy_rlm" / name) == \
        _ast_without_module_docstring(ORIG2 / "rag" / name)


def _normalised_driver_ast(path: Path, is_original: bool) -> str:
    return _normalised_from_source(path.read_text(encoding="utf-8"), is_original)


def test_run_browsecomp_differs_only_in_import_wiring():
    # src/ = the 2026-10-30 driver + the two v21 entries of 2026-10-31 (signature table, --sig choices). The
    # 2026-10-31 result-queue fix (read before join) is NOT in src/: it is a bug fix, kept in the second snapshot only.
    new = (SRC / "run_browsecomp.py").read_text(encoding="utf-8")
    v21 = '    "v21": __import__("dspy_rlm.signatures", fromlist=["x"]).BrowseCompSnippetPromptSignature,\n'
    assert new.count(v21) == 1 and new.count('"v20", "v21"], default="v1"') == 1
    new = new.replace(v21, "").replace('"v20", "v21"], default="v1"', '"v20"], default="v1"')
    assert _normalised_from_source(new, False) == \
        _normalised_driver_ast(ORIG / "examples" / "browsecomp_eval_dspy_rlm.py", True)


def test_second_snapshot_driver_adds_v21_and_the_queue_fix_only():
    """What the DSA launchers run (original/RLM_2026-11-11) vs the 2026-10-30 driver: v21 and the result-queue fix."""
    import difflib
    a = (ORIG / "examples" / "browsecomp_eval_dspy_rlm.py").read_text(encoding="utf-8").splitlines()
    b = (ORIG2 / "examples" / "browsecomp_eval_dspy_rlm.py").read_text(encoding="utf-8").splitlines()
    added = [l[1:] for l in difflib.unified_diff(a, b, lineterm="", n=0) if l.startswith("+") and not l.startswith("+++")]
    removed = [l[1:] for l in difflib.unified_diff(a, b, lineterm="", n=0) if l.startswith("-") and not l.startswith("---")]
    assert removed == ["        p.join(timeout)", "            status, *rest = q.get()",
                       [l for l in a if "--sig" in l and "choices" in l][0]]
    assert any("BrowseCompSnippetPromptSignature" in l for l in added)
    assert any("msg = q.get(timeout=5)" in l for l in added)


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
