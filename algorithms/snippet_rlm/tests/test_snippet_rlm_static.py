"""Static checks: the source-level diff between original/ and src/ is limited to wiring and documentation, the launchers
parse, and every config names an existing launcher."""
from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest

from snippet_rlm_testlib import ALG, ART, ORIG, SRC

RENAMES = {
    "harness.py": ORIG / "examples" / "browsecomp_eval_official_rlm.py",
    "run_browsecomp.py": ORIG / "scripts" / "run_bcp_snippet.py",
    "run_frames.py": ORIG / "examples" / "frames_eval_rlm.py",
    "run_obliq.py": ORIG / "scripts" / "run_obliq_mine.py",
    "prompt_modules.py": ORIG / "scripts" / "official_prompt_modules.py",
    **{f"hooks/{h}/sitecustomize.py": ORIG / "scripts" / "hooks" / h / "sitecustomize.py"
       for h in ("retry429", "azure_sampling", "run5d", "cap_ranking", "retry429_cap_ranking")},
}

# String constants allowed to appear only in src/ or only in original/ (import wiring and paths, rules 2-3).
ONLY_IN_SRC = {"shared", "dspy_rlm", "src", "prompt_modules.py", "FRAMES_DATA_DIR", "frames_rows.json"}
ONLY_IN_ORIG = {"/home/extstu_00/RLM", "official_prompt_modules.py", "scripts",
                "/home/extstu_00/RLM/data/frames/frames_rows.json"}


def _strings(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                docstrings.add(id(first.value))
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings]


@pytest.mark.parametrize("new", list(RENAMES))
def test_string_constants_unchanged(new):
    """Every prompt string, tool description, task text, env-var name and default is identical; the only differing
    string constants are the import/path wiring ones listed above."""
    a, b = _strings(RENAMES[new]), _strings(SRC / new)
    only_orig = set(a) - set(b)
    only_src = set(b) - set(a)
    assert only_orig <= ONLY_IN_ORIG, only_orig - ONLY_IN_ORIG
    assert only_src <= ONLY_IN_SRC, only_src - ONLY_IN_SRC
    # multiplicities of every shared constant are identical too
    for s in set(a) & set(b):
        assert a.count(s) == b.count(s), s


@pytest.mark.parametrize("new", [k for k in RENAMES if k.startswith("hooks/") or k == "prompt_modules.py"])
def test_hooks_and_prompt_modules_code_identical(new):
    """Hooks and prompt_modules: only the module docstring differs."""
    def body(p):
        t = ast.parse(p.read_text(encoding="utf-8"))
        return ast.dump(ast.Module(body=t.body[1:], type_ignores=[]))
    assert body(RENAMES[new]) == body(SRC / new)


def test_no_examples_imports_or_absolute_paths_left():
    for p in SRC.rglob("*.py"):
        code = "\n".join(l for l in p.read_text().splitlines() if not l.lstrip().startswith("#"))
        tree = ast.parse(p.read_text())
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                assert not (n.module or "").startswith(("examples", "rag")), (p, n.module)
            if isinstance(n, ast.Import):
                assert not any(a.name.startswith(("examples", "rag.")) for a in n.names), p
        assert "/home/extstu_00" not in code.split('"""', 2)[-1], p


def _sh_files():
    return sorted((ALG / "experiments").rglob("*.sh"))


@pytest.mark.parametrize("sh", _sh_files(), ids=lambda p: str(p.relative_to(ALG)))
def test_launcher_bash_n(sh):
    p = subprocess.run(["bash", "-n", str(sh)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr


@pytest.mark.parametrize("sh", _sh_files(), ids=lambda p: str(p.relative_to(ALG)))
def test_launcher_has_no_machine_paths(sh):
    code = "\n".join(l for l in sh.read_text().splitlines() if not l.lstrip().startswith("#"))
    assert "/home/extstu_00" not in code and "/tmp/claude" not in code and "ismail" not in code


ROWS = ["BCP-9", "BCP-11", "BCP-13", "BCP-15", "BCP-17", "BCP-19", "FR-7", "FR-9", "MAT-9", "TWI-9", "CON-9"]
ALLOWED_UNKNOWN = re.compile(r"^(LAUNCHER_RLMFIX\d+|NOORCH_QUESTION_LIST|AZURE_OPENAI_MODEL_ORIGINAL|"
                             r"AZURE_OPENAI_JUDGE_MODEL_ORIGINAL|OTHER_DOTENV_VARIABLES)=UNKNOWN\b")


def test_every_row_has_a_config():
    assert sorted(p.stem for p in (ALG / "configs").glob("*.env")) == sorted(ROWS)


@pytest.mark.parametrize("row", ROWS)
def test_config_parses_and_names_a_launcher(row, tmp_path):
    cfg = ALG / "configs" / f"{row}.env"
    assert subprocess.run(["bash", "-n", str(cfg)]).returncode == 0
    script = (f'set -eu; RUNS_DIR={tmp_path}; ALG={ALG}; AZURE_DEPLOYMENT_GPT5=x; . "{cfg}"; '
              'echo "ROW=$ROW"; echo "LAUNCHER=$LAUNCHER"; for a in "${LAUNCHER_ARGS[@]}"; do echo "ARG=$a"; done')
    p = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
    assert p.returncode == 0, p.stderr
    vals = dict(l.split("=", 1) for l in p.stdout.splitlines() if not l.startswith("ARG="))
    args = [l[4:] for l in p.stdout.splitlines() if l.startswith("ARG=")]
    assert vals["ROW"] == row
    launcher = ALG / vals["LAUNCHER"]
    assert launcher.is_file() and launcher.stat().st_mode & 0o111, launcher
    if args and args[0] == "snip":            # OBLIQ: ids file exists
        assert Path(args[2]).is_file(), args[2]
    for line in cfg.read_text().splitlines():
        if "UNKNOWN" in line and not line.lstrip().startswith("#"):
            assert ALLOWED_UNKNOWN.match(line) or "UNKNOWN" in line.split("#", 1)[1], line


def test_obliq_prompt_files_are_the_ones_the_runs_logged():
    """MAT-9 logged '[task] answer contract overridden from .../task_snippet.txt (1571 chars)' and
    '[strategy] BrowseComp+ block swapped for OBLIQ (932 chars)'."""
    m = ALG / "configs" / "obliq" / "math"
    assert len((m / "task_snippet.txt").read_text()) == 1571
    assert len((m / "strategy_snippet.txt").read_text().strip()) == 932


def test_shared_browsecomp_data_identical_to_snapshot():
    assert (ART / "shared" / "rlm_shared" / "browsecomp_data.py").read_bytes() == \
        (ORIG / "examples" / "browsecomp_data.py").read_bytes()
