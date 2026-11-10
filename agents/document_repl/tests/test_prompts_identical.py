"""Every text the model sees is unchanged: prompt blocks, tool descriptions, BM25 rewrites, prompt_id."""
from __future__ import annotations

import itertools
import json
import subprocess
import sys

import pytest

from conftest import ORIG_SCRIPTS, ORIG_SRC, SRC, load_file, subprocess_env


@pytest.fixture
def prompt_modules(clean_env):
    return load_file("_src_prompts", SRC / "prompts.py"), load_file("_orig_prompts", ORIG_SRC / "prompts.py")


def _strings(mod):
    return {k: v for k, v in vars(mod).items() if isinstance(v, str) and not k.startswith("__")}


def test_all_prompt_strings_equal(prompt_modules):
    new, old = prompt_modules
    a, b = _strings(new), _strings(old)
    assert a.keys() == b.keys()
    for k in a:
        assert a[k] == b[k], k


def test_build_prologue_equal_for_every_switch(prompt_modules):
    new, old = prompt_modules
    for bits in itertools.product([False, True], repeat=6):
        for budget in (0, 7):
            kw = dict(zip(["no_delegation", "lean", "solo", "single_pass", "constraints", "structured_plan"], bits))
            assert new.build_prologue(budget, **kw) == old.build_prologue(budget, **kw)


def test_prologue_id_matches_the_run_records(prompt_modules):
    new, old = prompt_modules
    # 94f31e91f6b7 is the prompt_id recorded by the dense and ColBERT rows (BCP-27, BCP-29).
    assert new.prologue_id() == old.prologue_id() == "94f31e91f6b7"


def test_tool_descriptions_equal(clean_env):
    new = load_file("_src_tools", SRC / "tools.py")
    old = load_file("_orig_tools", ORIG_SRC / "tools.py")
    for design, k_over, cap, k in (("A", None, 0, 5), ("B", None, 10, 5), ("B", False, 10, 20), ("D", None, 10, 5),
                                   ("B", None, 0, 5)):
        a = new.ToolBox(design=design, allow_k_override=k_over, max_searches=cap, default_k=k).descriptions()
        b = old.ToolBox(design=design, allow_k_override=k_over, max_searches=cap, default_k=k).descriptions()
        assert a == b, (design, k_over, cap)
    assert new.MAX_K == old.MAX_K == 50


_PROBE = r"""
import json, runpy, sys
runpy.run_path = lambda *a, **k: None          # stop the wrapper before it runs run_one.py
sys.argv = ["x"]
src = open({wrapper!r}).read()
src = src.replace({orig_dir!r}, {new_dir!r})   # point the original wrapper at the preserved original src/
code = compile(src, {wrapper!r}, "exec")
sys.path.insert(0, {new_dir!r})
exec(code, {{"__name__": "wrapper", "__file__": {wrapper!r}}})
import prompts, tools
d = tools.ToolBox(design="B", max_searches=10).descriptions()["search"]
print(json.dumps([d, prompts.USER_PROLOGUE, prompts.SOLO_READ_SENTENCE, prompts.prologue_id()]))
"""


def _bm25_texts(wrapper, prompts_dir, replace_from, corpus):
    code = _PROBE.format(wrapper=str(wrapper), orig_dir=replace_from, new_dir=str(prompts_dir))
    env = subprocess_env(**({"BM25_CORPUS": corpus} if corpus else {}))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("corpus", [None, "bcp"])
def test_bm25_wrapper_texts_equal(corpus):
    new = _bm25_texts(SRC / "run_bm25.py", SRC, "@@none@@", corpus)
    old = _bm25_texts(ORIG_SCRIPTS / "frames_sup_run_one.py", ORIG_SRC, "/home/extstu_00/suprlm/src", corpus)
    assert new == old
    assert "BM25 keyword search" in new[0] and "dense semantic" not in new[0]
    # 758e43ed5b95 is the prompt_id recorded by every BM25 row (BCP-25, BCP-31, FR-11, FR-13).
    assert new[3] == "758e43ed5b95"
