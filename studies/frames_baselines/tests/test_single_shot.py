"""FB-1: src/single_shot.py produces the same per-question records and summary as the original frames_eval.py, on a
synthetic FRAMES tree with a fake OpenAI client (no server, no network, no model).

The original hard-codes ROOT = "<original repository>"; it is run with that line pointed at the synthetic tree, which
also holds examples/browsecomp_data.py (the byte-identical original grader module, from agents/snippet_repl/original).
The readable copy reads FRAMES_DATA_DIR, LOGS_ROOT and ORIGINAL_REPO_ROOT. The BM25 top-100 cache is pre-written, so
no index is needed. Timing fields ("sec") are dropped before comparing.
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import runpy
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
ORIG = STUDY / "original" / "RLM" / "scripts" / "frames_eval.py"
SRC = STUDY / "src" / "single_shot.py"


class _Msg:
    def __init__(self, content): self.message = types.SimpleNamespace(content=content)


class _Completions:
    def create(self, model, messages, **kw):
        text = messages[0]["content"]
        h = int(hashlib.sha256(text.encode()).hexdigest(), 16)
        if kw.get("temperature") == 0:                       # judge call
            return types.SimpleNamespace(choices=[_Msg("correct: " + ("yes" if h % 3 else "no"))])
        return types.SimpleNamespace(choices=[_Msg(f"<think>Exact Answer: <x></think>\nExact Answer: A{h % 5}")])


class FakeOpenAI:
    def __init__(self, **kw): self.chat = types.SimpleNamespace(completions=_Completions())


def _tree(tmp: Path) -> Path:
    root = tmp / "R"
    data, logs = root / "data" / "frames", root / "logs" / "frames"
    data.mkdir(parents=True); logs.mkdir(parents=True); (root / "examples").mkdir()
    shutil.copy(REPO / "agents" / "snippet_repl" / "original" / "RLM" / "examples" / "browsecomp_data.py", root / "examples")
    titles = [f"Article {i}" for i in range(30)]
    with open(data / "corpus.jsonl", "w") as f:
        for t in titles:
            f.write(json.dumps({"title": t, "text": f"text of {t} " * 50}) + "\n")
    with open(data / "test.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Prompt", "Answer", "wiki_links", "reasoning_types"], delimiter="\t")
        w.writeheader()
        for i in range(12):
            links = [f"https://en.wikipedia.org/wiki/Article_{(i * 3 + k) % 30}" for k in range(1 + i % 3)]
            if i == 5:
                links.append("https://en.wikipedia.org/wiki/Missing_page")
            w.writerow({"Prompt": f"question {i}?", "Answer": f"A{i % 5}", "wiki_links": str(links), "reasoning_types": "Temporal reasoning"})
    np.save(logs / "bm25_top100.npy", np.array([[(i + j) % 30 for j in range(100)] for i in range(12)]))
    return root


def _strip(path: Path):
    return [{k: v for k, v in json.loads(l).items() if k != "sec"} for l in path.read_text().splitlines()]


def _run(fn, argv):
    old = sys.argv; sys.argv = ["x"] + argv
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            fn()
    finally:
        sys.argv = old
    return buf.getvalue()


def test_single_shot_identical(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    # One worker: Corpus.doc() shares one file handle across the worker threads (seek + readline), so with several
    # workers a read can start at another thread's offset (FB-I1 in CHANGES_FROM_ORIGINAL.md; kept, not fixed).
    argv = ["--conds", "closed,oracle,bm25_4", "--workers", "1"]
    a, b = _tree(tmp_path / "a"), _tree(tmp_path / "b")
    src = ORIG.read_text().replace('ROOT = "/home/extstu_00/RLM"', f"ROOT = {str(a)!r}")
    out_old = _run(lambda: exec(compile(src, str(ORIG), "exec"), {"__name__": "__main__"}), argv)
    monkeypatch.setenv("FRAMES_DATA_DIR", str(b / "data" / "frames"))
    monkeypatch.setenv("LOGS_ROOT", str(b / "logs")); monkeypatch.setenv("ORIGINAL_REPO_ROOT", str(b))
    out_new = _run(lambda: runpy.run_path(str(SRC), run_name="__main__"), argv)
    for c in ("closed", "oracle", "bm25_4"):
        assert _strip(a / "logs/frames" / f"llm_{c}.jsonl") == _strip(b / "logs/frames" / f"llm_{c}.jsonl"), c
    for f in ("llm_summary.json", "coverage_recall.json"):
        assert json.loads((a / "logs/frames" / f).read_text()) == json.loads((b / "logs/frames" / f).read_text()), f
    strip_t = lambda s: [l for l in s.splitlines() if "index loaded" not in l and "retrieved" not in l]
    assert strip_t(out_old) == strip_t(out_new)


def test_no_machine_paths():
    s = SRC.read_text()
    assert "/home/" not in s and "/tmp/claude" not in s and "From original/RLM/scripts/frames_eval.py" in s


def test_corpus_doc_shares_one_file_handle():
    """Pins FB-I1: one handle, seek + readline, no lock (the same in the original and the readable copy)."""
    for p in (ORIG, SRC):
        t = p.read_text()
        assert 'self.f = open(CORPUS, "rb")' in t
        assert "self.f.seek(int(self.offs[i])); return json.loads(self.f.readline())" in t
        assert "Lock(" not in t
