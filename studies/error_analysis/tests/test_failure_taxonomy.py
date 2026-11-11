"""ERR-1: src/failure_taxonomy.py and src/failure_drill_s25.py write and print exactly what the originals do, on a
synthetic FRAMES tree (824 questions, made-up rows; no model, no network).

The originals hard-code R = "<original repository>"; they are run here with that one line pointed at the synthetic
tree (logs/frames/... and data/frames/... below it), the readable copies with LOGS_ROOT / FRAMES_DATA_DIR.
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import random
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

STUDY = Path(__file__).resolve().parents[1]
ORIG = STUDY / "original" / "RLM" / "scripts"


def _tree(tmp: Path) -> Path:
    rng = random.Random(7)
    root = tmp / "R"
    fr, lg = root / "data" / "frames", root / "logs" / "frames"
    fr.mkdir(parents=True); lg.mkdir(parents=True)
    types = ["Numerical reasoning", "Tabular reasoning", "Multiple constraints", "Temporal reasoning", "Post processing"]
    rows, tsv = [], []
    for i in range(824):
        n = rng.randint(1, 3)
        gold = [rng.randint(0, 999) for _ in range(n - (1 if i % 17 == 0 else 0))]
        ans = rng.choice(["42", "1999", "Paris", "12.5 km"])
        rows.append({"i": i, "gold_ids": gold, "reasoning_types": rng.sample(types, 2), "answer": ans})
        tsv.append({"Prompt": f"q{i}", "Answer": ans, "wiki_links": str([f"https://en.wikipedia.org/wiki/T{i}_{k}" for k in range(n)])})
    (fr / "frames_rows.json").write_text(json.dumps(rows))
    with open(fr / "test.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Prompt", "Answer", "wiki_links"], delimiter="\t"); w.writeheader(); w.writerows(tsv)
    offs = np.cumsum([0] + [rng.randint(500, 30000) for _ in range(1000)])
    np.savez(fr / "corpus_meta.npz", titles=np.array([f"T{j} title" for j in range(1000)], dtype=object), offsets=offs)
    for cond in ("oracle", "closed", "bm25_4"):
        with open(lg / f"llm_{cond}.jsonl", "w") as f:
            for i in range(824):
                f.write(json.dumps({"i": i, "correct": rng.random() < 0.5}) + "\n")
    for d, ids in (("sup_q36_bm25f", range(824)), ("sup_q36_bm25f_s25", range(0, 200)),
                   ("sup_q36_bm25f_s25_rest0", range(200, 512)), ("sup_q36_bm25f_s25_rest1", range(512, 824))):
        dd = lg / d; dd.mkdir()
        graded = {}
        for i in ids:
            if i % 41 == 0:
                continue                                   # no record: killed at the time cap
            st = rng.choice(["completed"] * 8 + ["error", "empty_output"])
            raw = {"status": st, "tool_call_counts": {"search": rng.randint(1, 25)}, "n_iterations": rng.randint(3, 50),
                   "retrieved_docids": rows[i]["gold_ids"][:1] + rng.sample(range(1000), 5), "search_log": [{"query": "T1 title x"}, {"query": "y"}]}
            (dd / f"{i:03d}.json").write_text(json.dumps(raw))
            a = rng.choice(["Exact Answer: 42", "Exact Answer: ...", "", "Exact Answer: Paris"])
            graded[str(i)] = {"qid": str(i), "correct": rng.random() < 0.6, "answer": a, "gold_recall": rng.choice([0, 0.5, 1.0]),
                              "searches": rng.randint(1, 25)}
        (dd / "graded.json").write_text(json.dumps(graded))
    return root


def _exec_original(path: Path, root: Path, argv):
    src = path.read_text().replace('R = "/home/extstu_00/RLM"', f'R = {str(root)!r}')
    old_argv = sys.argv; sys.argv = [str(path)] + argv
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(src, str(path), "exec"), {"__name__": "__main__"})
    finally:
        sys.argv = old_argv
    return buf.getvalue()


def _run_src(path: Path, argv):
    old_argv = sys.argv; sys.argv = [str(path)] + argv
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.argv = old_argv
    return buf.getvalue()


@pytest.mark.parametrize("argv,out", [([], "failure_taxonomy.json"), (["--s25"], "failure_taxonomy_s25.json")])
def test_taxonomy_identical(tmp_path, monkeypatch, argv, out):
    root = _tree(tmp_path)
    monkeypatch.setenv("LOGS_ROOT", str(root / "logs")); monkeypatch.setenv("FRAMES_DATA_DIR", str(root / "data" / "frames"))
    old = _exec_original(ORIG / "frames_failure_taxonomy.py", root, argv)
    old_json = (root / "logs" / "frames" / out).read_bytes()
    new = _run_src(STUDY / "src" / "failure_taxonomy.py", argv)
    assert new == old
    assert (root / "logs" / "frames" / out).read_bytes() == old_json
    buckets = {r["bucket"] for r in json.loads(old_json).values()}
    assert "solved" in buckets and any(b.startswith("A1") for b in buckets) and any(b.startswith("C") for b in buckets)


def test_drill_identical(tmp_path, monkeypatch):
    root = _tree(tmp_path)
    monkeypatch.setenv("LOGS_ROOT", str(root / "logs")); monkeypatch.setenv("FRAMES_DATA_DIR", str(root / "data" / "frames"))
    _exec_original(ORIG / "frames_failure_taxonomy.py", root, ["--s25"])
    old = _exec_original(ORIG / "frames_s25_failure_drill.py", root, [])
    new = _run_src(STUDY / "src" / "failure_drill_s25.py", [])
    assert new == old and "D questions:" in new


def test_no_machine_paths():
    for p in (STUDY / "src").glob("*.py"):
        s = p.read_text()
        assert "/home/" not in s and "/tmp/claude" not in s and "From original/RLM/scripts/" in s, p
