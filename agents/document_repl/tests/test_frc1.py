"""FRC-1 (FRAMES, search cap 25 / 50 turns): the two launchers keep the original flags, the id lists cover the 824
questions exactly once, and src/frames_cap25_report.py prints what the original report printed (synthetic graded.json).
"""
from __future__ import annotations

import contextlib
import io
import json
import random
import runpy
import subprocess
from pathlib import Path

ALG = Path(__file__).resolve().parents[1]
SNAP = ALG / "original" / "RLM_2026-11-11" / "RLM" / "scripts"


def test_launchers_parse():
    for f in ("frames_qwen_cap25_test200.sh", "frames_qwen_cap25_rest.sh"):
        assert subprocess.run(["bash", "-n", str(ALG / "experiments" / f)]).returncode == 0


def test_launchers_keep_the_original_settings():
    t200 = (ALG / "experiments/frames_qwen_cap25_test200.sh").read_text()
    rest = (ALG / "experiments/frames_qwen_cap25_rest.sh").read_text()
    for s in ("BM25_PROMPT=1 MAX_SEARCHES=25 MAX_ITERS=50 BASE_URL=http://localhost:8030/v1 W=8",
              "export PYTHONPATH=$RLMLIB_PATH OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false"):
        assert s in t200, s
    assert "BM25_PROMPT=1 MAX_SEARCHES=25 MAX_ITERS=50 BASE_URL=http://localhost:$1/v1 W=8 IDS_FILE=$2 OUT=$3" in rest
    assert "run 8030 " in rest and "run 8031 " in rest and "sleep 120" in rest
    orig_chain = (SNAP / "frames_s25_chain.sh").read_text(); orig_full = (SNAP / "frames_s25_full.sh").read_text()
    assert "MAX_SEARCHES=25 MAX_ITERS=50 BASE_URL=http://localhost:8030/v1 W=8" in orig_chain
    assert "MAX_SEARCHES=25 MAX_ITERS=50 BASE_URL=http://localhost:$1/v1 W=8 IDS_FILE=$2 OUT=$3" in orig_full
    for s in (t200, rest):
        assert "/home/" not in s


def test_id_lists_partition_frames():
    d = ALG / "configs" / "frames_ids"
    lists = {f: [l.strip() for l in (d / f"{f}.txt").read_text().split()] for f in ("test_ids_shuffled", "s25_rest_gpu0", "s25_rest_gpu1")}
    assert [len(v) for v in lists.values()] == [200, 312, 312]
    allq = [q for v in lists.values() for q in v]
    assert sorted(map(int, allq)) == list(range(824))


def test_report_identical(tmp_path, monkeypatch):
    rng = random.Random(3)
    root = tmp_path / "R"; fr = root / "logs" / "frames"
    for d, ids in (("sup_q36_bm25f", range(824)), ("sup_q36_bm25f_s25", range(200)),
                   ("sup_q36_bm25f_s25_rest0", range(200, 512)), ("sup_q36_bm25f_s25_rest1", range(512, 824))):
        (fr / d).mkdir(parents=True)
        g = {str(i): {"qid": str(i), "correct": rng.random() < .65, "gold_recall": rng.random(), "searches": rng.randint(1, 25)}
             for i in ids if i % 50}
        (fr / d / "graded.json").write_text(json.dumps(g))
    src = (SNAP / "frames_s25_full_report.py").read_text().replace('R = "/home/extstu_00/RLM"', f"R = {str(root)!r}")
    a, b = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(a):
        exec(compile(src, "orig", "exec"), {"__name__": "__main__"})
    monkeypatch.setenv("FRAMES_RUNS_DIR", str(fr))
    with contextlib.redirect_stdout(b):
        runpy.run_path(str(ALG / "src" / "frames_cap25_report.py"), run_name="__main__")
    assert a.getvalue() == b.getvalue() and "25-search agent" in a.getvalue()
