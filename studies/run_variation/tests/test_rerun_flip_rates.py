"""VAR-2: src/rerun_flip_rates.py prints exactly what the original inline code printed, on a synthetic artifact.

Offline. The artifact is a tiny HTML file with an embedded JSON trace in the shape of docs/supervisor_artifact.html
(runs with "run", "label", "config" and per-question "qid" / "correct"); no BrowseComp-Plus data is involved.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import runpy
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
SRC = STUDY / "src" / "rerun_flip_rates.py"
ORIG = STUDY / "original" / "TRANSCRIPT" / "var2_flip_rates.py"


def _run(cfg, eff="medium", model="gpt-5-2025-08-07"):
    return [{"k": "reasoning effort", "v": eff}, {"k": "model actually served", "v": model}] + cfg


def _artifact(tmp: Path) -> Path:
    runs = []
    for name, eff, model, correct in [
        ("runA", "medium", "gpt-5-2025-08-07", [1] * 30 + [0] * 20),
        ("runB", "medium", "gpt-5-2025-08-07", [1] * 25 + [0] * 5 + [1] * 8 + [0] * 12),
        ("runC", "high", "gpt-5-2025-08-07", [1] * 50),              # other effort: never paired with A/B
        ("runD", "medium", "qwen3.8-flash", [0] * 50),               # other model: skipped
        ("runE", "medium", "gpt-5-2025-08-07", [1] * 10),            # < 40 common questions: skipped
    ]:
        qs = [{"qid": str(i), "correct": bool(c)} for i, c in enumerate(correct)]
        runs.append({"run": name, "label": name.upper(), "config": _run([], eff, model), "questions": qs})
    root = tmp / "repo"
    (root / "docs").mkdir(parents=True)
    trace = json.dumps({"runs": runs})
    (root / "docs" / "supervisor_artifact.html").write_text(
        f'<html><script id="trace" type="application/json">{trace}</script></html>', encoding="utf-8")
    return root


def _capture(fn):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn()
    return buf.getvalue()


def test_src_matches_original_output(tmp_path, monkeypatch):
    root = _artifact(tmp_path)
    monkeypatch.setenv("ORIGINAL_REPO_ROOT", str(root))
    new = _capture(lambda: runpy.run_path(str(SRC), run_name="__main__"))
    monkeypatch.chdir(root)          # the original ran with cwd = the original repository
    old = _capture(lambda: exec(compile(ORIG.read_text(), str(ORIG), "exec"), {"__name__": "__main__"}))
    assert new == old
    lines = new.splitlines()
    assert lines[0] == "5"           # runs with questions
    assert len(lines) == 2 and "runA" in lines[1] and "runB" in lines[1]
    assert "break 5/30 (0.17) fix 8/20 (0.40)" in lines[1]


def test_src_has_no_machine_paths():
    s = SRC.read_text()
    assert "/home/" not in s and "/tmp/claude" not in s
    assert "From original/TRANSCRIPT/var2_flip_rates.py" in s


def test_manifest_records_the_transcript_source():
    m = json.loads((STUDY / "original" / "MANIFEST.json").read_text())
    f = {x["copy"]: x for x in m["files"]}["TRANSCRIPT/var2_flip_rates.py"]
    assert f["source"].endswith(":3616") and f["tool_use_id"] and f["timestamp"].startswith("2026-10-27T02:05")
