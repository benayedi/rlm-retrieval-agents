"""DSA-3 / DSA-5 report: src/ablation_report.py writes the same report.json as the original dspy_ablation_report.py,
on synthetic records, with the gpt-4.1 judge and the HF dataset replaced by fakes (no network, no LLM).
"""
from __future__ import annotations

import json
import random
import runpy
import sys
import types
from pathlib import Path

ALGO = Path(__file__).resolve().parents[1]
ORIG = ALGO / "original" / "RLM_2026-11-11" / "RLM" / "scripts" / "run5d" / "dspy_ablation_report.py"
QIDS = [str(1000 + i) for i in range(830)]


def _fake_grade(outputs, cache_path, workers=32):
    return {q: bool(a) and int(q) % 3 != 0 for q, a in outputs.items()}


def _tree(root: Path):
    rng = random.Random(5)
    L = root / "logs"; (L / "dspy_ablation" / "arm5").mkdir(parents=True); (L / "bcp_rejudge_gpt41").mkdir()
    strata = ["S1", "S2", "S3", "S4", "S5"]
    qs = [{"qid": QIDS[i], "q_idx": i, "stratum": strata[j % 5]} for j, i in enumerate(range(0, 500, 5))]
    (L / "dspy_ablation" / "sample100.json").write_text(json.dumps(
        {"seed": 1, "population": 775, "strata": {s: {"pop": 155, "n": 20} for s in strata}, "questions": qs}))
    recs = [{"q_idx": x["q_idx"], "predicted": "ans" if x["q_idx"] % 7 else "", "iterations": rng.randint(2, 30),
             "search_calls": rng.randint(1, 40), "latency_s": rng.random() * 900} for x in qs[:95]]
    (L / "dspy_ablation" / "arm5" / "s0.json").write_text(json.dumps(recs))
    (L / "bcp_rejudge_gpt41" / "5.json").write_text(json.dumps({"verdicts": {q: int(q) % 2 == 0 for q in QIDS}}))
    (L / "dspy_ablation" / "snippet_bcp9_per_question.json").write_text(json.dumps({"qa": [
        {"sheet": "BrowseComp+", "row": 9, "qid": q, "correct": int(q) % 4 != 0, "turns": 5} for q in QIDS]}))
    for p in ["browsecomp_v2medium_baseline_first50.json", "browsecomp_v2medium_baseline_200.json",
              "browsecomp_v2_medium_q148_q830.json", "browsecomp_v2_medium_retry56.json"]:
        (L / p).write_text(json.dumps([{"q_idx": i, "iterations": 16, "search_calls": 10} for i in range(830)]))
    (root / "scripts").mkdir()
    (root / "scripts" / "bcp_judge.py").write_text(
        "def grade_outputs(outputs, cache_path, workers=32):\n    return {q: bool(a) and int(q) % 3 != 0 for q, a in outputs.items()}\n")
    return root


def test_report_identical(tmp_path, monkeypatch):
    fake_ds = types.ModuleType("datasets")
    fake_ds.load_dataset = lambda *a, **k: [{"query_id": q} for q in QIDS]
    monkeypatch.setitem(sys.modules, "datasets", fake_ds)
    a, b = _tree(tmp_path / "a"), _tree(tmp_path / "b")
    src = ORIG.read_text().replace('R = "/home/extstu_00/RLM"', f"R = {str(a)!r}") \
        .replace('"/home/extstu_00/RLM/logs/dspy_ablation/snippet_bcp9_per_question.json"',
                 repr(str(a / "logs/dspy_ablation/snippet_bcp9_per_question.json")))
    monkeypatch.setattr(sys, "argv", ["x", str(a / "logs/dspy_ablation/arm5")])
    sys.modules.pop("bcp_judge", None)
    exec(compile(src, "orig", "exec"), {"__name__": "__main__"})
    sys.modules.pop("bcp_judge", None)
    fake = types.ModuleType("rlm_shared.grading.bcp_judge"); fake.grade_outputs = _fake_grade
    monkeypatch.setitem(sys.modules, "rlm_shared.grading.bcp_judge", fake)
    monkeypatch.setenv("LOGS_ROOT", str(b / "logs"))
    monkeypatch.setattr(sys, "argv", ["x", str(b / "logs/dspy_ablation/arm5")])
    runpy.run_path(str(ALGO / "src" / "ablation_report.py"), run_name="__main__")
    ra = json.loads((a / "logs/dspy_ablation/arm5/report.json").read_text())
    rb = json.loads((b / "logs/dspy_ablation/arm5/report.json").read_text())
    assert ra == rb and ra["n"] == 100 and ra["results"] == 95
