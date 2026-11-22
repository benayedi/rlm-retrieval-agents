"""Shared code added with the follow-up studies (2026-11-11), offline:

- agents/shared/rlm_shared/grading/rejudge_bcp_rq4.py writes the same rq4_*.json files as the original
  bcp_rejudge_gpt41_rq4.py, with the gpt-4.1 judge replaced by a deterministic fake (no network, no LLM);
- benchmarks/models/serve_qwen36.sh keeps every argument and default of the original vLLM launcher.
"""
from __future__ import annotations

import hashlib
import json
import runpy
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SNAP = REPO / "agents" / "shared" / "original" / "RLM_2026-11-11" / "RLM" / "scripts"
FAKE_JUDGE = '''
import json, os, hashlib
def run_outputs(run_dir):
    o = {}
    for f in os.listdir(run_dir):
        if f.endswith(".json") and not f.startswith("_"):
            x = json.load(open(f"{run_dir}/{f}"))
            out = ((x.get("result") or [{}])[0] or {}).get("output", "") or ""
            o[str(x["query_id"])] = out if x.get("status") == "completed" else ""
    return o
def grade_outputs(outputs, cache_path, workers=32):
    v = {}
    for q, a in outputs.items():
        h = int(hashlib.sha256(a.encode()).hexdigest(), 16)
        v[q] = False if not a.strip() else ("filtered" if h % 11 == 0 else bool(h % 3))
    return v
'''


def _tree(root: Path, test_ids):
    logs = root / "logs"
    for d in ("bcp_policy_e2e_rerankonly/rollouts/108e3760dc", "gepa_bcp2/rollouts/d911ebbe23"):
        (logs / d).mkdir(parents=True)
        for q in list(test_ids) + ["9001", "9002"]:          # 9001/9002: not test200 (GEPA's own evaluation)
            st = "completed" if int(q) % 7 else "timeout"
            (logs / d / f"{q}.json").write_text(json.dumps({"query_id": q, "status": st,
                                                            "result": [{"output": f"answer {q} {d[:4]}"}]}))
    (logs / "bcp_rejudge_gpt41").mkdir()
    for row in ("16", "17"):
        (logs / "bcp_rejudge_gpt41" / f"{row}.json").write_text(json.dumps(
            {"verdicts": {q: bool((int(q) + int(row)) % 2) for q in test_ids}}))
    (root / "data" / "bcp_gepa_splits").mkdir(parents=True)
    split = json.loads((REPO / "studies/prompt_optimisation/splits/bcp_gepa_splits/split.json").read_text()) \
        if (REPO / "studies/prompt_optimisation/splits/bcp_gepa_splits/split.json").exists() else {"test200": list(test_ids)}
    (root / "data" / "bcp_gepa_splits" / "split.json").write_text(json.dumps(split))
    (root / "scripts").mkdir()
    (root / "scripts" / "bcp_judge.py").write_text(FAKE_JUDGE)
    return root


def test_rejudge_rq4_identical(tmp_path, monkeypatch):
    split = REPO / "studies/prompt_optimisation/splits/bcp_gepa_splits/split.json"
    test_ids = [str(q) for q in json.loads(split.read_text())["test200"]][:40] if split.exists() else [str(i) for i in range(40)]
    a, b = _tree(tmp_path / "a", test_ids), _tree(tmp_path / "b", test_ids)
    # original, with R pointed at tree a (its own scripts/bcp_judge.py is the fake)
    src = (SNAP / "bcp_rejudge_gpt41_rq4.py").read_text().replace('R = "/home/extstu_00/RLM"', f"R = {str(a)!r}")
    for k in [m for m in sys.modules if m == "bcp_judge"]:
        del sys.modules[k]
    exec(compile(src, "orig", "exec"), {"__name__": "__main__"})
    sys.modules.pop("bcp_judge", None)
    # readable copy, with the fake judge as rlm_shared.grading.bcp_judge and the same split file
    fake = types.ModuleType("rlm_shared.grading.bcp_judge"); exec(FAKE_JUDGE, fake.__dict__)
    monkeypatch.setitem(sys.modules, "rlm_shared.grading.bcp_judge", fake)
    monkeypatch.setenv("ORIGINAL_REPO_ROOT", str(b)); monkeypatch.setenv("LOGS_ROOT", str(b / "logs"))
    assert split.exists(), "the split copy studies/prompt_optimisation/splits/bcp_gepa_splits/split.json is missing"
    runpy.run_path(str(REPO / "agents/shared/rlm_shared/grading/rejudge_bcp_rq4.py"), run_name="__main__")
    for f in ("rq4_rerank_only.json", "rq4_gepa_winner.json", "rq4_summary.json"):
        assert json.loads((a / "logs/bcp_rejudge_gpt41" / f).read_text()) == \
            json.loads((b / "logs/bcp_rejudge_gpt41" / f).read_text()), f
    s = json.loads((b / "logs/bcp_rejudge_gpt41/rq4_summary.json").read_text())
    assert s["gepa_winner"]["n"] == len(test_ids)            # only test200 ids are judged
    assert s["rerank_only"]["rollout_dir"] == "logs/bcp_policy_e2e_rerankonly/rollouts/108e3760dc"


def test_serve_qwen36_keeps_the_original_arguments():
    orig = (SNAP / "serve_qwen36_moe.sh").read_text().splitlines()
    new = (REPO / "benchmarks/models/serve_qwen36.sh").read_text().splitlines()
    i, j = orig.index("set -euo pipefail"), new.index("set -euo pipefail")
    body_o = [l for l in orig[i + 1:] if not l.startswith(("ENV_BIN=", "export HF_HOME="))]
    body_n = [l for l in new[j + 1:] if not l.startswith(("ENV_BIN=", "export HF_HOME=", "REPO_ROOT=", "if [ ! -f",
                                                          "  echo \"missing", "fi", "set -a;"))]
    assert body_o == body_n
    text = "\n".join(new)
    assert "/home/" not in text and "MAX_LEN=${MAX_LEN:-262144}" in text and '"max_new_tokens": 16384' in text
