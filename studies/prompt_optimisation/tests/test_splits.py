"""The committed split files: byte-exact (manifest), ids only, and the properties each run relied on; the split
builders on synthetic data (they are never run against the committed files, which they would rewrite)."""
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys

import pytest

from .po_testlib import REPO, STUDY, VENV_PY

SPL = STUDY / "splits"


def test_manifest_matches():
    man = json.load(open(SPL / "MANIFEST.json"))
    for f in man["files"]:
        assert hashlib.sha256((SPL / f["copy"]).read_bytes()).hexdigest() == f["sha256"], f["copy"]


def _ids(v):
    return all(isinstance(x, str) and len(x) <= 6 for x in v)


def test_bcp_split_disjoint_and_ids_only():
    s = json.load(open(SPL / "bcp_gepa_splits" / "split.json"))
    parts = [set(s[k]) for k in ("pareto", "feedback", "test200", "test_rest")]
    assert [len(p) for p in parts] == [100, 200, 200, 330] and len(set().union(*parts)) == 830
    assert all(_ids(s[k]) for k in ("pareto", "feedback", "test200", "test_rest"))
    rest = json.load(open(SPL / "bcp_full830" / "split_rest300.json"))
    assert set(rest["rest300"]) == set(s["pareto"]) | set(s["feedback"])


def test_enriched_split():
    s = json.load(open(SPL / "bcp_gepa_splits" / "split.json"))
    e = json.load(open(SPL / "bcp_gepa_splits" / "split_enriched_q36.json"))
    assert len(e["strata"]["fail"]) == 70 and len(e["strata"]["guard"]) == 30
    assert sorted(e["pareto"], key=int) == sorted(e["strata"]["fail"] + e["strata"]["guard"], key=int)
    assert set(e["pareto"]) | set(e["feedback"]) == set(s["pareto"]) | set(s["feedback"])
    assert e["test200"] == s["test200"] and e["test_rest"] == s["test_rest"]


def test_frames_split_never_tests_q0_99():
    s = json.load(open(SPL / "frames" / "gepa_splits" / "split.json"))
    assert not set(s["test"]) & {str(i) for i in range(100)}
    used = set(s["feedback"]) | set(s["pareto"]) | set(s["test"])
    assert 824 - len(used) == 324            # the questions gepa_frames_confirm.py runs (PO-I3)


def test_obliq_splits():
    d = json.load(open(SPL / "obliq_splits" / "twitter_dev.json"))
    h = json.load(open(SPL / "obliq_splits" / "twitter_heldout.json"))
    assert len(d) == 100 and len(h) == 181 and not set(d) & set(h)


def test_split_builder_functions(synth_env):
    import make_bcp_splits as M
    assert sum(M.allocate({"a": 7, "b": 2, "c": 1}, 5).values()) == 5
    wall = {str(i): float(i) for i in range(50)}
    pool = {"x": [str(i) for i in range(30)], "y": [str(i) for i in range(30, 50)]}
    a = M.draw({k: list(v) for k, v in pool.items()}, 10, random.Random(1), wall)
    b = M.draw({k: list(v) for k, v in pool.items()}, 10, random.Random(1), wall)
    assert a == b and len(set(a)) == 10
    assert M.stratum({"status": "completed", "correct": True, "gold_recall": 1.0}) == "ok_rfull"
    assert M.stratum({"status": "timeout"}) == "error"


GRADED = os.path.join(os.environ.get("FULLDOC_ROOT", ""), "runs", "_graded_cache.json")


@pytest.mark.skipif(not os.path.exists(GRADED), reason="needs the 830-question grades (FULLDOC_ROOT/runs)")
def test_bcp_split_reproduces(monkeypatch, tmp_path):
    """main()'s draws, replayed with the original seed on the original grades, give the committed split."""
    import collections
    import make_bcp_splits as M
    G = json.load(open(GRADED))
    wall = {q: (r.get("wall") or 0.0) for q, r in G.items()}
    pool = collections.defaultdict(list)
    for q, r in G.items():
        pool[M.stratum(r)].append(q)
    rng = random.Random(20261006)
    got = [M.draw(pool, n, rng, wall) for n in (100, 200, 200)]
    s = json.load(open(SPL / "bcp_gepa_splits" / "split.json"))
    assert got == [s["pareto"], s["feedback"], s["test200"]]


def test_enrich_on_synthetic_cache(tmp_path):
    """Run make_bcp_enriched_split.py in a copy of the layout: 70 seed failures + 30 full-recall guards."""
    root = tmp_path / "repo"
    (root / "agents").mkdir(parents=True)
    os.symlink(REPO / "agents" / "shared", root / "agents" / "shared")
    src = root / "studies" / "prompt_optimisation" / "src"
    src.mkdir(parents=True)
    shutil.copy(STUDY / "src" / "make_bcp_enriched_split.py", src)
    spl = root / "studies" / "prompt_optimisation" / "splits" / "bcp_gepa_splits"
    spl.mkdir(parents=True)
    pool = [str(i) for i in range(300)]
    json.dump({"pareto": pool[:100], "feedback": pool[100:], "test200": ["900"], "test_rest": ["901"]},
              open(spl / "split.json", "w"))
    runs = tmp_path / "runs"; (runs / "gepa_bcp").mkdir(parents=True)
    cache = {f"ee5ea32a1c|{q}": {"status": "completed", "correct": int(q) % 3 != 0,
                                  "gold_recall": 1.0 if int(q) % 2 else 0.4} for q in pool}
    json.dump(cache, open(runs / "gepa_bcp" / "cache.json", "w"))
    p = subprocess.run([VENV_PY, str(src / "make_bcp_enriched_split.py")], env={**os.environ, "RUNS_DIR": str(runs)},
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    e = json.load(open(spl / "split_enriched_q36.json"))
    assert len(e["strata"]["fail"]) == 70 and len(e["strata"]["guard"]) == 30
    for q in e["strata"]["fail"]:
        r = cache[f"ee5ea32a1c|{q}"]; assert not r["correct"] or r["gold_recall"] < 0.5
    for q in e["strata"]["guard"]:
        r = cache[f"ee5ea32a1c|{q}"]; assert r["correct"] and r["gold_recall"] >= 0.999
    assert e["test200"] == ["900"]
