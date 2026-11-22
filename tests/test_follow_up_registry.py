"""experiments/registry.yaml, section follow_up_studies: every entry names files that exist in the repository
(launcher, config, verdicts, candidates, results, trajectory archive), controls that are registered IDs, and the
copied result files match their manifests. Offline; reads only repository files.
"""
from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
REG = yaml.safe_load((REPO / "experiments" / "registry.yaml").read_text())
FU = REG["follow_up_studies"]
IDS = {e["id"] for e in REG["experiments"]} | {e["id"] for e in FU}


def _paths(value):
    """Repository paths named in a registry field. A comma-separated list may continue with bare file names, which are
    relative to the first path's directory ("a/b/x.sh, y.sh"); anything after the first space of an item is prose."""
    out, base = [], None
    value = str(value).replace("{0,1}", "[01]").replace("{,_rest0,_rest1}", "*")   # brace lists -> globs
    for part in value.split(","):
        tok = part.strip().split(" ")[0]
        if not tok or tok.startswith(("logs/", "<", "docs/", "data/", "trajectories/")):
            continue
        if "/" in tok:
            out.append(tok); base = str(Path(tok).parent)
        elif base and "." in tok:
            out.append(f"{base}/{tok}")
    return [p for p in out if p.startswith(("agents/", "studies/", "results/", "benchmarks/"))]


@pytest.mark.parametrize("e", FU, ids=lambda e: e["id"])
def test_named_files_exist(e):
    for field in ("launcher", "config", "verdicts", "candidates", "results", "driver"):
        for p in _paths(e.get(field, "")):
            assert glob.glob(str(REPO / p)), (e["id"], field, p)


@pytest.mark.parametrize("e", FU, ids=lambda e: e["id"])
def test_controls_are_registered(e):
    c = e.get("control")
    if c and c[:3] in ("BCP", "FR-", "SP-", "TWQ", "FRC"):
        assert c.split(" ")[0] in IDS, (e["id"], c)


@pytest.mark.parametrize("e", [e for e in FU if "trajectories" in e], ids=lambda e: e["id"])
def test_trajectory_archive_present(e):
    d = REPO / e["trajectories"].split(" ")[0]
    if not (d / "MANIFEST.json").exists():
        pytest.skip(f"{e['id']}: archive not packed yet")
    m = json.loads((d / "MANIFEST.json").read_text())
    assert m["row"] == e["id"] and m["members"] > 0
    for part in m["archive_parts"]:
        assert (d / part["file"]).exists()


@pytest.mark.parametrize("dest", ["results/follow_up/verdicts", "studies/error_analysis/results",
                                  "studies/frames_baselines/results", "studies/prompt_optimisation/candidates"])
def test_copies_match_manifest(dest):
    m = json.loads((REPO / dest / "MANIFEST.json").read_text())
    for f in m["files"]:
        assert hashlib.sha256((REPO / dest / f["copy"]).read_bytes()).hexdigest() == f["sha256"], f["copy"]


def test_bcp_verdict_files_hold_only_ids_and_booleans():
    for f in glob.glob(str(REPO / "results/follow_up/verdicts/bcp/*.json")):
        d = json.loads(Path(f).read_text())
        v = d.get("verdicts") if isinstance(d, dict) else None
        if v is not None:
            assert all(isinstance(k, str) and k.isdigit() for k in v), f
            assert all(isinstance(x, bool) for x in v.values()), f
