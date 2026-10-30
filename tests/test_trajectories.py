"""Trajectory archives: manifests are consistent, and small archives round-trip byte-exactly through
scripts/provenance/unpack_trajectories.py (decrypt, decompress, hash-check). Skips when the LFS content is not pulled."""
import hashlib, json, subprocess, sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = sorted((ROOT / "trajectories").glob("*/*/MANIFEST.json"))


def is_pointer(p):
    return p.stat().st_size < 1000 and p.read_bytes().startswith(b"version https://git-lfs")


@pytest.mark.parametrize("m", MANIFESTS, ids=lambda p: p.parent.name)
def test_manifest_consistent(m):
    x = json.loads(m.read_text())
    assert x["row"] == m.parent.name and x["members"] == len(x["files"]) > 0
    assert x["encrypted"] == x["row"].startswith("BCP")            # every BrowseComp-Plus archive is encrypted
    assert len({f["path"] for f in x["files"]}) == len(x["files"])   # no duplicate paths
    assert x["member_bytes"] == sum(f["bytes"] for f in x["files"])
    for part in x["archive_parts"]:
        f = m.parent / part["file"]
        assert f.exists(), f
        assert part["bytes"] < 2_000_000_000                         # GitHub LFS file-size limit
        if not is_pointer(f):
            assert f.stat().st_size == part["bytes"]


@pytest.mark.parametrize("row", ["BCP-25", "FR-11"])                # small rows: one encrypted, one plain
def test_small_archive_round_trip(row, tmp_path):
    m = next((p for p in MANIFESTS if p.parent.name == row), None)
    if m is None:
        pytest.skip(f"no archive for {row}")
    x = json.loads(m.read_text())
    if any(is_pointer(m.parent / p["file"]) for p in x["archive_parts"]):
        pytest.skip("LFS content not pulled")
    r = subprocess.run([sys.executable, str(ROOT / "scripts/provenance/unpack_trajectories.py"), row, str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for f in x["files"]:
        assert hashlib.sha256((tmp_path / f["path"]).read_bytes()).hexdigest() == f["sha256"]
