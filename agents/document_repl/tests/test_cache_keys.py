"""split_runner.py computes the same FINGERPRINT / ckey as the original gepa_bcp.py.

The result caches of the Qwen3.6 BrowseComp+ rows are keyed "<ckey>|<qid>"; the keys and fingerprints below are the
ones in their progress logs. Each case runs in a fresh interpreter with the environment its wrapper set.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from conftest import SRC, subprocess_env

QWEN = {"GEPA_BCP_MODEL": "qwen36-35b-a3b", "SUP_FENCE_TOLERANT": "1", "SUP_MAX_BLOCKS": "5",
        "SUP_PRESENCE_PENALTY": "1.5", "GEPA_BCP_SERVERS": "http://localhost:8030/v1"}

CASES = [
    # row, entry module, extra env, fingerprint, cache key
    ("BCP-27", "bcp_qwen_dense.py", {}, "9e6ce99e", "e925f0f6c7"),
    ("BCP-31", "bcp_bm25_split.py", {}, "aa1d1029", "11d0c5bc6b"),
    ("BCP-29", "split_runner.py", QWEN, "29587b4e", "ee5ea32a1c"),
]

PROBE = """
import importlib.util, sys
s = importlib.util.spec_from_file_location("entry", {path!r}); m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
G = getattr(m, "G", m)
print(G.FINGERPRINT, G.ckey(G.seed_blocks()))
"""


@pytest.mark.parametrize("row,entry,extra,fp,key", CASES, ids=[c[0] for c in CASES])
def test_cache_key(tmp_path, row, entry, extra, fp, key):
    env = subprocess_env(RUNS_DIR=str(tmp_path / "runs"), FULLDOC_ROOT=str(tmp_path), RLMLIB_PATH=str(tmp_path),
                         PYTHON="python", **extra)
    env.pop("GEPA_PROMPTS", None)
    out = subprocess.run([sys.executable, "-c", PROBE.format(path=str(SRC / entry))], capture_output=True,
                         text=True, env=env, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split()[-2:] == [fp, key]
