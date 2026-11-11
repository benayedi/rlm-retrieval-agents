"""TWQ-1 / SP-6 configs: valid shell for experiments/run_config.sh, launcher present, settings equal to the launcher's."""
from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import pytest

ALG = Path(__file__).resolve().parents[1]


def _source(row: str, slice_: str = "dev100") -> dict:
    script = (f"set -u; SLICE={slice_}; RUNS_DIR=/r; ALG={shlex.quote(str(ALG))}; . {shlex.quote(str(ALG / 'configs' / (row + '.env')))}; "
              'echo "ROW=$ROW"; echo "LAUNCHER=$LAUNCHER"; echo "ARGS=${LAUNCHER_ARGS[*]:-}"; env')
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout
    return dict(l.split("=", 1) for l in out.splitlines() if "=" in l)


@pytest.mark.parametrize("row,arm,args", [("TWQ-1", "ctrl", ""), ("SP-6", "rr", "dev100")])
def test_config_sources_and_names_its_launcher(row, arm, args):
    e = _source(row)
    assert e["ROW"] == row and e["ARM"] == arm and e["ARGS"] == args
    assert (ALG / e["LAUNCHER"]).resolve().exists()
    text = (ALG / "configs" / f"{row}.env").read_text()
    assert "/home/" not in text and "/tmp/claude" not in text and "ismail" not in text.lower()


def test_slice_is_required():
    r = subprocess.run(["bash", "-c", f"set -u; unset SLICE; . {shlex.quote(str(ALG / 'configs' / 'TWQ-1.env'))}"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "SLICE" in r.stderr


@pytest.mark.parametrize("row", ["TWQ-1", "SP-6"])
def test_agent_settings_equal_the_launcher(row):
    e = _source(row)
    launcher = (ALG / "experiments" / "obliq_twitter_qwen36.sh").read_text()
    for k in ("OBLIQ_REPL_CUE", "OBLIQ_FENCE_TOLERANT", "OBLIQ_PRESENCE_PENALTY", "OBLIQ_TAG_SEARCH", "LOCAL_LM_TIMEOUT",
              "LOCAL_LLM_MODEL"):
        assert f"{k}={e[k]}" in launcher, k
    shard = (ALG / "experiments" / "obliq_shard.sh").read_text()
    assert "SEARCH_SNIPPET_TOKENS=512 OFFICIAL_TOP_K=$TOPK" in shard and e["OFFICIAL_TOP_K"] == "20"
    assert '20 12 10' in launcher and '20 12 8' in launcher            # top-k 20, 12 turns, 10 / 8 workers
    assert e["EMBEDDING_SERVER_URL"] == ("http://localhost:8010" if row == "TWQ-1" else "http://localhost:8775")
