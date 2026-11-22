"""DSA-3 / DSA-5 configs: valid shell, and every setting equal to the arm script it documents (original/RLM_2026-11-11)."""
from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

import pytest

ALGO = Path(__file__).resolve().parents[1]
ARMS = ALGO / "original" / "RLM_2026-11-11" / "RLM" / "scripts" / "run5d"


def _env(arm: str) -> dict:
    script = f"set -a; AZURE_DEPLOYMENT_GPT5=DEP; . {shlex.quote(str(ALGO / 'configs' / f'DSA-{arm}.env'))}; env -0"
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout
    return dict(x.split("=", 1) for x in out.split("\0") if "=" in x)


@pytest.mark.parametrize("arm", ["3", "5"])
def test_settings_equal_the_arm_script(arm):
    e = _env(arm)
    assert e["ROW_ID"] == f"DSA-{arm}" and e["CONTROL"] == "BCP-9"
    orig = (ARMS / f"dspy_ablation_arm{arm}.sh").read_text()
    exported = dict(re.findall(r"\b([A-Z_][A-Z0-9_]*)=([^\s;]+)", " ".join(l for l in orig.splitlines() if l.startswith("export "))))
    for k, v in exported.items():
        if k in ("PATH", "PYTHONUNBUFFERED", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"):
            continue
        want = "DEP" if v == "ismail-gpt-5" else v
        assert e.get(k) == want, (k, e.get(k), want)
    unset = re.search(r"^unset (.+)$", orig, re.M).group(1).split()
    assert e["ENV_UNSET"].split() == unset
    m = re.search(r"--sig v21 --top-k (\d+) --max-iters 100 --max-search 1000000", orig)
    assert f"--top-k {m.group(1)} --max-iters 100 --max-search 1000000 --timeout 3600" in e["RUN_ARGS"]
    assert (ALGO / e["LAUNCHER"]).exists() and (ALGO.parents[1] / e["SAMPLE"]).exists()
    text = (ALGO / "configs" / f"DSA-{arm}.env").read_text()
    assert "/home/" not in text and "/tmp/claude" not in text and "ismail" not in text.lower()
