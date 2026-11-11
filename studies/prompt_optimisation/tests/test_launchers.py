"""Every launcher passes the same flags and sets the same behaviour variables as the original wrapper or chain lines
it comes from (REFACTOR_RULES rule 5). Machine-location variables (PYV, PY, L, HF_HOME, PYTHONPATH, OUT, ...) differ
by design and are excluded."""
import re

import pytest

from .po_testlib import ORIG, STUDY

# launcher -> (original file, (first line, last line) or None for the whole file)
PAIRS = {
    "experiments/gepa1_bcp_run1.sh": ("scripts/run_gepa_bcp.sh", None),
    "experiments/gepa2_bcp_run2.sh": ("scripts/run_gepa_bcp2.sh", None),
    "experiments/gepa2_bcp_run2_continue.sh": ("logs/gepa_bcp2/continue_after_fix.sh", None),
    "experiments/gepa3_frames.sh": ("scripts/frames_gepa_chain.sh", None),
    "experiments/gepa3_frames_1gpu.sh": ("scripts/frames_gepa_chain_1gpu.sh", None),
    "obliq_twitter/experiments/run3a_gepa_night.sh": ("scripts/night_chain.sh", (79, 80)),
    "obliq_twitter/experiments/run3_gepa_paper.sh": ("scripts/overnight2.sh", (34, 38)),
    "obliq_twitter/experiments/run4_gepa_wide.sh": ("scripts/overnight3.sh", (27, 30)),
    "obliq_twitter/experiments/run4_validate.sh": ("scripts/night8_gepa27_finish.sh", (61, 61)),
    "hover/experiments/run_hover_gepa.sh": ("scripts/run_hover_gepa.sh", None),
}
LOCATION = {"PYV", "PY", "L", "HF_HOME", "PYTHONPATH", "OUT", "HOVER_OUT", "GEPA_OUT", "GEPA_SPLIT", "SERVE",
            "CUDA_VISIBLE_DEVICES", "GPU", "PORT", "UTIL", "M", "BUDGET", "WORKERS", "BEST_W", "others", "rc"}


def _text(path, rng):
    lines = path.read_text().splitlines()
    if rng:
        lines = lines[rng[0] - 1: rng[1]]
    return "\n".join(l for l in lines if not l.lstrip().startswith("#"))


def _flags(t):
    return set(re.findall(r"(?<![\w-])--[a-z][a-z0-9-]*", t))


def _env(t):
    return set(re.findall(r"\b([A-Z][A-Z0-9_]*)=", t)) - LOCATION


def _values(t):   # NAME=value pairs whose value is a literal, e.g. SUP_MAX_BLOCKS=5
    return {m for m in re.findall(r"\b([A-Z][A-Z0-9_]*=[A-Za-z0-9.:/{}\"'_-]+)", t)
            if m.split("=")[0] not in LOCATION and "/home/" not in m and "logs/" not in m and "scripts/" not in m
            and "data/" not in m}


@pytest.mark.parametrize("new,orig", sorted((k, v) for k, v in PAIRS.items()))
def test_same_flags_and_env(new, orig):
    o = _text(ORIG / orig[0], orig[1])
    n = _text(STUDY / new, None)
    assert _flags(o) <= _flags(n), _flags(o) - _flags(n)
    assert _env(o) <= _env(n), _env(o) - _env(n)
    assert _values(o) <= _values(n), _values(o) - _values(n)
