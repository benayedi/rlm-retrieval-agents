"""Shared constants of the prompt-optimisation tests (was in conftest.py; a module of its own so it does not
shadow other folders' conftest.py when the whole suite runs).

Test setup for the prompt-optimisation study: paths, the interpreter for subprocess tests, a synthetic environment.

No test starts a model, a server, the `claude` CLI or touches the network. Tests that need records of the original runs
(candidate prompts) read them from LOGS_ROOT and are skipped when it is not set.
"""
import os
import sys
from pathlib import Path

import pytest

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
for p in (REPO / "agents" / "shared", STUDY / "src", STUDY / "obliq_twitter" / "src", STUDY / "hover" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

ORIG = STUDY / "original" / "RLM"
# original file -> refactored copy (studies/prompt_optimisation/CHANGES_FROM_ORIGINAL.md, rename table)
PY_MAP = {
    "scripts/gepa_frames.py": "src/gepa_frames.py",
    "scripts/gepa_frames_confirm.py": "src/gepa_frames_confirm.py",
    "scripts/bcp_confirm.py": "src/bcp_confirm.py",
    "scripts/bcp_gepa_split.py": "src/make_bcp_splits.py",
    "scripts/bcp_enrich_split.py": "src/make_bcp_enriched_split.py",
    "scripts/validate_bcp.py": "src/validate_bcp.py",
    "scripts/bcp_trace_health.py": "src/bcp_trace_health.py",
    "scripts/bcp_search_ndcg.py": "src/bcp_search_ndcg.py",
    "scripts/bcp_recover_misfiled.py": "src/bcp_recover_misfiled.py",
    "scripts/gepa_official_obliq.py": "obliq_twitter/src/gepa_official_obliq.py",
    "scripts/gepa_official_obliq_v2.py": "obliq_twitter/src/gepa_official_obliq_v2.py",
    "scripts/gepa_night.py": "obliq_twitter/src/gepa_night.py",
    "scripts/gepa_paper.py": "obliq_twitter/src/gepa_paper.py",
    "scripts/gepa_wide.py": "obliq_twitter/src/gepa_wide.py",
    "scripts/validate_wide.py": "obliq_twitter/src/validate_wide.py",
    "scripts/trace_check.py": "obliq_twitter/src/trace_check.py",
    "scripts/hover_retriever.py": "hover/src/hover_retriever.py",
    "scripts/hover_build_index.py": "hover/src/build_index.py",
    "scripts/hover_make_splits.py": "hover/src/make_splits.py",
    "scripts/hover_baseline.py": "hover/src/baseline.py",
    "scripts/hover_gepa.py": "hover/src/hover_gepa.py",
}
VENV_PY = sys.executable
