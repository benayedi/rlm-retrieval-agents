"""Helpers and fixtures for the snippet REPL agent tests (mocks only: no network, no rlm server, no LM call).

The `rlm` library (github.com/alexzhang13/rlm @ 854e688) is located through RLMLIB_PATH, or, if that is unset, through
the interpreter's own sys.path; tests that need it are skipped with a reason when it cannot be found.
"""
from __future__ import annotations

import functools
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ALG = Path(__file__).resolve().parents[1]
ART = ALG.parents[1]
SRC = ALG / "src"
ORIG = ALG / "original" / "RLM"
PROBE = Path(__file__).with_name("_probe.py")


def rlmlib_dir() -> str | None:
    p = os.environ.get("RLMLIB_PATH")
    if p and (Path(p) / "rlm" / "__init__.py").exists():
        return p
    spec = importlib.util.find_spec("rlm")
    if spec and spec.origin:
        return str(Path(spec.origin).resolve().parents[1])
    return None


needs_rlm = pytest.mark.skipif(rlmlib_dir() is None,
                               reason="`rlm` is not importable (set RLMLIB_PATH to the rlm checkout at 854e688)")


def hermetic_env(extra: dict[str, str], workdir: Path) -> dict[str, str]:
    """A minimal environment: none of the caller's behaviour-selecting variables leak into a probe."""
    env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "TMPDIR") if k in os.environ}
    env.update(PYTHONPATH=rlmlib_dir() or "", PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               AZURE_OPENAI_API_KEY="probe-key", AZURE_OPENAI_ENDPOINT="http://azure.invalid/",
               AZURE_OPENAI_MODEL="probe-deployment", RLM_TRAJ_DIR=str(workdir / "traj"),
               OBLIQ_BENCH_ROOT=str(workdir / "obliq"))
    env.update(extra)
    return env


def _fake_obliq_root(root: Path) -> None:
    for task, subset, ids in (("descriptive", "twitter", ("111111111111111111", "222222222222222222")),
                              ("analogues", "math", ("p1___a", "p3___c"))):
        d = root / task / subset / "queries+qrels"
        d.mkdir(parents=True, exist_ok=True)
        (d / "qrels.tsv").write_text("query-id\tcorpus-id\tscore\n" + "".join(f"q1\t{i}\t1\n" for i in ids))


@functools.lru_cache(maxsize=None)
def _probe_cached(side: str, entry: str, env_items: tuple, workdir: str) -> dict:
    wd = Path(workdir)
    _fake_obliq_root(wd / "obliq")
    p = subprocess.run([sys.executable, str(PROBE), side, entry, workdir], env=hermetic_env(dict(env_items), wd),
                       capture_output=True, text=True, timeout=300, cwd=workdir)
    if p.returncode != 0 or "PROBE_JSON:" not in p.stdout:
        raise AssertionError(f"probe {side}/{entry} failed (exit {p.returncode}):\n{p.stdout[-3000:]}\n{p.stderr[-5000:]}")
    return json.loads(p.stdout.rsplit("PROBE_JSON:", 1)[1])


@pytest.fixture(scope="session")
def probe(tmp_path_factory):
    wd = tmp_path_factory.mktemp("probe")

    def run(side: str, entry: str, env: dict[str, str]) -> dict:
        return _probe_cached(side, entry, tuple(sorted(env.items())), str(wd))

    return run


# ── scenarios: (entry point, environment) pairs shared by the equivalence and behaviour tests ──
LIGHTON = {"LIGHTON_TOOL_DOCS": "1", "LIGHTON_TASK_TEXT": "1", "ORCHESTRATOR": "0", "OFFICIAL_TOP_K": "5"}
QWEN = {"LOCAL_LLM_BASE_URL": "http://127.0.0.1:9/v1", "LOCAL_LLM_MODEL": "qwen36-35b-a3b", "LOCAL_LLM_API_KEY": "EMPTY",
        "LOCAL_LM_TIMEOUT": "1800", "AZURE_OPENAI_REASONING_EFFORT": "medium", "SUB_LM_REASONING_EFFORT": "medium"}
MATH = ALG / "configs" / "obliq" / "math"

SCENARIOS = {
    "harness-defaults": ("harness", {}),
    "harness-run5d-switches": ("harness", dict(LIGHTON)),
    "harness-other-switches": ("harness", {"RETRIEVER_LINE": "0", "NO_UNKNOWN": "1", "DEPTH2_PROMPT": "1",
                                           "SEARCH_TRIAGE_RULE": "1", "RLM_MAX_OUTPUT_CHARS": "20000",
                                           "OBLIQ_NO_GETDOC": "1", "SEARCH_SNIPPET_LEN": "300",
                                           "SUB_LM_MODEL": "probe-sub", "MAX_CONCURRENT_SUBCALLS": "3"}),
    "bcp-gpt5 (BCP-13)": ("bcp", {**LIGHTON, "AZURE_OPENAI_REASONING_EFFORT": "high",
                                  "SUB_LM_REASONING_EFFORT": "medium"}),
    "bcp-qwen (BCP-15/17/19)": ("bcp", {**LIGHTON, **QWEN, "QWEN_PRESENCE_PENALTY": "1.5", "QWEN_FENCE_TOLERANT": "1",
                                        "QWEN_REPL_CUE": "1", "QWEN_PROSE_FINAL": "1", "QWEN_MAX_TOKENS": "8192",
                                        "BCP_SEARCH_CAP": "25"}),
    "frames-gpt5 (FR-7)": ("frames", {**LIGHTON, "RETRIEVER_LINE": "0", "FRAMES_BM25_LINE": "1",
                                      "FRAMES_SEARCH_CAP": "25", "AZURE_OPENAI_REASONING_EFFORT": "high",
                                      "SUB_LM_REASONING_EFFORT": "medium"}),
    "frames-qwen (FR-9)": ("frames", {**LIGHTON, **QWEN, "RETRIEVER_LINE": "0", "FRAMES_BM25_LINE": "1",
                                      "FRAMES_SEARCH_CAP": "25", "FRAMES_REPL_CUE": "1",
                                      "FRAMES_FENCE_TOLERANT": "1"}),
    "obliq-twitter (TWI-9)": ("obliq", {**LIGHTON, "OFFICIAL_TOP_K": "20", "LOCAL_LLM_BASE_URL": "",
                                        "OBLIQ_TASK": "descriptive", "OBLIQ_SUBSET": "twitter",
                                        "OBLIQ_RELATIVE_SCORES": "1", "OBLIQ_REPL_CUE": "1", "OBLIQ_TAG_SEARCH": "1",
                                        "AZURE_OPENAI_REASONING_EFFORT": "high", "SUB_LM_REASONING_EFFORT": "medium",
                                        "PROBE_SEARCH_IDS": "111111111111111111,333333333333333333",
                                        "PROBE_RESPONSE": "111111111111111111\n222222222222222222"}),
    "obliq-math (MAT-9 files + other switches)": ("obliq", {
        **LIGHTON, "OFFICIAL_TOP_K": "20", "LOCAL_LLM_BASE_URL": "", "OBLIQ_TASK": "analogues", "OBLIQ_SUBSET": "math",
        "OBLIQ_TASK_FILE": str(MATH / "task_snippet.txt"), "OBLIQ_STRATEGY_FILE": str(MATH / "strategy_snippet.txt"),
        "OBLIQ_RELATIVE_SCORES": "1", "OBLIQ_REPL_CUE": "1", "GPTOSS_FINAL_CUE": "1", "OBLIQ_NO_FORMAT_GATE": "1",
        "OBLIQ_FENCE_TOLERANT": "1", "OBLIQ_PRESENCE_PENALTY": "1.5", "OBLIQ_QRELS_POOL": "1",
        "PROBE_SEARCH_IDS": "p1___a,p2___b", "PROBE_RESPONSE": "p3___c\np2___b\np1___a"}),
}
