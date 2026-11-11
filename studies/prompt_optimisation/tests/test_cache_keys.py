"""Seeds and cache keys of the GEPA runs are reproduced by the refactored code (they hash prompt text and the harness
fingerprint, never a path). Keys from the original logs:
  GEPA-1/2 seed ee5ea32a1c, GEPA-2 winner d911ebbe23 (logs/gepa_bcp2/confirm_test200.log)
  GEPA-3 seed 1b5846380c, candidate 8ba345b481 = best 598a758e3a minus the score phrase (logs/gepa_frames_confirm/run.log)
  GEPA-4 seed STRATEGY 970 chars sha fef4457968; run 4 seed (STRATEGY + TASK_SUFFIX) de68174356 (logs/gepa_*/progress.txt)
Each runs in a subprocess, because importing the drivers sets process environment variables."""
import hashlib
import json
import os
import re
import subprocess
import textwrap

import pytest

from .po_testlib import REPO, STUDY, VENV_PY

RUN_OBLIQ = REPO / "agents" / "snippet_repl" / "src" / "run_obliq.py"


def _seed_strategy(src: str) -> str:   # the parsing every GEPA-4 driver does, verbatim
    seed_text = src.split("OBLIQ_STRATEGY = (", 1)[1].split("\n)", 1)[0]
    return "".join(l.strip().strip('"').replace('\\"', '"') for l in seed_text.splitlines())


def test_obliq_seed_strategy_parses_from_the_agent_source():
    s = _seed_strategy(RUN_OBLIQ.read_text())
    assert len(s) == 970 and hashlib.sha1(s.encode()).hexdigest()[:10] == "fef4457968"


def test_obliq_run4_seed_key():
    """gepa_wide.py keys the seed on STRATEGY + the A_control.txt answer contract (read from ORIGINAL_REPO_ROOT);
    that file is byte-identical to the agent's own _TASK literal, so the key is reproducible from this repository."""
    src = RUN_OBLIQ.read_text()
    task = re.search(r'^_TASK = """(.*?)"""', src, re.S | re.M).group(1)
    assert hashlib.sha1((_seed_strategy(src) + task).encode()).hexdigest()[:10] == "de68174356"


def _run(code, env):
    full = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/tmp"), **env}
    p = subprocess.run([VENV_PY, "-c", textwrap.dedent(code)], cwd=REPO, env=full, capture_output=True, text=True,
                       timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    return p.stdout


def _rlm_available():
    return bool(os.environ.get("RLMLIB_PATH")) and os.path.isdir(os.environ["RLMLIB_PATH"])


@pytest.mark.skipif(not _rlm_available(), reason="needs the rlm library (RLMLIB_PATH)")
def test_frames_seed_key(synth_env, tmp_path):
    (tmp_path / "frames_data_dir" / "frames_rows.json").write_text(
        json.dumps([{"i": 0, "question": "q", "answer": "a", "gold_ids": [1]}]))
    out = _run("""
        import sys; sys.argv = ["x"]; sys.path.insert(0, "studies/prompt_optimisation/src")
        import gepa_frames as gf
        print(gf.G.ckey(gf.seed_blocks()))
    """, {**synth_env, "PYTHONPATH": synth_env["RLMLIB_PATH"]})
    assert out.split()[-1] == "1b5846380c"


@pytest.mark.skipif(not _rlm_available(), reason="needs the rlm library (RLMLIB_PATH)")
def test_bcp_seed_key_under_run_env(synth_env):
    out = _run("""
        import sys; sys.argv = ["x"]; sys.path.insert(0, "agents/document_repl/src")
        import split_runner as G
        print(G.ckey(G.seed_blocks()))
    """, {**synth_env, "GEPA_BCP_MODEL": "qwen36-35b-a3b", "SUP_FENCE_TOLERANT": "1", "SUP_MAX_BLOCKS": "5",
          "SUP_PRESENCE_PENALTY": "1.5"})
    assert out.split()[-1] == "ee5ea32a1c"


LOGS = os.environ.get("LOGS_ROOT", "")


@pytest.mark.skipif(not (LOGS and os.path.exists(f"{LOGS}/gepa_bcp2/winner.json")),
                    reason="needs the original run records (LOGS_ROOT)")
def test_bcp_winner_key(synth_env):
    out = _run(f"""
        import sys, json; sys.argv = ["x"]; sys.path.insert(0, "agents/document_repl/src")
        import split_runner as G
        w, _ = G.repair(json.load(open("{LOGS}/gepa_bcp2/winner.json"))); print(G.ckey(w))
    """, {**synth_env, "GEPA_BCP_MODEL": "qwen36-35b-a3b", "SUP_FENCE_TOLERANT": "1", "SUP_MAX_BLOCKS": "5",
          "SUP_PRESENCE_PENALTY": "1.5"})
    assert out.split()[-1] == "d911ebbe23"


@pytest.mark.skipif(not (_rlm_available() and LOGS and os.path.exists(f"{LOGS}/gepa_frames/best_candidate.json")),
                    reason="needs the rlm library and the original run records (LOGS_ROOT)")
def test_frames_candidate_keys(synth_env, tmp_path):
    (tmp_path / "frames_data_dir" / "frames_rows.json").write_text("[]")
    os.makedirs(f"{synth_env['RUNS_DIR']}/gepa_frames")
    best = json.load(open(f"{LOGS}/gepa_frames/best_candidate.json"))
    json.dump(best, open(f"{synth_env['RUNS_DIR']}/gepa_frames/best_candidate.json", "w"))
    out = _run("""
        import sys, json, importlib.util as u; sys.argv = ["x"]
        s = u.spec_from_file_location("gfc", "studies/prompt_optimisation/src/gepa_frames_confirm.py")
        m = u.module_from_spec(s); s.loader.exec_module(m)
        print(m.G.ckey(json.load(open(m.RUNS + "/gepa_frames/best_candidate.json"))), m.G.ckey(m.candidate()))
    """, {**synth_env, "PYTHONPATH": synth_env["RLMLIB_PATH"]})
    assert out.split()[-2:] == ["598a758e3a", "8ba345b481"]


def test_frames_seed_anchor_present_in_the_bm25_wrapper():
    """gepa_frames.seed_blocks() execs the source text of the rollout entry between two anchors."""
    src = (REPO / "agents" / "document_repl" / "src" / "run_bm25.py").read_text()
    orig = (REPO / "agents" / "document_repl" / "original" / "RLM" / "scripts" / "frames_sup_run_one.py").read_text()
    cut = lambda s: s[s.index("PROLOGUE_OLD = "): s.index("# USER_PROLOGUE is a GEPA block")]
    a, b = {}, {}
    exec(cut(src), a); exec(cut(orig), b)
    assert (a["PROLOGUE_OLD"], a["PROLOGUE_NEW"]) == (b["PROLOGUE_OLD"], b["PROLOGUE_NEW"])
