"""Configs and input files of the follow-up runs (FRC-1, SP-1..SP-4, SP-7, TWQ-2), offline.

- every new <ID>.env is valid shell, names only repository files or placeholders, and holds no machine path;
- its RUN_ARGS carry the same flags (and the same values of the budget flags) as the launcher or split runner it names;
- the SP cache keys reproduce from src/split_runner.py with the proxy URL as retrieval;
- the OBLIQ prompt copies give the record prompt_id of the runs that used them, and every copied input file matches its
  MANIFEST.json.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ALG = Path(__file__).resolve().parents[1]
REPO = ALG.parents[1]
CFG = ALG / "configs"
NEW = ["FRC-1", "SP-1", "SP-2", "SP-3", "SP-4", "SP-7", "TWQ-2"]
PLACEHOLDERS = dict(FULLDOC_ROOT="/x", RLM_DATA_DIR="/x", AZURE_DEPLOYMENT_GPT5="dep", BASE_URL="u", IDS_FILE="i", DEP="d")


def _env(row: str) -> dict:
    """KEY=VALUE lines of a config, as bash reads them (placeholders for the referenced variables)."""
    script = "set -a; " + " ".join(f"{k}={v}" for k, v in PLACEHOLDERS.items()) + f"; . {shlex.quote(str(CFG / (row + '.env')))}; env -0"
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout
    return dict(x.split("=", 1) for x in out.split("\0") if "=" in x)


def _flags(args: str) -> dict:
    toks = shlex.split(args)
    out = {}
    for i, t in enumerate(toks):
        if t.startswith("--"):
            nxt = toks[i + 1] if i + 1 < len(toks) and not toks[i + 1].startswith("--") else None
            out[t] = nxt
    return out


@pytest.mark.parametrize("row", NEW)
def test_config_is_valid_shell_and_clean(row):
    text = (CFG / f"{row}.env").read_text()
    assert "/home/" not in text and "/tmp/claude" not in text and "ismail" not in text.lower()
    env = _env(row)
    assert env["ROW_ID"] == row
    for k in ("LAUNCHER", "RUN_ARGS", "ORIGINAL_OUTPUT", "ROOT_MODEL", "MAX_SEARCHES", "MAX_ITERATIONS"):
        assert env.get(k), (row, k)


@pytest.mark.parametrize("row", NEW)
def test_named_repository_files_exist(row):
    env = _env(row)
    for k in ("IDS_FILE", "GEPA_PROMPTS", "IDS_FILES", "SPLIT_FILES"):
        for p in re.split(r"[;\s]+", env.get(k, "")):
            if p and not p.startswith(("/x", "i", "<", "(", "$")) and "/" in p:
                cand = [REPO / p, ALG / p]
                assert any(c.exists() for c in cand), (row, k, p)
    for script in re.findall(r"(?:studies|experiments)/[\w./-]+\.sh", env["LAUNCHER"]):
        assert (REPO / script).exists() or (ALG / script).exists(), (row, script)


def _base_args() -> list[str]:
    tree = ast.parse((ALG / "src" / "split_runner.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "BASE_ARGS")
    return [e.value if isinstance(e, ast.Constant) else f"<{ast.unparse(e)}>" for e in node.value.elts]


def _launcher_flags(path: Path) -> dict:
    text = path.read_text()
    body = text[text.index("--design B"):]
    body = body.split("\n'\n")[0].split('> "${f%.json}.log"')[0]
    body = re.sub(r"\\\n", " ", body)
    return _flags(body)


@pytest.mark.parametrize("row", ["SP-1", "SP-2"])
def test_split_runner_rows_have_base_args(row):
    cfg = _flags(_env(row)["RUN_ARGS"])
    base = _flags(" ".join(_base_args()))
    assert set(cfg) == set(base)
    for k in ("--max-iterations", "--max-searches", "--max-timeout", "--lm-timeout", "--reasoning-effort"):
        assert cfg[k] == base[k], k


@pytest.mark.parametrize("row,server", [("FRC-1", "http://127.0.0.1:8041"), ("SP-4", "http://127.0.0.1:8042")])
def test_frames_rows_match_frames_qwen(row, server):
    cfg = _flags(_env(row)["RUN_ARGS"])
    lau = _launcher_flags(ALG / "experiments" / "frames_qwen.sh")
    assert set(cfg) == set(lau) | {"--base-url"}
    assert cfg["--server"] == server and cfg["--max-searches"] == "25" and cfg["--max-iterations"] == "50"
    assert cfg["--max-timeout"] == lau["--max-timeout"] == "5400"


def test_sp3_matches_the_gpt5_launcher():
    cfg = _flags(_env("SP-3")["RUN_ARGS"])
    lau = _launcher_flags(ALG / "experiments" / "bcp_gpt5_colbert.sh")
    assert set(cfg) == set(lau)
    assert cfg["--server"] == "http://127.0.0.1:8772" and cfg["--max-timeout"] == lau["--max-timeout"] == "2700"
    assert _env("SP-3")["EFFORT_SENT"] == "none"          # retry429 hook only (KNOWN_ISSUES K3)


@pytest.mark.parametrize("row,port", [("TWQ-2", "8767"), ("SP-7", "8774")])
def test_obliq_rows_match_their_launcher(row, port):
    cfg = _flags(_env(row)["RUN_ARGS"])
    text = (ALG / "experiments" / "obliq_twitter_qwen36.sh").read_text()
    body = re.sub(r"\\\n", " ", text[text.index('"$PYTHON" -u "$SRC/run_experiment.py"'):])
    lau = _flags(body)
    assert set(cfg) == set(lau)
    assert cfg["--server"] == f"http://127.0.0.1:{port}"
    for k in ("--max-searches", "--max-iterations", "--default-k", "--max-timeout", "--workers"):
        assert cfg[k] == lau[k], k


PROBE = """
import importlib.util
s = importlib.util.spec_from_file_location("e", {path!r}); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
print(m.FINGERPRINT, m.ckey(m.seed_blocks()))
"""


@pytest.mark.parametrize("port,fp,key", [("8768", "9ee941fc", "4512373504"), ("8769", "7c62edcd", "108e3760dc"),
                                         ("8772", "3146c6e8", "08b583387f")])
def test_sp_cache_keys(tmp_path, port, fp, key):
    import os
    env = {k: v for k, v in os.environ.items() if k != "GEPA_PROMPTS"}
    env.update(GEPA_BCP_MODEL="qwen36-35b-a3b", SUP_FENCE_TOLERANT="1", SUP_MAX_BLOCKS="5", SUP_PRESENCE_PENALTY="1.5",
               GEPA_BCP_SERVERS="http://localhost:8030/v1", GEPA_BCP_RETRIEVAL=f"http://127.0.0.1:{port}",
               RUNS_DIR=str(tmp_path / "runs"), FULLDOC_ROOT=str(tmp_path), RLMLIB_PATH=str(tmp_path), PYTHON="python",
               PYTHONPATH=str(REPO / "agents" / "shared"))
    out = subprocess.run([sys.executable, "-c", PROBE.format(path=str(ALG / "src" / "split_runner.py"))],
                         capture_output=True, text=True, env=env, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split()[-2:] == [fp, key]
    cfgs = (CFG / "SP-1.env").read_text() + (CFG / "SP-2.env").read_text()
    assert key in cfgs and fp in cfgs


@pytest.mark.parametrize("fname,pid,rows", [("prompts_math.json", "f92043bfd120", ["MAT-11"]),
                                            ("prompts_twitter.json", "299e86ce3e1e", ["TWI-11"]),
                                            ("prompts_congress.json", "891e1319f80c", ["CON-11"]),
                                            ("prompts_obliq.json", "00a9b02c72d6", ["TWQ-2", "SP-7"])])
def test_obliq_prompt_copies_give_the_record_prompt_id(fname, pid, rows):
    import os
    path = CFG / "obliq" / fname
    env = dict(os.environ, GEPA_PROMPTS=str(path))
    out = subprocess.run([sys.executable, "-c", "import prompts; print(prompts.prologue_id())"], capture_output=True,
                         text=True, env=env, cwd=str(ALG / "src"), timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split()[-1] == pid
    for r in rows:
        text = (CFG / f"{r}.env").read_text()
        assert f'GEPA_PROMPTS="agents/document_repl/configs/obliq/{fname}"' in text and pid in text


@pytest.mark.parametrize("d", ["agents/document_repl/configs/obliq", "agents/document_repl/configs/obliq_ids",
                               "agents/document_repl/configs/frames_ids", "agents/snippet_repl/configs/obliq/twitter/obliq_fd",
                               "studies/search_policy/experiments/inputs"])
def test_copied_files_match_their_manifest(d):
    man = json.loads((REPO / d / "MANIFEST.json").read_text())
    assert man["files"]
    for f in man["files"]:
        b = (REPO / d / f["file"]).read_bytes()
        assert hashlib.sha256(b).hexdigest() == f["sha256"] and len(b) == f["bytes"], f["file"]


def test_id_files_hold_ids_only():
    for p in [REPO / "agents/document_repl/configs/obliq_ids/ids281.json",
              *(REPO / "agents/snippet_repl/configs/obliq/twitter/obliq_fd").glob("rest181_g*.json")]:
        ids = json.loads(p.read_text())
        assert all(isinstance(x, str) and re.fullmatch(r"q\d{4}", x) for x in ids), p   # OBLIQ query ids "q0000"
    for p in [REPO / "studies/search_policy/experiments/inputs/pilot100_ids.txt",
              *(ALG / "configs" / "frames_ids").glob("*.txt")]:
        assert all(t.isdigit() for t in p.read_text().split()), p


def test_launchers_read_the_repository_copies():
    assert "$ALG/configs/obliq_ids/ids281.json" in (ALG / "experiments/obliq_twitter_qwen36.sh").read_text()
    assert "GEPA_PROMPTS=$ALG/configs/obliq/prompts_obliq.json" in (ALG / "experiments/obliq_twitter_qwen36.sh").read_text()
    assert "GEPA_PROMPTS=$ALG/configs/obliq/prompts_$TASK.json" in (ALG / "experiments/obliq_fd.sh").read_text()
    sp = REPO / "studies/search_policy/experiments"
    assert "$STUDY/experiments/inputs/pilot100_ids.txt" in (sp / "sp3_bcp_gpt5_rerank_pilot.sh").read_text()
    assert "FR_IDS=$REPO/agents/document_repl/configs/frames_ids" in (sp / "sp4_frames_qwen36_cap25_rerank.sh").read_text()
