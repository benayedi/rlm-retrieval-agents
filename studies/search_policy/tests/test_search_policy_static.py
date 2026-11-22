"""Static checks of the search-policy study code: compiles, documents its origin, carries no machine paths, and the
launchers set exactly the variables and arguments of the command lines they reproduce."""
from __future__ import annotations

import difflib
import py_compile
import re
import subprocess

import pytest

from search_policy_testlib import ANALYSIS, EXPERIMENTS, ORIG, REPO, SRC

PAIRS = {  # src copy -> original (studies/search_policy/original/RLM/...)
    "src/rerank_proxy.py": "scripts/bcp_rerank_proxy.py",
    "analysis/policy_replay.py": "scripts/bcp_policy_replay.py",
    "analysis/policy_cache.py": "scripts/bcp_policy_cache.py",
    "analysis/policy_eval.py": "scripts/bcp_policy_eval.py",
    "analysis/novelty_eval.py": "scripts/bcp_novelty_eval.py",
    "analysis/rerank_eval.py": "scripts/bcp_rerank_eval.py",
    "analysis/frames_rerank_eval.py": "scripts/frames_rerank_eval.py",
    "analysis/compare_bcp_split.py": "scripts/bcp_policy_compare.py",
    "analysis/compare_bcp_830.py": "scripts/bcp_full830_compare.py",
    "analysis/compare_frames_test200.py": "scripts/frames_rr_compare.py",
    "analysis/grade_gpt5_pilot.py": "scripts/bcp_pilot_grade.py",
    "analysis/compare_obliq_document_repl.py": "scripts/obliq_rr_compare.py",
    "analysis/compare_obliq_snippet_dev100.py": "scripts/obliq_mine_rr_compare.py",
    "analysis/compare_obliq_snippet_281.py": "scripts/obliq_q36_next_report.py",
}
STUDY = SRC.parent
NEW_AGENT_FILES = [REPO / "agents/snippet_repl/src/obliq_pick_best.py",
                   REPO / "agents/snippet_repl/experiments/obliq_twitter_qwen36.sh",
                   REPO / "agents/document_repl/experiments/obliq_twitter_qwen36.sh",
                   REPO / "agents/document_repl/experiments/bcp_qwen36_split.sh"]
PY = sorted(SRC.glob("*.py")) + sorted(ANALYSIS.glob("*.py")) + [NEW_AGENT_FILES[0]]
SH = sorted(EXPERIMENTS.glob("*.sh")) + NEW_AGENT_FILES[1:]


@pytest.mark.parametrize("p", PY, ids=lambda p: p.name)
def test_compiles_and_names_its_original(p):
    py_compile.compile(str(p), doraise=True)
    head = p.read_text().split('"""')[1]
    assert "From original/" in head


@pytest.mark.parametrize("p", PY + SH, ids=lambda p: p.name)
def test_no_machine_paths(p):
    t = p.read_text()
    assert "/home/" not in t and "/tmp/claude" not in t and "extstu" not in t


@pytest.mark.parametrize("p", SH, ids=lambda p: p.name)
def test_shell_syntax(p):
    assert subprocess.run(["bash", "-n", str(p)]).returncode == 0


def test_every_pair_exists():
    for new, old in PAIRS.items():
        assert (STUDY / new).exists() and (ORIG / old).exists()


# A removed original line may only be: a hard-coded location (an absolute path, or a path built on the original's
# repository-root variable R), an import or sys.path line (import wiring), or the first docstring line (the docstring
# gains a paragraph). Everything else of the original must survive verbatim.
ALLOWED_REMOVED = re.compile(r"/home/extstu_00|\{R\}|^\s*R = |^\s*(import |from |sys\.path)|^\"\"\"")


@pytest.mark.parametrize("new,old", PAIRS.items())
def test_diff_touches_only_paths_imports_docs(new, old):
    a = (ORIG / old).read_text().splitlines()
    b = (STUDY / new).read_text().splitlines()
    removed = [l[1:] for l in difflib.unified_diff(a, b, lineterm="", n=0) if l.startswith("-") and not l.startswith("---")]
    bad = [l for l in removed if not ALLOWED_REMOVED.search(l)]
    assert not bad, bad


def test_pick_best_diff():
    a = (REPO / "agents/snippet_repl/original/RLM_2026-11-11/RLM/scripts/obliq_best_full.py").read_text().splitlines()
    b = NEW_AGENT_FILES[0].read_text().splitlines()
    removed = [l[1:] for l in difflib.unified_diff(a, b, lineterm="", n=0) if l.startswith("-") and not l.startswith("---")]
    assert all(ALLOWED_REMOVED.search(l) for l in removed), removed


def test_reranker_prompt_unchanged():
    for new, old in PAIRS.items():
        o = (ORIG / old).read_text()
        for m in re.finditer(r'^(PREFIX|SUFFIX|INSTR) = .*$', o, re.M):
            assert m.group(0) in (STUDY / new).read_text()


# ---- launchers vs the command lines they reproduce ------------------------------------------------------------
def tokens(text):
    return set(re.findall(r"[A-Z_]+=\S+|--[a-z-]+(?: [^-\s\\][^\s\\]*)?", text))


def test_sp4_matches_original_chain():
    o = (ORIG / "scripts/frames_s25_rr_chain.sh").read_text()
    n = (EXPERIMENTS / "sp4_frames_qwen36_cap25_rerank.sh").read_text()
    for t in ("POLICY=novelty+rerank", "UPSTREAM=http://127.0.0.1:8041", "RERANK_MEM_FRACTION=0.20",
              "RERANK_TOK_BUDGET=8000", "CUDA_VISIBLE_DEVICES=1", "SERVER=http://127.0.0.1:8042", "BM25_PROMPT=1",
              "MAX_SEARCHES=25", "MAX_ITERS=50", "W=8", "--port 8042", "OMP_NUM_THREADS=4", "MKL_NUM_THREADS=4",
              "TOKENIZERS_PARALLELISM=false", "sleep 120", "all824_shuf_gpu0.txt", "all824_shuf_gpu1.txt",
              "sup_q36_s25_rr_g0", "sup_q36_s25_rr_g1", "run 8030", "run 8031", "ARM=rr"):
        assert t in o and t in n, t


# launch_argv of <FULLDOC_ROOT>/runs/obliq_q36_{ctrl,rr}/_provenance.json (2026-10-12), minus --name/--ids-file/--server
OBLIQ_FD_ARGV = ("--workers 10 --stagger 0.3 --design B --backend openai --base-url http://localhost:8030/v1 "
                 "--api-key-env ALIBABA_TOKEN --root-model qwen36-35b-a3b --sub-model qwen36-35b-a3b --reasoning-effort medium "
                 "--sub-reasoning-effort medium --default-k 30 --fixed-k --max-searches 8 --max-iterations 12 --seed-context "
                 "--compaction --lean-prologue --no-delegation --solo-prompt --no-orchestrator --compaction-threshold-pct 0.85 "
                 "--max-concurrent-subcalls 4 --max-timeout 3600 --lm-timeout 1800 --save-trajectory")


def test_obliq_document_launcher_argv():
    t = " ".join((REPO / "agents/document_repl/experiments/obliq_twitter_qwen36.sh").read_text().replace("\\\n", " ").split())
    flat = " ".join(OBLIQ_FD_ARGV.split())
    order = [x for x in flat.split(" --")]
    pos = -1
    for part in order:
        q = part if part.startswith("--") else "--" + part
        i = t.find(q, pos + 1)
        assert i > pos, q
        pos = i
    for env in ("ALIBABA_TOKEN=EMPTY", "SUP_FENCE_TOLERANT=1", "SUP_MAX_BLOCKS=5", "SUP_PRESENCE_PENALTY=1.5",
                "data/obliq_twitter/queries.jsonl", "data/obliq_twitter/prompts_obliq.json"):
        assert env in t


def test_split_launcher_env():
    t = (REPO / "agents/document_repl/experiments/bcp_qwen36_split.sh").read_text()
    body = t.split("set -u", 1)[1]
    for env in ("env -u GEPA_PROMPTS", "GEPA_BCP_MODEL=qwen36-35b-a3b", "SUP_FENCE_TOLERANT=1", "SUP_MAX_BLOCKS=5",
                "SUP_PRESENCE_PENALTY=1.5", "GEPA_BCP_SERVERS=http://localhost:8030/v1", "GEPA_BCP_RETRIEVAL=",
                "GEPA_OUT=", "GEPA_SPLIT=", "--seed-split", "--workers"):
        assert env in body, env


INSTR = ("Given a request for social media posts that implicitly express a described stance or pattern, judge whether "
         "this post matches the request")


def test_obliq_proxy_settings():
    sp6 = (EXPERIMENTS / "sp6_obliq_snippet_qwen36_rerank.sh").read_text()
    sp7 = (EXPERIMENTS / "sp7_obliq_document_qwen36_rerank.sh").read_text()
    for t in (sp6, sp7):
        assert f'INSTR="{INSTR}"' in t
        for e in ("POLICY=novelty+rerank", "DEPTH=100", "MAX_DOC_TOK=512", "RERANK_TOK_BUDGET=8000", "CUDA_VISIBLE_DEVICES=1"):
            assert e in t
    assert "WIRE=mine" in sp6 and "UPSTREAM=http://127.0.0.1:8010" in sp6 and "start_proxy 8775" in sp6
    assert "RERANK_MEM_FRACTION=0.15" in sp6 and "RERANK_MEM_FRACTION=0.20" in sp6
    assert "WIRE" not in sp7.split("set -u", 1)[1] and "UPSTREAM=http://127.0.0.1:8767" in sp7
    assert "RERANK_MEM_FRACTION=0.13" in sp7 and "start_proxy 8774" in sp7


def test_bcp_proxy_settings():
    sp1 = (EXPERIMENTS / "sp1_bcp_qwen36_rerank.sh").read_text().split("set -u", 1)[1]
    sp2 = (EXPERIMENTS / "sp2_bcp_qwen36_rerank_only.sh").read_text().split("set -u", 1)[1]
    sp3 = (EXPERIMENTS / "sp3_bcp_gpt5_rerank_pilot.sh").read_text().split("set -u", 1)[1]
    assert "POLICY=novelty+rerank DEPTH=30 RERANK_BATCH=6 RERANK_MEM_FRACTION=0.15 RERANK_XBATCH=0" in sp1
    assert "start_proxy 8768" in sp1 and "start_proxy 8772" in sp1 and "RERANK_TOK_BUDGET=8000" in sp1
    assert "OUT=bcp_policy_e2e SPLIT=test200 W=20" in sp1 and "OUT=bcp_rest300_rr SPLIT=rest300 W=24" in sp1
    assert "POLICY=rerank DEPTH=30 RERANK_BATCH=6 RERANK_MEM_FRACTION=0.15" in sp2 and "start_proxy 8769" in sp2
    assert "RERANK_MEM_FRACTION=0.5 RERANK_TOK_BUDGET=8000" in sp3 and "SERVER=http://127.0.0.1:8772" in sp3


def test_reconstructed_launchers_say_so():
    for p in ("sp1_bcp_qwen36_rerank.sh", "sp2_bcp_qwen36_rerank_only.sh", "sp3_bcp_gpt5_rerank_pilot.sh",
              "sp6_obliq_snippet_qwen36_rerank.sh", "sp7_obliq_document_qwen36_rerank.sh"):
        assert "RECONSTRUCTED" in (EXPERIMENTS / p).read_text()
    for p in NEW_AGENT_FILES[1:]:
        assert "RECONSTRUCTED" in p.read_text()
