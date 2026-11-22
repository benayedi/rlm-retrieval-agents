"""Analysis scripts: the original (its hard-coded locations pointed into a temporary tree) and the src copy (the same
tree through configs/paths.env variables) are run on the same synthetic inputs, and their output must be identical.
Synthetic inputs only: question ids and booleans, no benchmark text. No network."""
from __future__ import annotations

import io
import json
import shutil
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from search_policy_testlib import ANALYSIS, ORIG, REPO, write_json

SPLITS = REPO / "studies" / "prompt_optimisation" / "splits"


def run_text(text: str, file: str, argv=None) -> str:
    buf, old = io.StringIO(), sys.argv
    sys.argv = [file] + list(argv or [])
    try:
        with redirect_stdout(buf):
            exec(compile(text, file, "exec"), {"__name__": "__main__", "__file__": file})
    finally:
        sys.argv = old
    return buf.getvalue()


def run_original(name: str, tree: Path, argv=None) -> str:
    text = (ORIG / "scripts" / name).read_text()
    text = text.replace("/home/extstu_00/RLM", str(tree / "R")).replace("/home/extstu_00/suprlm", str(tree / "S"))
    return run_text(text, str(tree / "orig_" / name), argv)


def run_src(name: str, argv=None) -> str:
    p = ANALYSIS / name
    return run_text(p.read_text(), str(p), argv)


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """<tmp>/R = original repository root (logs/, data/), <tmp>/S = document REPL agent root; env for the src copies."""
    (tmp_path / "R" / "logs").mkdir(parents=True)
    (tmp_path / "R" / "data").mkdir(parents=True)
    (tmp_path / "S" / "runs").mkdir(parents=True)
    monkeypatch.setenv("LOGS_ROOT", str(tmp_path / "R" / "logs"))
    monkeypatch.setenv("RLM_DATA_DIR", str(tmp_path / "R" / "data"))
    monkeypatch.setenv("FULLDOC_ROOT", str(tmp_path / "S"))
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.delenv("GEPA_SPLIT", raising=False)
    return tmp_path


def bcp_split(tree: Path):
    """A synthetic 4-split file, at the original location and passed to the src copy through GEPA_SPLIT."""
    sp = {"pareto": ["1", "2"], "feedback": ["3", "4"], "test200": ["5", "6", "7"], "test_rest": ["8", "9"]}
    f = write_json(tree / "R" / "data" / "bcp_gepa_splits" / "split.json", sp)
    return sp, f


def rec(correct, recall=0.5, status="completed", searches=4):
    return {"correct": correct, "gold_recall": recall, "status": status, "searches": searches}


def test_compare_bcp_split_equivalent(tree, monkeypatch):
    sp, f = bcp_split(tree)
    monkeypatch.setenv("GEPA_SPLIT", str(f))
    L = tree / "R" / "logs"
    write_json(L / "gepa_bcp2" / "cache.json", {"ee5ea32a1c|5": rec(False, 0.2), "ee5ea32a1c|6": rec(True, 0.9),
                                                 "ee5ea32a1c|7": rec(False, 0.0, "killed")})
    write_json(L / "bcp_policy_e2e" / "cache.json", {"4512373504|5": rec(True, 0.6), "4512373504|6": rec(True, 1.0),
                                                      "4512373504|7": rec(True, 0.4, "timeout")})
    a = run_original("bcp_policy_compare.py", tree)
    b = run_src("compare_bcp_split.py")
    assert a == b and "paired n=2 of 3" in a and "won/lost 1/0" in a


def test_compare_bcp_830_equivalent(tree, monkeypatch):
    sp, f = bcp_split(tree)
    monkeypatch.setenv("GEPA_SPLIT", str(f))
    L = tree / "R" / "logs"
    ctrl = {f"ee5ea32a1c|{q}": rec(int(q) % 2 == 0, int(q) / 10) for q in ("1", "2", "3", "4", "5", "6", "7")}
    write_json(L / "gepa_bcp2" / "cache.json", ctrl)
    write_json(L / "bcp_testrest_ctrl" / "cache.json", {"k1|8": rec(True), "k1|9": rec(False)})
    write_json(L / "bcp_rest300_rr" / "cache.json", {"k2|1": rec(True, 0.9), "k2|2": rec(True, 0.1), "k2|3": rec(False, 0.35)})
    write_json(L / "bcp_policy_e2e" / "cache.json", {"k3|5": rec(True), "k3|6": rec(False), "k3|7": rec(True)})
    write_json(L / "bcp_testrest_rr" / "cache.json", {"k4|8": rec(True), "k4|9": rec(True)})
    a = run_original("bcp_full830_compare.py", tree)
    b = run_src("compare_bcp_830.py")
    assert a == b and "paired n=8" in a


def test_novelty_eval_equivalent(tree):
    S, L, D = tree / "S" / "runs", tree / "R" / "logs", tree / "R" / "data" / "bcp_policy"
    write_json(S / "qwen27b_colbert8192_first20" / "1.json",
               {"query_id": "1", "search_log": [{"query": "s0", "docids": ["a", "b"]}, {"query": "s1", "docids": ["b", "c"]}]})
    write_json(L / "gepa_bcp" / "rollouts" / "k" / "2.json",
               {"query_id": "2", "search_log": [{"query": "t0", "docids": ["x"]}, {"query": "t1", "docids": ["x", "y"]}]})
    D.mkdir(parents=True)
    lines = [{"qid": "1", "query": "s1", "seed_search": False, "agent_k": 2, "colbert": ["b", "c", "d"], "gold": ["c", "d"]},
             {"qid": "2", "query": "t1", "seed_search": False, "agent_k": 1, "colbert": ["x", "y"], "gold": ["y"]}]
    (D / "lists.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    a = run_original("bcp_novelty_eval.py", tree)
    h1 = (D / "hist.json").read_text()
    b = run_src("novelty_eval.py")
    assert a == b and (D / "hist.json").read_text() == h1 and "novelty 1.000" in a


def test_policy_eval_equivalent(tree):
    D = tree / "R" / "data" / "bcp_policy"; D.mkdir(parents=True)
    lines = [{"qid": str(i), "query": f"q{i}", "split": "val", "agent_k": 2, "seed_search": i == 0,
              "colbert": ["a", "b", "c", "g"], "bm25": ["g", "a", "z"], "gold": ["g"]} for i in range(3)]
    (D / "lists.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    a = run_original("bcp_policy_eval.py", tree, ["--split", "val"])
    b = run_src("policy_eval.py", ["--split", "val"])
    assert a == b and "HEADROOM" in a


def test_obliq_snippet_dev100_equivalent(tree):
    L = tree / "R" / "logs" / "obliq_mine_rr"
    for arm, nd in (("ctrl", 0.1), ("rr", 0.3)):
        write_json(L / f"{arm}_dev100_s0.json", [{"query_id": "q1", "ndcg@10": nd, "gold_recall": 0.2},
                                                  {"query_id": "q2", "ndcg@10": 0.0, "gold_recall": 0.1}])
    a = run_original("obliq_mine_rr_compare.py", tree)
    b = run_src("compare_obliq_snippet_dev100.py")
    assert a == b and "paired 2" in a


def test_obliq_snippet_281_equivalent(tree):
    dev = json.loads((SPLITS / "obliq_splits" / "twitter_dev.json").read_text())
    shutil.copy(SPLITS / "obliq_splits" / "twitter_dev.json", write_json(tree / "R" / "data" / "obliq_splits" / "twitter_dev.json", []))
    ids = [str(x) for x in dev[:3]] + ["q9998", "q9999"]
    write_json(tree / "R" / "data" / "obliq_fd" / "all281.json", ids)
    L = tree / "R" / "logs" / "obliq_mine_rr"
    write_json(L / "ctrl_dev100.json", [{"query_id": q, "ndcg@10": 0.1, "latency_s": 5} for q in ids[:3]])
    write_json(L / "snip_rest181_g0_s0.json", [{"query_id": "q9998", "ndcg@10": 0.0, "latency_s": 5}])
    write_json(L / "rr_dev100.json", [{"query_id": q, "ndcg@10": 0.2, "latency_s": 7} for q in ids[:3]])
    write_json(L / "rr_rest181_g0_s0_checkpoint.json", [{"query_id": "q9999", "ndcg@10": 0.5, "latency_s": 7}])
    for argv in ([], ["--final"]):
        a = run_original("obliq_q36_next_report.py", tree, argv)
        b = run_src("compare_obliq_snippet_281.py", argv)
        assert a == b and "RR:" in a


def test_frames_test200_equivalent(tree):
    shutil.copy(SPLITS / "frames" / "gepa_splits" / "split.json",
                write_json(tree / "R" / "data" / "frames" / "gepa_splits" / "split.json", {}))
    test = json.loads((SPLITS / "frames" / "gepa_splits" / "split.json").read_text())["test"]
    L = tree / "R" / "logs" / "frames"
    write_json(L / "sup_q36_bm25f" / "graded.json", {q: {"qid": q, "correct": i % 3 == 0, "gold_recall": 0.5} for i, q in enumerate(test)})
    write_json(L / "sup_q36_bm25f_rr" / "graded.json", {q: {"qid": q, "correct": i % 2 == 0, "gold_recall": 0.7} for i, q in enumerate(test)})
    a = run_original("frames_rr_compare.py", tree)
    b = run_src("compare_frames_test200.py")
    assert a == b and "McNemar" in a


def _obliq_tree(tree: Path, monkeypatch):
    """Synthetic OBLIQ-Bench qrels for both: the original's examples/obliq_data.py (copied into <tmp>/R/examples with
    its hard-coded root pointed at <tmp>/obliq) and rlm_shared.obliq_data (OBLIQ_ROOT patched)."""
    q = tree / "obliq" / "descriptive" / "twitter" / "queries+qrels"
    q.mkdir(parents=True)
    (q / "qrels.tsv").write_text("query_id\tcorpus_id\nq1\t11\nq1\t12\nq2\t21\n")
    ex = tree / "R" / "examples"; ex.mkdir(parents=True)
    src = (REPO / "agents" / "snippet_repl" / "original" / "RLM" / "examples" / "obliq_data.py").read_text()
    (ex / "obliq_data.py").write_text(src.replace("/home/extstu_00/OBLIQ-Bench", str(tree / "obliq")))
    (ex / "__init__.py").write_text("")
    from rlm_shared import obliq_data as OD
    monkeypatch.setattr(OD, "OBLIQ_ROOT", tree / "obliq")
    shutil.copy(SPLITS / "obliq_splits" / "twitter_dev.json",
                write_json(tree / "R" / "data" / "obliq_splits" / "twitter_dev.json", []))
    monkeypatch.syspath_prepend(str(tree / "R"))
    for m in [m for m in sys.modules if m == "examples" or m.startswith("examples.")]:
        monkeypatch.delitem(sys.modules, m)


def test_obliq_document_repl_equivalent(tree, monkeypatch):
    _obliq_tree(tree, monkeypatch)
    for arm, out in (("ctrl", "12\n99\n"), ("rr", "11\n12\n")):
        for qid, o, st in (("q1", out, "completed"), ("q2", "21\n", "completed" if arm == "rr" else "timeout")):
            write_json(tree / "S" / "runs" / f"obliq_q36_{arm}" / f"{qid}.json",
                       {"query_id": qid, "status": st, "result": [{"output": o}], "retrieved_docids": ["11", "21"],
                        "wall_time_s": 9, "n_iterations": 3, "tool_call_counts": {"search": 2}})
    a = run_original("obliq_rr_compare.py", tree)
    b = run_src("compare_obliq_document_repl.py")
    assert a == b and "== ALL: paired n=2" in a


def test_obliq_pick_best_equivalent(tree, monkeypatch):
    _obliq_tree(tree, monkeypatch)
    dev = [str(x) for x in json.loads((SPLITS / "obliq_splits" / "twitter_dev.json").read_text())]
    write_json(tree / "R" / "logs" / "obliq_rr" / "ids281.json", dev[:2] + ["q1"])
    L = tree / "R" / "logs" / "obliq_mine_rr"
    write_json(L / "ctrl_dev100.json", [{"query_id": dev[0], "ndcg@10": 0.4}, {"query_id": dev[1], "ndcg@10": 0.0}])
    write_json(L / "snip_rest181_g0_s0.json", [{"query_id": "q1", "ndcg@10": 0.5}])
    write_json(tree / "S" / "runs" / "obliq_q36_ctrl" / "q1.json",
               {"query_id": "q1", "status": "completed", "result": [{"output": "11\n"}]})
    orig = (REPO / "agents" / "snippet_repl" / "original" / "RLM_2026-11-11" / "RLM" / "scripts" / "obliq_best_full.py").read_text()
    orig = orig.replace("/home/extstu_00/RLM", str(tree / "R")).replace("/home/extstu_00/suprlm", str(tree / "S"))
    new_p = REPO / "agents" / "snippet_repl" / "src" / "obliq_pick_best.py"
    for argv in (["pick", "SNIP"], ["report", "SNIP"]):
        a = run_text(orig, str(tree / "orig_best.py"), argv)
        b = run_text(new_p.read_text(), str(new_p), argv)
        assert a == b and a.strip()
