"""Scoring rules of the held-out confirmation and the D_pareto validation, run on synthetic caches with a stand-in
for split_runner (no rollout, no judge). They pin down what the original code does, including PO-I3 (killed records
leave the paired set although the docstring says they count wrong)."""
import json
import runpy
import sys
import types

import pytest

from .po_testlib import STUDY


def fake_split_runner(cache, out, split_f, logs):
    G = types.ModuleType("split_runner")
    G._CACHE = cache
    G.OUT = str(out)
    G.SPLIT_F = str(split_f)
    G._cache_load = lambda: None
    G._log = logs.append
    G.seed_blocks = lambda: {"B": "seed"}
    G.repair = lambda c: (c, [])
    G.validate = lambda c, s: None
    G.ckey = lambda c: "SEED" if c.get("B") == "seed" else "WIN"
    G.run_batch = lambda *a, **k: None
    G.score = lambda r: (1.0 if r.get("correct") else 0.0) + 0.5 * (r.get("gold_recall") or 0.0)
    return G


def _rec(ok, status="completed", rec=1.0):
    return {"correct": ok, "status": status, "gold_recall": rec}


def test_bcp_confirm_pairs_and_counts(tmp_path, monkeypatch, synth_env):
    qs = [str(i) for i in range(6)]
    cache = {}
    for q, (s, w) in zip(qs, [(True, True), (False, True), (True, False), (False, False), (True, True), (True, True)]):
        cache[f"SEED|{q}"] = _rec(s); cache[f"WIN|{q}"] = _rec(w)
    cache["WIN|3"] = _rec(False, status="timeout")      # a timeout: an agent outcome, counted wrong
    cache["SEED|5"] = _rec(True, status="killed")       # killed by the runaway cap: dropped from the pairs
    split = tmp_path / "split.json"; json.dump({"test200": qs}, open(split, "w"))
    win = tmp_path / "winner.json"; json.dump({"B": "win"}, open(win, "w"))
    logs = []
    monkeypatch.setitem(sys.modules, "split_runner", fake_split_runner(cache, tmp_path, split, logs))
    monkeypatch.setattr(sys, "argv", ["bcp_confirm.py", "--winner", str(win), "--split-file", str(split), "--report"])
    runpy.run_path(str(STUDY / "src" / "bcp_confirm.py"), run_name="__main__")
    text = "\n".join(logs)
    assert "paired n=5 of 6" in text                     # q5 (killed) is not in the paired set
    assert "won/lost by the winner: 1/1" in text
    assert "timeouts+kills: original 0, winner 1" in text


def test_validate_bcp_winner_rule(tmp_path, monkeypatch, synth_env, capsys):
    par = [str(i) for i in range(4)]
    cache = {f"SEED|{q}": _rec(q != "0", rec=0.5) for q in par}
    cache.update({f"WIN|{q}": _rec(True, status="timeout" if q == "3" else "completed", rec=0.5) for q in par})
    cache["WIN|3"]["correct"] = False                    # timed out: wrong, and still in the comparison
    json.dump(cache, open(tmp_path / "cache.json", "w"))
    json.dump([{"B": "seed"}, {"B": "win"}], open(tmp_path / "candidates.json", "w"))
    split = tmp_path / "split.json"; json.dump({"pareto": par}, open(split, "w"))
    monkeypatch.setitem(sys.modules, "split_runner", fake_split_runner({}, tmp_path, split, []))
    monkeypatch.setenv("GEPA_OUT", str(tmp_path))
    runpy.run_path(str(STUDY / "src" / "validate_bcp.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert "cand 1 WIN: n=4" in out
    # seed 3/4 right, winner 3/4 right (q0 gained, q3 lost to the timeout): d_score 0 -> no winner
    assert json.load(open(tmp_path / "verdict_pareto.json")) == {"verdict": "no_improvement"}
