"""HoVer helpers and the critic plumbing GEPA-5 reuses from split_runner, with `claude -p` replaced by a stub."""
import json
import subprocess
import sys

import pytest


def test_normalize_text_matches_the_split_builder(synth_env):
    import hover_retriever as hr
    assert hr.normalize_text("The Beatles!") == "beatles"
    assert hr.recall_gold(["Beatles", "x"], ["The Beatles", "y"]) == 0.5
    assert hr.all_found(["a", "b"], ["A", "b"]) is True


def test_hover_critic_goes_through_claude_with_the_hover_system_prompt(synth_env, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["x"])
    import hover_gepa as hg
    GB = hg.GB
    monkeypatch.setattr(GB, "OUT", str(tmp_path)); monkeypatch.setattr(GB, "PROG", str(tmp_path / "progress.txt"))
    monkeypatch.setattr(GB, "CRITIC_SYSTEM", hg.HOVER_CRITIC)      # what hover_gepa.main() does
    seen = {}

    def fake_run(cmd, **kw):
        seen["system"] = cmd[cmd.index("--system-prompt") + 1]
        reply = "thinking...\n```\nfirst draft\n```\nbetter:\n```\nNEW INSTRUCTION\n```\n"
        return subprocess.CompletedProcess(cmd, 0, json.dumps({"result": reply, "total_cost_usd": 0.01}), "")
    monkeypatch.setattr(GB.subprocess, "run", fake_run)
    out = GB.reflection_lm("prompt")
    assert seen["system"] == hg.HOVER_CRITIC
    assert out == "```\nNEW INSTRUCTION\n```"           # the last fenced block is kept
    assert json.loads((tmp_path / "reflection_log.jsonl").read_text().splitlines()[-1])["critic"] == "claude"
