"""What the refactored Snippet RLM computes, on synthetic inputs (fake agent, fake retriever; no network, no LM)."""
from __future__ import annotations

import subprocess
import sys

import pytest

from snippet_rlm_testlib import ART, LIGHTON, SCENARIOS, SRC, hermetic_env, needs_rlm


@needs_rlm
def test_harness_imports_from_src(probe):
    out = probe("src", "harness", {})
    assert out["import_stdout"] == []
    assert out["default_top_k"] == 5 and out["snippet_len"] == 500 and out["repl_output_chars"] == 60000
    assert out["proxy"] == "http://localhost:8024"


@needs_rlm
def test_recall_on_synthetic_record(probe):
    """Fake agent: search 'first query' (d1,d2,d9), search with top_k=7, get_document('d3').
    Gold = {d1,d3,d5,d7}, evidence = {d2,d4}; retrieved = {d1,d2,d9,d3}."""
    rec = probe("src", "harness", {})["record"]
    assert rec["unique_docs"] == 4
    assert rec["gold_recall"] == pytest.approx(2 / 4)
    assert rec["evid_recall"] == pytest.approx(1 / 2)
    assert rec["search_calls"] == 2 and rec["get_document_calls"] == 1
    assert rec["topk_overrides"] == 1 and rec["topk_requested"] == [7]
    assert rec["correct"] is True              # heuristic match of "Paris" in the Exact Answer span; no judge call
    assert rec["error"] is None
    # `iterations` = LM calls in the usage summary minus llm_query calls (6 - 0 here). Known to be wrong on real runs
    # (CHANGES_FROM_ORIGINAL.md, issues): the summary keeps only one client's usage per model name.
    assert rec["iterations"] == 6 and rec["lm_calls"] == 6 and rec["input_tokens"] == 1000


@needs_rlm
def test_recall_frames_uses_gold_for_evidence(probe):
    rec = probe("src", "frames", SCENARIOS["frames-gpt5 (FR-7)"][1])["record"]
    assert rec["gold_recall"] == rec["evid_recall"] == pytest.approx(0.5)


@needs_rlm
def test_obliq_scoring_on_synthetic_record(probe):
    rec = probe("src", "obliq", SCENARIOS["obliq-twitter (TWI-9)"][1])["record"]
    # qrels q1 = {111.., 222..}; retrieved by the fake agent: 111.., 333.., d3 -> gold_recall 1/2
    assert rec["gold_recall"] == pytest.approx(0.5)
    assert rec["query_id"] == "q1" and rec["n_parsed"] == 2
    assert rec["ndcg@10"] == pytest.approx(1.0) and rec["correct"] is True


@needs_rlm
def test_format_gate(probe):
    out = probe("src", "harness", {})
    g = out["gate"]
    # three malformed submissions are refused (ready flipped back, notice printed) ...
    assert [x["ready"] for x in g[:3]] == [False, False, False]
    assert g[0]["printed"].startswith("[SUBMISSION REJECTED 1/3]") and "[SUBMISSION REJECTED 3/3]" in g[2]["printed"]
    # ... then the gate stops enforcing and accepts anything (MAX_REJECTS = 3)
    assert g[3]["ready"] is True and g[3]["printed"] == ""
    assert out["gate_fired"][0] == "still nothing"
    assert out["gate_wellformed"]["ready"] is True


@needs_rlm
def test_final_answer_extraction_and_fence_tolerance(probe):
    texts_out = probe("src", "bcp", SCENARIOS["bcp-qwen (BCP-15/17/19)"][1])["find_code_blocks"]
    plain = probe("src", "bcp", SCENARIOS["bcp-gpt5 (BCP-13)"][1])["find_code_blocks"]
    # unpatched library: only ```repl blocks run
    assert plain[0] == ["x = search_index('a')\nprint(x)"] and all(b == [] for b in plain[1:])
    # QWEN_FENCE_TOLERANT: a ```python / bare fence runs if it parses and does something
    assert texts_out[1] == ["x = search_index('a')"] and texts_out[2] == ["print(1)"]
    assert texts_out[3] == []                     # parses, does nothing
    # the answer template in a bare fence is not run as code; QWEN_PROSE_FINAL then submits it as prose, closing
    # fence included (the content still carries both lines the format gate checks)
    assert texts_out[4] == ['answer["content"] = \'Explanation: foo\\nExact Answer: bar\\n```\'\nanswer["ready"] = True']
    # QWEN_PROSE_FINAL: a code-less turn with a non-empty Exact Answer becomes the submission
    assert texts_out[5] == ['answer["content"] = \'Explanation: it is Paris.\\nExact Answer: Paris\\nConfidence: 80%\''
                            '\nanswer["ready"] = True']
    assert "Lyon" in texts_out[6][0] and "draft" not in texts_out[6][0]   # after </think>, from the last Explanation:
    assert texts_out[7] == []                     # empty template is not an answer
    assert texts_out[8] == [] and texts_out[9] == []


def test_final_answer_span():
    sys.path.insert(0, str(ART / "shared"))
    from rlm_shared.browsecomp_data import _final_answer_span, heuristic_match

    assert _final_answer_span("Explanation: x\nExact Answer: Queen Arwa\nConfidence: 3%") == "Queen Arwa"
    assert _final_answer_span("no format") == "no format"
    assert heuristic_match("Paris", "paris, France")


@needs_rlm
def test_snippet_tokenizer_comes_from_dspy_rlm(tmp_path):
    """SEARCH_SNIPPET_TOKENS truncation imports dspy_rlm.agent._get_snippet_tokenizer (was rag.agent)."""
    code = (
        "import sys, types\n"
        f"sys.path.insert(0, {str(SRC)!r})\n"
        "class T:\n"
        "    def encode(self, t, add_special_tokens=False): return t.split()\n"
        "    def decode(self, ids, skip_special_tokens=True): return ' '.join(ids)\n"
        "pkg = types.ModuleType('dspy_rlm'); pkg.__path__ = []\n"
        "ag = types.ModuleType('dspy_rlm.agent'); ag._get_snippet_tokenizer = lambda: T()\n"
        "sys.modules.update({'dspy_rlm': pkg, 'dspy_rlm.agent': ag})\n"
        "import harness as H\n"
        "H.SNIPPET_TOKENS = '3'\n"
        "print(repr(H._truncate('a b c d e')), repr(H._truncate('a b')))\n"
    )
    p = subprocess.run([sys.executable, "-c", code], env=hermetic_env({}, tmp_path), capture_output=True, text=True,
                       cwd=tmp_path, timeout=120)
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == "'a b c' 'a b'"


@needs_rlm
def test_run5d_launcher_settings_match_lighton_scenario():
    """The launchers export exactly the LIGHTON scenario's switches (keeps the equivalence scenarios honest)."""
    sh = (SRC.parent / "experiments" / "bcp_snip2.sh").read_text()
    for k, v in LIGHTON.items():
        assert f"{k}={v}" in sh
