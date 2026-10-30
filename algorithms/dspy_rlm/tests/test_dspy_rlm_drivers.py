"""The two drivers import from the new layout, keep the original signature maps, and read OBLIQ_BENCH_ROOT.

Nothing is run against a server: only module-level tables and the pure helpers are exercised, plus run_obliq's
loaders on a tiny temporary OBLIQ tree.
"""
from __future__ import annotations

import json

import pytest

import run_browsecomp as RB
import run_obliq as RO
from dspy_rlm import signatures as S


def test_browsecomp_signature_map():
    assert set(RB._SIGNATURES) == {"v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v13",
                                   "v14", "v15", "v16", "v17", "v20"}
    assert RB._SIGNATURES["v3"] is None                       # built per run from --max-iters/--max-search
    assert RB._SIGNATURES["v1"] is S.BrowseCompSignature
    assert RB._SIGNATURES["v20"] is S.BrowseCompDecompositionSignature
    for k, sig in RB._SIGNATURES.items():
        if sig is not None:
            assert sig._sig_key == k


def test_browsecomp_uses_shared_grading():
    import rlm_shared.browsecomp_data as BD
    assert RB.judge_match is BD.judge_match and RB.get_question is BD.get_question


def test_worker_map_lacks_v13_to_v20():
    """Documented issue: with --timeout > 0 the worker resolves v13-v20 to None (-> v1)."""
    import inspect
    src = inspect.getsource(RB._question_worker)
    for k in ("v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11"):
        assert f'"{k}":' in src
    for k in ("v13", "v14", "v15", "v16", "v17", "v20"):
        assert f'"{k}":' not in src


def test_obliq_signature_tables():
    assert RO.SUBSET_SIGNATURES["twitter"] is S.ObliqTwitterSig
    assert RO.SUBSET_SIGNATURES["congress"] is S.ObliqCongressSig
    assert RO.SUBSET_DATA["math-batch"] == "math"
    assert "wildchat-v7" not in RO.SUBSET_SIGNATURES          # documented issue: CLI accepts it -> KeyError


def test_obliq_uses_shared_score_advice():
    import inspect
    assert "from rlm_shared.obliq_score_advice import" in inspect.getsource(RO.run_eval)


def test_obliq_root_from_env(monkeypatch, tmp_path):
    monkeypatch.delenv("OBLIQ_BENCH_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="OBLIQ_BENCH_ROOT"):
        RO.load_queries("descriptive", "twitter")
    d = tmp_path / "descriptive" / "twitter" / "queries+qrels"
    d.mkdir(parents=True)
    (d / "queries.jsonl").write_text(json.dumps({"_id": "q1", "text": "t1"}) + "\n\n"
                                     + json.dumps({"_id": "q2", "text": "t2"}) + "\n")
    (d / "qrels.tsv").write_text("query-id\tcorpus-id\tscore\nq1\t11\t1\nq1\t12\t1\nq2\t21\t1\n")
    (d / "per_query_excluded_ids.json").write_text(json.dumps({"q1": [5]}))
    monkeypatch.setenv("OBLIQ_BENCH_ROOT", str(tmp_path))
    assert [q["_id"] for q in RO.load_queries("descriptive", "twitter")] == ["q1", "q2"]
    assert RO.load_qrels("descriptive", "twitter") == {"q1": {"11", "12"}, "q2": {"21"}}
    assert RO.load_qrels("descriptive", "twitter", pool=True) == {"q1": {"11", "12"}, "q2": {"21"}}  # fallback
    assert RO.load_excluded_ids("descriptive", "twitter") == {"q1": [5]}
    assert RO.load_excluded_ids("descriptive", "math") == {}


def test_obliq_metrics_and_id_parser():
    assert RO.ndcg_at(["a", "b"], {"a"}, 10) == 1.0
    assert RO.ndcg_at(["x", "a"], set(), 10) == 0.0
    assert RO.recall_at(["a", "x", "b"], {"a", "b"}, 2) == 0.5
    ans = "CHRG-116hhrg41317_p0001_09\n2034254279364489632\namerican-math-monthly___1998___10583"
    assert RO.extract_doc_ids_from_answer(ans) == [
        "american-math-monthly___1998___10583", "CHRG-116hhrg41317_p0001_09", "2034254279364489632"]
    assert RO.extract_doc_ids_from_answer("4, 9", subset="writing") == ["4", "9"]
    assert RO.extract_doc_ids_from_answer("4, 9", subset="twitter") == []


def test_heuristic_match_counts_empty_prediction_as_correct():
    """Documented issue (shared/rlm_shared/browsecomp_data.py): '' is a substring of every gold answer."""
    from rlm_shared.browsecomp_data import heuristic_match
    assert heuristic_match("Queen Arwa University", "") is True
