"""Metric correctness: the reference implementations against hand-computed values, and the implementations the
experiments actually used (shared/rlm_shared/obliq_data.py) against the reference. No network, no models."""
import math

import pytest

from rlm_shared import metrics as M
from rlm_shared import obliq_data as OD


# ---- hand-computed values ------------------------------------------------------------------------------------------

def test_ndcg_perfect_and_empty():
    assert M.ndcg_at_k(["a", "b"], {"a", "b"}, 10) == pytest.approx(1.0)
    assert M.ndcg_at_k([], {"a"}, 10) == 0.0
    assert M.ndcg_at_k(["a"], set(), 10) == 0.0


def test_ndcg_hand_computed():
    # gold {a, c}; ranking b, a, c -> DCG = 1/log2(3) + 1/log2(4); IDCG = 1 + 1/log2(3)
    dcg = 1 / math.log2(3) + 1 / math.log2(4)
    idcg = 1 + 1 / math.log2(3)
    assert M.ndcg_at_k(["b", "a", "c"], {"a", "c"}, 10) == pytest.approx(dcg / idcg)


def test_ndcg_cutoff_and_ideal_capped_at_k():
    # 3 gold, k = 2: ideal counts only 2 documents
    assert M.ndcg_at_k(["a", "b"], {"a", "b", "c"}, 2) == pytest.approx(1.0)
    assert M.ndcg_at_k(["x", "y", "a"], {"a"}, 2) == 0.0


def test_recall_hand_computed():
    assert M.recall_at_k(["a", "x", "b"], {"a", "b", "c", "d"}, 2) == pytest.approx(0.25)
    assert M.gold_recall(["a", "z", "a"], {"a", "b"}) == pytest.approx(0.5)
    assert M.gold_recall(["a"], set()) is None


def test_majority_and_accuracy():
    assert M.majority([True, True, False]) is True
    assert M.majority([True, False]) is False          # a tie is not a majority
    assert M.majority([]) is None
    # missing question and a non-boolean verdict both count as wrong
    assert M.accuracy({"1": True, "2": "filtered"}, ["1", "2", "3"]) == pytest.approx(1 / 3)


# ---- the implementations used by the experiments -------------------------------------------------------------------

CASES = [
    (["b", "a", "c"], {"a", "c"}),
    (["a"], {"a", "b", "c"}),
    ([], {"a"}),
    (["x", "y", "z"], {"a"}),
    ([str(i) for i in range(60)], {"3", "17", "55"}),
]


@pytest.mark.parametrize("ranking,gold", CASES)
@pytest.mark.parametrize("k", [1, 3, 5, 10, 20, 50])
def test_obliq_ndcg_recall_match_reference_on_deduplicated_input(ranking, gold, k):
    assert OD.ndcg_at(ranking, gold, k) == pytest.approx(M.ndcg_at_k(ranking, gold, k))
    assert OD.recall_at(ranking, gold, k) == pytest.approx(M.recall_at_k(ranking, gold, k))


def test_obliq_ndcg_requires_deduplicated_input():
    """obliq_data.ndcg_at does not deduplicate; it relies on extract_doc_ids_from_answer doing so. With a duplicated
    gold id it can exceed 1. score_answer always passes deduplicated ids, so reported values are unaffected."""
    assert OD.ndcg_at(["a", "a"], {"a"}, 10) > 1.0
    ids = OD.extract_doc_ids_from_answer("CHRG-1_a CHRG-1_a CHRG-2_b", "congress")
    assert ids == ["CHRG-1_a", "CHRG-2_b"]
    s = OD.score_answer("CHRG-1_a, CHRG-1_a, CHRG-2_b", {"CHRG-1_a"}, "congress")
    assert s["ndcg@10"] == pytest.approx(1.0)


def test_extract_doc_ids_per_task():
    assert OD.extract_doc_ids_from_answer("1. 1234567890123456789\n2. 1234567890123456780", "twitter") == \
        ["1234567890123456789", "1234567890123456780"]
    assert OD.extract_doc_ids_from_answer("[CHRG-117hhrg123_s4]", "congress") == ["CHRG-117hhrg123_s4"]
    # a ranking with no parsable ids scores 0 (OBLIQ failure category O1)
    s = OD.score_answer("```repl\nprint(results)\n```", {"x"}, "twitter")
    assert s["n_parsed"] == 0 and s["ndcg@10"] == 0.0
