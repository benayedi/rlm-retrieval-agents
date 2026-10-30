"""Signatures: every class is constructible and every prompt is identical to the original rag/signatures.py."""
from __future__ import annotations

import dspy
import pytest

from dspy_rlm import signatures as S


def _signature_classes(module):
    return {name: obj for name, obj in vars(module).items()
            if isinstance(obj, type) and issubclass(obj, dspy.Signature) and obj is not dspy.Signature}


def test_module_defines_the_expected_signatures():
    names = set(_signature_classes(S))
    # The ones the drivers select by --sig / --subset (spot check; the full set is compared below).
    for n in ("BrowseCompSignature", "BrowseCompMinimalSignature", "BrowseCompSubtractiveSignature",
              "BrowseCompDecompositionSignature", "ObliqTwitterSig", "ObliqCongressSig", "ObliqMathSig",
              "ObliqWritingSig", "ObliqWildchatV5Sig", "ReaderSignature"):
        assert n in names
    assert len(names) >= 35


@pytest.mark.parametrize("name", sorted(_signature_classes(S)))
def test_every_signature_is_constructible(name):
    cls = getattr(S, name)
    assert set(cls.input_fields) == {"question"}
    assert set(cls.output_fields) == {"answer"}
    assert cls.instructions.strip()
    # The import-time renaming loop gave every make_rlm_signature product its own name.
    assert cls.__name__ != "_Sig"
    # Constructible as dspy builds it: with_instructions returns a new class without touching the original.
    clone = cls.with_instructions(cls.instructions)
    assert clone.instructions == cls.instructions
    # A dspy.RLM can be built from it without any LM call.
    dspy.RLM(signature=cls, max_iters=2)


def test_same_signature_set_and_instructions_as_original(original_rag):
    new, old = _signature_classes(S), _signature_classes(original_rag.signatures)
    assert set(new) == set(old)
    for name in new:
        assert new[name].instructions == old[name].instructions, name
        assert new[name].__name__ == old[name].__name__, name
        for fld in ("question", "answer"):
            f_new = {**new[name].input_fields, **new[name].output_fields}[fld]
            f_old = {**old[name].input_fields, **old[name].output_fields}[fld]
            assert f_new.json_schema_extra == f_old.json_schema_extra, (name, fld)
        # (_sig_key is not compared: run_browsecomp sets it on the classes at import, so it depends on test order.)


def test_prompt_constants_identical(original_rag):
    old = original_rag.signatures
    consts = [k for k, v in vars(old).items() if k.startswith("_") and isinstance(v, str) and not k.startswith("__")]
    assert len(consts) > 20
    for k in consts:
        assert getattr(S, k) == getattr(old, k), k


@pytest.mark.parametrize("args", [
    dict(strategy="STRATEGY BLOCK\n", answer_desc="the answer"),
    dict(strategy="X", answer_desc="A", question_desc="Q?"),
    dict(strategy="X", answer_desc="A", repl_tools="TOOLS\n", repl_rules="RULES\n"),
])
def test_make_rlm_signature_matches_original(original_rag, args):
    new = S.make_rlm_signature(**args)
    old = original_rag.signatures.make_rlm_signature(**args)
    assert new.instructions == old.instructions
    assert new.input_fields["question"].json_schema_extra == old.input_fields["question"].json_schema_extra
    assert new.output_fields["answer"].json_schema_extra == old.output_fields["answer"].json_schema_extra
    # The prompt is tools + rules + strategy + orchestration, in that order.
    tools = args.get("repl_tools", S._REPL_TOOLS)
    rules = args.get("repl_rules", S._REPL_RULES)
    assert new.instructions.strip() == (tools + rules + args["strategy"] + S._ORCHESTRATION).strip()


@pytest.mark.parametrize("max_iters,max_search,commit_env", [
    (16, 35, None), (25, 50, None), (3, 5, None), (16, 35, "4"), (16, 35, "99"), (16, 35, "1"),
])
def test_browsecomp_minimal_budget_sig_matches_original(original_rag, monkeypatch, max_iters, max_search,
                                                        commit_env):
    if commit_env is None:
        monkeypatch.delenv("BROWSECOMP_COMMIT_TURN", raising=False)
    else:
        monkeypatch.setenv("BROWSECOMP_COMMIT_TURN", commit_env)
    new = S.browsecomp_minimal_budget_sig(max_iters, max_search)
    old = original_rag.signatures.browsecomp_minimal_budget_sig(max_iters, max_search)
    assert new.instructions == old.instructions
    assert new.__name__ == old.__name__ == "BrowseCompMinimalBudgetSignature"
    assert new._sig_key == old._sig_key == "v3"
    assert str(max_search) in new.instructions


def test_budget_commit_turn_default_is_75_percent(monkeypatch):
    monkeypatch.delenv("BROWSECOMP_COMMIT_TURN", raising=False)
    a = S.browsecomp_minimal_budget_sig(16, 35).instructions
    monkeypatch.setenv("BROWSECOMP_COMMIT_TURN", "12")   # round(16 * 0.75) == 12
    assert S.browsecomp_minimal_budget_sig(16, 35).instructions == a
