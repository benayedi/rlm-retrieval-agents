"""PUBLICATION COPY (refactor header). Original file: scripts/official_prompt_modules.py; unchanged apart from this
header. "examples/browsecomp_eval_official_rlm.py" below is src/harness.py here. Loaded by path (module name `_opm`,
importlib spec_from_file_location) from run_browsecomp.py (QWEN_REPL_CUE=1), run_frames.py (FRAMES_REPL_CUE=1) and
run_obliq.py (always). Its __main__ self-check reads whatever rlm prompt is loaded in that process (run alone, that is
the unpatched library prompt, as in the original).

Flatten the official harness's 8 chained monkeypatches into named, GEPA-optimisable blocks.

WHY. examples/browsecomp_eval_official_rlm.py builds its system prompt by applying eight
assert-anchored monkeypatches to the rlm library's RLM_SYSTEM_PROMPT at import time. That is
excellent for reproducibility -- an upstream reword fails loudly -- but it is hostile to a
prompt optimiser: there is no named parameter to mutate, only a 3,511-character blob produced
by side effects. GEPA_ASSESSMENT s4: "flattening to a *single* blob costs the module
structure, so re-split it into named sections ... or every reflection rewrites a 10k-character
wall with no attribution."

WHAT THIS DOES. Materialise the post-patch prompt once, split it on its natural paragraph
boundaries into five named blocks, and provide a rebuild() that reassembles them in order.
That gives the official harness the same shape the dspy harness already has
(_REPL_TOOLS + _REPL_RULES + strategy + _ORCHESTRATION), so a reflection can rewrite ONE
block and the others stay frozen and attributable.

THE SPLIT IS VERIFIED, NOT ASSUMED: rebuild() must reproduce the materialised prompt byte for
byte, and assert_roundtrip() fails loudly if a future upstream change moves the boundaries --
the same discipline the monkeypatches themselves use.

BLOCK ROLES (paragraph index in the materialised prompt):
  0 ROLE        what an RLM is                                  334 ch   freeze
  1 REPL_TOOLS  how to use the REPL, the tool surface         1,417 ch   freeze
  2 OUTPUT_CAP  the 60K truncation rule                          384 ch   freeze
  3 STRATEGY    how to search: the task-shaped advice            817 ch   *** MUTATE ***
  4 TURN_FORM   plan in prose, one repl block per turn           122 ch   freeze
  5 TURN_RULE   every turn must execute code                     427 ch   freeze

Only STRATEGY is the analogue of dspy's _OBLIQ_TWITTER_STRATEGY -- the block that carries
task knowledge rather than environment mechanics. It is the one GEPA should be given.
"""
from __future__ import annotations

SEP = "\n\n"
NAMES = ["ROLE", "REPL_TOOLS", "OUTPUT_CAP", "STRATEGY", "TURN_FORM", "TURN_RULE"]
MUTABLE = ["STRATEGY"]


def materialise() -> str:
    """The system prompt AFTER all eight monkeypatches have run."""
    import rlm.utils.prompts as PR
    return PR.RLM_SYSTEM_PROMPT


def split(prompt: str | None = None) -> dict[str, str]:
    p = materialise() if prompt is None else prompt
    paras = [x for x in p.split(SEP) if x.strip()]
    if len(paras) != len(NAMES):
        raise SystemExit(
            f"prompt no longer splits into {len(NAMES)} paragraphs (got {len(paras)}). "
            "An upstream reword or a new monkeypatch moved the boundaries: re-derive the "
            "split before trusting any optimisation built on it."
        )
    return dict(zip(NAMES, paras))


def rebuild(blocks: dict[str, str]) -> str:
    return SEP.join(blocks[n] for n in NAMES)


def assert_roundtrip() -> None:
    p = materialise()
    if rebuild(split(p)) != p:
        raise SystemExit("split/rebuild is NOT byte-identical -- the split is unsafe")


def apply(blocks: dict[str, str]) -> None:
    """Install a candidate back into the live prompt, in both places the harness reads it."""
    import rlm.utils.prompts as PR
    import rlm.core.rlm as core
    new = rebuild(blocks)
    PR.RLM_SYSTEM_PROMPT = new
    core.RLM_SYSTEM_PROMPT = new


if __name__ == "__main__":
    assert_roundtrip()
    b = split()
    print("roundtrip OK -- rebuild() is byte-identical to the materialised prompt\n")
    for n in NAMES:
        tag = "  *** MUTATE ***" if n in MUTABLE else ""
        print(f"  {n:<12} {len(b[n]):5d} ch{tag}")
    print(f"\n  TOTAL        {sum(len(v) for v in b.values()) + len(SEP)*(len(NAMES)-1):5d} ch")
    print("\n--- STRATEGY (the GEPA-mutable block) ---")
    print(b["STRATEGY"])
