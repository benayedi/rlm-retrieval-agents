"""Run the AUTHORS' RLM implementation (github.com/alexzhang13/rlm) on BrowseComp+.

WHY THIS EXISTS. Every arm in this project so far used dspy.RLM — a reimplementation of the
RLM paper, not the paper's own code. Reading the authors' repo surfaced differences that our
harness cannot express, several of which are exactly the levers our cost analysis identified:

  their RLM                          dspy.RLM (what we have been running)
  ─────────────────────────────────  ────────────────────────────────────
  max_budget       (USD ceiling)     — the per-question spend cap we concluded was the only
                                       lever the agent cannot route around. They shipped it.
  compaction + threshold_pct=0.85    — history compaction. Our repl_history grows unboundedly
                                       and cost is (content x turns / 2); this is the fix.
  orchestrator=True                  — the ORCHESTRATOR_ADDENDUM as a first-class toggle
  max_depth (real rlm_query)         — our llm_query is a flat sub-LM call; the "Recursive"
                                       part of RLM is something we never actually ran
  max_timeout / max_errors           — no equivalent
  on_iteration_* callbacks           — we reconstruct per-turn metrics by scraping logs

AND THE PROMPT CONTRADICTS OURS ON A POINT WE FLAGGED AS SUSPECT:
    theirs: "Plan in prose, then execute one ```repl``` block every turn"
            + "pause and plan: state explicitly how the task decomposes ... sketch the
               concrete sequence of turns ... before you execute them"
    ours:   "Planning in prose does nothing - write the code immediately."
Their paper ablates this: decomposition hints were worth +15.0 points (38.7 base -> 50.6 RLM
-> 65.6 RLM+hints). Our signature has zero worked examples; theirs has three.

WHAT IS HELD CONSTANT with our other arms: the same Reason-ModernColBERT index on port 8024,
the same gpt-5 root model, the same gpt-5.4-mini judge, the same q0-99, the same decryption
and gold-doc bookkeeping. Results are written in the same JSON shape so the existing analysis
scripts read them without modification.

WHAT IS NOT COMPARABLE: this is a different agent loop, a different prompt, and a different
submission protocol (answer["ready"] rather than SUBMIT()). Treat it as a separate experiment,
not another row in the knob table.

THEY SHIP NO RETRIEVAL TOOLS. RLM is a long-context framework -- their only custom_tools
example is stock prices. search_index / get_document below are ours, with the same
implementations and the same 8024 proxy every other arm used.

─────────────────────────────────────────────────────────────────────────────────────────────
HOW THIS FILE IS ORGANISED

The library is NEVER edited. Everything below is a monkeypatch applied at import time, so the
vendored clone at ~/rlmrepo stays byte-identical to upstream and can be re-pulled at any time.

  1. instrumentation   _wrap()               count llm_query properly (the library does not)
                       _d2_* + _D2           recursive-depth fixes, inert at max_depth=1
  2. submission        _install_contract()   refuse a malformed answer up to 3 times
  3. prompt patches    applied in this order, each asserting on its anchor text so a change
                       upstream fails loudly instead of silently doing nothing:
                         _strip_context_fiction()   drop the long-context framing
                         _raise_output_ceiling()    REPL output cap 20K -> 60K chars
                         _reanchor_prompt()         "context" -> retrieval
                         _final_prompt_fixes()      print-only REPL, no prose-only turns
                         _add_weak_result_rule()    what to do when a search returns nothing
                         _add_retriever_line()      the index is semantic  [RETRIEVER_LINE]
                         _add_depth2_line()         children can search    [DEPTH2_PROMPT]
                         _add_triage_rule()         wide-block triage      [SEARCH_TRIAGE_RULE]
  4. tools             make_tools() / _tool_specs() search_index + get_document over the proxy
  5. task text         _TASK_OURS / _TASK_LIGHTON   the user message      [LIGHTON_TASK_TEXT]
  6. the run           run_one() / main()

ENVIRONMENT SWITCHES — every one defaults to the frozen configuration used for the headline
830-question runs, so running with none of them set reproduces those runs.

  EMBEDDING_SERVER_URL        8024 = Reason-ModernColBERT (multi-vector) | 8001 = Qwen3-Embed-8B
  ORCHESTRATOR                0 drops the ORCHESTRATOR_ADDENDUM. Default 1 upstream; the runs
                              use 0 — same accuracy, -23% tokens, -37% latency
  LIGHTON_TOOL_DOCS           1 = LightOn's verbatim tool docstrings
  LIGHTON_TASK_TEXT           1 = LightOn's task wording and answer format
  RETRIEVER_LINE              0 removes the "index is a dense semantic retriever" sentences
  NO_UNKNOWN                  1 removes the abstention escape hatch
  SEARCH_TRIAGE_RULE          1 adds the wide-search-block triage rule (untested)
  DEPTH2_PROMPT               1 tells the model its children can search (only with max_depth 2)
  CHILD_MAX_ITERATIONS        iteration cap for a depth-2 child (default 15)
  MAX_CONCURRENT_SUBCALLS     parallel children (default 2; Azure 429s are scored as WRONG)
  SUB_LM_MODEL                route llm_query to a different deployment
  SUB_LM_REASONING_EFFORT     llm_query effort (default medium; high and low both measured worse)
  AZURE_OPENAI_REASONING_EFFORT  root effort (default high)
  SEARCH_SNIPPET_TOKENS / SEARCH_SNIPPET_LEN   snippet budget (runs use 512 tokens)
  OFFICIAL_TOP_K              search_index default top_k (5, matching the benchmark paper)
  RLM_MAX_OUTPUT_CHARS        REPL stdout ceiling (60000)
  RLM_TRAJ_DIR                where trajectories are written — the ONLY reliable source of
                              iteration counts and per-turn token volume

TWO FIELDS IN THE OUTPUT JSON LIE, and both are documented where they are produced:
  `iterations`    broken three different ways. Count `iteration` records in the trajectory.
  `input_tokens`  reports the SUB-LM only whenever llm_query ran, because the library merges
                  per-model usage with dict.update(). Corrected totals are ~2x the logged ones.
─────────────────────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import re
import time
from pathlib import Path

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# THE AUTHORS' RLM IS AN INSTALLED PACKAGE, NOT A sys.path HACK.
# It used to be appended to sys.path from a session-scoped scratchpad under /tmp, which meant
# every result in this project depended on a temp directory belonging to a dead session: if it
# were reaped, nothing here would import, let alone reproduce. The clone now lives at
# ~/rlmrepo and is installed editable into this venv:
#     pip install -e ~/rlmrepo --no-deps
# --no-deps is deliberate. The package pins openai>=2.14.0, anthropic, google-genai and
# portkey-ai; letting pip resolve those would upgrade the client libraries underneath a
# finished set of results. The venv already satisfies what the code actually uses.
# Pinned at github.com/alexzhang13/rlm commit 854e688 ("add headlong"), working tree clean —
# every modification we make is a monkeypatch below, never an edit to the library.

# Dataset access and grading. This used to import from the dspy harness, which meant
# executing `import dspy` and `from rag import ...` — an entire agent framework loaded so
# this file could base64-xor a question string. browsecomp_data has no such dependencies.
from examples.browsecomp_data import (  # noqa: E402
    get_question, get_gold_answer, get_gold_doc_ids, get_evidence_doc_ids, judge_match,
)
from rlm import RLM  # noqa: E402
from rlm.logger import RLMLogger  # noqa: E402
from rlm.environments.local_repl import LocalREPL  # noqa: E402

# ── COUNTING llm_query PROPERLY ───────────────────────────────────────────────────────────
# on_subcall_complete DOES NOT COUNT llm_query. It fires only when a recursive CHILD RLM
# finishes (rlm/core/rlm.py:865), so at max_depth=1 it can never fire at all. Plain
# llm_query / llm_query_batched go through LocalREPL._llm_query, append to
# _pending_llm_calls, and surface as ExecutionResult.rlm_calls — a path with no callback.
# The first 94 questions of this port therefore recorded llm_query_calls = 0 by construction,
# not by observation. That was an instrumentation artifact and it is retracted; these wrappers
# are the actual measurement. They also mean "iterations" (lm_calls - subcalls) was really
# TOTAL LM CALLS, sub-calls included — not turns.
_LLMQ = {"n": 0, "batched": 0, "prompts": 0}


def _wrap(name, is_batch):
    orig = getattr(LocalREPL, name)

    def wrapped(self, prompt, *a, **kw):
        if is_batch:
            _LLMQ["batched"] += 1
            _LLMQ["prompts"] += len(prompt)
        else:
            _LLMQ["n"] += 1
        return orig(self, prompt, *a, **kw)

    setattr(LocalREPL, name, wrapped)


_wrap("_llm_query", False)
_wrap("_llm_query_batched", True)


# ── RECURSIVE DEPTH (max_depth >= 2): THREE DEFECTS THAT MUST BE FIXED FIRST ──────────────
# Raising --max-depth is a one-word change, but the library's child-spawning path
# (rlm/core/rlm.py:800-870) has three properties that would make a naive depth=2 run measure
# the plumbing rather than the idea. All three are fixed here by monkeypatch; the vendored
# repo is left untouched.
#
# WHAT DEPTH=2 ACTUALLY BUYS. self.custom_sub_tools defaults to custom_tools (rlm.py:161) and
# the child is constructed with custom_tools=self.custom_sub_tools (rlm.py:827). So a child
# RLM INHERITS search_index and get_document and runs its own REPL loop over them. This is
# the one genuinely new capability: llm_query is a flat completion that cannot retrieve, so
# every reader intervention we measured was reading text the ROOT had already fetched. A
# depth-2 child can go and get its own.
#
# DEFECT 1 — THE CHILD GETS NO root_prompt. rlm.py:834 calls
#     child.completion(prompt, root_prompt=None)
# so the delegated task lands in the child's `context` variable and the child's entire user
# message is "Your context is a str of N total characters." This is byte-for-byte the bug
# that held root format compliance at 93/100 until we passed root_prompt (100/100 after).
# It is WORSE for the child, because _reanchor_prompt() has rewritten the system prompt to
# say the answer lives in the index rather than in `context` — so a child told nothing has
# no reason to print the one variable holding its instructions. Fixed in _completion below.
#
# DEFECT 2 — THE CHILD DOES NOT INHERIT OUR CONFIG. The child constructor (rlm.py:812-830)
# passes no `orchestrator`, no `compaction`, no `sampling_args` and no `sub_sampling_args`.
# Consequences: children run WITH the ORCHESTRATOR_ADDENDUM even under ORCHESTRATOR=0 (our
# best config), without compaction, and at the library default reasoning effort. A "depth=2"
# arm would silently be a different agent at depth 1 than at depth 0. Fixed in _init below.
#
# DEFECT 3 — CHILD TOKENS ARE INVISIBLE. The parent's usage_summary is built from ITS OWN
# lm_handler's clients (lm_handler.py:237-251). A child RLM spawns its own completion context
# with its own handler, so its tokens never reach the parent's summary; only cost is tracked,
# and only for the budget check (rlm.py:837). Unpatched, a depth=2 arm would report a token
# count that excludes every child — flattering and meaningless. _D2 accumulates it.
_D2 = {"child_in_tok": 0, "child_out_tok": 0, "children": 0, "child_turns": 0, "root_prompt_fixed": 0}
_D2_CFG: dict = {}

_orig_completion = RLM.completion
_orig_init = RLM.__init__


def _d2_init(self, *a, **kw):
    # DEFECT 2: make the child the same agent as the root.
    if kw.get("depth", 0) > 0 and _D2_CFG:
        # OVERRIDE, not setdefault: rlm.py:817 passes max_iterations=self.max_iterations, so
        # unpatched every child inherits the root's 100-turn budget. With history cost going
        # as (content x turns / 2), one rlm_query_batched of 4 children could cost four full
        # root questions. 15 turns is above the root's observed mean of 4.7 and well under
        # any runaway.
        kw["max_iterations"] = int(os.environ.get("CHILD_MAX_ITERATIONS", "15"))
        kw.setdefault("orchestrator", _D2_CFG["orchestrator"])
        kw.setdefault("compaction", _D2_CFG["compaction"])
        kw.setdefault("sampling_args", _D2_CFG["sampling_args"])
        kw.setdefault("sub_sampling_args", _D2_CFG["sub_sampling_args"])
        kw.setdefault("max_concurrent_subcalls", _D2_CFG["max_concurrent_subcalls"])
    return _orig_init(self, *a, **kw)


def _d2_completion(self, prompt, root_prompt=None):
    # DEFECT 1: a child called with no root_prompt cannot see its own task.
    if root_prompt is None and getattr(self, "depth", 0) > 0 and isinstance(prompt, str):
        root_prompt = prompt
        _D2["root_prompt_fixed"] += 1
    res = _orig_completion(self, prompt, root_prompt=root_prompt)
    # DEFECT 3: roll the child's usage up by hand.
    if getattr(self, "depth", 0) > 0:
        us = getattr(res, "usage_summary", None)
        if us is not None:
            _D2["child_in_tok"] += int(getattr(us, "total_input_tokens", 0) or 0)
            _D2["child_out_tok"] += int(getattr(us, "total_output_tokens", 0) or 0)
        _D2["children"] += 1
        md = getattr(res, "metadata", None) or {}
        try:
            _D2["child_turns"] += len(md.get("iterations", []))
        except Exception:
            pass
    return res


RLM.__init__ = _d2_init
RLM.completion = _d2_completion


# ── THE SUBMISSION CONTRACT ───────────────────────────────────────────────────────────────
# WHY. Their loop ends the moment `answer["ready"] = True`, whatever `answer["content"]`
# holds. Our dspy arms cannot do that: the answer is a typed signature field. Over 100
# questions the official port emitted a well-formed `Exact Answer:` line 93 times against
# snipmin's 100, and THREE of its twenty-one failures were purely that. q56's final output
# ended by asking ME a clarifying question — "With one of these, I'll re-run a focused search
# and return the exact sports title" — because a turn is allowed to end in free text addressed
# to a human. That is a missing contract, not worse reasoning: on the eight questions it lost
# to snipmin its mean gold_recall was HIGHER (0.88 vs 0.83 over the full run).
#
# WHAT THIS DOES. _AnswerDict fires on_ready the first time `ready` flips truthy
# (local_repl.py:42-47). We intercept that: if the content does not carry both an
# Explanation: and an Exact Answer: line, `ready` is set straight back to False, on_ready is
# never called, and a rejection notice is printed — which lands in the cell's captured stdout,
# so the model reads it on the next turn and gets to fix the format.
#
# THE CAP MATTERS. After MAX_REJECTS refusals the contract stops enforcing and accepts
# whatever it is given. Without that an agent that cannot produce the format would spin until
# max_iterations, converting a formatting failure into a 100-turn cost blowout — a far worse
# outcome than the malformed answer it started with.
#
# NOT ENFORCED: citations. TASK_SUFFIX asks for inline [doc_id] citations and the port
# produced at least one on 88/100 (snipmin 97/100), but our judge scores only the answer
# against the gold string — citations do not affect correctness here. Rejecting on a missing
# citation would spend turns on something the metric never reads.
MAX_REJECTS = 3
_REJECTS = {"n": 0}
_FORMAT_OK = re.compile(r"Exact Answer\s*:", re.I)
_EXPL_OK = re.compile(r"Explanation\s*:", re.I)


def _install_contract():
    from rlm.environments.local_repl import _AnswerDict

    orig = _AnswerDict.__setitem__

    def guarded(self, key, value):
        if key == "ready" and value:
            content = str(self.get("content", "") or "")
            ok = bool(_FORMAT_OK.search(content)) and bool(_EXPL_OK.search(content))
            if not ok and _REJECTS["n"] < MAX_REJECTS:
                _REJECTS["n"] += 1
                dict.__setitem__(self, "ready", False)
                print(
                    f"[SUBMISSION REJECTED {_REJECTS['n']}/{MAX_REJECTS}] answer['content'] must "
                    "contain BOTH an 'Explanation:' line and an 'Exact Answer:' line.\n"
                    "Do not ask the user questions and do not describe what you would search "
                    "next — you cannot receive a reply. Commit to your best answer, or write "
                    "'Exact Answer: Unknown'. Rewrite answer['content'] and set "
                    "answer['ready'] = True again."
                )
                return
        return orig(self, key, value)

    _AnswerDict.__setitem__ = guarded


_install_contract()


# ── REMOVING THE "context" FICTION ────────────────────────────────────────────────────────
# Their prompt builder tells the model, unconditionally:
#     "Your context is a str of N total characters. Each sub-LLM call can handle roughly
#      ~100k tokens at once."
# and on turn 0 adds:
#     "You have not interacted with the REPL environment or seen your prompt / context yet.
#      Look at the context first; do not provide a final answer yet."
#
# Both are correct for THEIR setting, where `context` holds the corpus — they load 1,000
# BrowseComp+ documents, 6-11M characters, into the REPL and the model must plan how to chunk
# it. In OUR setting the corpus is behind search_index and `context` can only hold the
# question itself (972 characters), so the model is told the size of something that is not
# data and instructed to go read it first. Measured in the previous run: nearly every question
# spent turn 1 on `print(type(context), len(context)); print(context[:400])` — one wasted LM
# call, ~15-20s, on a variable holding the question already present in its user message.
#
# Both strings are removed. This is the only edit that changes THEIR prompt rather than my
# harness around it, so it is narrow and asserted: if either string stops matching (a repo
# update), the patch raises instead of silently doing nothing.
def _strip_context_fiction():
    import rlm.core.rlm as core
    import rlm.utils.prompts as P

    orig_sys, orig_user = P.build_rlm_system_prompt, P.build_user_prompt
    marker_sys = "Your context is a"
    marker_user = "You have not interacted with the REPL environment"

    def patched_sys(*a, **kw):
        msgs = orig_sys(*a, **kw)
        for m in msgs:
            if m["role"] == "user" and marker_sys in m["content"]:
                m["content"] = re.sub(
                    r"\n*Your context is a .*?~100k tokens at once\.\s*", "", m["content"],
                    flags=re.S).strip()
        return msgs

    def patched_user(*a, **kw):
        msg = orig_user(*a, **kw)
        if marker_user in msg["content"]:
            msg["content"] = re.sub(
                r"You have not interacted with the REPL environment.*?final answer yet\.\s*",
                "", msg["content"], flags=re.S)
        return msg

    core.build_rlm_system_prompt = patched_sys
    core.build_user_prompt = patched_user


_strip_context_fiction()


# ── REPL OUTPUT CEILING: 20K -> 60K ───────────────────────────────────────────────────────
# This one is ENFORCED, not merely described. rlm/utils/parsing.py:format_iteration tail-trims
# every code block's output to max_character_length (default 20,000) before it enters message
# history, and rlm/core/rlm.py:456 calls it with the default. Measured on this run's own
# trajectories: 342 of 726 code cells — 47% — produced more than 20K of stdout, the largest
# 418,553 characters. Nearly half of all cell outputs were being cut before the model saw
# them, and the trajectory logs record the full text so the loss is invisible in the logs.
#
# 60,000 matches RLM_MAX_OUTPUT_CHARS=60000, the ceiling every dspy arm here has run with
# since the output-ceiling experiment. The prompt sentence is rewritten to match, because
# leaving it saying "~20K" would have the model budget its prints against a limit that no
# longer exists — that sentence is the reason it slices at 200 characters while snipmin,
# which is told its snippets are 2000 characters and given no ceiling, slices at 500-1000.
REPL_OUTPUT_CHARS = int(os.environ.get("RLM_MAX_OUTPUT_CHARS", "60000"))


def _raise_output_ceiling():
    import rlm.core.rlm as core
    import rlm.utils.parsing as P
    import rlm.utils.prompts as PR

    orig = P.format_iteration

    def patched(iteration, max_character_length=REPL_OUTPUT_CHARS):
        return orig(iteration, max_character_length=max_character_length)

    core.format_iteration = patched

    old = "REPL outputs over ~20K characters are truncated"
    new_txt = f"REPL outputs over ~{REPL_OUTPUT_CHARS // 1000}K characters are truncated"
    assert old in PR.RLM_SYSTEM_PROMPT, "20K sentence not found — repo changed"
    core.RLM_SYSTEM_PROMPT = PR.RLM_SYSTEM_PROMPT.replace(old, new_txt)
    PR.RLM_SYSTEM_PROMPT = core.RLM_SYSTEM_PROMPT


_raise_output_ceiling()


# ── RE-ANCHORING THE PROMPT FROM `context` TO RETRIEVAL ───────────────────────────────────
# The authors' system prompt says "context" 13 times across 8 of its 9 blocks, and in their
# setting every one is correct: `context` holds the corpus (they load 1,000 BrowseComp+
# documents, 6-11M characters, into the REPL). Here retrieval is behind search_index and
# `context` holds only the question, so those 13 references describe something that does not
# exist. Removing the size line and the turn-0 safeguard (see _strip_context_fiction) took out
# the two most visible symptoms and left the framing intact.
#
# THE ONE THAT MATTERS MOST is the parenthetical in the orchestrator addendum:
#     "if a Python keyword / regex search over `context` would already pin the answer, or if a
#      single visible passage already contains it, just read it directly"
# That sentence licenses exactly the behaviour measured as this arm's failure: of 212 code
# cells that called get_document in run 2, 102 did substring tests and short excerpts and only
# 24 delegated, while snipmin delegates 3x per question. It also directly contradicts the line
# added to TASK_SUFFIX in run 3 ("substring checks tell you a word is present, not what the
# document says"), so run 3 was arguing with itself.
#
# WHAT IS PRESERVED: the substance the paper's own ablation values — decompose before acting,
# sketch the sequence of turns, fat prompts, <=20 per batch, stage coarse-then-fine, verify
# before submitting. Only the object those instructions point at changes: `context` becomes
# the corpus reached through search_index and the document text those tools return.
#
# THIS IS THE LARGEST EDIT MADE TO THEIR PROMPT and it should be reported as such: after this
# the arm tests "the authors' agent loop with a retrieval-adapted prompt", not "the authors'
# prompt". Every substitution is asserted, so a repo change breaks the run instead of silently
# leaving the old text in place.
_CTX_REWRITES = [
    ("You are a Recursive Language Model (RLM): a language model with a prompt, and a very "
     "important context stored in a Python REPL related to that prompt.",
     "You are a Recursive Language Model (RLM): a language model with a prompt, and a very "
     "important document corpus reachable from a Python REPL through retrieval tools."),

    # SCOPED DELIBERATELY. An earlier wording said "the text they return lands in Python
    # variables, not in your context window" of BOTH tools, which reads as an invitation to
    # delegate search results too. That would be waste: a snippet is a few thousand tokens and
    # printing it IS reading it — stdout is appended to the message history, so the root model
    # sees it on the next turn with no extra LM call. Delegation exists for size, not for
    # capability: a document is 5K-900K characters, and because repl_history re-serialises
    # every prior turn, printing one is paid again on every later turn. So the two tools get
    # opposite advice, which is also the split snipmin runs and the split run 4 produced
    # (0 of 31 llm_query calls were on a search-result list).
    ("- `context`: the important, potentially very long information related to the prompt "
     "(typically `str` or `list[str]`).",
     "- the corpus: reached with `search_index` and `get_document` (described below). It is "
     "far too large to read; you see only what those tools return. Search results are short "
     "snippets meant to be printed and read directly — printing them puts them in front of "
     "you. `get_document` returns a document's full text into a Python variable: pass that "
     "variable to `llm_query` with a specific question rather than printing it, whatever its "
     "length. Use `llm_query_batched` to read several documents in one turn."),

    # NOTE the example given here. An earlier version named "a fetched document" as the
    # archetypal long payload, which contradicted the corpus line above once the print/delegate
    # rule became conditional at ~10,000 characters — most documents are 5,030 characters and
    # are meant to be printed. The example is now a genuinely long payload.
    # NOTE the interpolation. _raise_output_ceiling already rewrote the authors' "~20K" to
    # "~{REPL_OUTPUT_CHARS//1000}K", so this anchor must follow the configured ceiling rather
    # than hard-code 60. With RLM_MAX_OUTPUT_CHARS=20000 the earlier patch produces "~20K" and
    # a literal "~60K" anchor here raised AssertionError at import.
    (f"REPL outputs over ~{REPL_OUTPUT_CHARS // 1000}K characters are truncated, so for "
     "longer payloads slice `context` and pass slices through `llm_query` rather than "
     "`print`-ing them whole.",
     f"REPL outputs over ~{REPL_OUTPUT_CHARS // 1000}K characters are truncated, so anything "
     "that large — a long document, or many results at once — should go through `llm_query` "
     "rather than being `print`-ed whole."),

    ("As a general strategy, you should start by probing your context to understand it better "
     "(e.g. print a few lines, count them, etc.). Then, use the REPL to build up an answer to "
     "the query.",
     "As a general strategy, you should start by searching for the most distinctive detail in "
     "the question and looking at what comes back. Then, use the REPL to build up an answer to "
     "the query."),

    ("Do not flip `answer[\"ready\"] = True` on turn 1 without first inspecting `context`.",
     "Do not flip `answer[\"ready\"] = True` on turn 1 without first retrieving evidence."),

    ("Directly after you probe the `context` and understand your task, pause and plan:",
     "Directly after your first search and once you understand your task, pause and plan:"),

    ("(Conversely: if a Python keyword / regex search over `context` would already pin the "
     "answer, or if a single visible passage already contains it, just read it directly — "
     "sub-LMs are for when the raw text won't fit or the question needs semantic "
     "interpretation.)",
     "(A substring or regex test over a document tells you a word is present, not what the "
     "document says: use it to locate, never to conclude. Read directly only when the passage "
     "you need is already printed in front of you.)"),

    ("Long REPL stdout pollutes history the same way raw `context` does:",
     "Long REPL stdout pollutes history the same way raw document text does:"),

    ("Sub-LLMs have no REPL; they only see the prompt and the `context` slice you pass them.",
     "Sub-LLMs have no REPL and cannot retrieve; they only see the prompt and the document "
     "text you pass them."),

    ("If the raw workload exceeds both budgets at once (e.g. a context far larger than "
     "~20 × 100K chars),",
     "If the raw workload exceeds both budgets at once (e.g. far more candidate documents "
     "than ~20 × 100K chars),"),
]


def _reanchor_prompt():
    import rlm.core.rlm as core
    import rlm.utils.prompts as PR

    sysp, add = core.RLM_SYSTEM_PROMPT, PR.ORCHESTRATOR_ADDENDUM
    for old, newtxt in _CTX_REWRITES:
        if old in sysp:
            sysp = sysp.replace(old, newtxt, 1)
        elif old in add:
            add = add.replace(old, newtxt, 1)
        else:
            raise AssertionError(f"prompt anchor not found, repo changed: {old[:70]!r}")
    core.RLM_SYSTEM_PROMPT = sysp
    PR.RLM_SYSTEM_PROMPT = sysp
    core.ORCHESTRATOR_ADDENDUM = add
    PR.ORCHESTRATOR_ADDENDUM = add


_reanchor_prompt()


# ── TWO MORE PROMPT FIXES ─────────────────────────────────────────────────────────────────
# (a) EVERY TURN MUST EXECUTE CODE. The authors' prompt says "Plan in prose, then execute one
#     ```repl``` block every turn" — an intent, not a prohibition — and run 8's q38 shows what
#     that permits: the agent found the answer on turn 7, then emitted 94 CONSECUTIVE TURNS OF
#     PROSE with no code block and no answer["ready"], until the 100-iteration cap stopped it.
#     Each empty turn is a full LM call over the whole accumulated history, so the question
#     cost 2,195,679 input tokens; snipmin answered it in 59,775. Across run 8, 152 of 936
#     turns (16%) contained no code block. dspy's _REPL_RULES opens with exactly the missing
#     sentence — "Every turn must execute code. Never output plain text with no code block." —
#     one of the five rules I had written off as unenforced. It is restored here.
# (b) THE "6." HEADING. build_rlm_system_prompt injects the tool list as "6. Custom tools and
#     data available in the REPL:", a numbered continuation of a list that does not exist in
#     this prompt — there are no items 1-5. Cosmetic, but it has been in every run tonight.
def _final_prompt_fixes():
    import rlm.core.rlm as core
    import rlm.utils.prompts as PR

    must_code = ("\n\nEvery turn must execute code: never reply with prose alone. Plan briefly, "
                 "then put the plan into action in a ```repl``` block in the SAME turn. If you "
                 "already have the answer, submit it — set answer[\"content\"] and "
                 "answer[\"ready\"] = True inside a ```repl``` block. Restating a plan you have "
                 "already carried out wastes a turn and changes nothing.")
    marker = ("Plan in prose, then execute one ```repl``` block every turn, get feedback from "
              "the output, then continue on the next turn.")
    assert marker in core.RLM_SYSTEM_PROMPT, "plan-in-prose sentence not found"
    core.RLM_SYSTEM_PROMPT = core.RLM_SYSTEM_PROMPT.replace(marker, marker + must_code, 1)
    PR.RLM_SYSTEM_PROMPT = core.RLM_SYSTEM_PROMPT

    orig = core.build_rlm_system_prompt

    def patched(*a, **kw):
        msgs = orig(*a, **kw)
        for m in msgs:
            if m["role"] == "system":
                m["content"] = m["content"].replace(
                    "\n6. Custom tools and data available in the REPL:",
                    "\nRetrieval tools available in the REPL:")
        return msgs

    core.build_rlm_system_prompt = patched


_final_prompt_fixes()


# ── WEAK-RESULT STOPPING RULE ─────────────────────────────────────────────────────────────
# snipmin's _REPL_RULES ends with a rule the authors' prompt has no equivalent of:
#     "Do not reason over weak results. If scores are low (below ~0.30) and snippets are
#      clearly off-topic, skip llm_query on those results — they are noise. Try a different
#      query next turn."
# Nothing in this arm's prompt ever says when a line of searching is finished, and the traces
# show the cost of that. Run 8's losses to snipmin: q53 ran 114 searches with gold_recall 1.00
# and answered Unknown, its last turns still generating phrase variants; q79 ran 100 searches
# and wrote on turn 9 "our earlier phrase-matching strategy hasn't surfaced the target, I'll
# pivot to broader, less literal queries" — the right diagnosis, made with no turns left.
# Run 8 issues 23 searches per question to snipmin's 9.
#
# PHRASED AS A CONDITION ON RESULTS, NOT A BUDGET. A numeric cap on searches does not work
# here and was measured: at a cap of 50, nine questions run 8 currently gets RIGHT exceed it
# (q1 needed 140 searches and was correct), against eleven it gets wrong. Search count cannot
# separate converging from thrashing. Score and novelty can be read per result, which is what
# snipmin's rule keys on, so that is what this keys on too.
# THE SCORE CLAIM WAS FALSE AND IS CORRECTED HERE. Verified against the live index: four
# unrelated queries — including deliberate nonsense — all return the identical score vector
# [1.0, 0.5, 0.333333, 0.25, 0.2, 0.166667], i.e. 1/rank rounded to 6dp, with different
# doc_ids each time. The payload carries only doc_id, score and text; no similarity value
# exists anywhere. The proxy is the RRF fusion server running one arm, and it emits
# reciprocal rank whether it is fusing or not.
# So "scores are cosine similarities, below ~0.30 is noise" described a fixed positional cut:
# at top_k=5 ranks 4 and 5 ALWAYS score 0.25 and 0.20, on every query ever run. (snipmin has
# carried the same false claim twice — in _REPL_TOOLS and _REPL_RULES — all project, and at
# its top_k=20 the threshold would condemn ranks 4 through 20.)
# HOW MUCH HARM IT DID, measured on run 16's trajectories: of 140 document fetches traced
# back to their result list, 18% came from ranks 4-5 — the model largely ignored the
# threshold, which is the same pattern as every other unenforced instruction tonight. So this
# is a correctness fix to the prompt and the write-up, not an expected accuracy gain.
# SAY NOTHING ABOUT THE SCORE — which is what LightOn does, and theirs is the highest-scoring
# arm measured here (.8500). Their tool description names the field and stops: "Returns top-5
# hits with docid, score, and snippet." No explanation, no threshold, no advice, anywhere in
# their prompt.
# Both of our attempts to explain it went wrong in different directions:
#   runs 12-16  asserted something FALSE ("cosine similarities, below ~0.30 is noise" — the
#               value is 1/rank, so that condemned ranks 4-5 on every query ever run)  -> .8400
#   run 17      asserted something TRUE ("score is 1/rank, position not relevance")     -> .8100
#               3 wins / 6 losses vs run 16, and on five of the six losses gold_recall was
#               1.00 in BOTH runs — the evidence was in hand and a different candidate won.
# A true statement that redirects a stochastic process can still come out net negative. The
# entity-change advice is kept, since it is about queries rather than scores.
_WEAK_RESULTS = (
    " If a search returns only documents you have already seen, or only snippets that are "
    "plainly about something else, the wording of the query is not the problem: change which "
    "entity, event or detail you are searching for, or commit to the best answer your "
    "evidence supports. Rewording the same hypothesis does not make the corpus contain it."
)


def _add_weak_result_rule():
    import rlm.core.rlm as core
    import rlm.utils.prompts as PR

    marker = ("As a general strategy, you should start by searching for the most distinctive "
              "detail in the question and looking at what comes back.")
    assert marker in core.RLM_SYSTEM_PROMPT, "weak-result anchor not found"
    core.RLM_SYSTEM_PROMPT = core.RLM_SYSTEM_PROMPT.replace(marker, marker + _WEAK_RESULTS, 1)
    PR.RLM_SYSTEM_PROMPT = core.RLM_SYSTEM_PROMPT


_add_weak_result_rule()


# ── TELLING IT WHAT THE RETRIEVER IS ──────────────────────────────────────────────────────
# The single largest identified failure mode, and no arm has ever been told about it. The
# index is Reason-ModernColBERT — dense, late-interaction, NO lexical or exact-phrase
# matching. Quotation marks in a query are just characters. The agent does not know that and
# writes exact-phrase queries constantly:
#     q79  198 searches, 87% containing a quoted phrase, gold_recall 0.00
#     q71  130 searches, 74% quoted,                     gold_recall 0.12
# snipmin solved both neighbourhoods with 24 and 15 semantic queries. Across the project 51%
# of agent queries have been quoted exact phrases fired at a retriever that cannot honour
# them. Unlike every stopping rule tried tonight, this does not tell the agent WHEN to stop —
# it tells it what the tool can and cannot do, which is a fact it was missing.
_RETRIEVER_LINE = (
    " The index is a dense semantic retriever: it matches meaning, not literal strings, and "
    "quotation marks in a query have NO effect — a quoted phrase is not looked up verbatim. "
    "Describe the thing you are looking for in your own words rather than quoting text you "
    "expect the document to contain."
)


def _add_retriever_line():
    import rlm.core.rlm as core
    import rlm.utils.prompts as PR

    # anchors on the weak-result sentence, whose opening changed when the false
    # "scores are cosine similarities" claim was corrected to "score is 1/rank".
    marker = "If a search returns only documents you have already seen"
    assert marker in core.RLM_SYSTEM_PROMPT, "retriever-line anchor not found"
    core.RLM_SYSTEM_PROMPT = core.RLM_SYSTEM_PROMPT.replace(
        marker, _RETRIEVER_LINE.strip() + " " + marker, 1)
    PR.RLM_SYSTEM_PROMPT = core.RLM_SYSTEM_PROMPT


# RETRIEVER_LINE=0 removes the two sentences above and NOTHING else. It is on by default,
# so every arm from run 15 onward is unaffected; run 34 is the single-variable ablation.
# WHY IT IS BEING ABLATED: the runaway-search behaviour on puzzle questions (an agent
# enumerating candidate entities six per iteration for 15 iterations) was hypothesised to come
# from this line telling it not to quote. The pre/post comparison at runs 13/14 -> 16 says
# otherwise — searching went DOWN when the line was added, 32-34 -> 24-26 mean — but those
# runs changed more than one thing. This isolates it.
if os.environ.get("RETRIEVER_LINE", "1").strip() not in ("0", "false", "False"):
    _add_retriever_line()


# ── WHAT A RECURSIVE SUB-CALL IS (DEPTH2_PROMPT=1) ────────────────────────────────────────
# Under ORCHESTRATOR=0 — our best config — the live prompt says exactly one thing about
# recursion (rlm/utils/prompts.py:133): "recursive RLM sub-calls. Fall back to llm_query /
# llm_query_batched when recursion is disabled." It never says the child gets the retrieval
# tools. That omission matters because the model's belief is CORRECT at max_depth=1 and FALSE
# at max_depth=2: llm_query genuinely cannot search, which is why every reader intervention we
# measured (effort high/medium/low, gpt-5 vs mini — all null) was re-reading text the root had
# already fetched. self.custom_sub_tools defaults to custom_tools and the child is built with
# custom_tools=self.custom_sub_tools (rlm.py:161, 827), so at depth 2 a child CAN search.
#
# This is a fact that corrects a false belief, which is the only category of prompt change
# that has moved anything in this project (the dense-retriever line: quoting roughly halves,
# ~45% -> ~22%, and queries get ~40% longer, 9.9 -> 14.0 words;
# "every turn must execute code": one question 2.20M -> 74K tokens). Strategy instructions
# have all been null. So it states the capability and stops short of telling it when to use it.
#
# Gated so it CANNOT contaminate a max_depth=1 run: the caller sets DEPTH2_PROMPT=1 only
# alongside --max-depth 2.
_DEPTH2_LINE = (
    " `rlm_query(prompt)` and `rlm_query_batched(prompts)` spawn CHILD AGENTS, not sub-LLM "
    "calls: each child gets its own REPL and its own `search_index` and `get_document`, so "
    "unlike `llm_query` a child can run its own searches and open its own documents. A child "
    "sees only the prompt string you pass it — it inherits none of your variables and none of "
    "your findings — so each sub-question must be self-contained. It returns its answer as a "
    "string."
)


def _add_depth2_line():
    import rlm.core.rlm as core
    import rlm.utils.prompts as PR

    marker = "If a search returns only documents you have already seen"
    assert marker in core.RLM_SYSTEM_PROMPT, "depth2-line anchor not found"
    core.RLM_SYSTEM_PROMPT = core.RLM_SYSTEM_PROMPT.replace(
        marker, _DEPTH2_LINE.strip() + " " + marker, 1)
    PR.RLM_SYSTEM_PROMPT = core.RLM_SYSTEM_PROMPT


if os.environ.get("DEPTH2_PROMPT", "").strip() not in ("", "0", "false", "False"):
    _add_depth2_line()


# ── WHAT A SNIPPET IS — a fact worth knowing, and an intervention that FAILED ─────────────
# VERIFIED: a snippet is the document's OPENING, not the passage that matched. rag truncation
# is tk.decode(ids[:n]), confirmed on 45 of 45 sampled results. For a long document that is
# under 5% of it — q20's gold value sits in a 79,893-char document, q70's in a 27,094-char one,
# both outside the first 2,000 characters.
#
# Telling the model so did NOT help. On those two questions — never solved by any arm, both
# failing by never opening the gold document — adding the fact to the prompt changed nothing:
# q70 kept get_document at 0, q20 went 0 -> 1, both still wrong. The patch is deleted rather
# than left commented out, because its anchor ("Each result's `score` is 1/rank") no longer
# exists in the prompt: the score sentence was removed in run 19, so re-enabling it would
# assert rather than run.
#
# It remains the clearest example of the pattern this project kept hitting: an instruction of
# the form "here is what you should do" changes nothing, while a fact that CONTRADICTS a false
# belief changes behaviour — the retriever line (quoting roughly halves, ~45% -> ~22%, and
# queries get ~40% longer) and the score correction (q64: 144 searches -> 23) both did.


# ── OPTIONAL: TRIAGE RULE FOR WIDE SEARCH BLOCKS (SEARCH_TRIAGE_RULE=1) ───────────────────
# The problem it targets: this arm batches searches — median 2 per block, p90 6, max 9 — and
# each search prints up to 5 x 2,000 = 10,000 characters, so a wide block overruns the 60,000
# ceiling. Measured on run 8: median search cell 14,010 chars, p90 68,085, and 13% of search
# cells truncated. Truncation is silent: the trajectory log keeps the full text, so the model
# loses the tail of a wide block with no signal.
#
# Two designs already have evidence, and this is a third:
#   snipmin  - "Never loop over search_index", ~1 search per block, never overruns. Deleting
#              that rule (v17) took searches-per-turn 1.04 -> 7.00 and cost 4 questions of 54.
#   v16      - delegate EVERY search's snippets, print titles only: identical accuracy on the
#              hard 29 (17/29 both), -21% tokens, but +33% turns.
#   this     - keep cheap blocks cheap, triage only WIDE ones. Untested by anyone.
# It is deliberately a threshold rather than a blanket rule, because v16 showed the blanket
# version buys cost and not accuracy, and because narrow blocks never approach the ceiling.
if os.environ.get("SEARCH_TRIAGE_RULE", "").strip() not in ("", "0", "false", "False"):

    def _add_triage_rule():
        import rlm.core.rlm as core
        import rlm.utils.prompts as PR

        anchor_txt = ("Search results are short snippets meant to be printed and read "
                      "directly — printing them puts them in front of you.")
        # PHRASED AS AN OPERATION, NOT A GOAL. An earlier wording ended "...to triage which
        # documents are worth opening", which names the purpose and so narrows what the model
        # asks the sub-LM for — a relevance verdict rather than the content. The sub-call is
        # more useful doing the actual reading: extraction, comparison, whatever the question
        # needs. What the model then does with the answer is its own decision.
        addition = (" If one block issues more than three searches, do not print all their "
                    "snippets — that overruns the output limit and the tail is silently cut. "
                    "In that case print only score, doc_id and the first line of each hit, and "
                    "pass the snippet text through `llm_query` to read it and extract what you "
                    "need.")
        assert anchor_txt in core.RLM_SYSTEM_PROMPT, "triage anchor not found"
        core.RLM_SYSTEM_PROMPT = core.RLM_SYSTEM_PROMPT.replace(
            anchor_txt, anchor_txt + addition, 1)
        PR.RLM_SYSTEM_PROMPT = core.RLM_SYSTEM_PROMPT

    _add_triage_rule()

PROXY = os.environ.get("EMBEDDING_SERVER_URL", "http://localhost:8024")

# ── RETRIEVAL BUDGET, MATCHED TO THE DSPY ARMS ────────────────────────────────────────────
# The first run of this port was NOT comparable to snipmin on retrieval, and that was my
# error. Two settings were left at this file's own defaults instead of the arm's:
#
#              first run (uncomparable)        snipmin            this run
#   snippet    SEARCH_SNIPPET_LEN=500 chars    512 Qwen tokens    512 Qwen tokens (~2,000 ch)
#   top_k      20                              10                 10
#
# Per search that is ~10,000 chars of text for the official port against snipmin's ~20,500 —
# a 2x difference in what the agent can see before deciding, uncontrolled. It shows up in the
# behaviour: official searched 25 times per question to snipmin's 9 and opened 8 documents to
# snipmin's 3, which is what an agent does when a snippet cannot settle relevance. It is also
# why the port looked cheaper: 500-char snippets are small.
#
# SEARCH_SNIPPET_TOKENS truncates with Qwen/Qwen3-0.6B, the tokenizer the BrowseComp+ harness
# itself uses (searcher/tools.py:39-41), so "512" means here exactly what --snippet-max-tokens
# 512 means to LightOn. Falls back to the old character cap when unset so the first run stays
# reproducible byte-for-byte.
SNIPPET_TOKENS = os.environ.get("SEARCH_SNIPPET_TOKENS", "")
SNIPPET_LEN = int(os.environ.get("SEARCH_SNIPPET_LEN", "500"))
DEFAULT_TOP_K = int(os.environ.get("OFFICIAL_TOP_K", "5"))


def _truncate(text: str) -> str:
    if SNIPPET_TOKENS:
        from rag.agent import _get_snippet_tokenizer

        tk = _get_snippet_tokenizer()
        ids = tk.encode(text, add_special_tokens=False)
        n = int(SNIPPET_TOKENS)
        if len(ids) > n:
            return tk.decode(ids[:n], skip_special_tokens=True)
        return text
    if len(text) > SNIPPET_LEN:
        return text[:SNIPPET_LEN] + "\u2026"
    return text


def make_tools(state: dict):
    """search_index / get_document bound to a per-question metrics dict."""

    def search_index(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        # TOP_K IS A DEFAULT, NOT A CAP — deliberately. The agent decides how many results it
        # wants; 5 is only where it starts if it says nothing. An earlier version of this file
        # clamped it, which would have measured our budget rather than the agent's judgement.
        # Overrides are still counted (topk_overrides / topk_requested) because they are the
        # interesting behaviour: over 737 calls in the previous run the model set top_k
        # explicitly on 34% of them, most often to 20 — the number the docstring showed it —
        # and up to 50. The docstring now shows 5, so those counters also measure how strongly
        # the documented default anchors what it asks for.
        t0 = time.perf_counter()
        if top_k != DEFAULT_TOP_K:
            state["topk_overrides"] += 1
            state["topk_requested"].append(int(top_k))
        r = requests.post(f"{PROXY}/search", json={"query": query, "top_k": top_k}, timeout=120)
        r.raise_for_status()
        results = r.json()["results"]
        for x in results:
            if x.get("text"):
                x["text"] = _truncate(x["text"])
        state["search_calls"] += 1
        state["retrieval_s"] += time.perf_counter() - t0
        state["retrieved"].update(str(x["doc_id"]) for x in results)
        return results

    def get_document(doc_id: str) -> dict:
        t0 = time.perf_counter()
        r = requests.post(f"{PROXY}/document", json={"doc_id": str(doc_id)}, timeout=120)
        r.raise_for_status()
        state["getdoc_calls"] += 1
        state["retrieval_s"] += time.perf_counter() - t0
        state["retrieved"].add(str(doc_id))
        return r.json()

    return search_index, get_document


# TWO TASK LAYERS, selected by LIGHTON_TASK_TEXT.
#   default  — ours: "EXACTLY this format", the delegation paragraph telling it to pass
#              get_document text to llm_query, and the "Exact Answer: Unknown" escape.
#   =1       — LightOn's QUERY_TEMPLATE verbatim (bcp/search_agent/prompts.py), which has no
#              strategy, no delegation advice and NO Unknown escape, plus the answer["ready"]
#              mechanic this loop needs. Their run scores .85-.86 with nothing else.
_TASK_OURS = """

Answer the question using search_index and get_document over the corpus.

Set answer["content"] to a string in EXACTLY this format, then answer["ready"] = True:

  Explanation: <your explanation. Cite evidence documents inline by enclosing their
  doc_ids in square brackets at the end of sentences, for example [94480].>
  Exact Answer: <the exact name, date, title, or value the question asks for>
  Confidence: <your confidence score between 0% and 100%>

get_document returns the text in a Python variable, not into your context — you have not
read a document until you have actually looked at its text. Pass that variable to llm_query
with a specific question about it, whatever its length, rather than printing it or testing
it for a keyword: an excerpt or a substring check tells you a word is present, not what the
document says. Use llm_query_batched to read several documents in one turn.

If the answer cannot be found after exhausting all search angles, give Exact Answer: Unknown.
"""

_TASK_LIGHTON = """

You are a deep research agent. You need to answer the given question by interacting with a
search engine, using the search_index and get_document tools provided. Please perform
reasoning and use the tools step by step, in an interleaved manner. You may use the
search_index and get_document tools multiple times.

Your response should be in the following format. Set answer["content"] to that string, then
set answer["ready"] = True:

  Explanation: {your explanation for your final answer. For this explanation section only,
  you should cite your evidence documents inline by enclosing their doc_ids in square
  brackets [] at the end of sentences. For example, [94480].}
  Exact Answer: {the exact name, date, title, or value the question asks for}
  Confidence: {your confidence score between 0% and 100% for your answer}
"""

_LIGHTON_TASK = os.environ.get("LIGHTON_TASK_TEXT", "").strip() not in ("", "0", "false", "False")
TASK_SUFFIX = _TASK_LIGHTON if _LIGHTON_TASK else _TASK_OURS

# NO_UNKNOWN=1 strips only the abstention escape, leaving everything else in our task layer.
# LightOn never offers it and scores highest; 12 of run 8's 18 failures were give-ups, SEVEN
# of them holding the gold document (gold_recall 1.00). The risk is symmetric and must be
# read from the outcome, not assumed: abstentions may convert to correct answers, or to
# CONFIDENT WRONG ANSWERS, which score the same as an abstention but are worse to ship. The
# diagnostic is the give-up count — if it falls to ~0 and accuracy does not move, the second
# thing happened.
_UNKNOWN_LINE = ("\nIf the answer cannot be found after exhausting all search angles, "
                 "give Exact Answer: Unknown.\n")
if os.environ.get("NO_UNKNOWN", "").strip() not in ("", "0", "false", "False"):
    assert _UNKNOWN_LINE in TASK_SUFFIX or _LIGHTON_TASK, "Unknown-escape anchor not found"
    TASK_SUFFIX = TASK_SUFFIX.replace(_UNKNOWN_LINE, "")


def _tool_specs(search_index, get_document):
    """Tool descriptions: ours by default, LightOn's verbatim under LIGHTON_TOOL_DOCS=1.

    LightOn's are bcp/searcher/searchers/base.py:57-67 — the exact strings their MCP server
    registers, behind the 87.59 run. Note what is absent from them: snippet size, and any
    advice on when to search versus when to read. Nine words for get_document.
    """
    if os.environ.get("LIGHTON_TOOL_DOCS", "").strip() not in ("", "0", "false", "False"):
        return {
            "search_index": {"tool": search_index,
                             # LightOn's wording, with ONE addition: the snippet size.
                             # Run 13 had no number anywhere in its prompt and q94 cost it 201
                             # searches at gold_recall 0.00, where run 14 — which is told
                             # "~2000 characters" — needed 20 searches and reached 1.00.
                             "description": f"Perform a search on a knowledge source. Returns "
                                            f"top-{DEFAULT_TOP_K} hits with doc_id, score, and "
                                            f"text. The text contains the document's contents, "
                                            f"up to ~2000 characters (512 tokens) per hit."},
            "get_document": {"tool": get_document,
                             "description": "Retrieve a full document by its doc_id."},
        }
    return {
        "search_index": {"tool": search_index,
                         "description": "Search the document corpus. "
                                        "search_index(query: str, top_k: int = 5) -> "
                                        "list[{score, doc_id, text}]. text is a snippet of up "
                                        "to ~2000 characters (512 tokens) of the document's "
                                        "contents."},
        "get_document": {"tool": get_document,
                         "description": "get_document(doc_id: str) -> {doc_id, text}. "
                                        "Full text of one document. Can be very long."},
    }


# OBLIQ_NO_GETDOC=1 removes get_document from the tool surface entirely. A NO-OP unless set,
# so every existing run stays byte-identical.
#
# WHY. On OBLIQ-Twitter a "document" is a tweet: measured over 5,000 corpus entries the mean
# is 209 chars, median 219, MAX 868 -- and a 512-token snippet is ~2,000 chars. So ZERO of
# 5,000 documents are longer than the snippet search already returns. get_document cannot
# reveal text the agent does not already have, and it cannot surface a document search did
# not return, because you need a doc_id to call it.
#
# MEASURED COST of keeping it: 22.1 calls/question, and the prompt actively instructs their
# use ("pass that variable to llm_query with a specific question"), which also drives 7.1
# llm_query calls. corr(get_document_calls, gold_recall) = -0.071. Our harness takes 1308 s
# per question against the supervisor's 1026 s on a search-only surface, at an NDCG
# difference of +0.0054 (not significant).
#
# The matching prompt edit is REQUIRED, not cosmetic: REPL_TOOLS names get_document twice and
# tells the agent to use it, so removing the tool without the text would leave the agent
# calling a function that no longer exists.
def _strip_get_document(specs: dict) -> dict:
    if os.environ.get("OBLIQ_NO_GETDOC", "") != "1":
        return specs
    out = {k: v for k, v in specs.items() if k != "get_document"}
    import rlm.core.rlm as _core
    import rlm.utils.prompts as _PR
    _txt = _core.RLM_SYSTEM_PROMPT
    _old = ("- the corpus: reached with `search_index` and `get_document` (described below). "
            "It is far too large to read; you see only what those tools return. Search results "
            "are short snippets meant to be printed and read directly \u2014 printing them puts "
            "them in front of you. `get_document` returns a document's full text into a Python "
            "variable: pass that variable to `llm_query` with a specific question rather than "
            "printing it, whatever its length. Use `llm_query_batched` to read several "
            "documents in one turn.")
    _new = ("- the corpus: reached with `search_index` (described below). It is far too large to "
            "read; you see only what that tool returns. Search results are short snippets meant "
            "to be printed and read directly \u2014 printing them puts them in front of you, and "
            "on this corpus a snippet already contains the document in full.")
    import re as _re
    # Anchor on the distinctive opening so a reword upstream fails loudly rather than silently.
    _anchor = "- the corpus: reached with `search_index` and `get_document`"
    if _anchor not in _txt:
        raise SystemExit("OBLIQ_NO_GETDOC: REPL_TOOLS anchor not found; re-derive the edit")
    _start = _txt.index(_anchor)
    _end = _txt.index("- `llm_query(", _start)
    _core.RLM_SYSTEM_PROMPT = _txt[:_start] + _new + "\n" + _txt[_end:]
    _PR.RLM_SYSTEM_PROMPT = _core.RLM_SYSTEM_PROMPT
    print("[no-getdoc] get_document removed from tools and from REPL_TOOLS", flush=True)
    return out


def run_one(row: dict, q_idx: int, args) -> dict:
    question = get_question(row)
    gold = get_gold_answer(row)
    gold_ids = get_gold_doc_ids(row)
    evid_ids = get_evidence_doc_ids(row)

    state = {"search_calls": 0, "getdoc_calls": 0, "retrieval_s": 0.0,
             "retrieved": set(), "iterations": 0, "subcalls": 0,
             "in_tok": 0, "out_tok": 0, "cost": None, "lm_calls": 0,
             "topk_overrides": 0, "topk_requested": []}
    search_index, get_document = make_tools(state)
    _LLMQ.update(n=0, batched=0, prompts=0)
    _REJECTS["n"] = 0

    # NOTE ON CALLBACK SIGNATURES. rlm/core/rlm.py:114 documents
    #   on_iteration_complete(depth, iteration_num, duration)
    #   on_subcall_complete(depth, model, duration, error_or_none)
    # The FIRST argument is DEPTH, not the iteration index — the smoke test recorded
    # iters=0 because I read arg 0 as the iteration number. iteration_num is arg 1.
    def on_iter_complete(depth, iteration_num, duration):
        state["iterations"] = max(state["iterations"], int(iteration_num) + 1)

    def on_sub_complete(depth, model, duration, error):
        state["subcalls"] += 1

    # TRAJECTORY LOGGING. Without a log_dir the run produces only the one-line summary this
    # script prints, which is why every failure analysis so far has been done from final
    # answers rather than from what the agent actually did turn by turn. RLMLogger writes one
    # .jsonl per completion holding run config, every iteration, every code block and its
    # stdout, and every sub-LM call — the same files visualizer/ reads (Next.js, npm run dev,
    # localhost:3001). They are also plain JSONL, so they can be read directly without Node.
    logger = RLMLogger(log_dir=os.environ.get("RLM_TRAJ_DIR", "logs/rlm_traj"))
    # Children spawned by rlm_query inherit this exact config (see _d2_init). Azure quota
    # caps CONCURRENT calls, not calls/min, and rlm_query_batched fans out
    # max_concurrent_subcalls children at once — each running its own loop. Default 4 in
    # the library; 2 here, because 429s are recorded as wrong answers, not as retries.
    _D2_CFG.update(
        orchestrator=os.environ.get("ORCHESTRATOR", "1").strip() not in ("0", "false", "False"),
        compaction=args.compaction,
        sampling_args={"reasoning_effort": os.environ.get("AZURE_OPENAI_REASONING_EFFORT", "high")},
        sub_sampling_args={"reasoning_effort": os.environ.get("SUB_LM_REASONING_EFFORT", "medium")},
        max_concurrent_subcalls=int(os.environ.get("MAX_CONCURRENT_SUBCALLS", "2")),
    )
    for _k in _D2:
        _D2[_k] = 0
    # LOCAL_LLM_BASE_URL routes the root model (and llm_query, which inherits the root
    # client) to a local OpenAI-compatible server instead of Azure — the vLLM instance
    # serving Qwen3.8-27B int4 W4A16 on :8030. UNSET IS THE DEFAULT, so every existing
    # run reproduces unchanged; this is a switch, not a migration.
    #
    # REASONING EFFORT IS NOT PORTABLE. Azure gpt-5 takes low/medium/high; Qwen3.8 takes
    # low/medium/xhigh and returns 400 BadRequest on "high" — which is this harness's
    # own default (AZURE_OPENAI_REASONING_EFFORT=high). Set it explicitly when going local.
    _local_url = os.environ.get("LOCAL_LLM_BASE_URL", "").strip()
    if _local_url:
        _backend = "vllm"
        _backend_kwargs = {
            "base_url": _local_url,
            "api_key": os.environ.get("LOCAL_LLM_API_KEY", "EMPTY"),
            "model_name": os.environ.get("LOCAL_LLM_MODEL", "qwen3.8-27b"),
        }
    else:
        _backend = "azure_openai"
        _backend_kwargs = {
            "api_key": os.environ["AZURE_OPENAI_API_KEY"],
            "azure_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/"),
            "api_version": os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
            "model_name": os.environ["AZURE_OPENAI_MODEL"],
            "azure_deployment": os.environ["AZURE_OPENAI_MODEL"],
        }

    rlm = RLM(
        max_concurrent_subcalls=int(os.environ.get("MAX_CONCURRENT_SUBCALLS", "2")),
        logger=logger,
        backend=_backend,
        backend_kwargs=_backend_kwargs,
        environment="local",
        max_iterations=args.max_iterations,
        max_depth=args.max_depth,
        # ORCHESTRATOR_ADDENDUM, on by default in their library and on in every run here.
        # ~2,400 characters — about 40% of the system message and its largest single block —
        # entirely the authors' text, and NEVER ablated. ORCHESTRATOR=0 turns it off.
        # Worth testing because the highest-scoring arm measured in this project, hosted MCP
        # at .8500, has no orchestration guidance whatsoever: a four-sentence task prompt and
        # nine-word tool descriptions.
        orchestrator=os.environ.get("ORCHESTRATOR", "1").strip() not in ("0", "false", "False"),
        compaction=args.compaction,
        custom_tools=_strip_get_document(_tool_specs(search_index, get_document)),
        on_iteration_complete=on_iter_complete,
        on_subcall_complete=on_sub_complete,
        verbose=args.verbose,
        sampling_args={"reasoning_effort": os.environ.get("AZURE_OPENAI_REASONING_EFFORT", "high")},
        # SUB-LM EFFORT. Until now llm_query resolved to the SAME client as the root — no
        # `model` is ever passed, no other_backends were configured, so get_client fell
        # through to default_client and every sub-call ran gpt-5 at reasoning_effort=high.
        # snipmin has always run its reader at MEDIUM (SUB_LM_REASONING_EFFORT=medium).
        # High effort on "pull the name and date out of this document" is reasoning budget
        # spent on a reading task. sub_sampling_args is their supported way to do this: it
        # mirrors the root backend into a depth=1 client with its own sampling args
        # (rlm/core/rlm.py:128-138).
        # EXPECTED: cost and latency only. v12 measured the sub-LM MODEL upgrade to gpt-5 at
        # +1.3pp, p=1.00 — null — so the reader's capability is not what limits this task.
        sub_sampling_args={"reasoning_effort": os.environ.get("SUB_LM_REASONING_EFFORT", "medium")},
        # SUB_LM_MODEL routes llm_query (depth=1) to a DIFFERENT deployment from the root.
        # Without it, get_client falls through to default_client and sub-calls run the root
        # model (rlm/core/lm_handler.py:176-191). Passing other_backends explicitly creates a
        # depth-1 client; rlm/core/rlm.py:128-138 then merges sub_sampling_args into its kwargs,
        # so effort and model are set independently.
        # gpt-5.4-mini verified reachable on this endpoint and it DOES honour reasoning_effort
        # (109 / 71 / 244 reasoning tokens at low / medium / high on a probe task).
        **({"other_backends": ["azure_openai"],
            "other_backend_kwargs": [{
                "api_key": os.environ["AZURE_OPENAI_API_KEY"],
                "azure_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/"),
                "api_version": os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
                "model_name": os.environ["SUB_LM_MODEL"],
                "azure_deployment": os.environ["SUB_LM_MODEL"],
            }]} if os.environ.get("SUB_LM_MODEL") else {}),
    )

    t0 = time.perf_counter()
    err = None
    try:
        # PROMPT vs ROOT_PROMPT — these are not interchangeable and I had it wrong.
        #   prompt      -> becomes the REPL variable `context`. It is where the long document
        #                  goes in their setting. It is NOT shown to the model.
        #   root_prompt -> goes into the user message as "Answer the following: {root_prompt}"
        #                  (rlm/utils/prompts.py:237-244). This is the question.
        # The first run of this port passed question+TASK_SUFFIX as `prompt` only, so the
        # model's entire user message was "Your context is a str of N total characters. Each
        # sub-LLM call can handle roughly ~100k tokens at once." — the question and the whole
        # answer-format contract were reachable only by printing `context` in the REPL, and
        # turn 1 carries their safeguard "you have not interacted with the REPL environment or
        # seen your prompt / context yet ... do not provide a final answer yet." That is the
        # likeliest reason the format contract was followed on only 93/100 questions: it sat
        # in REPL history, not in the message history the model actually writes its answer from.
        result = rlm.completion(question + TASK_SUFFIX, root_prompt=question + TASK_SUFFIX)
        response = getattr(result, "response", "") or ""
        # UsageSummary aggregates across models (root + sub-LM), so this is the FULL bill,
        # the same scope as our dspy runs' track_usage() sum. rlm/core/types.py:86,91.
        us = getattr(result, "usage_summary", None)
        if us is not None:
            state["in_tok"] = int(getattr(us, "total_input_tokens", 0) or 0)
            state["out_tok"] = int(getattr(us, "total_output_tokens", 0) or 0)
            state["cost"] = getattr(us, "total_cost", None)
            calls = sum(int(m.total_calls or 0) for m in us.model_usage_summaries.values())
            state["lm_calls"] = calls
    except Exception as exc:  # noqa: BLE001
        response, err = "ERROR", f"{type(exc).__name__}: {exc}"
        # RLM_DEBUG_TB=1 prints the full traceback. The message alone ("expected string
        # or bytes-like object, got 'NoneType'") does not say WHICH call returned None,
        # which matters when swapping in a model whose response shape differs.
        if os.environ.get("RLM_DEBUG_TB") == "1":
            import traceback
            print("---- RLM_DEBUG_TB ----", flush=True)
            traceback.print_exc()
            print("---- end ----", flush=True)
    latency = time.perf_counter() - t0
    try:
        rlm.close()
    except Exception:
        pass

    correct = False if err else judge_match(question, response, gold)
    got = state["retrieved"]
    rec = {
        "q_idx": q_idx,
        "question": question,
        "gold": gold,
        "predicted": response,
        "correct": bool(correct),
        # ITERATIONS = ROOT TURNS. on_iteration_complete never fires regardless of its
        # documented signature, so this is reconstructed from the usage summary. The earlier
        # version subtracted state["subcalls"], which is fed by on_subcall_complete — and that
        # callback fires ONLY for recursive child RLMs (rlm/core/rlm.py:865), never for
        # llm_query, so at max_depth=1 it is always 0. The field therefore reported
        # root turns PLUS every llm_query call. Run 10 q1 recorded "iterations": 265 with 238
        # llm_query calls: 27 real turns. Cross-arm turn comparisons made with the old field
        # were apples-to-oranges, since dspy's iterations counts turns only.
        # llm_query_calls counts single calls plus batched PROMPTS, and each prompt is one LM
        # call in the usage summary, so this subtraction is exact.
        "iterations": max(0, state["lm_calls"] - (_LLMQ["n"] + _LLMQ["prompts"])),
        "lm_calls_total": state["lm_calls"],
        "search_calls": state["search_calls"],
        "get_document_calls": state["getdoc_calls"],
        "llm_query_calls": _LLMQ["n"] + _LLMQ["prompts"],
        "llm_query_single": _LLMQ["n"],
        "llm_query_batched_calls": _LLMQ["batched"],
        "llm_query_batched_prompts": _LLMQ["prompts"],
        "recursive_subcalls": state["subcalls"],
        "format_rejections": _REJECTS["n"],
        "topk_overrides": state["topk_overrides"],
        "topk_requested": state["topk_requested"][:50],
        "unique_docs": len(got),
        "latency_s": round(latency, 1),
        "retrieval_latency_s": round(state["retrieval_s"], 1),
        "gold_recall": (len(gold_ids & got) / len(gold_ids)) if gold_ids else 0.0,
        "evid_recall": (len(evid_ids & got) / len(evid_ids)) if evid_ids else 0.0,
        "input_tokens": state["in_tok"],
        "output_tokens": state["out_tok"],
        # Child RLM usage is NOT in usage_summary (its handler is separate). These are the
        # hand-rolled totals; total_input_tokens is the honest number at max_depth>1.
        "child_input_tokens": _D2["child_in_tok"],
        "child_output_tokens": _D2["child_out_tok"],
        "total_input_tokens": state["in_tok"] + _D2["child_in_tok"],
        "child_rlms": _D2["children"],
        "child_turns": _D2["child_turns"],
        "lm_calls": state["lm_calls"],
        "cost_usd": state["cost"],
        "error": err,
    }
    mark = "✓" if correct else "✗"
    print(f"[Q{q_idx + 1:03d}] [{mark}] iters={rec['iterations']} search={rec['search_calls']} "
          f"getdoc={rec['get_document_calls']} llmq={rec['llm_query_calls']} "
          f"lat={rec['latency_s']:.0f}s tok={rec['input_tokens']:,}/{rec['output_tokens']:,} "
          f"goldR={rec['gold_recall']:.2f} "
          f"pred={response[:70]!r}", flush=True)
    return rec


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=100)
    p.add_argument("--max-iterations", type=int, default=100)
    p.add_argument("--max-depth", type=int, default=1)
    p.add_argument("--compaction", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from datasets import load_dataset
    ds = load_dataset("Tevatron/browsecomp-plus", split="test")

    results = []
    for q_idx in range(args.start, args.end):
        results.append(run_one(ds[q_idx], q_idx, args))
        if (len(results)) % 2 == 0:
            Path(f"{args.out}_checkpoint.json").write_text(json.dumps(results, indent=2))
    Path(f"{args.out}.json").write_text(json.dumps(results, indent=2))
    ok = sum(1 for r in results if r["correct"])
    print(f"\n{args.out}: {ok}/{len(results)} = {ok / max(1, len(results)):.4f}")


if __name__ == "__main__":
    main()
