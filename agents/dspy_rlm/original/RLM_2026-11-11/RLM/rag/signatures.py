"""DSPy signatures for RAG RLM — BrowseComp+ and OBLIQ-Bench.

All signatures share the same REPL tools + rules block.
Task-specific strategy blocks are injected via make_rlm_signature().

Best performers per task (NDCG@10 gold, full run):
  BrowseComp+   : BrowseCompSignature          — 69.0% accuracy (573/830)
  Math          : ObliqMathSig        (v10)    — 0.132
  Congress      : ObliqCongressSig    (v2)     — 0.139
  Twitter       : ObliqTwitterSig     (v1)     — 0.0945
  WildChat      : ObliqWildchatV5Sig  (nested) — 0.0504
  Writing       : ObliqWritingSig     (v5)     — not fully evaluated yet
"""

import os

import dspy

# ─── Shared blocks (identical for every task) ─────────────────────────────────

_REPL_TOOLS = """\
You are a Recursive Language Model (RLM) with access to a document corpus via \
retrieval tools. You interact with a Python REPL iteratively — each turn you \
write a code block, see its output, then decide what to do next. State persists \
across turns.

════════════════════════════════════════════════════════════
TOOLS AVAILABLE IN THE REPL
════════════════════════════════════════════════════════════

  search_index(query: str, top_k: int = 20) -> list[dict]
      Search the corpus. Returns [{score, doc_id, text (≤2000 chars snippet)}, ...].
      Scores are cosine similarities — higher is more relevant.

  get_document(doc_id: str) -> dict
      Fetch the FULL text of one document. Use only when the snippet is promising
      but truncated. Never batch-fetch documents in a loop — fetch one, verify, proceed.

  llm_query(prompt: str) -> str
      Single sub-LLM call (~500K char capacity). Use for extraction, verification,
      or Q&A over a document you already have. Fast and cheap.

  llm_query_batched(prompts: list[str]) -> list[str]
      Run multiple llm_query calls concurrently. Same order out as in.
      Use fat prompts (~100K chars each), small batches (≤20). Avoid hundreds of
      tiny prompts — pack as much per call as possible.

  SUBMIT(answer=<str>)
      Submit final answer and terminate. Call when confident.

  print()
      The ONLY way to see output. Always wrap inspections in print(). Bare
      expressions on the last line are silently discarded.
"""


_REPL_RULES = """\
════════════════════════════════════════════════════════════
REPL RULES
════════════════════════════════════════════════════════════

**Every turn must execute code.** Never output plain text with no code block.
Planning in prose does nothing — write the code immediately.

**Never loop over search_index.** Call search_index once per code block, not in a
for-loop. One query → see results → decide next query in the next turn.

**Search before reasoning.** You have no documents until you call search_index.
Always retrieve first, then use llm_query to reason over what you retrieved.

**Check snippets before fetching full documents.**
Read the 2000-char snippet first. If it is clearly off-topic, do NOT call get_document.
Only fetch the full document when the snippet strongly suggests relevance.

**Do not reason over weak results.**
If scores are low (below ~0.30) and snippets are clearly off-topic, skip llm_query on
those results — they are noise. Try a different query next turn.
"""

_ORCHESTRATION = """\
════════════════════════════════════════════════════════════
ORCHESTRATION PRINCIPLES
════════════════════════════════════════════════════════════

Push every long-context operation into llm_query / llm_query_batched. Do not print
huge document texts into the REPL — your own context window is small and REPL stdout
pollutes history. If you want a summary, ask llm_query for a 1-2 sentence recap.

Use llm_query_batched with fat prompts (one whole document per prompt, ≤20 prompts per
batch) rather than many tiny single-sentence prompts.

Reserve your own tokens for high-level decisions: what to search next, how to combine
results, when to commit. Delegate everything else to tools.
"""

# Shared tools block for inner RLMs — no get_document, shorter snippets
_REPL_TOOLS_NO_GETDOC = """\
You are a Recursive Language Model (RLM) with access to a document corpus via \
retrieval tools. You interact with a Python REPL iteratively — each turn you \
write a code block, see its output, then decide what to do next. State persists \
across turns.

════════════════════════════════════════════════════════════
TOOLS AVAILABLE IN THE REPL
════════════════════════════════════════════════════════════

  search_index(query: str, top_k: int = 20) -> list[dict]
      Search the corpus. Returns [{score, doc_id, text (≤500 chars snippet)}, ...].
      Scores are cosine similarities — higher is more relevant.

  llm_query(prompt: str) -> str
      Single sub-LLM call (~500K char capacity). Use to analyze search results,
      judge relevance, and plan the next search angle.

  llm_query_batched(prompts: list[str]) -> list[str]
      Run multiple llm_query calls concurrently. Same order out as in.

  SUBMIT(answer=<str>)
      Submit final answer and terminate. Call when confident.

  print()
      The ONLY way to see output. Always wrap inspections in print().
"""

_REPL_RULES_NO_GETDOC = """\
════════════════════════════════════════════════════════════
REPL RULES
════════════════════════════════════════════════════════════

**Every turn must execute code.** Never output plain text with no code block.
Planning in prose does nothing — write the code immediately.

**Never loop over search_index.** Call search_index once per code block, not in a
for-loop. One query → see results → decide next query in the next turn.

**Always route results through llm_query.** Never print raw search snippets.
Pass them to llm_query for analysis and print only the compact response.

**Do not reason over weak results.**
If scores are low (below ~0.25) and snippets are clearly off-topic, try a different
query next turn.
"""

# ─── Factory ──────────────────────────────────────────────────────────────────

def make_rlm_signature(
    strategy: str,
    answer_desc: str,
    question_desc: str = "The question to answer.",
    repl_tools: str | None = None,
    repl_rules: str | None = None,
) -> type[dspy.Signature]:
    """Return a dspy.Signature class: shared REPL block + task strategy."""
    tools_block = repl_tools if repl_tools is not None else _REPL_TOOLS
    rules_block = repl_rules if repl_rules is not None else _REPL_RULES
    prompt = tools_block + rules_block + strategy + _ORCHESTRATION

    class _Sig(dspy.Signature):
        __doc__ = prompt
        question: str = dspy.InputField(desc=question_desc)
        answer: str = dspy.OutputField(desc=answer_desc)

    # Named at the call site below; without this every run logs "Sig: _Sig" and
    # the logs cannot say which signature produced them.
    return _Sig


_MULTI_DOC_ANSWER = (
    "Newline-separated list of doc_ids ranked from strongest to weakest match. "
    "Output ONLY doc_ids, one per line, no prose."
)

# ─── BrowseComp+ ──────────────────────────────────────────────────────────────
# BEST: 69.0% accuracy (573/830), gpt-5 + IVFSQ8

_BROWSECOMP_STRATEGY = """\
════════════════════════════════════════════════════════════
RETRIEVAL STRATEGY — READ CAREFULLY
════════════════════════════════════════════════════════════

**Search for named entities — NOT clue descriptions.**
Gold documents contain actual entity names. They do NOT contain your clue phrasing.
Infer the specific named entity each clue points to, then search for that name.

  BAD:  search_index("person born 1886 mistaken for shaman trip 1915")
  GOOD: search_index("We'wha Zuni Washington DC 1886")

  BAD:  search_index("learning institution 2002 three-day event graduation 2003")
  GOOD: search_index("Queen Arwa University Yemen 2002")

When you cannot infer the entity name, search for the specific event or fact instead —
the gold document will mention it by name.

**When a promising document is found, extract immediately.**
Do not keep searching while ignoring a relevant document. Fetch it and verify ALL
clues from the question at once — not just the one you searched for:

  doc = get_document(doc_id)
  verdict = llm_query(
      f"Answer each of these questions about the document:\\n"
      f"1. [clue 1 from question]\\n"
      f"2. [clue 2 from question]\\n"
      f"3. [all other clues]\\n\\nDocument:\\n{doc['text']}"
  )
  print(verdict)

If the verdict confirms most clues, commit. Do not anchor on a candidate and keep
searching endlessly to confirm — if multiple clues match, that IS your answer.

**When a document mentions multiple similar values, identify WHICH one before extracting.**
If the document lists multiple people, dates, or values of the same type:
- First resolve WHICH entity the question is asking about (not just the first one listed).
- Never extract the first matching value — verify it belongs to the right entity/event.

**After turn 12, commit your best inference.**
If you have not found a confirmed answer by turn 12 out of 16, stop searching and commit
your best-supported candidate. Do not let the rollout exhaust itself unsubmitted.


════════════════════════════════════════════════════════════
FINAL ANSWER
════════════════════════════════════════════════════════════

Call SUBMIT(answer=...) with a concise string: the exact name, date, title, or value
the question asks for. No explanation, no hedging. If genuinely impossible after
exhausting all strategies, call SUBMIT(answer="Unknown").
"""

BrowseCompSignature = make_rlm_signature(
    strategy=_BROWSECOMP_STRATEGY,
    answer_desc="Concise final answer (name, date, title, etc.). 'Unknown' if not found.",
    question_desc="The research question to answer.",
)

# v2 — Minimal free-form strategy inspired by paper Appendix G.
# No prescribed steps: the model decides what to search, when to fetch, when to submit.
_BROWSECOMP_MINIMAL_STRATEGY = """\
════════════════════════════════════════════════════════════
YOUR TASK
════════════════════════════════════════════════════════════

You are a deep research agent. Answer the question by interacting with the corpus
using search_index and get_document. Reason step by step and use the tools in an
interleaved manner — search, read, decide what to search next. You may search as
many times as needed.

Call SUBMIT(answer=...) with your succinct final answer: the exact name, date,
title, or value the question asks for. No explanation.
If the answer cannot be found after exhausting all search angles, call SUBMIT(answer="Unknown").
"""

# v3 = v2 minimal + an explicit budget and a soft-commit deadline.
#
# Why: v2 tells the agent "you may search as many times as needed" and never
# mentions a turn budget, so 24% of questions ran into the max_iters ceiling and
# were truncated mid-rollout, scoring 20.6% against 92.3% for questions that
# finished cleanly. v1 has a soft-commit line and is truncated only 12.9% of the
# time — but v1 also scores only 83.2% on the questions it does finish, so we add
# ONLY the budget line, not the rest of v1's 2.3k-character strategy.
#
# The turn numbers are injected from the harness's actual --max-iters rather than
# written into the prose. v1 hardcodes "turn 12 out of 16", which was simply false
# in the shards that ran with max_iters=25: the prompt and the harness disagreed
# and nothing detected it.
_BROWSECOMP_MINIMAL_BUDGET_STRATEGY = """\
════════════════════════════════════════════════════════════
YOUR TASK
════════════════════════════════════════════════════════════

You are a deep research agent. Answer the question by interacting with the corpus
using search_index and get_document. Reason step by step and use the tools in an
interleaved manner — search, read, decide what to search next.

**You have {max_iters} turns and {max_search} search calls. This is a hard limit.**
If the rollout ends before you submit, the answer counts as WRONG — an unsubmitted
rollout scores zero, so never let the budget run out while you are still searching.

**After turn {commit_turn}, commit.**
If you do not have a confirmed answer by turn {commit_turn} of {max_iters}, stop
searching and SUBMIT your best-supported candidate anyway. A considered guess from
partial evidence beats no answer at all. Only submit "Unknown" if you have no
candidate whatsoever.

Call SUBMIT(answer=...) with your succinct final answer: the exact name, date,
title, or value the question asks for. No explanation.
"""


# v4 = v2 minimal + ONLY the scoring rule. No turn numbers, no deadline.
#
# Why: v3 bundled four changes and won big on truncated questions (15.6% -> 34.4%,
# p=0.0005) but drags slightly everywhere else, and it spends MORE turns almost
# everywhere (+0.5 to +1.7 iterations on non-capped questions). Two facts say the
# turn numbers are the suspect half:
#   1. dspy.RLM already feeds the agent an `iteration` field reading "13/16" every
#      turn, so "You have 16 turns" is redundant information.
#   2. A deadline at turn 12 can only FIRE on 9.7% of non-capped questions, yet the
#      drag shows up in the 2-6 iteration band too — so the harm comes from the text
#      being present, not from the deadline triggering.
# v4 therefore keeps only the sentences that address the observed failure (68% of
# truncated questions submit "Unknown" for a guaranteed zero) and drops the budget
# framing that plausibly raises time pressure on easy questions.
_BROWSECOMP_MINIMAL_SCORING_STRATEGY = """\
════════════════════════════════════════════════════════════
YOUR TASK
════════════════════════════════════════════════════════════

You are a deep research agent. Answer the question by interacting with the corpus
using search_index and get_document. Reason step by step and use the tools in an
interleaved manner — search, read, decide what to search next.

**An unsubmitted or empty answer is scored WRONG.**
A considered guess from partial evidence is worth more than no answer: if you have
a plausible candidate but cannot fully confirm it, SUBMIT the candidate. Only submit
"Unknown" if you have no candidate whatsoever.

Call SUBMIT(answer=...) with your succinct final answer: the exact name, date,
title, or value the question asks for. No explanation.
"""

# v5 = v2 minimal + an OVERVIEW first round. Untested idea on BrowseComp; the
# closest prior test is Twitter's batched first round (3 searches in one block),
# which cost 4% latency and changed nothing. This is a different intervention: not
# "same searches, fewer turns" but "deliberately BROADER first look, then drill".
#
# Why it might work here specifically. BrowseComp questions are entity-inference:
# several indirect clues, one named answer. 52% of successful questions surface a
# gold doc on search #1 and 75% within 3, yet the agent averages 9.5 searches and
# 24% of rollouts hit the 16-turn ceiling scoring 20.6% (vs 89% for rollouts that
# finish). So the loss is turns burned before converging, not missing documents.
# Covering the clue set in parallel on turn 1 may identify the entity sooner and
# leave budget to spare.
_BROWSECOMP_OVERVIEW_STRATEGY = """\
════════════════════════════════════════════════════════════
YOUR TASK
════════════════════════════════════════════════════════════

You are a deep research agent. Answer the question by interacting with the corpus
using search_index and get_document.

**FIRST TURN — OVERVIEW PASS.** The question gives you several independent clues.
Do not chase them one at a time. In your first code block, split the question into
3 distinct clues and search each one separately, in the same block:

    for q in [clue1_query, clue2_query, clue3_query]:
        for r in search_index(q, top_k=20):
            print(r['doc_id'], round(r['score'], 3), r['text'][:200])

Then look for a doc_id that appears under MORE THAN ONE clue — an entity satisfying
two independent clues is almost certainly the answer. That intersection is the
signal; a high score on a single clue is not.

**AFTER the overview**, work normally: one search per code block, each query
informed by what you just saw. Use get_document to verify a promising candidate
against ALL the clues before submitting.

Call SUBMIT(answer=...) with your succinct final answer: the exact name, date,
title, or value the question asks for. No explanation.
If the answer cannot be found after exhausting all search angles, call SUBMIT(answer="Unknown").
"""

# v6 = batch-only control for v5. Same 3 searches in the first block, but WITHOUT
# the intersection instruction. v5 changes two things at once (fewer turns AND
# "look for a doc_id appearing under more than one clue"), so on its own it cannot
# say which half matters. v6 isolates the turn saving; v5 minus v6 isolates the
# intersection reasoning.
_BROWSECOMP_BATCH_STRATEGY = """\
════════════════════════════════════════════════════════════
YOUR TASK
════════════════════════════════════════════════════════════

You are a deep research agent. Answer the question by interacting with the corpus
using search_index and get_document.

**In your FIRST code block only**, issue 3 search_index calls together — pick 3
genuinely different angles on the question and print all their results. This costs
one turn instead of three.

**AFTER that first block**, work normally: one search per code block, each query
informed by what you just saw. Use get_document to verify a promising candidate
before submitting.

Call SUBMIT(answer=...) with your succinct final answer: the exact name, date,
title, or value the question asks for. No explanation.
If the answer cannot be found after exhausting all search angles, call SUBMIT(answer="Unknown").
"""

# v7 = FREE-FORM BATCHING. v6 mandates exactly three searches in the first block,
# before the agent has seen anything — the moment it is least equipped to pick angles.
# HANDOFF section 5 already found the general form of this: agent-directed delegation
# beats MANDATED delegation, on all three OBLIQ tasks and across five structured
# congress variants (v4 .1330, v5e .1465, v6 .1407, v7 .1579, v8 .1144, every one below
# free-form v2 .1812). And measured here: v6+fusion LOSES to v2+fusion by 8.0 points
# (5 win / 21 loss, p=0.002) with identical recall, so the cost is behavioural.
# So this permits batching without requiring it, and says WHEN it is worth doing.
_BROWSECOMP_FREEBATCH_STRATEGY = """\
════════════════════════════════════════════════════════════════════════
YOUR TASK
════════════════════════════════════════════════════════════════════════

You are a deep research agent. Answer the question by interacting with the corpus
using search_index and get_document. Reason step by step and use the tools in an
interleaved manner — search, read, decide what to search next. You may search as
many times as needed.

You may issue SEVERAL search_index calls in a SINGLE code block whenever that is
the better move, and you decide when it is. Searching together costs one turn
instead of several, so it pays when you have independent angles that do not depend
on each other's results — for example distinct clues in the question, or alternative
phrasings of the same entity. Searching one at a time is better when what you find
should shape the next query. Use your judgement each turn; there is no required
number, and no requirement to batch at all.

Call SUBMIT(answer=...) with your succinct final answer: the exact name, date,
title, or value the question asks for. No explanation.
If the answer cannot be found after exhausting all search angles, call SUBMIT(answer="Unknown").
"""

# v8 = READ-FIRST. v2 names get_document in a list of tools and says nothing more, and
# the agent calls it ~2.0 times per question against 8.5 searches, skipping it entirely
# on 16-20% of questions. LightOn's BrowseComp+ table (blog: "the bloated retriever era
# is over") reports the SAME retriever and LLM at 87.59% with a scaffold built around
# reading, against 79.52% without — and notably their reading scaffold uses FEWER
# searches (13.27 vs 19.31) and LOWER recall (81.55% vs 83.52%). More reading, less
# searching, better answers.
# That direction matters because the two prompt changes tried here so far both pushed
# toward MORE searching and both lost accuracy: v6 mandated batching -8.0pp (p=0.002),
# v7 agent-directed batching -9.8pp (p=0.003). This one trades searches for reads.
# Phrasing stays free-form and agent-directed: HANDOFF section 5 and section 13 of the
# Retrieval sheet both found mandated structure loses to letting the agent decide.
_BROWSECOMP_READFIRST_STRATEGY = """\
════════════════════════════════════════════════════════════════════════
YOUR TASK
════════════════════════════════════════════════════════════════════════

You are a deep research agent. Answer the question by interacting with the corpus
using search_index and get_document. Reason step by step and use the tools in an
interleaved manner — search, read, decide what to search next. You may search as
many times as needed.

**search_index returns only a PREVIEW of each document — the opening passage, not
the whole thing.** The detail that settles a question is very often further down,
so a preview that merely looks promising is not evidence. When a result could
plausibly contain the answer, call get_document(doc_id) and read it in full before
deciding. Reading one strong candidate is usually worth more than another search:
if you have already retrieved the right document, no additional query will improve
on it, and a preview alone can be enough to make a wrong answer look supported.

Do not submit an answer that rests only on a preview. Either open the document and
confirm the detail, or say so and keep looking.

Call SUBMIT(answer=...) with your succinct final answer: the exact name, date,
title, or value the question asks for. No explanation.
If the answer cannot be found after exhausting all search angles, call SUBMIT(answer="Unknown").
"""

BrowseCompReadFirstSignature = make_rlm_signature(
    strategy=_BROWSECOMP_READFIRST_STRATEGY,
    answer_desc="Concise final answer (name, date, title, etc.). 'Unknown' if not found.",
    question_desc="The research question to answer.",
)
BrowseCompReadFirstSignature.__name__ = "BrowseCompReadFirstSignature"
BrowseCompReadFirstSignature._sig_key = "v8"


BrowseCompFreeBatchSignature = make_rlm_signature(
    strategy=_BROWSECOMP_FREEBATCH_STRATEGY,
    answer_desc="Concise final answer (name, date, title, etc.). 'Unknown' if not found.",
    question_desc="The research question to answer.",
)
BrowseCompFreeBatchSignature.__name__ = "BrowseCompFreeBatchSignature"
BrowseCompFreeBatchSignature._sig_key = "v7"


BrowseCompBatchSignature = make_rlm_signature(
    strategy=_BROWSECOMP_BATCH_STRATEGY,
    answer_desc="Concise final answer (name, date, title, etc.). 'Unknown' if not found.",
    question_desc="The research question to answer.",
)
BrowseCompBatchSignature.__name__ = "BrowseCompBatchSignature"
BrowseCompBatchSignature._sig_key = "v6"


BrowseCompOverviewSignature = make_rlm_signature(
    strategy=_BROWSECOMP_OVERVIEW_STRATEGY,
    answer_desc="Concise final answer (name, date, title, etc.). 'Unknown' if not found.",
    question_desc="The research question to answer.",
)
BrowseCompOverviewSignature.__name__ = "BrowseCompOverviewSignature"
BrowseCompOverviewSignature._sig_key = "v5"


BrowseCompMinimalScoringSignature = make_rlm_signature(
    strategy=_BROWSECOMP_MINIMAL_SCORING_STRATEGY,
    answer_desc="Concise final answer (name, date, title, etc.). 'Unknown' if not found.",
    question_desc="The research question to answer.",
)
BrowseCompMinimalScoringSignature.__name__ = "BrowseCompMinimalScoringSignature"
BrowseCompMinimalScoringSignature._sig_key = "v4"


# v13 = v2 minimal with the ABSTENTION LICENCE REMOVED. One variable.
#
# THE MEASUREMENT THAT MOTIVATES IT (Retrieval section 29). On the .7880 run, 95 of 830
# questions (11.4%) submitted "Unknown". BrowseComp+ scores an abstention as WRONG and
# imposes NO PENALTY for a wrong guess, so an abstention is strictly dominated by any
# named candidate. Sixty percent of those 95 had a gold document retrieved, and section 28
# showed that for 47% of the failure set the answer string was already inside the 500-char
# snippet the agent had in context. The agent is declining to commit to answers it holds.
#
# WHY NOT v4. v4 already carries a commitment rule, but a SOFT one: it permits "Unknown"
# "if you have no candidate whatsoever", and an agent that has searched sixteen times and
# convinced itself it found nothing claims exactly that. Measured: v4 still returned 24%
# "Unknown" on capped questions (logs/browsecomp_v4_med_capped40_checkpoint.json, n=25).
# v4 also predates Reason-MCB entirely — it was only ever run on the dense arm, at n=25/50,
# never paired. This is the hard version and it removes the escape hatch outright.
#
# TWO PLACES LICENSE THE ABSTENTION, and both are changed here: the strategy prose AND the
# output field description, which read "'Unknown' if not found." in every signature in this
# file. Changing only the prose would leave the field description contradicting it.
#
# EXPECTED FAILURE MODE, recorded before the run: forcing a guess converts abstentions into
# wrong answers at no cost, but it may also disturb questions the agent would have got
# right, by pushing it to commit early on a candidate it would otherwise have kept
# searching past. That is why the arm is scored paired against arm A on the same questions
# rather than by counting recovered abstentions.
_BROWSECOMP_MINIMAL_COMMIT_STRATEGY = """\
════════════════════════════════════════════════════════════
YOUR TASK
════════════════════════════════════════════════════════════

You are a deep research agent. Answer the question by interacting with the corpus
using search_index and get_document. Reason step by step and use the tools in an
interleaved manner — search, read, decide what to search next. You may search as
many times as needed.

**NEVER submit "Unknown". There is no credit for abstaining and no penalty for
being wrong, so a named guess is always worth more than no answer.**
Every question in this corpus HAS an answer and the evidence for it is reachable.
If you cannot confirm a candidate, name the single most plausible one you have seen
— a partially-supported entity, a date from a document that matched most of the
clues, the best of several competing candidates. Commit to it.

Call SUBMIT(answer=...) with your succinct final answer: the exact name, date,
title, or value the question asks for. No explanation.
"""

BrowseCompMinimalCommitSignature = make_rlm_signature(
    strategy=_BROWSECOMP_MINIMAL_COMMIT_STRATEGY,
    answer_desc=(
        "Concise final answer (name, date, title, etc.). "
        "Always a concrete candidate — never 'Unknown', never empty."
    ),
    question_desc="The research question to answer.",
)
BrowseCompMinimalCommitSignature.__name__ = "BrowseCompMinimalCommitSignature"
BrowseCompMinimalCommitSignature._sig_key = "v13"


# v14 = v2 minimal with EVERY MENTION OF get_document REMOVED. Search-only ablation.
#
# Pairs with DISABLE_GET_DOCUMENT=1 in rag/agent.py. v2 opens "interacting with the corpus
# using search_index and get_document" — leaving that in while the tool is absent would
# instruct the agent to call something that does not exist, which is a confound, not an
# ablation. The benchmark paper handles this the same way: Appendix G is a SEPARATE prompt
# for the get-doc condition.
#
# WHY THIS ARM EXISTS. The leaderboard runs this exact A/B three times and gets the same
# answer: LightOn 79.52 -> 87.59 (+8.07), Mixedbread 80.00 -> 90.48 (+10.48), and the paper's
# gpt-4.1 35.42 -> 43.61 (+8.19, Table 5). This system HAS get_document, calls it 1.93 times
# per question — MORE than the paper's 1.85 — and scores 78.80, which is their NO-get_doc
# level. Two readings fit that: either our reader works and our search-only baseline is ~70
# (so the deficit is retrieval), or our reader contributes nothing (so the deficit IS the
# reader). They predict opposite fixes and only this ablation separates them.
_BROWSECOMP_SEARCHONLY_STRATEGY = """\
════════════════════════════════════════════════════════════
YOUR TASK
════════════════════════════════════════════════════════════

You are a deep research agent. Answer the question by interacting with the corpus
using search_index. Reason step by step and use the tool in an interleaved manner —
search, read the results, decide what to search next. You may search as many times
as needed.

Call SUBMIT(answer=...) with your succinct final answer: the exact name, date,
title, or value the question asks for. No explanation.
If the answer cannot be found after exhausting all search angles, call SUBMIT(answer="Unknown").
"""

BrowseCompSearchOnlySignature = make_rlm_signature(
    strategy=_BROWSECOMP_SEARCHONLY_STRATEGY,
    answer_desc="Concise final answer (name, date, title, etc.). 'Unknown' if not found.",
    question_desc="The research question to answer.",
)
BrowseCompSearchOnlySignature.__name__ = "BrowseCompSearchOnlySignature"
BrowseCompSearchOnlySignature._sig_key = "v14"


def browsecomp_minimal_budget_sig(max_iters: int, max_search: int):
    """v3: v2 minimal with the REAL budget injected, so prompt and harness agree.

    Built per-run rather than at import time — the commit deadline is 75% of the
    iteration budget, which only makes sense once max_iters is known.
    """
    # Deadline defaults to 75% of the budget (turn 12 of 16), which is where the
    # iteration analysis shows marginal accuracy collapsing. Override with
    # BROWSECOMP_COMMIT_TURN to ablate the fraction — an env var rather than an
    # argument so it also reaches the multiprocessing worker, which rebuilds the
    # signature in a fresh interpreter from a fixed argument list.
    _env = os.environ.get("BROWSECOMP_COMMIT_TURN", "").strip()
    if _env:
        commit_turn = max(2, min(int(_env), max_iters))
    else:
        commit_turn = max(2, int(round(max_iters * 0.75)))
    sig = make_rlm_signature(
        strategy=_BROWSECOMP_MINIMAL_BUDGET_STRATEGY.format(
            max_iters=max_iters, max_search=max_search, commit_turn=commit_turn),
        answer_desc="Concise final answer (name, date, title, etc.). 'Unknown' if not found.",
        question_desc="The research question to answer.",
    )
    sig.__name__ = "BrowseCompMinimalBudgetSignature"
    sig._sig_key = "v3"
    return sig


BrowseCompMinimalSignature = make_rlm_signature(
    strategy=_BROWSECOMP_MINIMAL_STRATEGY,
    answer_desc="Concise final answer (name, date, title, etc.). 'Unknown' if not found.",
    question_desc="The research question to answer.",
)

# ─── Congress (tip-of-tongue) ─────────────────────────────────────────────────
# BEST: ObliqCongressSig (v2) — NDCG@10=0.139, 254 questions
# Beats all dense retrievers + Query Rewriter; gap to Multi-Hop = −0.044

_OBLIQ_CONGRESS_STRATEGY = """\
════════════════════════════════════════════════════════════
YOUR TASK — TIP-OF-TONGUE: CONGRESSIONAL HEARING
════════════════════════════════════════════════════════════

The user vaguely remembers a specific moment from a US congressional hearing.
There is exactly ONE correct transcript passage to find. Names, dates, and
committee identifiers have been removed from the query — focus on the rhetorical
dynamic, the topic, and the type of exchange described.

Each call to search_index() shows your remaining budget: [search N/8  K remaining].
Use your full budget — do not submit while angles remain unexplored.

**Strategy:**
1. Identify the KEY ELEMENTS of the described moment:
   • What is the topic/policy area being discussed?
   • What is the rhetorical dynamic? (trap question, confrontation, revelation,
     calm cross-examination, witness cornered, senator loses patience...)
   • What is the witness's role? (CEO, cabinet secretary, regulator, expert...)
   • Are there any specific phrases, numbers, or rhetorical moves described?
   Name these concretely before searching.
2. Search for the passage using DIFFERENT angles each hop. Vary the vocabulary:
   formal transcript language, policy terms, witness role, the rhetorical move.
   Scores are cosine similarities — if the top result is below 0.25, the angle
   is wrong: try a completely different framing.
   If 2 consecutive searches on the same angle return nothing above 0.25,
   move on — do not keep rephrasing the same dead-end.
3. After each search, scan the snippets and write a short note for any promising
   doc: doc_id → which clues from the description it matches. Use get_document
   to read the full passage when a snippet looks strong and needs verification.
   Keep your notes updated — this is your working memory.
4. SUBMIT a newline-separated list of doc_ids (format: CHRG-...) ranked strongest
   to weakest based on your accumulated notes. Put your best candidate first.
   Include all plausible matches — the gold is ONE specific passage.
   No prose — ONLY doc_ids, one per line.
"""

ObliqCongressSig = make_rlm_signature(  # BEST: NDCG@10=0.139 (254q)
    strategy=_OBLIQ_CONGRESS_STRATEGY,
    answer_desc=_MULTI_DOC_ANSWER,
    question_desc="A tip-of-tongue description of a congressional hearing moment to find.",
)

# EXPERIMENTAL: v2 base + explicit llm_query per hop for next-query gen (writing-style, no JSON)
_OBLIQ_CONGRESS_STRATEGY_V4 = """\
════════════════════════════════════════════════════════════
YOUR TASK — TIP-OF-TONGUE: CONGRESSIONAL HEARING (v4)
════════════════════════════════════════════════════════════

The user vaguely remembers a specific moment from a US congressional hearing.
There is exactly ONE correct transcript passage to find. Names, dates, and
committee identifiers have been removed from the query — focus on the rhetorical
dynamic, the topic, and the type of exchange described.

Each call to search_index() shows your remaining budget: [search N/8  K remaining].
Use your full budget — do not submit while angles remain unexplored.

**Strategy:**
1. Identify the KEY ELEMENTS of the described moment:
   • What is the topic/policy area being discussed?
   • What is the rhetorical dynamic? (trap question, confrontation, revelation,
     calm cross-examination, witness cornered, senator loses patience...)
   • What is the witness's role? (CEO, cabinet secretary, regulator, expert...)
   • Are there any specific phrases, numbers, or rhetorical moves described?
   Name these concretely before searching.

2. Each hop: search, then use llm_query to analyze the snippets and generate
   the next search query in TRANSCRIPT VOCABULARY — language that would appear
   verbatim in an official US congressional transcript. Use this exact pattern:

  notes = {}  # Initialize once: doc_id -> matched clues (update each hop)

  # After each search_index call:
  snippets = "\\n".join(f"{r['doc_id']}: {r['text'][:300]}" for r in results[:10])
  next_query = llm_query(
      f"WHAT THE USER REMEMBERS: {question}\\n"
      f"MY NOTES SO FAR: {notes}\\n"
      f"SNIPPETS FROM THIS HOP:\\n{snippets}\\n\\n"
      "Which passages could be the moment described? Update notes with any matches.\\n"
      "Then generate the next search query using TRANSCRIPT VOCABULARY:\\n"
      "  • Speaker titles and names (Chairman, Ranking Member, Mr. Smith, Ms. Jones)\\n"
      "  • Committee names (Judiciary, Commerce, Intelligence, Armed Services)\\n"
      "  • Legislation and policy terms (Section 230, antitrust, GDPR, sanctions)\\n"
      "  • Specific phrases a senator or witness might say verbatim\\n"
      "Target a DIFFERENT angle than previous hops.\\n"
      "Return ONLY the next search query string, nothing else."
  )
  print(next_query[:100])
  # notes[doc_id] = 'matched clues...'  ← update for any promising snippets

3. Keep notes and candidate doc_ids as working memory across all hops.
   Scores are cosine similarities — if top score < 0.25, the angle is wrong:
   try a completely different framing.

4. SUBMIT a newline-separated list of doc_ids (format: CHRG-...) ranked
   strongest to weakest based on accumulated notes. No prose — ONLY doc_ids.
"""

ObliqCongressV4Sig = make_rlm_signature(  # EXPERIMENTAL: v2 + llm_query next-query gen (writing-style)
    strategy=_OBLIQ_CONGRESS_STRATEGY_V4,
    answer_desc=_MULTI_DOC_ANSWER,
    question_desc="A tip-of-tongue description of a congressional hearing moment to find.",
)

# v5: v4 + get_document on top candidates per hop (combines transcript-vocab steering with full-doc reading)
_OBLIQ_CONGRESS_STRATEGY_V5 = """\
════════════════════════════════════════════════════════════
YOUR TASK — TIP-OF-TONGUE: CONGRESSIONAL HEARING (v5)
════════════════════════════════════════════════════════════

The user vaguely remembers a specific moment from a US congressional hearing.
There is exactly ONE correct transcript passage to find. Names, dates, and
committee identifiers have been removed from the query — focus on the rhetorical
dynamic, the topic, and the type of exchange described.

Each call to search_index() shows your remaining budget: [search N/8  K remaining].
Use your full budget — do not submit while angles remain unexplored.

**Strategy:**
1. Identify the KEY ELEMENTS of the described moment:
   • What is the topic/policy area being discussed?
   • What is the rhetorical dynamic? (trap question, confrontation, revelation,
     calm cross-examination, witness cornered, senator loses patience...)
   • What is the witness's role? (CEO, cabinet secretary, regulator, expert...)
   • Are there any specific phrases, numbers, or rhetorical moves described?
   Name these concretely before searching.

2. Each hop: search → llm_query to analyze snippets, fetch top candidates, generate
   next query in TRANSCRIPT VOCABULARY. Use this exact pattern:

  import json as _json, re as _re
  notes = {}   # doc_id -> matched clues (update each hop)
  fetched = set()  # doc_ids already read in full — never re-fetch

  # After each search_index call:
  snippets = "\\n".join(f"{r['doc_id']}: score={r['score']:.3f}: {r['text'][:300]}" for r in results[:10])
  already_read = sorted(fetched) or 'none'
  parsed = _json.loads(_re.search(r'\\{[\\s\\S]*\\}', llm_query(
      f"WHAT THE USER REMEMBERS: {question}\\n"
      f"MY NOTES SO FAR: {notes}\\n"
      f"ALREADY FETCHED IN FULL (do NOT re-fetch these): {already_read}\\n"
      f"SNIPPETS FROM THIS HOP:\\n{snippets}\\n\\n"
      "1. Update notes with any passages that could match the described moment.\\n"
      "   CRITICAL: Use the EXACT doc_id from the snippet line (e.g., CHRG-115hhrg30956_p0000_19).\\n"
      "   NEVER truncate to hearing-level (e.g., NOT CHRG-115hhrg30956 without the _p0000_NN suffix).\\n"
      "2. Should you fetch any doc in full? Only if ALL of these hold:\\n"
      "   • The doc is NOT in ALREADY FETCHED list above\\n"
      "   • AND: snippet is cut off so you cannot judge the match, OR you need to\\n"
      "     verify which exact adjacent passage is the gold moment\\n"
      "   If fetching, use the EXACT passage doc_id (e.g., CHRG-115hhrg30956_p0000_19).\\n"
      "   Default to fetch_docs=[].\\n"
      "3. Generate the next search query in TRANSCRIPT VOCABULARY:\\n"
      "   • Speaker titles/names (Chairman, Ranking Member, Mr. Smith, Ms. Jones)\\n"
      "   • Committee names (Judiciary, Commerce, Intelligence, Armed Services)\\n"
      "   • Legislation and policy terms (Section 230, antitrust, GDPR, sanctions)\\n"
      "   • Specific phrases a senator or witness might say verbatim\\n"
      "   Target a DIFFERENT angle than previous hops.\\n"
      'Return JSON: {"notes_update": {"EXACT_doc_id_with_suffix": "why it matches"}, '
      '"fetch_docs": ["EXACT_doc_id_with_suffix_if_needed"], "next_query": "transcript query here"}'
  ), _re.DOTALL).group())
  for doc_id, clue in parsed.get('notes_update', {}).items():
      notes[doc_id] = clue
  for doc_id in parsed.get('fetch_docs', [])[:1]:
      if doc_id not in fetched:
          get_document(doc_id)
          fetched.add(doc_id)
          notes[doc_id] = (notes.get(doc_id, '') + ' [FULL DOC READ]')[:200]
  query = parsed.get('next_query', '')
  print(f"fetched={sorted(fetched)}  next={query[:80]!r}")

3. Keep notes and candidate doc_ids as working memory across all hops.
   Scores are cosine similarities — if top score < 0.25, the angle is wrong:
   try a completely different framing.

4. SUBMIT a newline-separated list of EXACT doc_ids ranked strongest to weakest
   based on accumulated notes. Use the FULL passage ID including suffix
   (e.g., CHRG-115hhrg30956_p0000_19, NOT CHRG-115hhrg30956).
   No prose — ONLY doc_ids, one per line.
"""

ObliqCongressV5Sig = make_rlm_signature(  # v4 + get_document on top candidates per hop
    strategy=_OBLIQ_CONGRESS_STRATEGY_V5,
    answer_desc=_MULTI_DOC_ANSWER,
    question_desc="A tip-of-tongue description of a congressional hearing moment to find.",
)

# v6: v2 base + final reranking step (mirrors OBLIQ paper's 3rd-stage LLM call across all candidates)
_OBLIQ_CONGRESS_STRATEGY_V6 = """\
════════════════════════════════════════════════════════════
YOUR TASK — TIP-OF-TONGUE: CONGRESSIONAL HEARING (v6)
════════════════════════════════════════════════════════════

The user vaguely remembers a specific moment from a US congressional hearing.
There is exactly ONE correct transcript passage to find. Names, dates, and
committee identifiers have been removed from the query — focus on the rhetorical
dynamic, the topic, and the type of exchange described.

Each call to search_index() shows your remaining budget: [search N/8  K remaining].
Use your full budget — do not submit while angles remain unexplored.

**Strategy:**
1. Identify the KEY ELEMENTS of the described moment:
   • What is the topic/policy area being discussed?
   • What is the rhetorical dynamic? (trap question, confrontation, revelation,
     calm cross-examination, witness cornered, senator loses patience...)
   • What is the witness's role? (CEO, cabinet secretary, regulator, expert...)
   • Are there any specific phrases, numbers, or rhetorical moves described?
   Name these concretely before searching.

2. Search for the passage using DIFFERENT angles each hop. Vary the vocabulary:
   formal transcript language, policy terms, witness role, the rhetorical move.
   Scores are cosine similarities — if the top result is below 0.25, the angle
   is wrong: try a completely different framing.
   If 2 consecutive searches on the same angle return nothing above 0.25,
   move on — do not keep rephrasing the same dead-end.

3. After each search, scan the snippets and write a short note for any promising
   doc: doc_id → which clues from the description it matches. Use get_document
   to read the full passage when a snippet looks strong and needs verification.
   Keep your notes updated — this is your working memory.

   Use this pattern to accumulate candidates:
     notes = {}      # doc_id -> evidence string (initialize once, update each hop)
     cand_text = {}  # doc_id -> passage text, needed by the rerank in step 4

     # After EVERY search_index call, keep the text of everything you saw:
     for r in results:
         cand_text[r['doc_id']] = r['text']

   Add a doc to notes if it is even plausibly the moment. Scoring looks only at
   where the ONE gold passage lands in your top 10 — a wrong doc in slot 7 costs
   you nothing at all. So keep every plausible candidate and never prune the pool
   to make it look tidy; a candidate you drop here can no longer be ranked.

4. FINAL RERANK — after exhausting your search budget, before SUBMIT.

   Judge the PASSAGES, not your notes about them: notes were written hop by hop
   and carry the order you happened to find things in, which is not the same as
   how well each passage matches. Rank by candidate NUMBER, so a mistyped or
   truncated doc_id cannot silently drop a candidate out of your answer.

     import json as _json, re as _re
     cands = list(notes)
     final = cands   # fallback order, so a failed rerank never costs the SUBMIT
     if len(cands) >= 2:
       try:
         numbered = "\\n".join(
             f"[{i}] {cand_text.get(d, '')[:900]}" for i, d in enumerate(cands))
         raw = llm_query(
             f"WHAT THEY REMEMBER:\\n{question}\\n\\n"
             f"CANDIDATE PASSAGES:\\n{numbered}\\n\\n"
             "Rank every candidate by how likely it is the exact moment described.\\n"
             "The description was written from hazy memory: 1-2 details in it are\\n"
             "probably wrong and adjacent moments may be conflated. Prefer the\\n"
             "passage matching the core exchange over one matching every detail.\\n"
             f"Output all {len(cands)} candidates. Every number from 0 to "
             f"{len(cands) - 1} must appear exactly once — omit none.\\n"
             'Return ONLY JSON: {"ranked":[{"candidate_num":<int>,"reason":"<brief>"}]}'
         )
         order = _json.loads(_re.search(r'\\{[\\s\\S]*\\}', raw, _re.DOTALL).group())["ranked"]
         seen, ranked = set(), []
         for item in order:
             i = int(item["candidate_num"])
             if 0 <= i < len(cands) and i not in seen:
                 seen.add(i)
                 ranked.append(cands[i])
         # Safety net: anything the reranker omitted still gets submitted, at the tail.
         final = ranked + [d for i, d in enumerate(cands) if i not in seen]
       except Exception as _e:
         print("rerank failed:", _e)
     print(f"[rerank] {len(final)} candidates, top3={final[:3]}")

5. SUBMIT final — one doc_id per line, no prose. Use the FULL passage id
   including its suffix (e.g. CHRG-115hhrg30956_p0000_19, NOT CHRG-115hhrg30956).

   You have 10 scoring slots and there is exactly one gold passage, so a short
   list wastes slots for free. If final has fewer than 10 entries, append the
   next most plausible doc_ids you saw during search before submitting:

     if len(final) < 10:
         final += [d for d in cand_text if d not in final][:10 - len(final)]
     SUBMIT("\\n".join(final))
"""

ObliqCongressV6Sig = make_rlm_signature(  # v2 + final reranking via llm_query
    strategy=_OBLIQ_CONGRESS_STRATEGY_V6,
    answer_desc=_MULTI_DOC_ANSWER,
    question_desc="A tip-of-tongue description of a congressional hearing moment to find.",
)

# v7: v2 + a per-hop CANDIDATE SELECTION call. Motivated by transcript evidence:
# on 6 questions where search retrieved the gold passage, 4 had it printed into the
# agent's own stdout (one of them on 5 separate turns) and never written into notes.
# Every retrieved doc was printed, so this is recognition, not visibility. Worth
# +.0315 on the gpt-5.2 254q run, vs +.0106 for any reordering of the shortlist.
#
# Deliberately NOT like v4/v5/v6: the sub-call only triages this hop's snippets. It
# does not generate the next query, decide fetches, or remove anything. Search
# planning stays with the outer agent, which is what v2 does best.
_OBLIQ_CONGRESS_STRATEGY_V7 = """\
════════════════════════════════════════════════════════════
YOUR TASK — TIP-OF-TONGUE: CONGRESSIONAL HEARING (v7)
════════════════════════════════════════════════════════════

The user vaguely remembers a specific moment from a US congressional hearing.
There is exactly ONE correct transcript passage to find. Names, dates, and
committee identifiers have been removed from the query — focus on the rhetorical
dynamic, the topic, and the type of exchange described.

Each call to search_index() shows your remaining budget: [search N/8  K remaining].
Use your full budget — do not submit while angles remain unexplored.

**Strategy:**
1. Identify the KEY ELEMENTS of the described moment:
   • What is the topic/policy area being discussed?
   • What is the rhetorical dynamic? (trap question, confrontation, revelation,
     calm cross-examination, witness cornered, senator loses patience...)
   • What is the witness's role? (CEO, cabinet secretary, regulator, expert...)
   • Are there any specific phrases, numbers, or rhetorical moves described?
   Name these concretely before searching.

2. Search for the passage using DIFFERENT angles each hop. Vary the vocabulary:
   formal transcript language, policy terms, witness role, the rhetorical move.
   Scores are cosine similarities — if the top result is below 0.25, the angle
   is wrong: try a completely different framing.
   If 2 consecutive searches on the same angle return nothing above 0.25,
   move on — do not keep rephrasing the same dead-end.
   You decide every query yourself, from the description and what you have seen.

3. After EVERY search, run one CANDIDATE SELECTION call. Judging 20 passages
   while also planning your next query is how the right passage gets overlooked,
   so this call does nothing but spot possible matches — hand it the snippets and
   let it answer that one question.

  import json as _json, re as _re
  candidates = {}   # doc_id -> why it might match. Initialize ONCE. It only GROWS.
  cand_text = {}    # doc_id -> passage text, for the final ordering

  # After each search_index call:
  for r in results:
      cand_text[r['doc_id']] = r['text']
  snippets = "\\n".join(f"{r['doc_id']}\\n{r['text'][:600]}\\n" for r in results)
  picked = llm_query(
      f"SOMEONE IS TRYING TO RECALL A MOMENT FROM A CONGRESSIONAL HEARING:\\n"
      f"{question}\\n\\n"
      f"TRANSCRIPT PASSAGES:\\n{snippets}\\n\\n"
      "Which of these passages could plausibly be the moment described?\\n"
      "Be INCLUSIVE. Exactly one passage is correct. Naming a wrong one costs\\n"
      "nothing, but missing the right one loses the question outright — so list\\n"
      "anything with a real chance, not only your single best guess.\\n"
      "This is a hazy recollection, so some details may be misremembered. Judge\\n"
      "on the central exchange rather than requiring every detail to line up.\\n"
      "Use the EXACT doc_id shown above, including its _pXXXX_NN suffix.\\n"
      'Return ONLY JSON: {"picks":[{"doc_id":"CHRG-...","why":"<brief>"}]}'
  )
  try:
      for p in _json.loads(_re.search(r'\\{[\\s\\S]*\\}', picked, _re.DOTALL).group())["picks"]:
          candidates.setdefault(p["doc_id"], p.get("why", ""))   # never overwrite, never delete
  except Exception as _e:
      print("selection parse failed:", _e)
  print(f"[select] {len(candidates)} candidates so far")

   Never remove anything from candidates. A candidate you drop can no longer be
   ranked, and a wrong one sitting in your list costs you nothing.
   Use get_document only when a snippet is cut off and you need the rest.

4. SUBMIT the accumulated candidates, ordered by how well each matches — best
   first — using the FULL passage id including its suffix (e.g.
   CHRG-115hhrg30956_p0000_19, NOT CHRG-115hhrg30956).

   Passages are consecutive slices of a hearing, so the moment may sit in the
   slice next to one you picked. Append the immediate neighbours of your top
   candidates, then any other passages you saw, until you have 10:

  final = [<your candidate doc_ids, best first>]
  for d in list(final):
      m = _re.match(r'^(.*_p\\d+)_(\\d+)$', d)
      if m:
          for off in (1, -1):
              nb = f"{m.group(1)}_{int(m.group(2)) + off:02d}"
              if int(m.group(2)) + off >= 0 and nb not in final:
                  final.append(nb)
  final += [d for d in cand_text if d not in final]
  SUBMIT("\\n".join(final[:10]))
"""

ObliqCongressV7Sig = make_rlm_signature(  # v2 + inclusive per-hop candidate selection
    strategy=_OBLIQ_CONGRESS_STRATEGY_V7,
    answer_desc=_MULTI_DOC_ANSWER,
    question_desc="A tip-of-tongue description of a congressional hearing moment to find.",
)

# v8: a FAITHFUL implementation of the paper's Congress multi-hop design (appendix
# D.5.4), which none of v4-v7 actually followed. The paper keeps three concerns in
# three separate prompts; every earlier variant bundled them:
#   v4  fused query-generation with note-taking in one call
#   v5  added fetch decisions to that same JSON
#   v6  reranked from NOTES rather than passages, and told the model to PRUNE
#   v7  used inclusive selection but let the agent pick its own queries
# The paper instead does, per hop: (1) generate the query from the description +
# accumulated notes ONLY — never from the retrieved snippets, so the query cannot
# anchor on what a bad search already returned; (2) extract candidate_ids + notes
# from the snippets, additively. Then once at the end: (3) a listwise rerank that
# sees only the original description and the passages, not the notes.
_OBLIQ_CONGRESS_STRATEGY_V8 = """\
════════════════════════════════════════════════════════════
YOUR TASK — TIP-OF-TONGUE: CONGRESSIONAL HEARING (v8, paper design)
════════════════════════════════════════════════════════════

The user vaguely remembers a specific moment from a US congressional hearing.
There is exactly ONE correct transcript passage to find. Names, dates, and
committee identifiers have been removed from the query.

Follow this three-stage structure EXACTLY. Each stage is a separate llm_query with
one job. Do not merge them — the separation is the point.

  import json as _json, re as _re
  notes = ""        # free-text notes, carried across hops
  candidates = {}   # doc_id -> passage text. Initialize ONCE. Only GROWS.
  hop = 0

── PER HOP (repeat until your search budget is gone) ────────────────────────────

STAGE 1 — generate the query from the DESCRIPTION AND NOTES ONLY.
Do not show it the snippets. A query derived from what a failing search returned
just chases that failure.

  hop += 1
  q = _json.loads(_re.search(r'\\{[\\s\\S]*\\}', llm_query(
      "You are an advanced retrieval system. You will be given a tip-of-tongue query "
      "describing a user's hazy memory of a specific moment from a US congressional "
      "hearing. Given their description and any notes from previous search attempts, "
      "generate a search query that would match the actual hearing transcript. Think "
      "about what the transcript would actually contain - speaker names, committee "
      "procedures, policy terms, specific phrases.\\n\\n"
      f"ORIGINAL DESCRIPTION:\\n{question}\\n\\n"
      f"NOTES FROM PREVIOUS HOPS:\\n{notes or 'none yet'}\\n\\n"
      f"HOP: {hop} of 8\\n\\n"
      'Return ONLY JSON: {"search_query": "<query using transcript language>", '
      '"rationale": "<what you are looking for this hop>"}'
  ), _re.DOTALL).group())
  results = search_index(q["search_query"], top_k=20)

STAGE 2 — extract candidate ids and notes from the snippets. Additive only.

  snips = "\\n".join(f"{r['doc_id']}: {r['text'][:400]}" for r in results)
  for r in results:
      pass
  ex = _json.loads(_re.search(r'\\{[\\s\\S]*\\}', llm_query(
      "You are trying to find a specific congressional hearing moment that someone "
      "vaguely remembers. You will see their description and some candidate transcript "
      "passages. Select any passages that could plausibly be the moment they are "
      "describing. Write notes about what you have learned so far - what matches, what "
      "does not, what to search for next.\\n\\n"
      f"WHAT THEY REMEMBER:\\n{question}\\n\\n"
      f"PREVIOUS NOTES:\\n{notes or 'none yet'}\\n\\n"
      f"CANDIDATE PASSAGES (hop {hop}):\\n{snips}\\n\\n"
      "Use the EXACT doc_id shown, including its _pXXXX_NN suffix.\\n"
      'Return ONLY JSON: {"candidate_ids": ["id1"], "notes": "<what matched, what did '
      'not, what to try next>", "summary": "<brief summary of this hop>"}'
  ), _re.DOTALL).group())
  text_of = {r['doc_id']: r['text'] for r in results}
  for cid in ex.get("candidate_ids", []):
      candidates.setdefault(cid, text_of.get(cid, ''))    # never remove
  notes = (notes + "\\n" + ex.get("notes", "")).strip()[-3000:]
  print(f"hop {hop}: {len(candidates)} candidates")

── AFTER THE LAST HOP ───────────────────────────────────────────────────────────

STAGE 3 — listwise rerank. It sees the description and the PASSAGES only, never
your notes, so hop-order bias cannot leak into the final ordering.

  cands = list(candidates)
  final = cands
  if len(cands) >= 2:
    try:
      numbered = "\\n".join(f"[{i}] {candidates[d][:600]}" for i, d in enumerate(cands))
      raw = llm_query(
          "You are trying to find a specific congressional hearing moment. Rank "
          "passages by how likely they are the moment being described.\\n\\n"
          f"WHAT THEY REMEMBER:\\n{question}\\n\\n"
          f"CANDIDATE PASSAGES:\\n{numbered}\\n\\n"
          f"Output exactly {len(cands)} items. Every candidate number must appear "
          "exactly once.\\n"
          'Return ONLY JSON: {"ranked": [{"candidate_num": <int>, "reason": "<brief>"}]}'
      )
      order = _json.loads(_re.search(r'\\{[\\s\\S]*\\}', raw, _re.DOTALL).group())["ranked"]
      seen, ranked = set(), []
      for it in order:
          i = int(it["candidate_num"])
          if 0 <= i < len(cands) and i not in seen:
              seen.add(i); ranked.append(cands[i])
      final = ranked + [d for i, d in enumerate(cands) if i not in seen]
    except Exception as _e:
      print("rerank failed:", _e)
  print(f"[rerank] {len(final)} candidates, top3={final[:3]}")
  SUBMIT("\\n".join(final))

Use the FULL passage id including its suffix (e.g. CHRG-115hhrg30956_p0000_19).
No prose - ONLY doc_ids, one per line.
"""

ObliqCongressV8Sig = make_rlm_signature(  # faithful port of the paper's D.5.4 multi-hop
    strategy=_OBLIQ_CONGRESS_STRATEGY_V8,
    answer_desc=_MULTI_DOC_ANSWER,
    question_desc="A tip-of-tongue description of a congressional hearing moment to find.",
)


# ─── Math (analogues) ─────────────────────────────────────────────────────────
# BEST: ObliqMathSig (v10) — NDCG@10=0.132, 151 questions
# Beats all Qwen static + LateOn; just below Gemini (−0.012) and Multi-Hop (−0.029)

_OBLIQ_MATH_STRATEGY = """\
════════════════════════════════════════════════════════════
YOUR TASK — ANALOGUES: MATHEMATICAL META-PROGRAM
════════════════════════════════════════════════════════════

Given a math problem, find OTHER problems in the corpus that require the SAME
abstract proof strategy — the same "meta-move" or "aha moment" — even if they
come from a completely different field or use different notation.

Two problems match if their solutions use the same HIGH-LEVEL proof skeleton:
  • "Exploit uniqueness of X to constrain the answer"
  • "Finite domain: injective ⇔ surjective"
  • "Transform representation → swap order of operations → collapse"
  • "Symmetry averaging to project onto invariants"
Problems can share a meta-move despite having NOTHING in common on the surface.

Each call to search_index() shows your remaining budget: [search N/8  K remaining].
Use your full budget — do not submit while angles remain unexplored.

**Strategy:**
1. Initialize a retrieval tracker at the very start:

   all_docs = {}  # doc_id -> best cosine score seen

2. Identify the KEY INSIGHT: what is the single proof step that makes the
   problem work? Name it concretely (e.g. "apply the map to both sides and use
   uniqueness", "multiplication by a is injective on a finite set so surjective").

3. Search for that same proof move in OTHER mathematical settings. Vary the
   domain (analysis, algebra, combinatorics, geometry) and the vocabulary.
   Scores are cosine similarities — if the top result is below 0.30, the
   angle is wrong: abandon it immediately and try a different framing.
   If 2 consecutive searches on the same angle return nothing above 0.30,
   move on — do not keep rephrasing the same dead-end query.

   After EVERY search_index call, record all results:

   for r in results:
       all_docs[r['doc_id']] = max(all_docs.get(r['doc_id'], 0), r['score'])

4. After each search, scan the snippets and write a short note for any
   promising doc: doc_id → why it might share the meta-move. Use get_document
   + llm_query only when the snippet is truncated and the doc looks strong.
   Keep your notes updated as you search — this is your working memory.

5. SUBMIT all retrieved docs. First rank your top candidates by relevance,
   then append the rest sorted by cosine score:

   top = [<your best doc_ids in order>]
   rest = sorted((d for d in all_docs if d not in top), key=all_docs.get, reverse=True)
   SUBMIT("\\n".join(top + rest))

   Never submit fewer docs than you retrieved — every retrieved doc must appear.
"""

# ── Math ablation A: explicit LLM rerank before submitting ────────────────────
# Targets the largest measured gap in the whole project. On math the agent submits
# NDCG@10 .1721 while a perfect ranking of the documents it ALREADY retrieved would
# score .5307 (oracle analysis over 151 questions). So retrieval is not the limit —
# ordering is. The baseline ranks by raw cosine score after a hand-picked head;
# this variant scores candidates with the sub-LM against the actual matching
# criterion (same proof meta-move) and sorts by that instead.
#
# Note the risk: the equivalent change on Twitter (accumulator + top-30 listwise
# rerank) REGRESSED, .1115 -> .1030. Math is the better bet because its headroom is
# 2x Twitter's and "same proof strategy" is a judgement an LLM can make from text,
# where "implicit stance" apparently is not.
# Build by SPLITTING at a stable marker rather than matching the code block: the
# baseline text contains escaped backslashes (SUBMIT("\\n".join(...))) and an exact
# .replace() on it silently no-ops, which is how the first attempt at this variant
# ended up identical to the baseline.
_MATH_STEP5 = "5. SUBMIT all retrieved docs."
assert _MATH_STEP5 in _OBLIQ_MATH_STRATEGY
# math-batch = the math meta-program strategy + the SAME permissive batching paragraph
# that v7 (BrowseCompFreeBatchSignature) added to browsecomp. Tests whether
# agent-directed batching transfers from an accuracy task to a graded NDCG task.
# NOTE the regime differs sharply: math runs with --max-search 8 against browsecomp's
# 35, so batching five queries in one turn spends most of the budget before the agent
# has seen anything. The budget is held at 8 so batching is the only variable.
_OBLIQ_MATH_STRATEGY_BATCH = _OBLIQ_MATH_STRATEGY + """

**Issuing several searches at once.** You may call search_index() SEVERAL times in a
SINGLE code block whenever that is the better move, and you decide when it is.
Searching together costs one turn instead of several, so it pays when you have
independent angles that do not depend on each other's results — for example the same
meta-move phrased for different domains (analysis, algebra, combinatorics, geometry).
Searching one at a time is better when what you find should shape the next query.
Remember the budget is shared: every call in a block counts against [search N/8].
Use your judgement each turn; there is no required number, and no requirement to
batch at all.
"""

ObliqMathBatchSig = make_rlm_signature(
    strategy=_OBLIQ_MATH_STRATEGY_BATCH,
    answer_desc="Ranked list of doc_ids of analogous problems, best first.",
    question_desc="The mathematical problem to find analogues for.",
)
ObliqMathBatchSig.__name__ = "ObliqMathBatchSig"


_OBLIQ_MATH_STRATEGY_RERANK = _OBLIQ_MATH_STRATEGY.split(_MATH_STEP5)[0] + """\
5. RERANK before submitting. Cosine score measures topical similarity, but the task
   is shared PROOF STRATEGY — a different thing. Cosine order is a poor final
   ranking. Spend your last 1-2 turns fixing the order.

   Keep the snippet text as you search, alongside all_docs:

   snippets = {}   # doc_id -> snippet text

   Then score your top 30 candidates with the sub-LM, about 10 per call, asking for
   a number per doc:

   cand = sorted(all_docs, key=all_docs.get, reverse=True)[:30]
   prompt = (
       "KEY INSIGHT of the query problem: <the meta-move you named in step 2>.\\n"
       "For each candidate below output one line 'doc_id: N', N from 0-10, for how\\n"
       "strongly its solution would use that SAME abstract proof move. Judge the\\n"
       "MOVE, not the topic - a combinatorics and an analysis problem can both\\n"
       "score 10. Output only those lines.\\n\\n"
       + "\\n\\n".join(d + ": " + snippets.get(d, "")[:600] for d in cand[:10]))
   print(llm_query(prompt))

   Parse the numbers, sort those 30 by score descending (ties broken by cosine),
   then append every remaining doc by cosine:

   top = [<the 30 reordered by the parsed scores>]
   rest = sorted((d for d in all_docs if d not in top), key=all_docs.get, reverse=True)
   SUBMIT(chr(10).join(top + rest))

   Never submit fewer docs than you retrieved — every retrieved doc must appear.
"""

ObliqMathRerankSig = make_rlm_signature(
    strategy=_OBLIQ_MATH_STRATEGY_RERANK,
    answer_desc=_MULTI_DOC_ANSWER,
    question_desc="A mathematical problem; find others sharing its proof meta-move.",
)


# ── Math ablation B: overview search before drilling in ───────────────────────
# The baseline fires 8 narrow searches, each conditioned on the last. This variant
# spends its first turn on a deliberately COARSE survey at high top_k to map the
# corpus regions, then drills into the regions that look live. Rationale: gold
# discovery is heavily front-loaded (72% of math questions surface a gold doc on
# search #1), which suggests the first query already lands in roughly the right
# place — so a wider first look may map the neighbourhood more cheaply than 8
# narrow probes.
_OBLIQ_MATH_STRATEGY_OVERVIEW = _OBLIQ_MATH_STRATEGY.replace(
"""3. Search for that same proof move in OTHER mathematical settings. Vary the
   domain (analysis, algebra, combinatorics, geometry) and the vocabulary.""",
"""3. FIRST TURN — OVERVIEW PASS. Before any narrow query, map the space. Issue 3
   deliberately BROAD searches in one code block at top_k=50, one per level of
   abstraction: (a) the bare meta-move in plain words, (b) the move stated in the
   query's own domain, (c) the move stated in a deliberately DIFFERENT domain.
   Print only doc_id, score and the first 120 chars of each — you are mapping
   regions, not reading yet.

   for q in [q_abstract, q_same_domain, q_other_domain]:
       for r in search_index(q, top_k=50):
           all_docs[r['doc_id']] = max(all_docs.get(r['doc_id'], 0), r['score'])
           print(r['doc_id'], round(r['score'],3), r['text'][:120])

   Note which score bands and which phrasings produced clusters. THEN spend your
   remaining searches drilling into the productive regions with narrow queries.

   Search for that same proof move in OTHER mathematical settings. Vary the
   domain (analysis, algebra, combinatorics, geometry) and the vocabulary.""")

ObliqMathOverviewSig = make_rlm_signature(
    strategy=_OBLIQ_MATH_STRATEGY_OVERVIEW,
    answer_desc=_MULTI_DOC_ANSWER,
    question_desc="A mathematical problem; find others sharing its proof meta-move.",
)


ObliqMathSig = make_rlm_signature(  # BEST: NDCG@10=0.132 (151q)
    strategy=_OBLIQ_MATH_STRATEGY,
    answer_desc=_MULTI_DOC_ANSWER,
    question_desc="A math problem whose abstract proof strategy you should find analogues for.",
)

# ─── Writing (analogues) ──────────────────────────────────────────────────────
# BEST: ObliqWritingSig (v5) — not fully evaluated yet
# Key insight: author-name search >> style-based search when name is inferable
# Embedding retrieval ceiling is low (8B NDCG@10=0.035); gap is a retrieval problem

_OBLIQ_WRITING_STRATEGY = """\
════════════════════════════════════════════════════════════
YOUR TASK — ANALOGUES: AUTHORSHIP (WRITING STYLE)
════════════════════════════════════════════════════════════

Given a text snippet, find OTHER snippets in the corpus written by the SAME
AUTHOR. The author may write about completely different topics — focus on
STYLE, not content.

**Step 1 — Identify the author.**
Read the snippet carefully. If you can infer the author's name, blog, or
affiliation from the text — search for that directly. If not, extract their
distinctive stylistic markers:
  • Characteristic vocabulary and phrasing choices
  • Sentence rhythm and syntactic patterns (long/short, subordinate clauses, etc.)
  • Hedging language and epistemic stance ("it seems", "one might argue", "I suspect")
  • Tone and rhetorical posture (ironic, earnest, dry, conversational, formal)
  • Punctuation habits and paragraph structure
  • Thematic preferences (recurring concerns, analogies, framings)

**Step 2 — Search using all available signals per hop.**
Each hop, pick ONE angle and write a query that captures it. Good angles include:
author name or blog, characteristic vocabulary, sentence rhythm, hedging style,
epistemic stance, tone, rhetorical posture, thematic preferences. Let your notes
from the previous hop guide which angle to try next.
NEVER repeat an angle already tried.

Each call to search_index() shows your remaining budget: [search N/8  K remaining].
Use your full budget — do not submit while angles remain unexplored.

**Step 3 — After each hop, take notes and generate the next query.**
Scan the retrieved snippets and note which look promising. Then use llm_query
to analyze the results and generate the next search angle:

  next_query = llm_query(
      f"ORIGINAL SNIPPET:\\n{question}\\n\\n"
      f"NOTES FROM PREVIOUS HOPS: {notes or 'None yet'}\\n"
      f"SNIPPETS FROM THIS HOP:\\n{snippets}\\n\\n"
      "What stylistic patterns do you observe? What search angle should we try "
      "next to find more texts by this author? Return ONLY the next search query."
  )

Keep notes and candidate doc_ids as working memory across all hops.

**Step 4 — SUBMIT** a newline-separated list of doc_ids ranked strongest to
weakest based on accumulated scores. No prose — ONLY doc_ids, one per line.
"""

ObliqWritingSig = make_rlm_signature(  # BEST: current writing strategy
    strategy=_OBLIQ_WRITING_STRATEGY,
    answer_desc=_MULTI_DOC_ANSWER,
    question_desc="A text snippet whose author you should find other writing samples for.",
)

# ─── Twitter (descriptive) ────────────────────────────────────────────────────
# BEST: ObliqTwitterSig (v1) — NDCG@10=0.0945, 281 questions
# Beats Gemini (+0.026) and Query Rewriter (+0.028); gap to Multi-Hop = −0.046

_OBLIQ_TWITTER_STRATEGY = """\
════════════════════════════════════════════════════════════
YOUR TASK — DESCRIPTIVE: IMPLICIT POLITICAL STANCE (TWITTER)
════════════════════════════════════════════════════════════

Find tweets that express a specific IMPLICIT political stance through irony,
hedging, or selective framing — WITHOUT stating it directly.

Each call to search_index() shows your remaining budget: [search N/8  K remaining].
Use your full budget — do not submit while angles remain unexplored.

**Strategy:**
1. Before searching, identify the implicit stance:
   • What belief is implied WITHOUT being stated?
   • What rhetorical move signals it? (irony, mock-praise, "so much for X",
     selective juxtaposition, conspicuous omission, sarcastic agreement)
   • What event/topic/actor is the context?

2. Each hop: search a DIFFERENT angle — vary the rhetorical device, the
   topic framing, the implied target. If top score < 0.20, angle is wrong:
   switch immediately.

3. SUBMIT a newline-separated list of doc_ids ONLY, ranked strongest to
   weakest. No prose — ONLY doc_ids.
"""

ObliqTwitterSig = make_rlm_signature(  # BEST: NDCG@10=0.0945 (281q)
    strategy=_OBLIQ_TWITTER_STRATEGY,
    answer_desc=_MULTI_DOC_ANSWER,
    question_desc="Description of an implicit political stance to find in tweets.",
)

# v2: v1 + cross-hop accumulator + listwise rerank of the top 30 before submitting.
# Motivated by measurement on the `submitall` run (183q): 61% of queries already had
# a gold somewhere in the submitted list, but the median gold sat at rank 27 — just
# outside NDCG@10's window. Perfect reranking of the top 30 alone is worth
# .0730 -> .1666, past the paper's Multi-Hop (.141). Submitting MORE does not help
# (submitall put out 121 ids and scored .0714, BELOW the 6-id baseline's .0946),
# because NDCG@10 only reads the first 10 slots. So the lever is ORDERING, not volume.
#
# Twitter is multi-gold (median 6 golds/query, up to 132), unlike Congress's single
# gold — every gold placed in the top 10 adds score, so the rerank has far more to
# win here than it did on Congress (+.094 vs +.011 ceiling).
_OBLIQ_TWITTER_STRATEGY_V2 = """\
════════════════════════════════════════════════════════════
YOUR TASK — DESCRIPTIVE: IMPLICIT POLITICAL STANCE (TWITTER, v2)
════════════════════════════════════════════════════════════

Find tweets that express a specific IMPLICIT political stance through irony,
hedging, or selective framing — WITHOUT stating it directly.

The relevant tweets will NOT state the position directly and will likely share
very little vocabulary with the query. For example:
  Query: "tweets implicitly defending strikes as legitimate retaliation"
  Match: "So when you poke the bear for decades, you're surprised it bites?"
Do not expect keyword overlap. Expect the STANCE to match while the words differ.

MANY tweets match each stance — typically 5 to 10, sometimes dozens. Scoring reads
only your FIRST 10 submitted ids, and every matching tweet you place there adds
score. A wrong tweet in slot 8 costs you nothing. So the goal is not to find one
perfect tweet, nor to submit hundreds — it is to get as many genuine matches as
possible into the first 10 positions.

Each call to search_index() shows your remaining budget: [search N/8  K remaining].
Use your full budget — do not submit while angles remain unexplored.

**Strategy:**
1. Before searching, identify the implicit stance:
   • What belief is implied WITHOUT being stated?
   • What rhetorical move signals it? (irony, mock-praise, "so much for X",
     selective juxtaposition, conspicuous omission, sarcastic agreement)
   • What event/topic/actor is the context?

2. Initialize an accumulator ONCE, before your first search:

  all_docs = {}    # doc_id -> best cosine score seen
  cand_text = {}   # doc_id -> tweet text, for the rerank in step 4

   After EVERY search_index call, record everything you saw:

  for r in results:
      all_docs[r['doc_id']] = max(all_docs.get(r['doc_id'], 0), r['score'])
      cand_text[r['doc_id']] = r['text']

   Without this, tweets found in early hops are lost by the time you submit.

3. Each hop: search a DIFFERENT angle — vary the rhetorical device, the topic
   framing, the implied target. Cosine scores on this task run LOW even for good
   matches, because the query describes a stance while the tweet performs it.
   Do NOT abandon an angle on a low score alone; judge the snippets themselves.

4. FINAL RERANK — after your last hop, before SUBMIT.

   Cosine order is a poor guide to whether a tweet *performs* the stance, so
   re-judge the strongest candidates by reading them. Rank by candidate NUMBER so
   a mistyped id cannot drop anyone.

  import json as _json, re as _re
  top30 = sorted(all_docs, key=all_docs.get, reverse=True)[:30]
  final = top30
  if len(top30) >= 2:
    try:
      numbered = "\\n".join(f"[{i}] {cand_text.get(d,'')[:300]}" for i, d in enumerate(top30))
      raw = llm_query(
          f"STANCE BEING SEARCHED FOR:\\n{question}\\n\\n"
          f"CANDIDATE TWEETS:\\n{numbered}\\n\\n"
          "Which of these tweets EXPRESS that stance implicitly — through sarcasm,\\n"
          "irony, hedging, euphemism, selective framing, understatement or a\\n"
          "rhetorical question? A matching tweet will NOT state the position\\n"
          "directly and may share almost no vocabulary with the description.\\n"
          "Rank ALL candidates, best match first. Several will genuinely match.\\n"
          f"Output all {len(top30)} candidates; every number 0 to {len(top30)-1}\\n"
          "must appear exactly once.\\n"
          'Return ONLY JSON: {"ranked":[{"candidate_num":<int>,"reason":"<brief>"}]}'
      )
      order = _json.loads(_re.search(r'\\{[\\s\\S]*\\}', raw, _re.DOTALL).group())["ranked"]
      seen, ranked = set(), []
      for item in order:
          i = int(item["candidate_num"])
          if 0 <= i < len(top30) and i not in seen:
              seen.add(i); ranked.append(top30[i])
      final = ranked + [d for i, d in enumerate(top30) if i not in seen]
    except Exception as _e:
      print("rerank failed:", _e)
  print(f"[rerank] {len(final)} reranked, top3={final[:3]}")

5. SUBMIT the reranked candidates first, then everything else by cosine score.
   The first 10 are what count, so the rerank's ordering matters most:

  rest = sorted((d for d in all_docs if d not in final), key=all_docs.get, reverse=True)
  SUBMIT("\\n".join(final + rest))
"""

# Twitter, batched first round. Identical to ObliqTwitterSig except for one REPL
# rule: the first code block may issue up to 3 searches together, then it reverts
# to one-per-block.
#
# Rationale from the logs: gold discovery is heavily front-loaded — 35% of
# successful Twitter questions surface a gold doc on search #1 and 75% within 3 —
# and the strategy tells the agent to try DIFFERENT angles each hop, so the first
# few queries are derived from the question rather than from prior results. Packing
# them into one turn removes 2 of ~10.5 turns (~19%) without reducing the number of
# searches, so no discovery is lost; only the adaptivity of queries 2-3 is given up.
_REPL_RULES_BATCH1 = _REPL_RULES.replace(
    "**Never loop over search_index.** Call search_index once per code block, not in a\n"
    "for-loop. One query → see results → decide next query in the next turn.",
    "**Batch your FIRST round of searches, then go one at a time.**\n"
    "In your FIRST code block only, issue up to 3 search_index calls together — pick 3\n"
    "genuinely different angles on the question and print all their results. This costs\n"
    "one turn instead of three.\n"
    "After that first block, call search_index at most once per code block: one query →\n"
    "see results → decide the next query in the next turn. Never use a for-loop over\n"
    "search_index beyond the first block.")

ObliqTwitterBatch1Sig = make_rlm_signature(
    strategy=_OBLIQ_TWITTER_STRATEGY,
    answer_desc=_MULTI_DOC_ANSWER,
    question_desc="Description of an implicit political stance to find in tweets.",
    repl_rules=_REPL_RULES_BATCH1,
)


ObliqTwitterV2Sig = make_rlm_signature(  # v1 + accumulator + top-30 listwise rerank
    strategy=_OBLIQ_TWITTER_STRATEGY_V2,
    answer_desc=_MULTI_DOC_ANSWER,
    question_desc="Description of an implicit political stance to find in tweets.",
)

# ─── WildChat (descriptive) ───────────────────────────────────────────────────
# BEST: ObliqWildchatV5Sig (nested RLM) — NDCG@10=0.0504, 40 questions
# +131% over flat agent; just below Gemini (−0.007); gap to Multi-Hop = −0.020

_READER_STRATEGY = """\
════════════════════════════════════════════════════════════
YOUR TASK — DOCUMENT VERIFICATION
════════════════════════════════════════════════════════════

You will receive a FAILURE MODE description, previous notes, and a list of
CANDIDATE doc_ids from a search hop. Your job: fetch each document, read the
full conversation, and score it for how strongly the AI exhibits the failure.

TOOLS: get_document(doc_id), llm_query(prompt).

**Strategy:**
1. For each doc_id in the candidates list, call get_document(doc_id).
2. Use llm_query to judge the full conversation against the failure description.
   Score: 1.0 = unmistakable failure, 0.5 = possible, 0.0 = clearly absent.
3. SUBMIT a JSON object — nothing else:
   {"candidates": {"doc_id": score, ...}, "notes": "what matched, what didn't, what to search for next"}
"""

_OBLIQ_WILDCHAT_STRATEGY_V5 = """\
════════════════════════════════════════════════════════════
YOUR TASK — DESCRIPTIVE: LLM FAILURE MODES (WILDCHAT)
════════════════════════════════════════════════════════════

Find Human–AI conversations where the AI exhibits a specific BEHAVIORAL
FAILURE MODE. The failure is visible in the transcript but never explicitly
named or acknowledged by either party.

You have TWO tools: search_index and llm_query. There is NO get_document.
Each hop uses TWO llm_query calls: one to generate an independent query,
one to extract notes from results. Never derive the next query from search
results — always generate it from the failure description and accumulated notes.

Each call to search_index() shows your remaining budget: [search N/8  K remaining].
Use your full budget — do not submit while angles remain unexplored.

════════════════════════════════════════════════════════════
MANDATORY PER-HOP PATTERN — TWO llm_query CALLS PER HOP
════════════════════════════════════════════════════════════

  # Initialize once at the very first turn:
  import json, re as _re
  notes = ""
  candidates = {}  # doc_id -> best relevance score seen (0.0–1.0)
  hop = 0

  # Each hop (one REPL turn):
  hop += 1

  # Call 1 — generate next query from failure description + notes only.
  # Do NOT derive from previous search results — stay independent.
  query = llm_query(
      f"You are finding conversations where an AI exhibits a behavioral failure.\\n"
      f"FAILURE MODE: {question}\\n"
      f"NOTES FROM PREVIOUS HOPS (includes what to search for next): {notes or 'None yet'}\\n"
      f"HOP: {hop} of 8\\n\\n"
      "Generate ONE search query following the 'what to search for next' guidance in the notes.\\n"
      "If no notes yet, start with the most direct signal of the failure.\\n"
      "Return ONLY the query string, nothing else."
  )
  print(f"Hop {hop} query: {query.strip()[:100]!r}")

  # Search
  results = search_index(query.strip(), k=20)

  # Call 2 — spawn inner RLM to read full documents and score them.
  import json as _json
  snippet_ids = [r['doc_id'] for r in results][:10]
  extraction = rlm_query(
      f"FAILURE MODE: {question}\\n"
      f"PREVIOUS NOTES: {notes or 'None yet'}\\n"
      f"CANDIDATE_IDS = {_json.dumps(snippet_ids)}\\n\\n"
      "Fetch ONLY the doc_ids listed in CANDIDATE_IDS above using get_document. "
      "Do NOT fetch any other IDs. Read each full conversation. "
      "Score 0.0-1.0 how strongly it exhibits the failure mode.\\n"
      'Return JSON: {"candidates": {"doc_id": score, ...}, '
      '"notes": "what matched, what didn\'t, what to search for next"}'
  )
  parsed = json.loads(_re.search(r'\\{[\\s\\S]*\\}', extraction).group())
  for doc_id, score in parsed.get("candidates", {}).items():
      candidates[doc_id] = max(candidates.get(doc_id, 0.0), float(score))
  notes = (notes + "\\n" + parsed.get("notes", "")).strip()
  print(f"  top={sorted(candidates, key=candidates.get, reverse=True)[:3]}  notes={parsed.get('notes','')[:80]!r}")

════════════════════════════════════════════════════════════
SUBMIT
════════════════════════════════════════════════════════════

After all 8 hops, sort by accumulated scores and SUBMIT:

  ranked = sorted(candidates, key=candidates.get, reverse=True)
  SUBMIT("\\n".join(ranked))

No prose — ONLY doc_ids, one per line.
"""

# EXPERIMENTAL: query reformulation pass before searching (not fully evaluated)
_OBLIQ_WILDCHAT_STRATEGY_V6 = """\
════════════════════════════════════════════════════════════
YOUR TASK — DESCRIPTIVE: LLM FAILURE MODES (WILDCHAT v6)
════════════════════════════════════════════════════════════

Find Human–AI conversations where the AI exhibits a specific BEHAVIORAL FAILURE MODE.
The failure is visible in the transcript but never explicitly named by either party.

CORE INSIGHT: The failure description uses evaluator/researcher language. The actual
conversations use natural, casual language. You must bridge this gap BEFORE searching.

TOOLS: search_index, llm_query, rlm_query. No get_document in outer.

════════════════════════════════════════════════════════════
TURN 0 — REFORMULATION (run once before any search)
════════════════════════════════════════════════════════════

  import json as _json, re as _re
  notes = ""
  candidates = {}  # doc_id -> best score seen
  hop = 0

  # Translate the abstract failure into concrete conversation signals
  reformulated = llm_query(
      f"TASK: Find raw Human-AI chat logs where the AI commits this failure.\\n"
      f"FAILURE DESCRIPTION: {question}\\n\\n"
      "The corpus is RAW conversations — users type casually, AIs respond naturally.\\n"
      "Neither party names or acknowledges the failure. You must find it from surface signals.\\n"
      "IMPORTANT: The corpus uses NATURAL LANGUAGE — NOT the abstract/technical vocabulary\\n"
      "of the failure description above. Map the description to actual conversation text.\\n\\n"
      "Do THREE things:\\n"
      "1. EXAMPLE: Write a realistic 2-turn conversation where this failure occurs\\n"
      "   (User: ... AI: ...). Use casual language, NOT research vocabulary.\\n"
      "2. SIGNALS: List 6 short phrases or words that would appear VERBATIM in matching\\n"
      "   conversations — things you could Ctrl+F for in the actual chat log.\\n"
      "3. ANGLES: Generate 8 short search queries (5-10 words each) using the NATURAL\\n"
      "   vocabulary from step 1 and 2, NOT the abstract vocabulary of the description.\\n\\n"
      '   BAD query: "machine-parseable structured output human framing"\\n'
      '   GOOD query: "here is the json of course happy to help"\\n\\n'
      'Return JSON: {"example": "User: ...\\nAI: ...", '
      '"signals": ["phrase1", ...], "angles": ["query1", ..., "query8"]}'
  )
  parsed0 = _json.loads(_re.search(r'\\{[\\s\\S]*\\}', reformulated, _re.DOTALL).group())
  pre_angles = parsed0.get("angles", [])
  signals    = parsed0.get("signals", [])
  conv_hint  = parsed0.get("example", "")
  print(f"  signals={signals[:3]}")
  print(f"  angles={pre_angles[:2]}")

════════════════════════════════════════════════════════════
PER-HOP PATTERN (8 hops, one REPL turn each)
════════════════════════════════════════════════════════════

  hop += 1

  # Use pre-generated angle (already in conversation vocabulary)
  query = pre_angles[hop - 1] if hop <= len(pre_angles) else llm_query(
      f"FAILURE MODE: {question}\\n"
      f"CONVERSATION EXAMPLE: {conv_hint}\\n"
      f"SIGNALS: {signals}\\nNOTES: {notes}\\n\\n"
      "Generate ONE search query using words from ACTUAL conversations (not research terms).\\n"
      "Return ONLY the query string."
  )
  results = search_index(query.strip(), k=20)

  snippet_ids = [r['doc_id'] for r in results][:10]
  extraction = rlm_query(
      f"FAILURE MODE: {question}\\n"
      f"PREVIOUS NOTES: {notes or 'None yet'}\\n"
      f"CANDIDATE_IDS = {_json.dumps(snippet_ids)}\\n\\n"
      "Fetch ONLY the doc_ids listed in CANDIDATE_IDS above using get_document. "
      "Do NOT fetch any other IDs. Read each full conversation. "
      "Score 0.0-1.0 how strongly it exhibits the failure mode.\\n"
      'Return JSON: {"candidates": {"doc_id": score, ...}, '
      '"notes": "what matched, what didn\'t, what to search for next"}'
  )
  parsed = _json.loads(_re.search(r'\\{[\\s\\S]*\\}', extraction).group())
  for doc_id, score in parsed.get("candidates", {}).items():
      candidates[doc_id] = max(candidates.get(doc_id, 0.0), float(score))
  notes = (notes + "\\n" + parsed.get("notes", "")).strip()[-2000:]
  print(f"  hop={hop} q={query.strip()[:60]!r}  top={sorted(candidates, key=candidates.get, reverse=True)[:3]}")

════════════════════════════════════════════════════════════
SUBMIT
════════════════════════════════════════════════════════════

After all 8 hops:

  ranked = sorted(candidates, key=candidates.get, reverse=True)
  SUBMIT("\\n".join(ranked))

No prose — ONLY doc_ids, one per line.
"""

ReaderSignature = make_rlm_signature(  # inner RLM for V5/V6: reads full docs, scores 0-1
    strategy=_READER_STRATEGY,
    answer_desc='JSON object mapping doc_id to relevance score, e.g. {"candidates": {"id1": 0.9, "id2": 0.3}, "notes": "..."}',
    question_desc="Failure mode description + list of candidate doc_ids to verify.",
    repl_tools=_REPL_TOOLS_NO_GETDOC,
    repl_rules=_REPL_RULES_NO_GETDOC,
)

ObliqWildchatV5Sig = make_rlm_signature(  # BEST: nested RLM, NDCG@10=0.0504 (40q)
    strategy=_OBLIQ_WILDCHAT_STRATEGY_V5,
    answer_desc=_MULTI_DOC_ANSWER,
    question_desc="Description of an LLM behavioral failure mode to find in conversations.",
    repl_tools=_REPL_TOOLS_NO_GETDOC,
    repl_rules=_REPL_RULES_NO_GETDOC,
)

ObliqWildchatV6Sig = make_rlm_signature(  # EXPERIMENTAL: reformulation pass + nested RLM
    strategy=_OBLIQ_WILDCHAT_STRATEGY_V6,
    answer_desc=_MULTI_DOC_ANSWER,
    question_desc="Description of an LLM behavioral failure mode to find in conversations.",
    repl_tools=_REPL_TOOLS_NO_GETDOC,
    repl_rules=_REPL_RULES_NO_GETDOC,
)


# v8: Free-form agent with full-document search (no get_document needed).
# search_index returns the entire conversation text — no snippet truncation.
# Model reads all top-10 docs per hop and scores them directly.
_OBLIQ_WILDCHAT_STRATEGY_V8 = """\
════════════════════════════════════════════════════════════
YOUR TASK — DESCRIPTIVE: LLM FAILURE MODES (WILDCHAT)
════════════════════════════════════════════════════════════

Find Human–AI conversations where the AI exhibits a specific BEHAVIORAL FAILURE MODE.
The failure is visible in what the AI says or does — but is never explicitly named or
acknowledged by either party. You are looking for conversations where it just HAPPENS,
silently.

Each call to search_index() shows your remaining budget: [search N/8  K remaining].
Use your full budget — do not submit while angles remain unexplored.

**Key insight:** search_index returns the FULL conversation text for each result.
Read every result carefully — the failure may appear anywhere in the exchange,
not just at the start.

**Strategy:**
1. Identify what LINGUISTIC SIGNALS in the conversation would reveal this failure:
   • What would the USER say that sets up the situation?
   • What would the AI OUTPUT look like when it exhibits the failure?
     (The failure is in the AI's behavior, not explicitly labelled.)
   • What phrasing, topic, or exchange pattern is distinctive?
   Name these signals concretely before your first search.
2. Search across multiple angles each hop. Vary between:
   • User-side signals: what the human asked or set up
   • AI-side signals: what the AI said or did in response
   • The topic/domain where the failure tends to occur
   If scores are below 0.25 and texts are clearly off-topic, switch angle immediately.
3. After each search, use llm_query to read the full conversations and score each one.
   Keep a running note: doc_id → score (0.0–1.0) based on how clearly the full
   conversation exhibits the failure. Pass entire doc texts to llm_query — it has
   large capacity and can handle multiple conversations at once.
4. After exhausting your search budget, SUBMIT your top 10 doc_ids ranked by score.
   No prose — ONLY doc_ids, one per line, strongest first.
"""

ObliqWildchatV8Sig = make_rlm_signature(  # v8: flat free-form agent, standard tools
    strategy=_OBLIQ_WILDCHAT_STRATEGY_V8,
    answer_desc=_MULTI_DOC_ANSWER,
    question_desc="Description of an LLM behavioral failure mode to find in conversations.",
)


# v9: Like paper's multi-hop — outer RLM fetches all top-10 docs per hop with
# get_document, then scores them jointly in ONE llm_query call (no inner RLM).
# Eliminates rlm_query overhead, enables cross-doc comparison per hop.
_OBLIQ_WILDCHAT_STRATEGY_V9 = """\
════════════════════════════════════════════════════════════
YOUR TASK — DESCRIPTIVE: LLM FAILURE MODES (WILDCHAT)
════════════════════════════════════════════════════════════

Find Human–AI conversations where the AI exhibits a specific BEHAVIORAL FAILURE MODE.
The failure is visible in what the AI says or does — but is never explicitly named or
acknowledged by either party.

Each call to search_index() shows your remaining budget: [search N/8  K remaining].
Use your full budget — do not submit while angles remain unexplored.

════════════════════════════════════════════════════════════
MANDATORY PER-HOP PATTERN
════════════════════════════════════════════════════════════

  import json as _json, re as _re
  notes = ""
  candidates = {}  # doc_id -> best score seen (0.0–1.0)
  hop = 0

  # ── Each hop (one REPL turn) ──────────────────────────────
  hop += 1

  # Step 1: generate next query from failure description + notes only
  query = llm_query(
      f"FAILURE MODE: {question}\\n"
      f"NOTES FROM PREVIOUS HOPS: {notes or 'None yet'}\\n"
      f"HOP: {hop}/8\\n\\n"
      "Generate ONE search query targeting the most distinctive signal of the failure.\\n"
      "Use natural conversation language — words that would appear verbatim in a chat log.\\n"
      "Return ONLY the query string."
  )
  print(f"Hop {hop} query: {query.strip()[:80]!r}")

  # Step 2: search
  results = search_index(query.strip())

  # Step 3: fetch ALL top-10 docs in full
  top10 = results[:10]
  fetched = {}
  for r in top10:
      doc = get_document(r['doc_id'])
      fetched[r['doc_id']] = doc.get('text', '')

  # Step 4: score ALL fetched docs jointly in ONE llm_query call
  docs_block = "\\n\\n---\\n\\n".join(
      f"DOC_ID: {doc_id}\\n{text[:4000]}" for doc_id, text in fetched.items()
  )
  parsed = _json.loads(_re.search(r'\\{[\\s\\S]*\\}', llm_query(
      f"FAILURE MODE: {question}\\n"
      f"PREVIOUS NOTES: {notes or 'None yet'}\\n\\n"
      f"Below are {len(fetched)} Human-AI conversations. Read ALL of them, then score\\n"
      f"each one 0.0–1.0 for how clearly the AI exhibits the failure mode.\\n"
      f"Compare them against each other — 1.0 = unmistakable, 0.0 = clearly absent.\\n\\n"
      f"{docs_block}\\n\\n"
      'Return JSON: {{"scores": {{"doc_id": score, ...}}, '
      '"notes": "what patterns matched, what angle to try next"}}'
  ), _re.DOTALL).group())

  for doc_id, score in parsed.get('scores', {}).items():
      candidates[doc_id] = max(candidates.get(doc_id, 0.0), float(score))
  notes = (notes + "\\n" + parsed.get('notes', '')).strip()[-2000:]
  top3 = sorted(candidates, key=candidates.get, reverse=True)[:3]
  print(f"  hop={hop}  top3={top3}  notes={parsed.get('notes','')[:60]!r}")

════════════════════════════════════════════════════════════
SUBMIT
════════════════════════════════════════════════════════════

After all 8 hops:

  ranked = sorted(candidates, key=candidates.get, reverse=True)
  SUBMIT("\\n".join(ranked))

No prose — ONLY doc_ids, one per line.
"""

ObliqWildchatV9Sig = make_rlm_signature(  # v9: get_document + joint llm_query per hop (no rlm_query)
    strategy=_OBLIQ_WILDCHAT_STRATEGY_V9,
    answer_desc=_MULTI_DOC_ANSWER,
    question_desc="Description of an LLM behavioral failure mode to find in conversations.",
)


# Give every signature class its own __name__ so runs record which one was used.
_NAME_SIGNATURES = {k: v for k, v in list(globals().items())
                    if isinstance(v, type) and getattr(v, "__name__", "") == "_Sig"}
for _n, _cls in _NAME_SIGNATURES.items():
    _cls.__name__ = _n
    _cls.__qualname__ = _n


# ─── v9: SUBTRACTIVE ─────────────────────────────────────────────────────────
# Every prompt change before this one ADDED instruction, and none of them helped:
# v6 mandated batching -8.0pp (p=0.002), v7 agent-directed batching -9.8pp (p=0.003),
# v8 read-first +0.5pp (p=1.00, 175/200 ties). v9 goes the other way and REMOVES.
#
# The reason, found 2026-08-29 in the BrowseComp+ paper (arXiv 2508.06600, App. E/G):
# the harness LightOn report 87.59% against gives its agent ~95 WORDS in total, and
# says NOTHING about when to call get_document — the tool is simply present. Our v2 is
# 512 words and contains four separate instructions that suppress reading:
#   "Use only when the snippet is promising but truncated"
#   "Never batch-fetch documents in a loop"
#   "If it is clearly off-topic, do NOT call get_document"
#   "Do not print huge document texts into the REPL" + "Push every long-context
#    operation into llm_query"   <- this one moves READING out of the agent entirely
# The last pair is architectural, not stylistic: our agent is told to delegate reading
# to an isolated sub-LLM call, while theirs reads the document in its own context.
#
# Our v2 already CONTAINS their prompt almost verbatim, in its YOUR TASK block. v9 is
# that block, plus only the REPL mechanics our harness genuinely requires, and nothing
# else. Identical toolset to v2 — the ONLY variable is how much we tell the agent.
#
# _ORCHESTRATION is appended unconditionally by make_rlm_signature, and it is one of the
# blocks being removed, so v9 is built directly instead of through that helper.

_BROWSECOMP_V9_PROMPT = """\
You interact with a Python REPL. Each turn you write a code block and see its output.
State persists across turns. print() is the only way to see anything.

  search_index(query: str, top_k: int = 5) -> list[dict]   # [{score, doc_id, text}, ...]
  get_document(doc_id: str) -> dict                        # full text of one document
  llm_query(prompt: str) -> str                            # one sub-LLM call
  llm_query_batched(prompts: list[str]) -> list[str]       # several, concurrently
  SUBMIT(answer=<str>)                                     # submit and stop

Every turn must execute code.
Call search_index once per code block, not in a for-loop.

You are a deep research agent. You need to answer the given question by interacting
with a search engine, using the search_index and get_document tools provided. Please
perform reasoning and use the tools step by step, in an interleaved manner.

Call SUBMIT(answer=...) with the exact name, date, title, or value the question asks
for. If it cannot be found, call SUBMIT(answer="Unknown").
"""


class BrowseCompSubtractiveSignature(dspy.Signature):
    __doc__ = _BROWSECOMP_V9_PROMPT
    question: str = dspy.InputField(desc="The research question to answer.")
    answer: str = dspy.OutputField(
        desc="Concise final answer (name, date, title, etc.). 'Unknown' if not found.")


BrowseCompSubtractiveSignature._sig_key = "v9"


# ─── v10: THE OUTPUT CONTRACT ────────────────────────────────────────────────
# The one variable from the BrowseComp+ harness that has never been tested here.
#
# Read verbatim 2026-08-29 from arXiv 2508.06600 Appendix G ("Search Prompt with
# Get-Doc") — the prompt behind the 87.59% row. It is 112 words, and HALF of it is a
# required response format:
#
#   Explanation: {{...cite your evidence documents inline by enclosing their docids in
#                 square brackets [] at the end of sentences. For example, [20].}}
#   Exact Answer: {{your succinct, final answer}}
#   Confidence: {{your confidence score between 0% and 100% for your answer}}
#
# Their agent CANNOT answer without grounding each claim in a document id and committing
# to a confidence. Ours says, in v2, literally "No explanation." — the exact inverse.
#
# Why this is a better candidate than the two mechanisms already tested and rejected:
# it does not ask the agent to read MORE (v8 did that: reading +40%, accuracy null, CI
# excluding +8pp) and it does not shorten the prompt (v9 did that: reading doubled,
# accuracy trending -8pp at n=75). It asks the agent to PROVE each claim against a
# specific document — which is the recognition step the failure analysis found broken:
# 63 of 359 questions where the gold WAS retrieved were still answered wrong.
#
# SINGLE VARIABLE. v10 is v2 with ONLY the submit block replaced. All 512 words remain,
# including the four reading suppressors, so any difference is attributable to the
# output contract and not to prompt volume. Our judge is already the official
# BrowseComp+ grader, which EXTRACTS the final answer from the response, so a verbose
# response is what it expects and scoring is unaffected.

_BROWSECOMP_V10_STRATEGY = _BROWSECOMP_MINIMAL_STRATEGY.replace(
    """Call SUBMIT(answer=...) with your succinct final answer: the exact name, date,
title, or value the question asks for. No explanation.
If the answer cannot be found after exhausting all search angles, call SUBMIT(answer="Unknown").""",
    """Call SUBMIT(answer=...) with a string in EXACTLY this format:

  Explanation: <your explanation for your final answer. For this explanation section
  only, cite your evidence documents inline by enclosing their doc_ids in square
  brackets [] at the end of sentences. For example, [94480].>
  Exact Answer: <your succinct, final answer — the exact name, date, title, or value
  the question asks for>
  Confidence: <your confidence score between 0% and 100% for your answer>

Every sentence in the Explanation that states a fact must carry the doc_id it came
from. If you cannot cite a document for a claim, you do not yet have the evidence for
it. If the answer cannot be found after exhausting all search angles, give
Exact Answer: Unknown.""")

assert _BROWSECOMP_V10_STRATEGY != _BROWSECOMP_MINIMAL_STRATEGY, "v10 submit-block replace failed"

BrowseCompCitationSignature = make_rlm_signature(
    strategy=_BROWSECOMP_V10_STRATEGY,
    answer_desc="Explanation with inline [doc_id] citations, then Exact Answer, then Confidence.",
    question_desc="The research question to answer.",
)
BrowseCompCitationSignature.__name__ = "BrowseCompCitationSignature"
BrowseCompCitationSignature._sig_key = "v10"


# ── v15: v10 + a SIZE-BASED rule against printing search snippets ───────────────────────
# THE DEFECT THIS TARGETS (measured on nobatch1 Q67, 2026-09-08). repl_history is an
# InputField: every prior turn's code AND printed output is re-serialised into every later
# prompt, so cost is the sum over turns of a growing history — quadratic in turns. On that
# question:
#     18 output blocks   399,140 chars   mean 22,174 each   <- printed SEARCH RESULTS
#     20 code blocks       9,300 chars                      <- 2% of history
#     predicted cost if re-sent every turn: 1,096,384 tok   ACTUAL: 1,020,830 tok  (7% error)
# 98% of what we pay to re-serialise is raw search text the model printed itself. Hosted MCP's
# whole message history for the same question holds 40,063 chars — 10x less material.
#
# WHY v10 DOES NOT ALREADY PREVENT IT. v10's ORCHESTRATION block says "Do not print huge
# DOCUMENT texts into the REPL". The model obeys that exactly — it never prints documents
# (llm_query fires ~1:1 with get_document). The rule simply does not mention search results,
# and search results are the 98%.
#
# WHY A SIZE THRESHOLD RATHER THAN A BAN. The get_document rule works because it is concrete
# and leaves an alternative. A blanket "never print snippets" removes the agent's ability to
# triage at all; measured, it already declines to print result text on 33% of searches, so it
# triages deliberately. This mirrors the rule it demonstrably complies with.
#
# NOT A MECHANICAL CLAMP. Five constraints on information-per-action have each been routed
# around with more actions (top_k 20->5: +84% searches/+64% tok; ceiling 60k->10k: +1.3 turns;
# snippet depth: model re-slices; search-per-turn 4 and 1: +7.5% iters/+19.8% tok). This one
# restricts no action, so there is nothing to route around — and the prompt layer is known to
# steer this agent hard (section 32: llm_query fired 1.84x/question purely because a docstring
# mentioned it).
_BROWSECOMP_V15_STRATEGY = _BROWSECOMP_V10_STRATEGY + """

════════════════════════════════════════════════════════════
KEEP SEARCH RESULTS IN VARIABLES, NOT IN STDOUT
════════════════════════════════════════════════════════════

search_index() RETURNS a list — the results are already in a Python variable. Printing
them copies every snippet into your history, where it is re-sent on every later turn.

To triage, print ONLY the compact fields:
    for r in results:
        print(r['score'], r['doc_id'])

Do NOT print snippet text longer than 300 characters. If you need to judge or extract
from the text of several results, pass the VARIABLE to llm_query and print only its
compact answer:
    verdict = llm_query(f"Which of these is about X? {results}")
    print(verdict)

A single printed search of 20 results costs ~22,000 characters of history. Twenty turns
of that is a million tokens. Keep stdout small; keep the data in variables.
"""

BrowseCompNoPrintSignature = make_rlm_signature(
    strategy=_BROWSECOMP_V15_STRATEGY,
    answer_desc="Explanation with inline [doc_id] citations, then Exact Answer, then Confidence.",
    question_desc="The research question to answer.",
)
BrowseCompNoPrintSignature.__name__ = "BrowseCompNoPrintSignature"
BrowseCompNoPrintSignature._sig_key = "v15"


# ── v16: v15 with the defect that blinded it removed ───────────────────────────────────
# WHAT WENT WRONG IN v15 (Q21, read from the trace 2026-09-08). The instruction's worked
# example was `for r in results: print(r['score'], r['doc_id'])`, and the agent treated it as
# the COMPLETE recipe -- it printed nothing else. Its stdout looked like:
#     1 55516  0.5 16091  0.333 38231  0.25 68944  0.2 84585  0.167 64519 ...
# Doc 68944 IS THE GOLD DOCUMENT and it sat at rank 4 of the FIRST search. The agent could not
# tell, because a bare id carries no signal. It searched 89 times over 86 turns looking at
# lists of numbers, never opened 68944, and answered "Unknown". gold_recall=1.00,
# evid_recall=1.00, 3,578,973 input tokens, 72 minutes, WRONG.
# Measured across v15: at least 31% of its search-and-print blocks printed score+doc_id with no
# text at all; 8 of 25 questions did it on nearly every block (Q14 32/37, Q34 27/29, Q53 30/31,
# Q67 30/32, Q80 36/44, Q84 15/16, Q32 26/26, Q72 8/8).
#
# THE FIX. Keep the size discipline -- that part worked, blocks fell from ~22,000 chars to ~650
# -- but require the TITLE LINE so triage is still possible. These corpus documents open with a
# header, `--- title: ... date: ... ---`, so text[:120] carries the title. That is what hosted
# MCP effectively shows: its search returns {docid, score, snippet} where snippet is the raw
# document opening, title included, ~1,743 chars x 5 results, all of it automatically in
# context. 20 results x 120 chars is ~2,400 chars per search -- still a 9x cut against v10's
# 22,000, and it restores the one thing v15 threw away.
_BROWSECOMP_V16_STRATEGY = _BROWSECOMP_V10_STRATEGY + """

════════════════════════════════════════════════════════════
KEEP SEARCH RESULTS IN VARIABLES — BUT ALWAYS PRINT THE TITLE
════════════════════════════════════════════════════════════

search_index() RETURNS a list — the results are already in a Python variable. Printing
whole snippets copies them into your history, where they are re-sent on every later turn.

Print a COMPACT LINE PER RESULT that always includes the title. Every document in this
corpus starts with a header like `--- title: ... date: ... ---`, so the first ~120
characters carry the title:

    for r in results:
        print(r['score'], r['doc_id'], r['text'][:120].replace('\n', ' '))

NEVER print only the score and doc_id. A bare id tells you nothing, and you will search
forever past a document you already retrieved. You MUST be able to read the title.

Do not print more than 120 characters of text per result while triaging. When a title
looks relevant, act on it: call get_document(doc_id) and read that document, or pass the
results VARIABLE to llm_query and print only its compact answer.

If the same documents keep coming back, stop searching and open the most promising one.
"""

BrowseCompTitleTriageSignature = make_rlm_signature(
    strategy=_BROWSECOMP_V16_STRATEGY,
    answer_desc="Explanation with inline [doc_id] citations, then Exact Answer, then Confidence.",
    question_desc="The research question to answer.",
)
BrowseCompTitleTriageSignature.__name__ = "BrowseCompTitleTriageSignature"
BrowseCompTitleTriageSignature._sig_key = "v16"


# ── v17: v10 WITH _REPL_RULES REMOVED ──────────────────────────────────────────────────
# WHY. Hosted MCP gives its agent TEN LINES total (bcp/search_agent/prompts.py
# QUERY_TEMPLATE) plus a nine-word get_document description, and no behavioural rules at
# all. We give ~4,300 characters, of which _REPL_RULES (992 ch) is five bolded imperatives
# with no counterpart on their side:
#     "Every turn must execute code. Planning in prose does nothing."
#     "Never loop over search_index."                      <- VIOLATED: 12% of turns batch,
#                                                             up to 20 searches per block,
#                                                             max 98 in one question
#     "Search before reasoning. Always retrieve first."    <- our invention; the RLM paper
#                                                             says nothing about action
#                                                             ordering, and dspy fires a
#                                                             reasoning field every turn
#                                                             regardless, so this suppresses
#                                                             nothing except pre-search prose
#     "Check snippets before fetching full documents."     <- VIOLATED on Q93: fetched a
#                                                             929,059-char memoir first and
#                                                             never cited it; MCP opened the
#                                                             two small docs first and won
#     "Do not reason over weak results (below ~0.30)."     <- an unmeasured threshold I asserted
#
# TWO OF FIVE RULES ARE DEMONSTRABLY IGNORED, which is the case for testing removal: unenforced
# instruction may be worse than none, since it consumes prompt without shaping behaviour.
#
# THE RATIO THIS TARGETS. Tool calls per reasoning step: ours 1.57, hosted MCP 0.94. MCP gets
# one reasoning turn per tool call because the Responses API enforces it. Ours comes from
# batching inside a turn — which "Never loop over search_index" already forbids and does not
# prevent.
#
# EXPECTATION, ON THE RECORD: .586, like the seven interventions before it. Five of eight arms
# on the hard 29 land on exactly 17/29, including hosted MCP. If v17 also lands there it is
# further evidence the ~.59 ceiling on those questions is not a prompt property at all.
_BROWSECOMP_V17_STRATEGY = _BROWSECOMP_V10_STRATEGY

BrowseCompNoRulesSignature = make_rlm_signature(
    strategy=_BROWSECOMP_V17_STRATEGY,
    answer_desc="Explanation with inline [doc_id] citations, then Exact Answer, then Confidence.",
    question_desc="The research question to answer.",
    repl_rules="",          # <- THE SINGLE VARIABLE: _REPL_RULES removed entirely
)
BrowseCompNoRulesSignature.__name__ = "BrowseCompNoRulesSignature"
BrowseCompNoRulesSignature._sig_key = "v17"


# ── v20: THE PAPER'S DECOMPOSITION GUIDANCE, ported to retrieval ───────────────────────
# THE ONE COMPONENT OF THE AUTHORS' PROMPT WE NEVER IMPLEMENTED, and the one their own
# ablation values highest. From the RLM paper (Figure 4a and Table): "we ablate how sensitive
# RLM behavior is to in-context decomposition examples in its system prompt... in-context RLM
# trajectories greatly improve both overall performance and the initial decomposition attempt,
# EVEN IF THE EXAMPLE IS UNRELATED TO THE ACTUAL TASK."
#     GPT-5.2 (base)                                38.7
#     RLM (GPT-5.2, depth=1)                        50.6
#     RLM (GPT-5.2, depth=1) + decomposition hints  65.6     +15.0
# Their prompt carries FOUR worked ```repl``` examples. Ours carries zero.
#
# TWO CHANGES, one hypothesis (adopt the paper's decomposition guidance):
#   1. REPLACE our "Every turn must execute code. Planning in prose does nothing — write the
#      code immediately." with their instruction, which says the OPPOSITE:
#        "Plan in prose, then execute one ```repl``` block every turn" (RLM_SYSTEM_PROMPT)
#        "pause and plan: state explicitly how the task decomposes into sub-LLM / REPL steps,
#         and sketch the concrete sequence of turns ... before you execute them"
#         (ORCHESTRATOR_ADDENDUM)
#   2. ADD worked examples.
#
# THE EXAMPLES ARE ADAPTED, NOT COPIED. Every example in their prompt chunks a `context`
# variable, because their setting puts the corpus IN the prompt (their BrowseComp+ run loads
# 1,000 documents, 6-11M tokens). We have a 100K-document corpus behind a retrieval tool and
# no `context` variable at all, so a verbatim copy would be incoherent. What ports is the
# STRUCTURE they demonstrate: plan first, write a programmatic strategy rather than ad-hoc
# turns, accumulate into buffers, branch on results, aggregate at the end.
#
# THE FAILURE THIS TARGETS is a decomposition failure, not a budget one. On Q93 our agent
# fetched a 929,059-char memoir as its FIRST document and never cited it, while hosted MCP
# opened the two small documents that held the answer and got it right. Trace review found
# the same shape repeatedly: 11 of snipmin's 17 failures had gold_recall = 1.00.
_BROWSECOMP_V20_STRATEGY = _BROWSECOMP_V10_STRATEGY + """

════════════════════════════════════════════════════════════
PLAN FIRST, THEN EXECUTE A PROGRAMMATIC STRATEGY
════════════════════════════════════════════════════════════

Before your first search, state your plan in prose: what the question is really asking,
which distinguishing detail is most searchable, and the sequence of turns you expect —
what each turn computes and what would make you change course. Then execute one code
block per turn, check the output, and continue.

Do not open documents until you have narrowed WHICH entity or event the question is
about. A search result list is for narrowing; a document is for confirming. Opening a
large document early is the most expensive mistake available to you.

WORKED EXAMPLE — narrow, then confirm, accumulating into a buffer:

    # Turn 1: the most distinctive phrase first, and look only at titles.
    hits = search_index('"open-air factory" Wexford Bull Ring')
    for r in hits:
        print(r['score'], r['doc_id'], r['text'][:120].replace('\\n', ' '))

    # Turn 2: a title looked right — confirm it, and keep only the verdict.
    doc = get_document('34106')
    facts = llm_query(f"From this document, extract: name, town, dates, "
                      f"and any mention of an open-air factory. "
                      f"Answer in under 80 words.\\n\\n{doc['text']}")
    print(facts)
    buffer = {'34106': facts}

    # Turn 3: branch on what you learned rather than repeating the same query.
    if 'Wexford' in facts:
        hits2 = search_index('Bull Ring Wexford 1798 monument sculptor')
    else:
        hits2 = search_index('tenterfields open-air cloth drying Wexford')

WORKED EXAMPLE — several candidates at once, one call:

    cands = [r['doc_id'] for r in hits if r['score'] > 0.4][:8]
    docs = [get_document(d)['text'][:40000] for d in cands]
    verdicts = llm_query_batched([
        f"Does this document describe a person who {criterion}? "
        f"Answer YES or NO and quote the sentence.\\n\\n{t}" for t in docs])
    for d, v in zip(cands, verdicts):
        print(d, v[:100])

WHEN NOT TO DELEGATE: if a snippet already contains the answer, or a Python substring
check over the results settles it, just read it — llm_query is for text that will not
fit or questions that need interpretation.

If two searches in a row return documents you have already seen, your query is not the
problem — your hypothesis is. Change the entity you are searching for, not the wording.
"""

BrowseCompDecompositionSignature = make_rlm_signature(
    strategy=_BROWSECOMP_V20_STRATEGY,
    answer_desc="Explanation with inline [doc_id] citations, then Exact Answer, then Confidence.",
    question_desc="The research question to answer.",
    # OUR RULES ARE REPLACED, NOT SUPPLEMENTED. The point of this arm is to run the paper's
    # guidance INSTEAD OF ours, so _REPL_RULES is deleted in full — the same 990-char deletion
    # v17 makes — and the PLAN FIRST block above stands in its place. Layering the paper's
    # examples on top of our five imperatives (the first version of this arm) would have kept
    # exactly the two rules the examples contradict:
    #   "Never loop over search_index"          vs  the paper's batched, programmatic strategy
    #   "Check snippets before fetching" (2000-char snippet)  — stale: these runs send 512 tokens
    # and would have carried v10's turn structure, hence v10's cost. v17 showed that deletion
    # alone is worth a median 0.45x on input tokens (fewer turns, more searches per turn), so
    # keeping the rules would also have confounded any cost reading of this arm.
    repl_rules="",
)
BrowseCompDecompositionSignature.__name__ = "BrowseCompDecompositionSignature"
BrowseCompDecompositionSignature._sig_key = "v20"


# ─── v11: FIX THE DELEGATION BOUNDARY ────────────────────────────────────────
# Not another instruction to read more. v8 (read-first), v9 (subtractive) and v10 (output
# contract) all changed WHAT the agent was told and all came back null, with their
# manipulations verified to have worked. v11 changes WHERE THE READING HAPPENS.
#
# The defect, found by tracing an actual run: search_index results ARE printed into the
# reasoning model's context (the model writes the print loop itself), but get_document
# results were NOT. The agent fetched a document into a Python variable, handed it to
# llm_query with its own framing of what to look for, and only the summary came back. The
# reasoning model never saw the raw text. If the agent's framing missed the answer, it was
# unrecoverable — the model saw "not found" and moved on. That is the boundary behind the
# 57 of 830 questions answered "Unknown" DESPITE the gold document having been retrieved.
#
# Three changes, paired with the harness change in rag/agent.py:
#   1. get_document now PRINTS documents under GETDOC_INLINE_CHARS (6000) in full, so the
#      reasoning model reads them itself. Gold documents are short — p50 1,297 tokens —
#      so most land in context. This mirrors the BrowseComp+ harness, whose tool results
#      are appended to the conversation by construction.
#   2. The orchestration block no longer tells the agent to push everything into llm_query
#      and not to print document text. That instruction is what created the boundary.
#   3. When llm_query IS used (long documents only), the agent must include the ORIGINAL
#      QUESTION VERBATIM alongside its sub-question, so the sub-LLM can recognise a
#      relevant answer even when the agent's framing is wrong.
#
# Everything else is v2 verbatim, so this is comparable to the .7880 arm.

_V11_ORCHESTRATION = """\
════════════════════════════════════════════════════════════
READING AND DELEGATION
════════════════════════════════════════════════════════════

get_document PRINTS short documents in full. When you call it, the text appears in
your own context between BEGIN/END markers — read it yourself. Do not summarise a
document you can already see; you are the one answering the question, and a summary
can only lose detail you would have noticed.

Only long documents are withheld (the call tells you which). For those, use
llm_query — and ALWAYS include the user's original question verbatim in the prompt,
not only your current sub-question:

    llm_query(f'''ORIGINAL QUESTION: {question}

    My specific sub-question: <what you are trying to establish right now>

    Document:
    {doc['text']}

    Answer the sub-question. ALSO report anything in this document that bears on the
    ORIGINAL QUESTION, even if I did not ask about it.''')

That last instruction matters: your sub-question encodes a guess about where the
answer lives. If the guess is wrong and the sub-LLM answers only what you asked, the
answer is lost and you will never know it was there.

Reserve your own tokens for deciding what to search next and when to commit.
"""

_BROWSECOMP_V11_STRATEGY = _BROWSECOMP_MINIMAL_STRATEGY

BrowseCompReadInContextSignature = make_rlm_signature(
    strategy=_BROWSECOMP_V11_STRATEGY,
    answer_desc="Concise final answer (name, date, title, etc.). 'Unknown' if not found.",
    question_desc="The research question to answer.",
    repl_rules=_REPL_RULES.replace(
        """**Check snippets before fetching full documents.**
Read the 2000-char snippet first. If it is clearly off-topic, do NOT call get_document.
Only fetch the full document when the snippet strongly suggests relevance.""",
        """**Fetch and READ promising documents.**
Read the snippet first. If it could plausibly bear on the question, call get_document —
short documents are printed in full for you to read directly. A snippet is a preview,
not evidence."""),
)
# make_rlm_signature appends _ORCHESTRATION unconditionally; swap it for the v11 block,
# which is the actual fix — the old text told the agent not to print document text and to
# push every long-context operation into llm_query.
BrowseCompReadInContextSignature.__doc__ = (
    BrowseCompReadInContextSignature.__doc__.replace(_ORCHESTRATION, _V11_ORCHESTRATION))
BrowseCompReadInContextSignature.__name__ = "BrowseCompReadInContextSignature"
BrowseCompReadInContextSignature._sig_key = "v11"


# v21 = the snippet RLM's prompt (2026-10-31 DSPy ablation, arm 3). The task text is LightOn's QUERY_TEMPLATE as
# the snippet RLM runs it (LIGHTON_TASK_TEXT=1), with SUBMIT in place of answer["ready"] and a short answer field
# instead of the Explanation / Exact Answer / Confidence block. None of the shared blocks (_REPL_TOOLS, _REPL_RULES,
# _ORCHESTRATION): no "one search per code block" rule, no score-threshold claim, no "Unknown" escape. Tool
# descriptions come from MINIMAL_TOOL_DOCS=1; the REPL description is dspy.RLM's own.
class BrowseCompSnippetPromptSignature(dspy.Signature):
    """You are a deep research agent. You need to answer the given question by interacting with a
search engine, using the search_index and get_document tools provided. Please perform
reasoning and use the tools step by step, in an interleaved manner. You may use the
search_index and get_document tools multiple times.

When you are done, call SUBMIT(answer=...) with the exact name, date, title, or value the question asks for."""
    question: str = dspy.InputField(desc="The research question to answer.")
    answer: str = dspy.OutputField(desc="Concise final answer (name, date, title, etc.).")


BrowseCompSnippetPromptSignature._sig_key = "v21"
