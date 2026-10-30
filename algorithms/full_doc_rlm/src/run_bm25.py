"""The Full-Doc RLM harness (run_one.py) with its retriever description made true for a BM25
index. run_one.py is not edited: this patches the strings in memory, then runs run_one.py as
__main__ with the same argv. From original/RLM/scripts/frames_sup_run_one.py.

Used by experiments/frames_qwen.sh (BM25_PROMPT=1), experiments/frames_gpt5.sh, experiments/bcp_gpt5_bm25.sh and
bcp_bm25_split.py (BM25_CORPUS=bcp): rows FR-11, FR-13, BCP-25, BCP-31.

The harness prompt was written for BrowseComp+ over a dense index (ColBERT / Qwen3-Embedding), and states
as fact things that are false over BM25: "dense semantic index ... not keywords", "write each
query as a descriptive sentence ... rather than a short keyword string", BrowseComp+ recall
figures "measured on this corpus", "the index embedded only the first part of each document",
"Retrieval is semantic", "long web pages". Only those sentences change; every other word, the
tools, caps and flags are as in the dense baseline. Each replacement is asserted, so an upstream
reword fails loudly instead of silently running the old text.

  cd $FULLDOC_ROOT && python <repo>/algorithms/full_doc_rlm/src/run_bm25.py --query-id 0 ... (run_one.py args)
"""
import os, runpy, sys

SUP_SRC = os.path.dirname(os.path.abspath(__file__))  # directory of run_one.py (original: /home/extstu_00/suprlm/src)
sys.path.insert(0, SUP_SRC)

# Shared with examples/frames_eval_rlm.py (FRAMES_BM25_LINE) so both harnesses are told the same.
# Facts only. The first version (2026-10-14) also advised "queries built from distinctive names
# ... work best; long descriptive sentences dilute those terms": on q0-99 it shortened queries
# 6.3 -> 4.7 words, raised exact-repeat queries 7% -> 12% and 30-turn-cap hits 28 -> 37, with no
# gain in gold recall (0.588 -> 0.580), and the dilution claim did not hold (full-question seed
# query hits gold in top-5 50% of the time, as often as short queries). Logs: sup_q36_bm25.
# BM25_CORPUS=bcp: the same facts for BrowseComp-Plus's ~100k web pages (scripts/bcp_bm25_test.py);
# only the corpus noun changes. Unset = FRAMES (English Wikipedia).
BCP = os.environ.get("BM25_CORPUS", "frames") == "bcp"
_CORPUS_DESC = ("a fixed corpus of about 100,000 web pages (one page per document, the whole page "
                "indexed)") if BCP else "English Wikipedia (one article per document, the whole article indexed)"
BM25_FACT = (
    f"The retriever is BM25 keyword search over {_CORPUS_DESC}: it scores documents by the words of "
    "your query, after stemming and stopword removal, not by their meaning, and quotation marks are "
    "ignored (a quoted phrase is not matched verbatim)."
)
# FRAMES_TITLE_SEARCH=1 (2026-10-21): one prompt clause, nothing else. The 25-search agent's retrieval misses
# (79 of 824 questions) are mostly long main articles; the agent names the entity inside a longer query and
# BM25 ranks shorter pages that mention it higher. Offline, the missed gold article's exact title ALONE puts
# it in the top 5 for 98 of the 177 missed articles (58 of the 79 questions). Unset = text unchanged.
TITLE_CLAUSE = (
    " To get the article about a named entity (a person, place, work, event or organisation), search its "
    "exact Wikipedia article title on its own, with no other words: with extra words in the query, BM25 "
    "often ranks shorter pages that mention the entity above its own long article."
)
if os.environ.get("FRAMES_TITLE_SEARCH") == "1" and not BCP:
    BM25_FACT += TITLE_CLAUSE
    print("[title-search] title-only search clause added to the search description", flush=True)

SEARCH_REPLACEMENTS = [
    ("Documents are long (tens of thousands of characters)",
     "Documents are " + ("web pages" if BCP else "Wikipedia articles") + ", often long (tens of thousands of characters)"),
    ("'embedded_span' is [start, end] character offsets into 'text' marking the region the "
     "retrieval vector was actually computed over: the index embedded only the first part of "
     "each document, so matches are driven by that region and the rest of the text was never "
     "seen by the retriever. ",
     "'embedded_span' is [start, end] character offsets into 'text' marking the region the "
     "retriever indexed, which is the whole document. "),
    ("The retriever is a dense semantic index (one embedding per document): it matches "
     "the MEANING of a full natural-language description, not keywords. Measured on this "
     "corpus, queries of 6-12 words recover about 10% of the relevant documents in their "
     "top-5; queries of 13-20 words about 15%; a complete question-length description "
     "about 15% with far higher chance of at least one hit. Write each query as a "
     "descriptive sentence or two that states the entity type and the constraints you "
     "are matching, rather than a short keyword string.",
     BM25_FACT),
    ("Measured on this corpus with question-length queries: "
     "the top-5 hold about 15% of the relevant documents (at least one of them 48% of the "
     "time), the top-30 about 33% (74%), the top-50 about 39% (80%). ",
     ""),
]

import prompts  # noqa: E402  (same module object run_one.py imports)
import tools  # noqa: E402

PROLOGUE_OLD = ("Retrieval is semantic, driven by the wording of each query, so a "
                "near-identical query returns near-identical documents.")
PROLOGUE_NEW = ("Retrieval is BM25 keyword search, driven by the words of each query, so a query "
                "with the same key terms returns near-identical documents.")
# USER_PROLOGUE is a GEPA block (scripts/gepa_frames.py), so under GEPA_PROMPTS it arrives already
# rewritten -- GEPA's seed carries PROLOGUE_NEW. Patch only when the dense sentence is present.
if os.environ.get("GEPA_PROMPTS"):
    if PROLOGUE_OLD in prompts.USER_PROLOGUE:
        prompts.USER_PROLOGUE = prompts.USER_PROLOGUE.replace(PROLOGUE_OLD, PROLOGUE_NEW)
else:
    assert PROLOGUE_OLD in prompts.USER_PROLOGUE, "USER_PROLOGUE anchor moved"
    prompts.USER_PROLOGUE = prompts.USER_PROLOGUE.replace(PROLOGUE_OLD, PROLOGUE_NEW)
    assert "Retrieval is semantic" not in prompts.USER_PROLOGUE, "USER_PROLOGUE not patched"
_old = "long web pages — tens of thousands of characters each —"
assert _old in prompts.SOLO_READ_SENTENCE, "SOLO_READ_SENTENCE anchor moved"
if not BCP:   # on BrowseComp-Plus "long web pages" is already true
    prompts.SOLO_READ_SENTENCE = prompts.SOLO_READ_SENTENCE.replace(
        _old, "Wikipedia articles, often tens of thousands of characters long —")

_orig_desc = tools.ToolBox.descriptions


def _bm25_descriptions(self):
    d = _orig_desc(self)
    s = d["search"]
    for old, new in SEARCH_REPLACEMENTS:
        assert old in s, f"search description anchor moved: {old[:60]!r}"
        s = s.replace(old, new, 1)
    assert "dense semantic" not in s and "Measured on this corpus" not in s
    d["search"] = s
    return d


tools.ToolBox.descriptions = _bm25_descriptions
print("[bm25-prompt] retriever description patched for BM25", flush=True)

# run_one.py's final json.dump(record) has no default=, so a trajectory holding a non-JSON value
# (FRAMES q42: the model left an Ellipsis `...` in a REPL variable) crashed the save after a
# finished run and lost the rollout. Fall back to repr() for such values; JSON output is otherwise
# byte-identical.
import json as _json  # noqa: E402

_orig_dump = _json.dump


def _dump(obj, fp, *a, **kw):
    kw.setdefault("default", repr)
    return _orig_dump(obj, fp, *a, **kw)


_json.dump = _dump

sys.argv[0] = f"{SUP_SRC}/run_one.py"
runpy.run_path(f"{SUP_SRC}/run_one.py", run_name="__main__")
