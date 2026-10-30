"""Staged patch — Design E: RLM-side only. The retrieval index and its scoring
are untouched; every change is to how the RLM harness describes and uses them.

  1. Tool description states the retriever's CONTRACT, as measured: it is a dense
     semantic retriever that rewards descriptive natural-language queries. On 2,500
     agent-issued queries, <=12-word queries got R@5 10% / hit@10 32%; 13-20-word
     queries 15% / 48%; the full question 15% / 58%. A fact about the tool.
  2. DOCS description names what the REPL can do with retrieved text (string
     search, regex, slicing) — 0/100 runs used embedded_span, 24% used regex.
  3. `grep_docs(pattern, window)` helper: pure Python over DOCS, no retrieval, no
     LLM tokens. Returns {docid, offset, snippet} for every match.
  4. --seed-context (from the D patch): search #1 = the raw question, counted.
  5. --compaction / --max-timeout (from the rlm-side patch).

Apply only with no sweep in flight. Requires apply_design_d_patch.py and
apply_rlm_side_patch.py to have run first (they add the flags this relies on).
Design "D" (hybrid + passages) is NOT applied — out of scope by decision.
"""

import ast
import re

P = "/home/extstu_00/suprlm/src/tools.py"
s = open(P).read()
assert "grep_docs" not in s, "already patched"

# --- 1. retriever contract appended to every design's search description --------
CONTRACT = (
    '            search_desc += (\n'
    '                " The retriever is a dense semantic index (one embedding per document): it matches "\n'
    '                "the MEANING of a full natural-language description, not keywords. Measured on this "\n'
    '                "corpus, queries of 6-12 words recover about 10% of the relevant documents in their "\n'
    '                "top-5; queries of 13-20 words about 15%; a complete question-length description "\n'
    '                "about 15% with far higher chance of at least one hit. Write each query as a "\n'
    '                "descriptive sentence or two that states the entity type and the constraints you "\n'
    '                "are matching, rather than a short keyword string."\n'
    '            )\n'
)
anchor = "        if self.max_searches:\n            search_desc += ("
assert anchor in s
s = s.replace(anchor, CONTRACT + anchor, 1)

# --- 2. grep_docs description, anchored on the stable SEARCH_LOG entry --------------
#     (the D patch already rewrote the DOCS text to name string search / regex / spans,
#      so we do not touch DOCS here; we only add the grep_docs entry beside it)
anchor2 = '            "SEARCH_LOG": ('
assert anchor2 in s, "SEARCH_LOG description anchor missing"
s = s.replace(anchor2, '''            "grep_docs": (
                "grep_docs(pattern: str, window: int = 400, max_hits: int = 40, flags=None) -> "
                "list[dict]. Regex-search the 'text' of every document in DOCS and return "
                "{'docid', 'offset', 'snippet'} for each match, snippet = the +/- window characters "
                "around it (case-insensitive by default). Pure Python over what you already "
                "retrieved: no search call, no LLM tokens. Use it to check a constraint (a date, a "
                "name, a number) across all retrieved pages at once before deciding what to search "
                "for next."
            ),
''' + anchor2, 1)
assert s.count('"grep_docs": (') == 1, "grep_docs description not inserted exactly once"

# --- 3. grep_docs implementation + injection --------------------------------------
s = s.replace(
    "    def get_document(self, docid: str) -> dict | None:",
    '''    def grep_docs(self, pattern: str, window: int = 400, max_hits: int = 40, flags=None) -> list[dict]:
        """Regex over every retrieved document's text. No retrieval, no LLM."""
        import re as _re
        fl = _re.IGNORECASE if flags is None else flags
        try:
            rx = _re.compile(pattern, fl)
        except _re.error as e:
            return [{"error": f"bad regex: {e}"}]
        out = []
        for docid, rec in self.DOCS.items():
            text = rec.get("text") or rec.get("snippet") or ""
            for m in rx.finditer(text):
                a, b = max(0, m.start() - window), min(len(text), m.end() + window)
                out.append({"docid": docid, "offset": m.start(), "snippet": text[a:b]})
                if len(out) >= max_hits:
                    return out
        return out

    def get_document(self, docid: str) -> dict | None:''',
)
s = s.replace(
    '''        _search, _get_document = self.search, self.get_document

        def search(query, k=None):
            return _search(query, k)

        def get_document(docid):
            return _get_document(docid)
''',
    '''        _search, _get_document, _grep = self.search, self.get_document, self.grep_docs

        def search(query, k=None):
            return _search(query, k)

        def get_document(docid):
            return _get_document(docid)

        def grep_docs(pattern, window=400, max_hits=40, flags=None):
            return _grep(pattern, window, max_hits, flags)
''',
)
s = s.replace(
    '''            "get_document": {"tool": get_document, "description": d["get_document"]},''',
    '''            "get_document": {"tool": get_document, "description": d["get_document"]},
            "grep_docs": {"tool": grep_docs, "description": d["grep_docs"]},''',
)
assert '"grep_docs": {"tool": grep_docs' in s and '"grep_docs": (' in s, "grep_docs injection/description mismatch"
open(P, "w").write(s)
ast.parse(s)
print("tools.py: retriever contract + grep_docs (desc + injection)")

# --- prompt id must change so the judge's config check separates E from earlier runs
P2 = "/home/extstu_00/suprlm/src/prompts.py"
s2 = open(P2).read()
if "DESIGN_E" not in s2:
    s2 += '\n# Marker so prologue_id() distinguishes runs made with the Design E tool surface.\nDESIGN_E = "tool-contract+grep_docs+seed"\n'
    s2 = s2.replace("    blob = (QUERY_TEMPLATE + CONTEXT_NOTE + USER_PROLOGUE).encode()",
                    "    blob = (QUERY_TEMPLATE + CONTEXT_NOTE + USER_PROLOGUE + globals().get('DESIGN_E', '')).encode()")
    open(P2, "w").write(s2)
    ast.parse(s2)
    print("prompts.py: prompt_id now distinguishes Design E")
print("STAGED. Order: apply_design_d_patch.py -> apply_rlm_side_patch.py -> apply_design_e_patch.py (D's hybrid/passages mode is present in code but unused).")
