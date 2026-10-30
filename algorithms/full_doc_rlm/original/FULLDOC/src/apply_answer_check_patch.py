"""Staged patch — `verify_answer(text)` REPL helper targeting the F1_surface_form failure class.

Seven of 24 failures in eval100_E_high were the right entity in the wrong surface form
(plural, capitalisation, punctuation, truncated title, short vs full name). The task
template asks for an "Exact Answer"; the honest fix is fidelity to the source text the
agent already holds. This tool checks a candidate string against every retrieved
document and returns the source's own spellings — no gold, no judge knowledge, general.

Apply only with no sweep in flight; requires the E patch (grep_docs) to be present.
"""

import ast

P = "/home/extstu_00/suprlm/src/tools.py"
s = open(P).read()
assert "def grep_docs" in s and "verify_answer" not in s

# --- implementation, next to grep_docs --------------------------------------------
s = s.replace(
    "    def get_document(self, docid: str) -> dict | None:",
    '''    def verify_answer(self, text: str, max_hits: int = 8) -> dict:
        """Does this exact string occur in any retrieved document? If not, what are the
        closest spellings the sources actually use? Pure Python over DOCS."""
        import difflib
        import re as _re
        text = (text or "").strip()
        if not text:
            return {"exact": [], "case_insensitive": [], "closest_source_spellings": [], "note": "empty"}
        exact, ci, cands = [], [], {}
        low = text.lower()
        # a loose regex from the words of the candidate, tolerant to punctuation/plurals
        words = [_re.escape(w) for w in _re.findall(r"\\w+", text) if len(w) > 2]
        loose = _re.compile(r"[^\\n]{0,40}".join(words[:6]), _re.I) if words else None
        for docid, rec in self.DOCS.items():
            body = rec.get("text") or rec.get("snippet") or ""
            if text in body:
                exact.append(docid)
            elif low in body.lower():
                ci.append(docid)
            if loose and len(exact) + len(ci) < max_hits:
                for m in loose.finditer(body):
                    span = body[max(0, m.start() - 20): m.end() + 20].replace("\\n", " ")
                    cands[span.strip()] = docid
                    if len(cands) >= 3 * max_hits:
                        break
        # rank candidate spans by similarity to the candidate string
        ranked = sorted(cands.items(), key=lambda kv: -difflib.SequenceMatcher(None, low, kv[0].lower()).ratio())
        closest = [{"docid": d, "text": t} for t, d in ranked[:max_hits]]
        note = ("found verbatim" if exact else
                "found only with different capitalisation — copy the source's casing" if ci else
                "not found verbatim in any retrieved document — if this is a name or title, "
                "copy it exactly as a source writes it (see closest_source_spellings)")
        return {"exact": exact[:max_hits], "case_insensitive": ci[:max_hits],
                "closest_source_spellings": closest, "note": note}

    def get_document(self, docid: str) -> dict | None:''',
)

# --- closure injection ---------------------------------------------------------------
s = s.replace(
    "        _search, _get_document, _grep = self.search, self.get_document, self.grep_docs",
    "        _search, _get_document, _grep, _verify = self.search, self.get_document, self.grep_docs, self.verify_answer",
)
s = s.replace(
    '''        def grep_docs(pattern, window=400, max_hits=40, flags=None):
            return _grep(pattern, window, max_hits, flags)
''',
    '''        def grep_docs(pattern, window=400, max_hits=40, flags=None):
            return _grep(pattern, window, max_hits, flags)

        def verify_answer(text, max_hits=8):
            return _verify(text, max_hits)
''',
)
s = s.replace(
    '''            "grep_docs": {"tool": grep_docs, "description": d["grep_docs"]},''',
    '''            "grep_docs": {"tool": grep_docs, "description": d["grep_docs"]},
            "verify_answer": {"tool": verify_answer, "description": d["verify_answer"]},''',
)

# --- description (environment fact: the answer is compared as a string) -----------
anchor2 = '            "SEARCH_LOG": ('
s = s.replace(anchor2, '''            "verify_answer": (
                "verify_answer(text: str) -> dict. Check a candidate final answer against every "
                "retrieved document: returns the docids where it occurs verbatim, where it occurs "
                "only with different capitalisation, and the closest spellings the sources actually "
                "use. Your 'Exact Answer' is compared as a string, so names, titles and terms should "
                "be copied exactly as a source writes them (full form, original punctuation and "
                "casing, singular/plural as written). Call this before setting answer['ready']."
            ),
''' + anchor2, 1)
assert s.count('"verify_answer": (') == 1 and '"verify_answer": {"tool": verify_answer' in s
open(P, "w").write(s)
ast.parse(s)
print("tools.py: verify_answer helper + description")

P2 = "/home/extstu_00/suprlm/src/prompts.py"
s2 = open(P2).read()
if "DESIGN_F" not in s2:
    s2 += '\nDESIGN_F = "verify_answer"\n'
    s2 = s2.replace("globals().get('DESIGN_E', '')", "globals().get('DESIGN_E', '') + globals().get('DESIGN_F', '')")
    open(P2, "w").write(s2)
    ast.parse(s2)
    print("prompts.py: prompt_id distinguishes the verify_answer surface")
