"""Staged patch — Design G: batch-1 review findings folded into the Design F surface.

Source: the third reviewer batch (traces 116,181,210,264,293,330,347,450) of eval100_E_high,
which completed the 24/24 taxonomy. Ranked general fixes it proposed and how each becomes
an environment fact or a tool (never per-question content, never gold-derived):

  G1  Answer form / name variants (F1 116, J1 264, plus batch-3's short-vs-full-name and
      alias-vs-formal-title cases): verify_answer() now also returns 'variants' — the
      distinct longer forms the retrieved documents use around the candidate's key word
      (full names, formal titles), with counts and docids. Pure Python over DOCS.
  G2  Locator queries at k<=10 missed documents that a larger k would have returned
      (R1 181, 293, 347): the search description states the measured recall-vs-k facts for
      this corpus (top-5 15%, top-30 33%, top-50 39% for a question-length query), that k
      does not change the cost of a call, and that grep_docs / unread_docs reach hits at
      ranks the agent has not read.
  G3  Constraint drop (D2 210) and premature commit (D3 330): the prologue states the
      task-general facts that stated conditions hold conjunctively, that a fit not seen in a
      retrieved document is a guess, and that answer['ready'] ends the run.

Apply only with no sweep in flight; after apply_design_f_patch.py.

POST-AUDIT EDITS (13 Sep 12:20, applied by hand after this script; the live source is
src/tools.py / src/run_one.py / src/prompts.py, not the strings below):
  - variants regex: capitalised token = initial or word not ending in "."; bounded {0,40}
    quantifiers (corpus has 50K-char whitespace-free tokens -> quadratic backtracking);
    hyphenated key ("Adams-Smith") captured.
  - verify_answer / unread_docs / grounding gate skip non-dict DOCS records; the gate falls
    back to the original capture on any exception (it must never block a submission).
  - search-description wording: recall@k (share of relevant documents) stated as such, with
    hit@k (chance of at least one) beside it — both from runs/first_stage_eval.json.
  - prologue: "ready ends the run" qualified by the once-only gate hold.
"""
import ast

P = "/home/extstu_00/suprlm/src/tools.py"
s = open(P).read()
assert "def verify_answer" in s and "def unread_docs" in s and "variants" not in s

# ---------------------------------------------------------------- G1: name/title variants
s = s.replace(
    '''        return {"exact": exact[:max_hits], "case_insensitive": ci[:max_hits],
                "closest_source_spellings": closest, "note": note}''',
    '''        # G1: the distinct longer forms the sources use around the candidate's key word
        # (e.g. "J. R. Smith" / "John Smith" / "Smith") — surface forms, no judgement.
        variants: dict[str, dict] = {}
        alpha = [w for w in _re.findall(r"[^\\W\\d_]+", text)]
        key = None
        if alpha:
            longs = [w for w in alpha if len(w) >= 4]
            key = longs[-1] if longs else max(alpha, key=len)
        if key:
            cap = r"[A-Z\\u00C0-\\u024F][\\w'\\u2019.-]*"
            pat = _re.compile(r"(?:%s\\s+){0,3}%s(?:\\s+%s){0,3}" % (cap, _re.escape(key), cap))
            for docid, rec in dict.items(self.DOCS):
                body = rec.get("text") or rec.get("snippet") or ""
                for m in pat.finditer(body):
                    v = " ".join(m.group(0).split())
                    if v.lower() == key.lower() and len(alpha) > 1:
                        continue  # the bare key word alone is not a variant of a multi-word candidate
                    ent = variants.setdefault(v, {"text": v, "count": 0, "docids": []})
                    ent["count"] += 1
                    if docid not in ent["docids"] and len(ent["docids"]) < 5:
                        ent["docids"].append(docid)
                    if len(variants) > 200:
                        break
        var_list = sorted(variants.values(), key=lambda e: -e["count"])[:max_hits]
        return {"exact": exact[:max_hits], "case_insensitive": ci[:max_hits],
                "closest_source_spellings": closest, "variants": var_list, "note": note}''',
)
assert '"variants": var_list' in s

# description of the new field
s = s.replace(
    '''                "the work rather than a nickname for it. Call this before setting answer['ready']."
            ),''',
    '''                "the work rather than a nickname for it. 'variants' lists, with counts and docids, "
                "the distinct longer forms the sources use around the candidate's key word — e.g. "
                "the full name behind a surname or the formal title behind a short one — so you "
                "can see which form the sources themselves use. Call this before setting "
                "answer['ready']."
            ),''',
)
assert "'variants' lists" in s

# ---------------------------------------------------------------- G2: recall-vs-k facts
anchor = "        if self.max_searches:\n            search_desc += ("
assert anchor in s
KFACTS = (
    '            search_desc += (\n'
    '                " One call costs one call whatever k is: hits are ranked, so k=30 returns the same "\n'
    '                "first five plus ranks 6-30. Measured on this corpus with a question-length query, a "\n'
    '                "relevant document is among the top-5 about 15% of the time, the top-30 about 33%, "\n'
    '                "the top-50 about 39%. Hits at ranks you have not read are still in DOCS, where "\n'
    '                "grep_docs (for a literal string from the question: a year, a number, a quoted "\n'
    '                "phrase, a distinctive term) and unread_docs reach them without another call."\n'
    '            )\n'
)
s = s.replace(anchor, KFACTS + anchor, 1)
open(P, "w").write(s)
ast.parse(s)
print("tools.py: verify_answer variants + recall-vs-k facts")

# ---------------------------------------------------------------- G3: task facts in the prologue
P2 = "/home/extstu_00/suprlm/src/prompts.py"
s2 = open(P2).read()
assert "TASK_FACTS" not in s2
s2 = s2.replace(
    '''    "results."
)
''',
    '''    "results."
)

# Task-general facts (Design G). True of any question that states several conditions;
# nothing here describes what BrowseComp questions or answers look like.
TASK_FACTS = (
    " When a question states several conditions, the answer satisfies all of them at once: a "
    "candidate that fits most of them and contradicts one is not the answer, and a condition "
    "you have not seen satisfied in a retrieved document is unverified, whatever your prior "
    "belief. Setting answer['ready'] ends the run; nothing can be checked afterwards."
)
USER_PROLOGUE = USER_PROLOGUE + TASK_FACTS
''',
    1,
)
s2 += '\nDESIGN_G = "variants+k-facts+task-facts"\n'
s2 = s2.replace("globals().get('DESIGN_F2', '')", "globals().get('DESIGN_F2', '') + globals().get('DESIGN_G', '')")
open(P2, "w").write(s2)
ast.parse(s2)
print("prompts.py: TASK_FACTS appended to USER_PROLOGUE; DESIGN_G marker. DESIGN G APPLIED.")
