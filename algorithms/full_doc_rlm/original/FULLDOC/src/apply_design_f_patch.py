"""Staged patch — Design F: fixes for the failure taxonomy, all RLM/harness-side.

Derived from a per-trace review of the 24 failures in eval100_E_high (three independent
reviewers). Every change is either a harness bug fix or a statement of fact about the
environment; none encodes anything about any question or answer.

  H1  Over-cap search() returned [{"error": ...}] -> `[h["docid"] for h in hits]` raised
      KeyError and discarded the whole turn (710 t19, 751 t20, 871 t23). Now returns []
      and PRINTS the budget message into the REPL stdout.
  H2  The agent miscounted its own searches by 1-2 and lost turns to it. Every search
      now prints "[search n/cap] k=.. -> m new documents (DOCS holds N)". Also exposes
      how many docs a query actually added (the R2 wasted-budget signal), as a fact.
  H3  globals()/eval/exec are None in the sandbox; probing them raised
      "TypeError: 'NoneType' object is not callable" and the agent concluded its tools
      were broken (751 t1-7, 754 t2-5). The tool description now states it.
  R3  Gold documents sat in DOCS unread for 13 turns (1094). DOCS now records which
      records the REPL actually accessed; `unread_docs()` lists the rest.
  F1  Right entity, wrong surface form (5/24 + 1 misspelling): `verify_answer` (separate
      staged patch) plus a SUBMISSION GROUNDING GATE in run_one: when answer["ready"]
      flips, if the "Exact Answer:" string occurs in no retrieved document, the harness
      prints a notice into the REPL and un-sets ready ONCE, so the agent can check the
      source spelling. No gold is consulted — only DOCS.
  R1/R2 Query-writing facts appended to the search description: what a query built
      around a name absent from question and DOCS is; what adding types/roles/outlets
      the question does not state does; that a query adding <2 new docs is exhausted.

Apply only with no sweep in flight; after apply_design_e_patch.py and
apply_answer_check_patch.py.
"""

import ast

P = "/home/extstu_00/suprlm/src/tools.py"
s = open(P).read()
assert "def grep_docs" in s and "def verify_answer" in s and "unread_docs" not in s

# ---------------------------------------------------------------- R3: access-tracking DOCS
s = s.replace(
    "class ToolBox:",
    '''class _TrackedDocs(dict):
    """A dict that remembers which docids the REPL has looked at. Lets the harness
    answer "which retrieved documents have you never opened?" — a fact, not a hint."""

    def __init__(self):
        super().__init__()
        self.accessed: set = set()

    def __getitem__(self, k):
        self.accessed.add(k)
        return super().__getitem__(k)

    def get(self, k, default=None):
        if k in self:
            self.accessed.add(k)
        return super().get(k, default)

    def items(self):
        # iterating all items counts as looking at all of them
        self.accessed.update(self.keys())
        return super().items()

    def values(self):
        self.accessed.update(self.keys())
        return super().values()


class ToolBox:''',
)
s = s.replace("        self.DOCS: dict[str, dict] = {}", "        self.DOCS: dict[str, dict] = _TrackedDocs()")

# ---------------------------------------------------------------- H1 + H2: search() return shape & counter print
s = s.replace(
    '''        if self.max_searches and self.tool_call_counts["search"] >= self.max_searches:
            self.search_refused += 1
            return [{
                "error": "retrieval_budget_exhausted",
                "message": (
                    f"You have used all {self.max_searches} search calls for this question. "
                    f"No further retrieval is available. Answer from the "
                    f"{len(self.DOCS)} documents already in DOCS."
                ),
            }]
        self.tool_call_counts["search"] += 1''',
    '''        if self.max_searches and self.tool_call_counts["search"] >= self.max_searches:
            self.search_refused += 1
            # Return an EMPTY list so `[h["docid"] for h in hits]` keeps working; the
            # message goes to the REPL's stdout, which the model reads.
            print(f"[search refused] all {self.max_searches} search calls are used for this "
                  f"question; no further retrieval. Answer from the {len(self.DOCS)} documents "
                  f"already in DOCS (unread_docs() lists the ones you have not opened).")
            return []
        self.tool_call_counts["search"] += 1''',
)
s = s.replace(
    '''        results = resp.get("results", [])
        self._record([r["docid"] for r in results])''',
    '''        results = resp.get("results", [])
        before = len(self._seen)
        self._record([r["docid"] for r in results])
        new_docs = len(self._seen) - before
        cap = f"/{self.max_searches}" if self.max_searches else ""
        print(f"[search {self.tool_call_counts['search']}{cap}] k={k_eff} -> {len(results)} hits, "
              f"{new_docs} new documents (DOCS holds {len(self._seen)})")''',
)

# ---------------------------------------------------------------- unread_docs helper + injection
s = s.replace(
    "    def verify_answer(self, text: str, max_hits: int = 8) -> dict:",
    '''    def unread_docs(self) -> list[dict]:
        """Retrieved documents the REPL has never accessed, with their heads."""
        out = []
        for docid in self.DOCS.keys():
            if docid in self.DOCS.accessed:
                continue
            rec = dict.__getitem__(self.DOCS, docid)
            text = rec.get("text") or rec.get("snippet") or ""
            out.append({"docid": docid, "n_chars": len(text), "head": text[:200].replace("\\n", " ")})
        return out

    def verify_answer(self, text: str, max_hits: int = 8) -> dict:''',
)
s = s.replace(
    "        _search, _get_document, _grep, _verify = self.search, self.get_document, self.grep_docs, self.verify_answer",
    "        _search, _get_document, _grep, _verify, _unread = self.search, self.get_document, self.grep_docs, self.verify_answer, self.unread_docs",
)
s = s.replace(
    '''        def verify_answer(text, max_hits=8):
            return _verify(text, max_hits)
''',
    '''        def verify_answer(text, max_hits=8):
            return _verify(text, max_hits)

        def unread_docs():
            return _unread()
''',
)
s = s.replace(
    '''            "verify_answer": {"tool": verify_answer, "description": d["verify_answer"]},''',
    '''            "verify_answer": {"tool": verify_answer, "description": d["verify_answer"]},
            "unread_docs": {"tool": unread_docs, "description": d["unread_docs"]},''',
)

# ---------------------------------------------------------------- descriptions: facts only
anchor2 = '            "SEARCH_LOG": ('
s = s.replace(anchor2, '''            "unread_docs": (
                "unread_docs() -> list[dict]. The retrieved documents this REPL session has never "
                "accessed (never indexed DOCS[docid] or iterated), as {'docid', 'n_chars', 'head'}. "
                "Every search adds documents; the ones you did not open are evidence you have not "
                "looked at."
            ),
''' + anchor2, 1)

# environment facts about the sandbox and about queries, appended to the search description
FACTS = (
    '            search_desc += (\n'
    '                " Each call prints \'[search n/cap] k=.. -> m new documents\' into the REPL output; a "\n'
    '                "query that adds fewer than two new documents has exhausted its facet, and another "\n'
    '                "paraphrase of it returns the same documents. Retrieval matches the words you send: a "\n'
    '                "query built around a name that appears in neither the question nor DOCS retrieves "\n'
    '                "pages about that name, i.e. it verifies your own guess rather than searching the "\n'
    '                "question; adding a type, role, outlet or place the question does not state (house, "\n'
    '                "governor, actor, a newspaper) restricts results to that addition. Sandbox note: "\n'
    '                "globals(), locals(), eval() and exec() are disabled in this REPL and return None; "\n'
    '                "the tools are ordinary names (search, get_document, grep_docs, verify_answer, "\n'
    '                "unread_docs, llm_query, llm_query_batched) and DOCS / SEARCH_LOG are ordinary "\n'
    '                "variables."\n'
    '            )\n'
)
anchor = "        if self.max_searches:\n            search_desc += ("
assert anchor in s
s = s.replace(anchor, FACTS + anchor, 1)

open(P, "w").write(s)
ast.parse(s)
print("tools.py: tracked DOCS + unread_docs, search() empty-return + counter print, sandbox/query facts")

# ---------------------------------------------------------------- F1: submission grounding gate (run_one)
P2 = "/home/extstu_00/suprlm/src/run_one.py"
s2 = open(P2).read()
assert "grounding" not in s2
s2 = s2.replace(
    "from rlm import RLM  # noqa: E402",
    '''from rlm import RLM  # noqa: E402
from rlm.environments import local_repl as _lr  # noqa: E402


def install_grounding_gate(toolbox):
    """Harness behaviour: when the agent flips answer["ready"], check that the string on
    its "Exact Answer:" line occurs in some retrieved document. If it does not, print a
    notice into the REPL and un-set ready ONCE so the agent can check the source spelling
    (verify_answer) before resubmitting. Consults only DOCS — never the gold answer."""
    import re as _re

    state = {"warned": False}
    orig = _lr.LocalREPL._capture_answer

    def _extract(content: str) -> str:
        m = _re.search(r"Exact Answer:\\s*(.+)", content or "", _re.I)
        return (m.group(1) if m else (content or "")).strip().strip("*").strip()

    def gated(self, content):
        ans = _extract(str(content))
        docs = toolbox.DOCS
        found = False
        if ans and len(ans) >= 2:
            low = ans.lower()
            for rec in dict.values(docs):
                body = (rec.get("text") or rec.get("snippet") or "")
                if low in body.lower():
                    found = True
                    break
        if not found and not state["warned"] and ans:
            state["warned"] = True
            print(f"[answer check] The string {ans!r} does not occur in any retrieved document. "
                  f"If it is a name, title or term, copy it exactly as a source writes it — run "
                  f"verify_answer({ans!r}) to see the sources' spellings — then set answer['ready'] "
                  f"again. This notice is shown once; a second submission is accepted as-is.")
            a = self.locals.get("answer")
            if isinstance(a, dict):
                dict.__setitem__(a, "ready", False)
            return  # do not capture; the run continues
        return orig(self, content)

    _lr.LocalREPL._capture_answer = gated''',
)
s2 = s2.replace(
    "    logger = RLMLogger()\n",
    "    logger = RLMLogger()\n    if args.grounding_gate:\n        install_grounding_gate(toolbox)\n",
)
s2 = s2.replace(
    '    ap.add_argument("--compaction", action="store_true",',
    '    ap.add_argument("--grounding-gate", action="store_true",\n'
    '                    help="once per run, hold a submission whose Exact Answer string occurs in no "\n'
    '                         "retrieved document and ask the agent to check the source spelling")\n'
    '    ap.add_argument("--compaction", action="store_true",',
)
s2 = s2.replace(
    '            "compaction": args.compaction,',
    '            "compaction": args.compaction,\n            "grounding_gate": args.grounding_gate,',
)
open(P2, "w").write(s2)
ast.parse(s2)
print("run_one.py: --grounding-gate (submission check against DOCS, once)")

P3 = "/home/extstu_00/suprlm/src/run_experiment.py"
s3 = open(P3).read()
if "--grounding-gate" not in s3:
    s3 = s3.replace('    ap.add_argument("--compaction", action="store_true")',
                    '    ap.add_argument("--compaction", action="store_true")\n    ap.add_argument("--grounding-gate", action="store_true")')
    s3 = s3.replace('        if args.compaction:\n            cmd.append("--compaction")',
                    '        if args.compaction:\n            cmd.append("--compaction")\n        if args.grounding_gate:\n            cmd.append("--grounding-gate")')
    open(P3, "w").write(s3)
    ast.parse(s3)
    print("run_experiment.py: --grounding-gate plumbed")

P4 = "/home/extstu_00/suprlm/src/judge.py"
s4 = open(P4).read()
s4 = s4.replace('"seed_context", "compaction", "max_timeout_s")},',
                '"seed_context", "compaction", "max_timeout_s", "grounding_gate")},')
open(P4, "w").write(s4)
ast.parse(s4)

P5 = "/home/extstu_00/suprlm/src/prompts.py"
s5 = open(P5).read()
if "DESIGN_F2" not in s5:
    s5 += '\nDESIGN_F2 = "tracked-docs+unread+counter-print+facts"\n'
    s5 = s5.replace("globals().get('DESIGN_F', '')", "globals().get('DESIGN_F', '') + globals().get('DESIGN_F2', '')")
    open(P5, "w").write(s5)
    ast.parse(s5)
print("judge/prompts: config key + prompt_id updated. DESIGN F STAGED.")
