"""Staged patch: Design D tool surface + context seeding + fixed_k recording fix.

Apply only when no sweep is in flight (worker subprocesses import tools.py /
run_one.py at start, so editing mid-sweep forks the configuration).

Design D = hybrid first stage + passage selection (see hybrid.py). The tool
returns, per document, the query-relevant windows with character offsets, and
the complete text still lands in DOCS. Everything the description says is a
statement about the environment: what the tool returns, what the REPL can do
with it. Nothing about the task or its answers.
"""

import ast

# ---------------------------------------------------------------------------
P = "/home/extstu_00/suprlm/src/tools.py"
s = open(P).read()
assert '"D"' not in s, "design D already present"

s = s.replace('        if design not in ("A", "B"):\n            raise ValueError(f"design must be \'A\' or \'B\', got {design!r}")',
              '        if design not in ("A", "B", "D"):\n            raise ValueError(f"design must be A, B or D, got {design!r}")')
s = s.replace('        self.allow_k_override = (design == "B") if allow_k_override is None else allow_k_override',
              '        # Design A keeps the official fixed k=5; B lets the model choose; D is a fixed\n'
              '        # first-stage k with passage selection, so k is not a model choice there either.\n'
              '        self.allow_k_override = (design == "B") if allow_k_override is None else allow_k_override')

# request: D uses hybrid first stage + passages mode
s = s.replace(
    '''                    "mode": "full" if self.design == "B" else "snippet",
                    "snippet_max_tokens": self.snippet_max_tokens,
                    "tag": self.tag,''',
    '''                    "mode": {"A": "snippet", "B": "full", "D": "passages"}[self.design],
                    "snippet_max_tokens": self.snippet_max_tokens,
                    "tag": self.tag,
                    "retriever": "hybrid" if self.design == "D" else "dense",
                    "n_passages": 15,
                    "per_doc": 3,''',
)

# description for D — inserted before the final `else:` (design B) branch
anchor = '        else:\n            search_desc = (\n                f"search(query: str, k: int = {self.default_k}) -> list[dict]. "'
assert anchor in s
s = s.replace(anchor, '''        elif self.design == "D":
            search_desc = (
                f"search(query: str) -> list[dict]. Perform a hybrid (dense + keyword) search on a "
                f"knowledge source. Retrieves the top-{self.default_k} documents, then selects the "
                f"passages inside them that best match the query, and returns one record per "
                f"document, ordered by best passage: {{'docid', 'score', 'head', 'n_chars', "
                f"'passages', 'text'}}. 'passages' is a list of up to 3 query-relevant windows, "
                f"each {{'offset', 'end', 'score', 'text'}} — 'offset'/'end' are character positions "
                f"into the document's complete 'text', so text[offset:end] is exactly that "
                f"passage. About 15 passages come back per call. 'head' is the first 300 characters "
                f"(title/lede). 'text' is the COMPLETE document; documents are long (median 10K, "
                f"mean 34K characters), so read passages directly, and use Python string search "
                f"(`in`, .find, re) or llm_query over 'text' rather than printing it. Every "
                f"returned document is merged into the REPL variable DOCS (docid -> record) and the "
                f"call is appended to SEARCH_LOG."
            )
''' + anchor)

# DOCS description: say what can be done with it (environment facts)
s = s.replace(
    '''            "DOCS": (
                "dict mapping docid -> the record for every document retrieved so far in this "
                "task (search hits and fetched documents). Grows as you search; inspect with "
                "list(DOCS) or DOCS[docid]."
            ),''',
    '''            "DOCS": (
                "dict mapping docid -> the record for every document retrieved so far in this "
                "task (search hits and fetched documents). Grows as you search; inspect with "
                "list(DOCS) or DOCS[docid]. Each record's 'text' is a plain Python string: "
                "keyword tests (`'x' in t`), t.find(), t.lower().count(), and re.search / "
                "re.finditer over it cost nothing and run over every document at once in a loop."
            ),''',
)
open(P, "w").write(s)
ast.parse(s)
print("tools.py: design D + environment-level DOCS description")

# ---------------------------------------------------------------------------
P2 = "/home/extstu_00/suprlm/src/run_one.py"
s2 = open(P2).read()
s2 = s2.replace('    ap.add_argument("--design", default="A", choices=["A", "B"])',
                '    ap.add_argument("--design", default="A", choices=["A", "B", "D"])')
# fixed_k recording (was never inserted)
s2 = s2.replace('            "hard_search_cap": args.max_searches or None,',
                '            "hard_search_cap": args.max_searches or None,\n'
                '            "fixed_k": args.default_k if args.fixed_k else None,')
# context seeding: run search #1 on the raw question before turn 1 (counted)
s2 = s2.replace('    ap.add_argument("--save-trajectory", action="store_true")',
                '    ap.add_argument("--seed-context", action="store_true",\n'
                '                    help="run search(question) before turn 1 and load the result as `context` "\n'
                '                         "(counted as search #1). Makes the stock RLM prompt\'s `context` framing "\n'
                '                         "true instead of a stub, and every leaderboard agent starts this way.")\n'
                '    ap.add_argument("--save-trajectory", action="store_true")')
s2 = s2.replace(
    '''        result = agent.completion(
            CONTEXT_NOTE, root_prompt=QUERY_TEMPLATE.format(Question=question)
        )''',
    '''        if args.seed_context:
            seed = toolbox.search(question)  # counted; goes through the same tag
            payload = {
                "note": ("`context` holds the results of the first search, issued with the "
                         "question verbatim. The same records are already in DOCS. Further "
                         "retrieval is via search / get_document; DOCS and SEARCH_LOG grow."),
                "first_search": seed,
            }
        else:
            payload = CONTEXT_NOTE
        result = agent.completion(
            payload, root_prompt=QUERY_TEMPLATE.format(Question=question)
        )''',
)
s2 = s2.replace('            "fixed_k": args.default_k if args.fixed_k else None,',
                '            "fixed_k": args.default_k if args.fixed_k else None,\n'
                '            "seed_context": args.seed_context,')
open(P2, "w").write(s2)
ast.parse(s2)
print("run_one.py: design D, fixed_k recorded, --seed-context")

# ---------------------------------------------------------------------------
P3 = "/home/extstu_00/suprlm/src/run_experiment.py"
s3 = open(P3).read()
s3 = s3.replace('    ap.add_argument("--design", default="A", choices=["A", "B"])',
                '    ap.add_argument("--design", default="A", choices=["A", "B", "D"])')
s3 = s3.replace('    ap.add_argument("--fixed-k", action="store_true")',
                '    ap.add_argument("--fixed-k", action="store_true")\n'
                '    ap.add_argument("--seed-context", action="store_true")')
s3 = s3.replace('        if args.fixed_k:\n            cmd.append("--fixed-k")',
                '        if args.fixed_k:\n            cmd.append("--fixed-k")\n'
                '        if args.seed_context:\n            cmd.append("--seed-context")')
open(P3, "w").write(s3)
ast.parse(s3)
print("run_experiment.py: design D, --seed-context")

# ---------------------------------------------------------------------------
P4 = "/home/extstu_00/suprlm/src/judge.py"
s4 = open(P4).read()
s4 = s4.replace('"search_budget_in_prompt", "orchestrator_addendum", "hard_search_cap", "fixed_k")},',
                '"search_budget_in_prompt", "orchestrator_addendum", "hard_search_cap", "fixed_k",\n'
                '                                 "seed_context")},')
open(P4, "w").write(s4)
ast.parse(s4)
print("judge.py: seed_context in config key")
print("ALL STAGED — apply only with no sweep in flight, then restart the server (apply_hybrid_patch first).")
