"""Staged patch: add a hard retrieval cap to ToolBox, and remove the last k-nudge.

Applied only after the in-flight runs finish, because editing tools.py mid-run
would give later worker subprocesses a different tool surface than earlier ones.

A hard cap is a declared scaffold constraint — the same kind of thing as the
official k=5 — not a prompt trick. It is enforced in the harness, stated
truthfully in the tool description, and every refused call is recorded, so the
reported average is bounded by construction and the reader can see exactly how
often the agent wanted more than it got.
"""

import re

P = "/home/extstu_00/suprlm/src/tools.py"
s = open(P).read()

# 1. Remove the residual "widening k is free" nudge flagged by the verification
#    audit. k stays available; the harness no longer editorialises about cost.
s = s.replace(
    "f\"merged into the REPL variable DOCS and appended to SEARCH_LOG. You choose k \"\n"
    "                f\"per call (1..{MAX_K}); larger k costs nothing extra per call.\"",
    "f\"merged into the REPL variable DOCS and appended to SEARCH_LOG. k is yours to \"\n"
    "                f\"choose per call, from 1 to {MAX_K}.\"",
)

# 2. Constructor: max_searches
s = s.replace(
    "        allow_k_override: bool | None = None,\n        tag: str | None = None,\n    ):",
    "        allow_k_override: bool | None = None,\n        tag: str | None = None,\n"
    "        max_searches: int = 0,\n    ):",
)
s = s.replace(
    "        self.tag = tag\n",
    "        self.tag = tag\n"
    "        # Hard retrieval cap. 0 = unlimited. A declared scaffold constraint:\n"
    "        # enforced here, stated in the tool description, and every refusal counted.\n"
    "        self.max_searches = max_searches\n",
)
s = s.replace(
    '        self.tool_call_counts = {"search": 0, "get_document": 0}',
    '        self.tool_call_counts = {"search": 0, "get_document": 0}\n'
    "        self.search_refused = 0",
)

# 3. Enforce the cap before any retrieval happens, and before the counter moves,
#    so a refused call is never counted as a search.
s = s.replace(
    '''    def search(self, query: str, k: int | None = None) -> list[dict]:
        """Retrieve from the corpus. Counted; results mirrored into DOCS/SEARCH_LOG."""
        self.tool_call_counts["search"] += 1''',
    '''    def search(self, query: str, k: int | None = None) -> list[dict]:
        """Retrieve from the corpus. Counted; results mirrored into DOCS/SEARCH_LOG."""
        if self.max_searches and self.tool_call_counts["search"] >= self.max_searches:
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
)

# 4. State the cap truthfully in the tool description.
s = s.replace(
    "        return {\n            \"search\": search_desc,",
    "        if self.max_searches:\n"
    "            search_desc += (\n"
    "                f\" You have a hard budget of {self.max_searches} search calls for this \"\n"
    "                f\"question; calls beyond that are refused and return an error instead of \"\n"
    "                f\"results, so plan the queries you spend them on.\"\n"
    "            )\n"
    "        return {\n            \"search\": search_desc,",
)

# 5. Retry only when the server cannot have served the request. An HTTPError or a
#    read timeout may mean the server already counted the call, so retrying there
#    would inflate the authoritative server-side counter.
s = s.replace(
    """        last = None
        for _ in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.load(r)
            except Exception as e:  # transient localhost hiccup
                last = e
        raise RuntimeError(f"retrieval server unreachable: {last}")""",
    """        last = None
        for _ in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.load(r)
            except urllib.error.URLError as e:
                # Connection never established: the server cannot have counted it,
                # so retrying is safe.
                if isinstance(e, urllib.error.HTTPError):
                    raise RuntimeError(f"retrieval server error {e.code}: {e}") from e
                last = e
            except Exception as e:
                # Anything else (read timeout, bad payload) may have been served
                # and counted already; retrying would double-count.
                raise RuntimeError(f"retrieval call failed: {e}") from e
        raise RuntimeError(f"retrieval server unreachable: {last}")""",
)

# 6. Fixed-k contract: when the model cannot choose k, say what the tool returns
#    instead of offering a choice. Mirrors how the official k=5 is described.
s = s.replace(
    "        else:\n            search_desc = (\n                f\"search(query: str, k: int = {self.default_k}) -> list[dict]. \"\n                f\"Perform a search on a knowledge source. Returns the top-k hits, each as \"",
    "        elif not self.allow_k_override:\n            search_desc = (\n                f\"search(query: str) -> list[dict]. \"\n                f\"Perform a search on a knowledge source. Returns the top-{self.default_k} hits, each as \"\n                f\"{{'docid', 'score', 'text', 'n_chars', 'embedded_span', 'snippet_span'}}. \"\n                f\"'text' is the COMPLETE document, not a snippet, so a hit needs no follow-up \"\n                f\"fetch. Documents are long (tens of thousands of characters) and {self.default_k} arrive \"\n                f\"per call — keep them in REPL variables and delegate reading to llm_query / \"\n                f\"llm_query_batched rather than printing them. 'embedded_span' is [start, end] \"\n                f\"character offsets into 'text' marking the region the retrieval vector was \"\n                f\"computed over: the index embedded only the first part of each document, so \"\n                f\"matches are driven by that region. 'snippet_span' marks the first \"\n                f\"{self.snippet_max_tokens} tokens. Results are merged into the REPL variable \"\n                f\"DOCS and appended to SEARCH_LOG.\"\n            )\n        else:\n            search_desc = (\n                f\"search(query: str, k: int = {self.default_k}) -> list[dict]. \"\n                f\"Perform a search on a knowledge source. Returns the top-k hits, each as \"",
)
assert "elif not self.allow_k_override" in s, "fixed-k branch not inserted"

open(P, "w").write(s)
print("tools.py patched: hard cap + k-nudge removed + retry semantics + fixed-k description")

# --- run_one.py / run_experiment.py: plumb --max-searches --------------------
P2 = "/home/extstu_00/suprlm/src/run_one.py"
s2 = open(P2).read()
if "--max-searches" not in s2:
    s2 = s2.replace(
        '    ap.add_argument("--search-budget", type=int, default=0,',
        '    ap.add_argument("--max-searches", type=int, default=0,\n'
        '                    help="hard cap on search calls; 0 = uncapped. Enforced by the harness "\n'
        '                         "and declared in the tool description.")\n'
        '    ap.add_argument("--search-budget", type=int, default=0,',
    )
    s2 = s2.replace(
        "        snippet_max_tokens=args.snippet_max_tokens,\n        tag=tag,\n    )",
        "        snippet_max_tokens=args.snippet_max_tokens,\n        tag=tag,\n"
        "        max_searches=args.max_searches,\n"
        "        allow_k_override=(False if args.fixed_k else None),\n    )",
    )
    s2 = s2.replace(
        '    ap.add_argument("--max-searches", type=int, default=0,',
        '    ap.add_argument("--fixed-k", action="store_true",\n'
        '                    help="pin k to --default-k for every search; the model cannot change it. "\n'
        '                         "A declared tool contract, like the official k=5.")\n'
        '    ap.add_argument("--max-searches", type=int, default=0,',
    )
    s2 = s2.replace(
        '            "hard_search_cap": args.max_searches or None,',
        '            "hard_search_cap": args.max_searches or None,\n'
        '            "fixed_k": args.default_k if args.fixed_k else None,',
    )
    s2 = s2.replace(
        '            "search_budget_in_prompt": args.search_budget or None,',
        '            "search_budget_in_prompt": args.search_budget or None,\n'
        '            "hard_search_cap": args.max_searches or None,',
    )
    s2 = s2.replace(
        '    record["tool_errors"] = toolbox.errors',
        '    record["search_refused"] = toolbox.search_refused\n'
        '    record["tool_errors"] = toolbox.errors',
    )
    s2 = s2.replace(
        'f"iters={record.get(\'n_iterations\')} {record[\'wall_time_s\']}s"',
        'f"refused={record[\'search_refused\']} "\n'
        '        f"iters={record.get(\'n_iterations\')} {record[\'wall_time_s\']}s"',
    )
    open(P2, "w").write(s2)
    print("run_one.py patched")

P3 = "/home/extstu_00/suprlm/src/run_experiment.py"
s3 = open(P3).read()
if "--max-searches" not in s3:
    s3 = s3.replace(
        '    ap.add_argument("--search-budget", type=int, default=0)',
        '    ap.add_argument("--search-budget", type=int, default=0)\n'
        '    ap.add_argument("--max-searches", type=int, default=0)',
    )
    s3 = s3.replace(
        '    ap.add_argument("--max-searches", type=int, default=0)',
        '    ap.add_argument("--max-searches", type=int, default=0)\n'
        '    ap.add_argument("--fixed-k", action="store_true")',
    )
    s3 = s3.replace(
        '        cmd += ["--search-budget", str(args.search_budget)]',
        '        cmd += ["--search-budget", str(args.search_budget),\n'
        '                "--max-searches", str(args.max_searches)]\n'
        '        if args.fixed_k:\n            cmd.append("--fixed-k")',
    )
    open(P3, "w").write(s3)
    print("run_experiment.py patched")

# --- judge.py: surface refusals and the cap ---------------------------------
P4 = "/home/extstu_00/suprlm/src/judge.py"
s4 = open(P4).read()
s4 = s4.replace(
    '"search_budget_in_prompt", "orchestrator_addendum")},',
    '"search_budget_in_prompt", "orchestrator_addendum", "hard_search_cap", "fixed_k")},',
)
if "search_refused" not in s4:
    s4 = s4.replace(
        '        "n_timeouts": sum(r.get("status") == "timeout" for r in runs),',
        '        "n_timeouts": sum(r.get("status") == "timeout" for r in runs),\n'
        '        "avg_search_refused": round(\n'
        '            sum(r.get("search_refused", 0) for r in real) / max(1, n_real), 2),\n'
        '        "n_hit_cap": sum(bool(r.get("search_refused")) for r in real),',
    )
    s4 = s4.replace(
        "    print(f\"  timeouts             {summary['n_timeouts']}\")",
        "    print(f\"  refused (over cap)   {summary['avg_search_refused']} avg, \"\n"
        "          f\"{summary['n_hit_cap']}/{n_real} queries hit the cap\")\n"
        "    print(f\"  timeouts             {summary['n_timeouts']}\")",
    )
    open(P4, "w").write(s4)
    print("judge.py patched")

# sanity
import ast
for f in ("tools.py", "run_one.py", "run_experiment.py", "judge.py"):
    ast.parse(open(f"/home/extstu_00/suprlm/src/{f}").read())
print("all parse OK")
assert "costs nothing extra per call" not in open(P).read(), "k-nudge still present!"
print("k-nudge confirmed removed")
