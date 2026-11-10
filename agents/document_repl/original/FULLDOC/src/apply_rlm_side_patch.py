"""Staged patch: plumb two RLM harness settings for the RLM-side round.

  --compaction     RLM(compaction=True): the library's own history compaction —
                   summarises the root model's message history when it nears the
                   context limit and keeps the full log in a REPL `history`
                   variable. Attacks the 96%-of-cost root-history churn.
  --max-timeout    already a run_one arg (default 1800 s) but not exposed by the
                   orchestrator. Two capped runs died at the 30-minute wall; the
                   official harness has no wall limit at all.

Both are recorded in the run config so the judge's consistency check covers them.
Apply only when no sweep is in flight.
"""

import ast

P = "/home/extstu_00/suprlm/src/run_one.py"
s = open(P).read()
assert "--compaction" not in s, "already patched"

s = s.replace(
    '    ap.add_argument("--seed-context", action="store_true",',
    '    ap.add_argument("--compaction", action="store_true",\n'
    '                    help="enable rlm\'s built-in history compaction (library feature).")\n'
    '    ap.add_argument("--seed-context", action="store_true",',
)
s = s.replace(
    "        orchestrator=args.orchestrator,\n",
    "        orchestrator=args.orchestrator,\n        compaction=args.compaction,\n",
)
s = s.replace(
    '            "seed_context": args.seed_context,',
    '            "seed_context": args.seed_context,\n'
    '            "compaction": args.compaction,\n'
    '            "max_timeout_s": args.max_timeout,',
)
open(P, "w").write(s)
ast.parse(s)
print("run_one.py: --compaction, config records compaction + max_timeout_s")

P2 = "/home/extstu_00/suprlm/src/run_experiment.py"
s2 = open(P2).read()
s2 = s2.replace(
    '    ap.add_argument("--seed-context", action="store_true")',
    '    ap.add_argument("--seed-context", action="store_true")\n'
    '    ap.add_argument("--compaction", action="store_true")\n'
    '    ap.add_argument("--max-timeout", type=float, default=1800.0,\n'
    '                    help="RLM wall limit per query in seconds (harness setting)")',
)
s2 = s2.replace(
    '        if args.seed_context:\n            cmd.append("--seed-context")',
    '        if args.seed_context:\n            cmd.append("--seed-context")\n'
    '        if args.compaction:\n            cmd.append("--compaction")\n'
    '        cmd += ["--max-timeout", str(args.max_timeout)]',
)
# the subprocess guard must exceed the RLM's own wall limit, or a stub is written
# while the worker is still finishing
s2 = s2.replace(
    '    ap.add_argument("--per-query-timeout", type=int, default=2400)',
    '    ap.add_argument("--per-query-timeout", type=int, default=None,\n'
    '                    help="subprocess guard; default = max_timeout + 600")',
)
s2 = s2.replace(
    '    run_dir = f"{ROOT}/runs/{args.name}"',
    '    if args.per_query_timeout is None:\n'
    '        args.per_query_timeout = int(args.max_timeout) + 600\n'
    '    run_dir = f"{ROOT}/runs/{args.name}"',
)
open(P2, "w").write(s2)
ast.parse(s2)
print("run_experiment.py: --compaction, --max-timeout, guard derived from it")

P3 = "/home/extstu_00/suprlm/src/judge.py"
s3 = open(P3).read()
s3 = s3.replace(
    '                                 "seed_context")},',
    '                                 "seed_context", "compaction", "max_timeout_s")},',
)
open(P3, "w").write(s3)
ast.parse(s3)
print("judge.py: compaction + max_timeout_s in config key")
