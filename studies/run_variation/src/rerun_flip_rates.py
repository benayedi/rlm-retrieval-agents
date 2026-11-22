"""VAR-2: how much a plain re-run of the GPT-5 document REPL agent changes its answers on BrowseComp-Plus.

Every pair of runs in docs/supervisor_artifact.html (the embedded per-question trace of the earlier GPT-5 document REPL
runs) with the same reasoning effort and the same served model (gpt-5-2025-08-07) and at least 40 common questions is
compared: "break" = right in the first run, wrong in the second; "fix" = wrong in the first, right in the second.
It prints one line per pair (run names, n, break and fix counts and rates) and no question text. This is the
comparison point of SP-3 (the GPT-5 reranker pilot).

Caveat (studies/run_variation/README.md, KNOWN_ISSUES): pairs are matched only on effort and served model, not on
the whole configuration; e.g. gpt5_colbert_pair50 (the published 512-token ColBERT index) vs gpt5_colbert8192_pair50
(the 8,192-token index). All eight pairs ran at medium effort; the pilot ran at high.

From original/TRANSCRIPT/var2_flip_rates.py: the code was never saved as a file in the original repository. It was run
as a `python - <<'EOF'` heredoc in a Claude Code session (2026-10-27 02:05 UTC; source, line, tool_use id and SHA-256
in original/MANIFEST.json). original/TRANSCRIPT/var2_flip_rates_draft.py is the previous attempt in that session (same
day, 02:05:18; it filtered on six configuration fields and effort high and printed nothing). Changed only: the location
of the artifact (ORIGINAL_REPO_ROOT) and this documentation (studies/run_variation/CHANGES_FROM_ORIGINAL.md). Run:
  set -a; . configs/paths.env; set +a
  python studies/run_variation/src/rerun_flip_rates.py
"""
import re,json,itertools
import os,sys
# Refactor (paths): the original ran with cwd = the original repository and opened docs/supervisor_artifact.html.
# Here the file is read from ORIGINAL_REPO_ROOT/docs/; the encrypted copy in trajectories/run_variation/VAR-2/ unpacks
# to that layout (scripts/provenance/unpack_trajectories.py).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "agents", "shared"))
from rlm_shared.paths import path  # noqa: E402
t=open(os.path.join(str(path("ORIGINAL_REPO_ROOT")), "docs", "supervisor_artifact.html")).read(); dec=json.JSONDecoder()
runs={}
for m in re.finditer(r'"run": "([^"]+)", "label"', t):
    j=m.start(); depth=0
    while j>0:
        j-=1
        if t[j]=='}': depth+=1
        elif t[j]=='{':
            if depth==0: break
            depth-=1
    try: o,_=dec.raw_decode(t[j:])
    except Exception: continue
    if not o.get('questions'): continue
    cfg={c['k']:c['v'] for c in o['config']}
    runs[o['run']]=(cfg.get('reasoning effort'),cfg.get('model actually served'),{q['qid']:bool(q.get('correct')) for q in o['questions']})
print(len(runs))
for a,b in itertools.combinations(runs,2):
    ea,ma,A=runs[a]; eb,mb,B=runs[b]
    common=set(A)&set(B)
    if len(common)<40 or ea!=eb or ma!=mb or ma!='gpt-5-2025-08-07': continue
    ra=[q for q in common if A[q]]; wa=[q for q in common if not A[q]]
    brk=sum(not B[q] for q in ra); fix=sum(B[q] for q in wa)
    print(f"{ea:6s} {a:24s} vs {b:24s} n={len(common):3d} break {brk}/{len(ra)} ({brk/max(1,len(ra)):.2f}) fix {fix}/{len(wa)} ({fix/max(1,len(wa)):.2f})")
