#!/usr/bin/env python
"""Automatic health check on a finished (or running) OBLIQ arm.

Refactored copy of original/RLM/scripts/trace_check.py (studies/prompt_optimisation/CHANGES_FROM_ORIGINAL.md).
Run: python obliq_twitter/src/trace_check.py '<result glob>' --traj <trajectory dir> --label X (called by the
  night7/night8 chains).
Changed: nothing but this paragraph (the script holds no machine-specific path).
Names: 'supervisor harness' / 'full-doc RLM' in the text below is the document REPL agent (agents/document_repl);
  'official RLM harness' / 'our harness' / run_obliq_mine.py is the snippet REPL agent (agents/snippet_repl,
  src/run_obliq.py).

Exit 0 = healthy, 1 = warnings, 2 = BROKEN (caller should stop rather than
produce numbers that look like results but are not).

The checks are the failure signatures this project actually hit:
  * harness exceptions  (`error` set)                       -> gpt-oss find_code_blocks(None)
  * lm_calls == 0                                           -> agent loop never ran
  * search_calls == 0                                       -> retrieval never ran
  * n_parsed == 0                                           -> nothing submitted, scores a fake 0
  * latency_s < 10                                          -> instant crash file masquerading as a result
  * every ndcg@10 == 0                                      -> scoring or index wired wrong
  * wasted-turn rate (turns emitting no code block)         -> the model talks instead of acting
  * empty model responses                                   -> harmony/channel-style breakage

NOT checked: `iterations`. It is computed at browsecomp_eval_official_rlm.py:1119 as
max(0, lm_calls - (llm_query_calls + llm_query_batched_prompts)), which double-counts
llm_query and clamps to 0 on perfectly healthy runs. lm_calls is the honest field.
"""
import json, glob, os, sys, time, argparse, statistics as st
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("results", help="glob for the arm's result JSONs, e.g. 'logs/q36_base_*_s*.json'")
ap.add_argument("--traj", default="", help="trajectory dir to sample")
ap.add_argument("--label", default="arm")
ap.add_argument("--min-n", type=int, default=0, help="BROKEN if fewer rows than this")
ap.add_argument("--traj-sample", type=int, default=40)
ap.add_argument("--settle-s", type=int, default=600,
                help="ignore trajectory files touched this recently -- they are still running")
a = ap.parse_args()

rows, bad_files = [], []
for f in sorted(glob.glob(a.results)):
    try:
        rows += json.load(open(f))
    except Exception as e:
        bad_files.append((f, str(e)[:60]))

P = lambda *x: print(*x, flush=True)
P(f"\n===== TRACE CHECK: {a.label} =====")
P(f"  pattern {a.results}  ->  {len(rows)} rows")
for f, e in bad_files:
    P(f"  !! unreadable {os.path.basename(f)}: {e}")

if not rows:
    P("  !! BROKEN: no results at all")
    sys.exit(2)

broken, warn = [], []
if bad_files:
    warn.append(f"{len(bad_files)} unreadable result file(s)")

n = len(rows)
g = lambda r, k, d=0: (r.get(k) if r.get(k) is not None else d)
errs   = [r for r in rows if r.get("error")]
nolm   = [r for r in rows if g(r, "lm_calls") == 0]
nosrch = [r for r in rows if g(r, "search_calls") == 0]
noparse= [r for r in rows if g(r, "n_parsed") == 0]
fast   = [r for r in rows if 0 < g(r, "latency_s", 999) < 10]
nd     = [float(g(r, "ndcg@10")) for r in rows]
dup    = [q for q, c in Counter(r.get("query_id") for r in rows).items() if c > 1]

P(f"  NDCG@10  mean={st.mean(nd):.4f}  zeros={sum(1 for x in nd if x==0)}/{n}"
  f"  gold_recall={st.mean([float(g(r,'gold_recall')) for r in rows]):.4f}")
P(f"  searches/q={st.mean([g(r,'search_calls') for r in rows]):.1f}"
  f"  lm_calls/q={st.mean([g(r,'lm_calls') for r in rows]):.1f}"
  f"  parsed/q={st.mean([g(r,'n_parsed') for r in rows]):.1f}"
  f"  latency={st.mean([g(r,'latency_s') for r in rows]):.0f}s")
fr = st.mean([g(r, "format_rejections") for r in rows])
P(f"  format_rejections/q={fr:.2f}  errors={len(errs)}  lm_calls==0: {len(nolm)}"
  f"  search==0: {len(nosrch)}  n_parsed==0: {len(noparse)}  sub-10s: {len(fast)}")

def frac(lst): return len(lst) / n
if a.min_n and n < a.min_n:        broken.append(f"only {n} rows, expected >= {a.min_n}")
if frac(errs)   > 0.30:            broken.append(f"{len(errs)}/{n} rows carry a harness error")
elif errs:                         warn.append(f"{len(errs)}/{n} rows carry a harness error")
if frac(nolm)   > 0.30:            broken.append(f"{len(nolm)}/{n} never called the LM")
elif nolm:                         warn.append(f"{len(nolm)}/{n} never called the LM")
if frac(nosrch) > 0.30:            broken.append(f"{len(nosrch)}/{n} never issued a search")
elif nosrch:                       warn.append(f"{len(nosrch)}/{n} never issued a search")
if frac(noparse)> 0.30:            broken.append(f"{len(noparse)}/{n} submitted nothing parseable")
elif noparse:                      warn.append(f"{len(noparse)}/{n} submitted nothing parseable")
# All-zero NDCG is only decisive when n is large enough. The 27B baseline scores exactly
# 0 on 62% of questions, so 2/2 zeros happens 38% of the time on a perfectly healthy model.
# Calling that BROKEN at n=3 would fail a smoke gate by coin flip.
if max(nd) == 0:
    p_chance = 0.62 ** n
    if p_chance < 0.05:
        broken.append(f"every one of {n} questions scored NDCG@10 = 0 "
                      f"(p={p_chance:.3f} under the 62% baseline zero rate)")
    else:
        warn.append(f"all {n} questions scored 0, but that is {p_chance:.0%} likely by chance "
                    f"at this n -- not decisive; the other signals look {'healthy' if not broken else 'bad'}")
if fast:                           warn.append(f"{len(fast)} sub-10s rows (likely crash files)")
if dup:                            warn.append(f"{len(dup)} duplicate query_id(s), e.g. {dup[:3]}")
if fr > 3:                         warn.append(f"format_rejections/q = {fr:.1f}: model often emits no usable code block")
if errs:
    for m, c in Counter(str(r["error"])[:70] for r in errs).most_common(3):
        P(f"    error x{c}: {m}")

# ---- trajectories: what the agent actually did, turn by turn ----
if a.traj and os.path.isdir(a.traj):
    # Exclude trajectories still being written: an in-flight question has no final answer
    # yet, and counting it as "never answered" turns every hourly check on a live run into
    # a false alarm.
    _now = time.time()
    files = [f for f in sorted(glob.glob(os.path.join(a.traj, "*.jsonl")), key=os.path.getmtime)
             if _now - os.path.getmtime(f) > a.settle_s][-a.traj_sample:]
    turns = wasted = empty = 0; nofinal = 0; looped = 0
    for f in files:
        last, rep, saw_final, tn = None, 0, False, 0
        for line in open(f, errors="ignore"):
            try: d = json.loads(line)
            except Exception: continue
            if d.get("type") != "iteration": continue
            turns += 1; tn += 1
            cb = d.get("code_blocks") or []
            if not cb: wasted += 1
            resp = d.get("response")
            if resp is None or (isinstance(resp, str) and not resp.strip()): empty += 1
            key = json.dumps(cb)[:400]
            rep = rep + 1 if (cb and key == last) else 0
            last = key
            if rep >= 4: looped += 1; rep = 0
            if d.get("final_answer"): saw_final = True
        if tn and not saw_final: nofinal += 1
    if turns:
        P(f"  traj: {len(files)} files, {turns} turns | no-code-block {wasted} ({wasted/turns:.0%})"
          f" | empty response {empty} | never answered {nofinal}/{len(files)} | stuck-loops {looped}")
        if wasted / turns > 0.50: broken.append(f"{wasted/turns:.0%} of turns emit no code block")
        elif wasted / turns > 0.20: warn.append(f"{wasted/turns:.0%} of turns emit no code block")
        if empty / turns > 0.10: broken.append(f"{empty/turns:.0%} of turns got an EMPTY model response")
        if looped: warn.append(f"{looped} stuck loops (same code block 5x running)")
        # `final_answer` is not populated on every harness path: GEPA rollouts routinely show
        # 40% "never answered" while their result rows carry n_parsed=30 and no errors. The
        # authoritative signal is n_parsed, so only trust this when the results agree.
        if files and nofinal / len(files) > 0.30:
            if noparse:
                warn.append(f"{nofinal}/{len(files)} trajectories never produced a final answer "
                            f"(corroborated: {len(noparse)} rows submitted nothing parseable)")
            else:
                P(f"  (note: {nofinal}/{len(files)} trajectories have no final_answer field, but "
                  f"every result row submitted doc_ids -- field not populated on this path)")
    else:
        warn.append(f"no trajectory turns found under {a.traj}")

for w in warn:   P(f"  ~~ WARN: {w}")
for b in broken: P(f"  !! BROKEN: {b}")
P(f"  VERDICT: {'BROKEN' if broken else ('WARN' if warn else 'OK')}")
sys.exit(2 if broken else (1 if warn else 0))
