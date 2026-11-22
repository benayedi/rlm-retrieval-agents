# Reporting scripts

These scripts produced the per-question failure analysis (`Failure_Analysis.xlsx`) and two column sets of
`Experiments.xlsx`: Avg. Iterations Per Query and Input / Output Tokens Per Query. They are refactored copies of
scripts in the original repository. The logic is unchanged. Absolute paths became the variables below, and
`examples.*` imports became `rlm_shared.*` (see `agents/shared/CHANGES_FROM_ORIGINAL.md`).

The six `failure_analysis_{extract,build,findings,extras,turns_merge,turns_sheet}.py` follow the 2026-10-31 state of the
original scripts (`agents/shared/original/RLM_2026-11-11/`), which adds the completed Qwen3.6 snippet REPL rows BCP-15 and
BCP-17 and the Qwen3.6 pairs; `results/original_baselines/Failure_Analysis.xlsx` is the workbook they produced on
2026-10-31 (sanitised). The 2026-10-30 versions are in `agents/shared/original/RLM/` (KNOWN_ISSUES K35).

**They read run logs that exist only on the original machine.** Point the variables at a copy of those logs:

| variable | what is read |
|---|---|
| `LOGS_ROOT` | root of the original run logs (logs/ of the original repository): `run5d/full/<run>/`, `bcp_rejudge_gpt41/`, `bcp_gpt5_bm25/`, `frames/sup_q36_bm25f/`, the DSPy `*_q*.json` files, … |
| `ORIGINAL_REPO_ROOT` | `docs/supervisor_artifact.html`, plus `scripts/workbook/tokmaps/root_tok_*.json` (the token and turn maps of the September snippet REPL runs) |
| `FULLDOC_ROOT` | `data/prepared/qrel_evidence.json`, plus `runs/obliq_fd_<task>/` |
| `FRAMES_DATA_DIR` | `frames_rows.json` |
| `OBLIQ_BENCH_ROOT` | OBLIQ qrels (through `rlm_shared.obliq_data`) |

`failure_analysis_extract.py` also loads `Tevatron/browsecomp-plus` from Hugging Face to decrypt the questions and
gold answers in memory. The resulting `per_question.json` therefore contains BrowseComp-Plus questions and answers in
plain text. **Do not publish it.**

Load the settings first:

```bash
set -a; . configs/paths.env; set +a
```

## Failure analysis, in order

The order is extract → turns → turns_merge → build (+ findings, extras, turns_sheet) → eval → inject.

```bash
S=scripts/reporting; W=/path/to/workdir
# 1. extract: one record per question for every filled Experiments.xlsx row
python $S/failure_analysis_extract.py $W/per_question.json
# 2. turns: per-question turn counts of each run5d snippet REPL run, counted from its trajectory JSONL
for r in bcp_snip_gpt5_bm25 bcp_snip_qwen_bm25 frames_snip_gpt5 frames_snip_qwen \
         obliq_snip_math obliq_snip_twitter obliq_snip_congress; do
  python $S/failure_analysis_turns.py $LOGS_ROOT/run5d/full/$r/traj $W/turns/$r.json
done
# 3. turns_merge: add turns, turn budget and turn outcome to per_question.json (in place)
python $S/failure_analysis_turns_merge.py $W/per_question.json $W/turns
# 4. build: the workbook; the three fragments are exec()'d by build.py in this order
python $S/failure_analysis_build.py $W/per_question.json $W/Failure_Analysis.xlsx \
    $S/failure_analysis_findings.py $S/failure_analysis_extras.py $S/failure_analysis_turns_sheet.py
# 5. eval: evaluate the summary formulas (no LibreOffice needed)
python $S/failure_analysis_eval.py $W/Failure_Analysis.xlsx $W/vals.json
# 6. inject: write the evaluated values into the workbook as cached formula results
python $S/failure_analysis_inject.py $W/Failure_Analysis.xlsx $W/vals.json $W/Failure_Analysis.final.xlsx
```

`failure_analysis_turns.py` needs one output file per run, named as `failure_analysis_turns_merge.py` expects
(`<turns_dir>/<run>.json`). The two September snippet REPL rows (BrowseComp+ 9 and 11) do not use it. Their turns come from
`$ORIGINAL_REPO_ROOT/scripts/workbook/tokmaps/root_tok_rlm_traj{32,24}.json`.

`build.py` also writes `<out>.meta.json`.

**The published copy, `results/original_baselines/Failure_Analysis.xlsx`, is produced by
`scripts/provenance/sanitize_failure_analysis.py`.** It takes the step 4 output and `vals.json` from step 5, removes
the plain-text BrowseComp-Plus content and injects the values itself:

```bash
python scripts/provenance/sanitize_failure_analysis.py $W/Failure_Analysis.xlsx $W/vals.json $W/Failure_Analysis.public.xlsx
```

Its output must then pass `scripts/provenance/bcp_leak_scan.py`. Do not publish the output of step 6.

## Experiments.xlsx columns

```bash
python scripts/reporting/iteration_counts.py > iterations.json    # {sheet: {row: [avg turns, n]}}
python scripts/reporting/token_columns.py   > tokens.json         # {sheet: {row: [input, output, exact|estimate]}}
```

Neither script calls an LLM. Both print JSON to stdout. Copying the values into the workbook was a manual step.
