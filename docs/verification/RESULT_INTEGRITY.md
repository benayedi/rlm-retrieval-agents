# Result integrity report

Date: 2026-10-30. Scope: every row of [`results/original_baselines/Experiments.xlsx`](../../results/original_baselines/Experiments.xlsx).

**What was done.** The headline numbers were recomputed from the per-question records the runs wrote, using code
written for this audit:
- [`scripts/audit/recompute_experiments.py`](../../scripts/audit/recompute_experiments.py), with the reference metrics
  in [`agents/shared/rlm_shared/metrics.py`](../../agents/shared/rlm_shared/metrics.py), which are tested in
  [`tests/test_metrics.py`](../../tests/test_metrics.py).
- Correctness comes from the gpt-4.1 verdicts already on disk. No judge or model was called, and no experiment was
  re-run.
- The per-question records live outside the repository, in the original machine's `logs/` (431 GB). The machine-readable
  output is in [`data/recompute_experiments.json`](data/recompute_experiments.json).

**What this establishes.** The workbook numbers follow from the stored per-question results under the stated
definitions.

**What it does not establish.**
- That the per-question results would be reproduced by re-running the agents: the agents are stochastic API calls with
  no seeds.
- That the gpt-4.1 verdicts themselves are right.

## 1. Recomputed vs reported

| Row | Agent | Reported accuracy / NDCG@10 | Recomputed | Reported recall | Recomputed recall (definition) | Reported searches | Recomputed | Verdict |
|---|---|---|---|---|---|---|---|---|
| BCP-9 | Snippet REPL | 0.794 | 0.7940 | 0.804 | 0.8040 (over results) | 29.5 | 29.46 | match |
| BCP-11 | Snippet REPL | 0.8578 | 0.8578 | 0.824 | 0.8239 (over results) | 22.3 | 22.33 | match |
| BCP-13 | Snippet REPL | 0.6747 | 0.6747 | 0.6815 | 0.6815 (over the 828 results) | 60.87 | 60.87 | match |
| BCP-15 | Snippet REPL | 0.380 † (276 done) | — | 0.4979 | — | 19.23 | — | still running, not audited |
| BCP-17 | Snippet REPL | — | — | — | — | — | — | not run yet |
| BCP-19 | Snippet REPL | 0.2494 | 0.2494 | 0.3571 | 0.3571 (over the 587 results) | 20.15 | 20.15 | match |
| BCP-21 | Document REPL | 0.8313 | 0.8313 | 0.7619 | 0.7619 (**evidence** recall) | 7.9 | 7.90 | match |
| BCP-23 | Document REPL | 0.8747 | 0.8747 | 0.7875 | 0.7875 (**evidence** recall) | 8.12 | 8.12 | match |
| BCP-25 | Document REPL | 0.6 | 0.6000 | 0.4973 | 0.4973 (gold recall, all 830 records incl. 4 unfinished) | 9.2 | 9.19 | match |
| BCP-27 | Document REPL | 0.4422 | 0.4422 | 0.3973 | 0.3973 (over the 817 records written; 13 killed runs left none) | 8.66 | 8.66 | match |
| BCP-29 | Document REPL | 0.5795 | 0.5795 | 0.5123 | 0.5123 (all 830 cache entries, incl. 53 errored or killed) | 8.6 | 8.57 | match |
| BCP-31 | Document REPL | 0.2241 | 0.2241 | 0.1806 | 0.1806 (all 830 cache entries, incl. 18 errored or killed) | 9.7 | 9.71 | match |
| FR-7 | Snippet REPL | 0.7633 | 0.7633 | 0.6716 | 0.6716 (over results = all 824) | 6.59 | 6.59 | match |
| FR-9 | Snippet REPL | 0.6201 | 0.6201 | 0.5752 | 0.5752 (over **all 824**; 0.6311 over the 751 results) | 12.85 | 12.85 | match |
| FR-11 | Document REPL | 0.807 | 0.8070 | 0.7039 | 0.7039 (over results = all 824) | 6.17 | 6.17 | match |
| FR-13 | Document REPL | 0.6104 | 0.6104 | 0.5981 | 0.5981 (over **all 824**; 0.6025 over the 818 results) | 7.6 | 7.62 | match |
| MAT-9 | Snippet REPL | 0.1489 | 0.1489 | 0.1409 | 0.1409 (Recall@10) | — | 28.26 | match |
| MAT-11 | Document REPL | 0.1590 | 0.1590 | 0.1361 | 0.1361 | — | 10.00 | match |
| TWI-9 | Snippet REPL | 0.1019 | 0.1019 | 0.1058 | 0.1058 | — | 34.36 | match |
| TWI-11 | Document REPL | 0.0950 | 0.0950 | 0.0951 | 0.0951 | — | 10.00 | match |
| CON-9 | Snippet REPL | 0.1876 | 0.1876 | 0.2008 | 0.2008 | — | 54.54 | match |
| CON-11 | Document REPL | 0.1332 | 0.1332 | 0.1496 | 0.1496 | — | 10.00 | match |

All 20 complete rows reproduce every audited number.

- Accuracy is correct answers over all questions; a missing result counts as wrong. This holds for every BrowseComp+
  and FRAMES row.
- OBLIQ NDCG@10 and Recall@10 were recomputed from the answer text with the binary NDCG of
  [`agents/shared/rlm_shared/obliq_data.py`](../../agents/shared/rlm_shared/obliq_data.py), checked against the independent reference.

## 2. Findings

### R1. The "Recall" column mixes five definitions (affects interpretation, not accuracy)

| Definition | Rows |
|---|---|
| Gold-document recall, mean over questions with a result | BCP-9, -11, -13, -19, FR-7, FR-11 |
| Gold-document recall, mean over all questions (missing = 0) | FR-9, FR-13 |
| Gold-document recall, mean over the records that exist (killed runs wrote none) | BCP-27 |
| Gold-document recall, mean over all records, including errored or killed runs with partial retrieval | BCP-25, -29, -31 |
| **Evidence**-document recall (evidence ⊇ gold), from the trace page | BCP-21, BCP-23 |

The differences within a row are small (≤ 0.06), but recall is not comparable across rows without restating it under one
definition. The document REPL GPT-5 dense and ColBERT rows (BCP-21, -23) can only be restated with access to their raw
records, which are not in the original repository: they come from the trace embedded in `docs/supervisor_artifact.html`.

**Recommendation:** footnote the column in the paper table, or restate it as gold recall over all questions where the
records allow.

### R2. Content-filtered verdicts counted as correct: latent, no affected question

`agents/shared/original/RLM/scripts/run5d/bcp_snip_grade.py:24` stores `bool(verdict)`. The judge returns the string
`"filtered"` when Azure's content filter blocks the prompt, so a filtered question would be counted correct.

Checked offline: filtered verdicts are never cached, so every answered question missing from `_judge_cache_gpt41.json`
would be one. **0** questions match in BCP-13, BCP-19 and the partial BCP-15. The re-judged rows (BCP-9, -11, -25, -29,
-31) hold only booleans. No reported number is affected. The bug is recorded with a proposed fix in [KNOWN_ISSUES.md](KNOWN_ISSUES.md);
the refactored grader keeps the original behaviour, like all refactored code.

### R3. Unfinished questions are wrong by construction (as documented)

Every row counts a question with no completed answer (time limit, watchdog, error) as wrong. Counts per row, from
the records:

| Row | Questions without a completed answer | Breakdown |
|---|---|---|
| BCP-19 | 243 of 830 | no record (1 h limit) |
| BCP-27 | 121 | 108 error + 13 killed |
| FR-9 | 73 of 824 | no record (2 h limit) |
| BCP-29 | 53 | errored, killed or empty |
| BCP-31 | 18 | errored, killed or empty |
| BCP-13 | 9 | 2 no record + 7 with an error |
| FR-13 | 6 | no record |
| BCP-25 | 4 | not completed |
| all other rows | 0 | |

This is the stated convention, not an error, but it means the Qwen3.6 rows measure "correct within the time budget".

### R4. Failure analysis treats BCP-25 recall as evidence recall

`Failure_Analysis.xlsx` uses evidence recall for BCP-25, while `Experiments.xlsx` reports gold recall (0.4973; evidence
recall 0.4606). Both are computed correctly; the failure-analysis note should say which one it uses. Its categories
F3 (no gold seen) and F5 (all gold seen) are the same under either definition only at 0 and 1, as its notes already say.

### R5. Provenance gaps

- **BCP-21 / BCP-23:** per-question data exist only as the trace embedded in the results page. The raw run records
  were not available to the audit, and the trace truncates answers at 1,500 characters.
- **BCP-11:** the launchers of the runs `rlmfix25/29/30/31` are not in the repository; only `rlmfix24` is.
- **No run records the random seed:** none is set; API models were sampled at their defaults.
- **The library version is assumed:** the `rlm` commit (`854e688`) is not recorded by September runs.

## 3. File integrity

- `scripts/provenance/preserve_originals.py --check`: every preserved source file matches its SHA-256 manifest.
- Duplicates: no question appears twice within a row's final record set.
  - Snippet REPL September runs: a retried question's later record replaces the earlier one, by file time.
  - BCP-21/23 trace: 0 questions appear in two runs.
- Verdict files contain only booleans, apart from the latent R2 path, which was never taken.
- [`results/original_baselines/Failure_Analysis.xlsx`](../../results/original_baselines/Failure_Analysis.xlsx) is
  redacted: no BrowseComp-Plus plain text, verified by
  [`scripts/provenance/bcp_leak_scan.py`](../../scripts/provenance/bcp_leak_scan.py).
