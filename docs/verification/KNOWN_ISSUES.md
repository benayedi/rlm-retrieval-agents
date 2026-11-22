# Known issues

Every issue found in the review (Phases 6–8), with what it means for the reported results.

**No issue was fixed in `src/`.** The refactored code must behave exactly like the code that produced the results
([REFACTOR_RULES.md](../architecture/REFACTOR_RULES.md)). Each fix is proposed here and should land as its own commit,
followed by a re-run of the affected experiment.

The detailed entries, with file:line for both the original and the refactored code, are in:
- [agents/snippet_repl/CHANGES_FROM_ORIGINAL.md](../../agents/snippet_repl/CHANGES_FROM_ORIGINAL.md) (**S**n)
- [agents/document_repl/CHANGES_FROM_ORIGINAL.md](../../agents/document_repl/CHANGES_FROM_ORIGINAL.md) (**F**n)
- [agents/dspy_rlm/CHANGES_FROM_ORIGINAL.md](../../agents/dspy_rlm/CHANGES_FROM_ORIGINAL.md) (**D**n)
- [agents/shared/CHANGES_FROM_ORIGINAL.md](../../agents/shared/CHANGES_FROM_ORIGINAL.md) (**G**n)

Result-level checks are in [RESULT_INTEGRITY.md](RESULT_INTEGRITY.md) (**R**n).

Severity:
- **A**: changes how a reported number must be read, or which configuration it belongs to.
- **B**: a documentation or methodology statement is wrong; the numbers are unaffected.
- **C**: latent bug, robustness or hygiene; no effect on any reported number.

## A. Affects how reported results must be read

| ID | Issue | Rows affected | Evidence | Proposed action |
|---|---|---|---|---|
| K1 | **Document REPL agent retrieves a model-chosen number of whole documents per search (1–50), not 5.** Design B allows `k` to be overridden (`MAX_K = 50`), and no launcher passes `--fixed-k`. | all document REPL BrowseComp+ / FRAMES rows | F1. Measured: mean 24.5 docs/search (BCP-25), 13.2 (FR-11), max 50 | Correct the method description (done in this repository's docs); report k next to the search count |
| K2 | **Snippet REPL vs document REPL rows differ in search cap, turn limit, time limit and ColBERT index length.** | every snippet REPL ↔ document REPL pair | S7, F1, registry | State that a pair compares packages, not reading strategies; for a causal claim, re-run with matched budgets |
| K3 | **`reasoning_effort` is dropped on Azure unless a hook forwards it.** rlm's Azure client sends only model + messages. | BCP-9, BCP-11 (root and sub-LLM at the deployment default, ≈ medium, though labelled high / medium); BCP-25 (default, already footnoted); BCP-21, BCP-23 unverified | S4, F3, `src/hooks/azure_sampling` | Correct the footnote for BCP-9/11 (done in the experiment documents); load the hook in every Azure launcher |
| K4 | **The Recall column mixes five definitions** (gold vs evidence documents, three denominators). | all BrowseComp+ / FRAMES rows | R1, G2 | Footnote the column, or restate as gold recall over all questions |
| K5 | **BCP-21 and BCP-23 ran on a patched build of the rlm library that was not preserved.** Their raw run records are not on this host; only a summary trace survives. BCP-23 logs up to 19 searches against a cap of 10, and BCP-21's launchers are lost. | BCP-21, BCP-23 | F12, R5, verified on the trace (2 questions above 10) | Obtain the run records and the patched build, or mark the two rows as not reproducible from this repository |
| K6 | **The OBLIQ snippet REPL rows ran with the BrowseComp answer-format gate on.** It rejects doc-id-only rankings up to 3 times per question. | MAT-9, TWI-9, CON-9 | S3. Verified: at least one rejection on 140/151, 279/281, 237/254 questions | A likely contributor to the 9–15% unusable rankings (O1). Re-run with `OBLIQ_NO_FORMAT_GATE=1` to measure it |
| K7 | **Unfinished questions are scored wrong** (time limit, watchdog, API error after retries). | BCP-19 (243), BCP-15 (190), BCP-27 (121), FR-9 (73), BCP-17 (68), BCP-29 (53), BCP-31 (18), BCP-13 (9), FR-13 (6), BCP-25 (4) | R3, S2 | Documented convention; the Qwen3.6 rows measure accuracy within the time budget |
| K8 | **The September snippet REPL rows (BCP-9, BCP-11) were produced by a harness version that is not preserved.** The closest copy (commit 7685538) builds identical prompts and tools under their settings. Four of five BCP-11 launchers are lost. | BCP-9, BCP-11 | S11 | Treat as a provenance gap; configs mark the unknown values |

## B. Wrong statements in documentation or methodology (numbers unaffected)

| ID | Issue | Where | Note |
|---|---|---|---|
| K9 | **"No sub-LLM" in document REPL agent is prompt-only.** `--no-delegation` is ignored by the stock library, which installs `llm_query` / `rlm_query` anyway. | F2 | A scan of every OBLIQ document REPL trajectory finds no call to them. The BrowseComp+/FRAMES trajectories were not scanned (records are 70 MB each) |
| K10 | **Document REPL tool text advertises tools and a retriever that are not there** (`read_docs`, `llm_query`, a dense retriever for ColBERT runs). The BM25 rows correct it in memory (`run_bm25.py`). | F8 | The model saw these sentences; they are part of the evaluated prompt. Measured on the OBLIQ trajectories: model code calls the non-existent `read_docs(` in 33/151 MAT-11, 17/281 TWI-11 and 37/254 CON-11 runs (each call raises `NameError`, costing a turn) |
| K11 | **Snippet REPL `iterations` / `input_tokens` fields are wrong** (usage dicts merged per model name). | S1 | The workbook already uses trajectory counts and `usage.jsonl`; documented in its footnote |
| K12 | **Document REPL `n_iterations` counts the forced final-answer call** (31 at the cap). | F4 | 154 BCP-25 runs. The workbook footnote states it |
| K13 | **`bcp_snip.sh` says GPT runs have no compaction, but passes `--compaction`.** | S8 | BCP-13 ran with compaction, BCP-9/11 without (registry corrected) |
| K14 | **The README and setup notes say `rlm` is pip-installed;** it is a checkout on `PYTHONPATH`. | inventory | This repository documents `RLMLIB_PATH` |
| K15 | **An in-run judge model came from an implicit `.env`** (gpt-5.4-mini), via `dspy_rlm.agent`'s `load_dotenv()`. | S9 | Only the in-run `correct` field; reported numbers are gpt-4.1 re-grades |

| K26 | **The BrowseComp-Plus BM25 snippet REPL rows (BCP-13, BCP-19) told the model the index is "a dense semantic retriever"** (`RETRIEVER_LINE` defaults to on; FRAMES swaps in a BM25 sentence). | Snippet REPL review N2 | Part of the evaluated prompt for those two rows; may have discouraged exact-phrase queries that BM25 rewards. Effect not measured |
| K27 | **Snippet REPL `top_k = 5` is a default the model can raise, and the harness itself has no character cap.** The effective snippet is min(2,000 chars, 512 tokens) because every preserved search server cuts `text` to 2,000 chars first; the :8001 dense server used by BCP-9 is not preserved, so its cut is unverified. | Snippet REPL review N4 (corrected) | BCP-19 raised k (up to 10) in 9 questions; BCP-13, FR-7 never did |

## C. Latent bugs and hygiene (no reported number affected)

| ID | Issue | Where | Checked |
|---|---|---|---|
| K16 | **A content-filtered verdict becomes "correct"** (`bool("filtered")`) in the snippet and document REPL graders. | G1, R2 | 0 affected questions in every graded run |
| K17 | **`heuristic_match` counts an empty prediction (or any substring of the gold) as correct.** | S12, D2 | Only the harness's in-run field; final grading never uses it |
| K18 | **Graders differ:** FRAMES has no retry round when all votes fail, uses `eval()` on gold ids, and detects the content filter more narrowly; there are three voting loops; ties count wrong. | G3–G5 | Verdict files hold only booleans (R2) |
| K19 | **Every OBLIQ snippet REPL question appears twice in `all.json`** (the shard merge picks up checkpoint files). | S10 | Means unaffected (every question exactly twice); the audit deduplicates |
| K20 | **DSPy.RLM:** v13–v20 signatures fall back to v1 with `--timeout > 0`; an OBLIQ timeout leaves its thread running; a large result can be recorded as TIMEOUT; Qwen runs log the Azure model name; `wildchat-v7` raises KeyError; the duplicated OBLIQ loaders differ. | D1, D4–D11 | DSPy.RLM has no row in the final table |
| K21 | **Document REPL hygiene:** `served_models` is always `None`; the partial answer is dropped on timeout; `prompt_id` does not hash the solo prompt; `run_bm25.py` patches `json.dump` process-wide; temporary prompt files are not deleted; the client-side search cap counts failed calls. | F5–F7, F10, F15, F16 | Provenance and robustness only |
| K22 | **Missing pieces:** the BCP-23 launchers call helper scripts that are not preserved (`assert_served_model.py`, `report.py`, `budget.py`); the :8001 dense server (`embed_server_new.py`) and two index builders are not in the original repository. | F13, G11, G12, [benchmarks/README.md](../../benchmarks/README.md) | Documented in the launchers and configs |
| K23 | **Duplicated code:** fence tolerance (×3), the Qwen REPL cue (×3), the search cap (×2), LM-timeout injection (×2, implemented differently), OBLIQ loaders (×2). | S5, D11 | Kept to preserve behaviour; consolidate after a re-run validates it |
| K24 | **Watchdog pricing** charges every model GPT-5 rates. | S13 | Affects only when the $10 per-question watchdog fires |
| K25 | **`benchmarks/retrieval/build_bm25_index.py` uses bm25s defaults (k1 1.5, b 0.75), not k1 0.9 / b 0.4.** | G13 | No reported row uses it: BrowseComp+ BM25 is served from the index built by `bcp_build_bm25_default.py` (k1 0.9, b 0.4, Lucene), FRAMES from `frames_build_bm25.py` (k1 0.9, b 0.4). The tuned k1 10 / b 1 index in the document REPL data directory is not used either. |
| K28 | **The snippet REPL format gate can be bypassed** by rebinding `answer`, and the answer forced at the turn limit is never gated. | Snippet REPL review N1 | Robustness of the gate only |
| K29 | **Qwen3.6 compaction threshold is computed against a 128k-token context** (library name table), about 108.8k tokens. | Snippet REPL review N3 | Applies to the Qwen3.6 snippet REPL rows with `--compaction` |
| K30 | **Document REPL minor findings:** the grounding gate always holds a one-character answer once; the over-cap tool text says "error" but returns `[]`; the `retriever` field is ignored by servers; the judge's one-configuration check ignores most flags; Design D is unused dead code. | Document REPL review F-new-1 to F-new-5 | No reported number affected |

## D. Follow-up studies (migrated 2026-11-11)

Found while migrating the follow-up studies ([FOLLOW_UP_STUDIES.md](../experiments/FOLLOW_UP_STUDIES.md)). Details,
with file:line, are in each study's `CHANGES_FROM_ORIGINAL.md` (**SP-I**n search policy, **PO-I**n prompt
optimisation, **RV-I**n run variation, **EA-I**n error analysis, **FB-I**n FRAMES baselines) and in the agents'
"Second snapshot" sections. Severity as above.

| ID | Sev. | Issue | Affects | Evidence | Proposed action |
|---|---|---|---|---|---|
| K31 | A | **Two sources disagree on the n of the FRAMES GEPA confirmation (GEPA-3).** The thesis evidence ledger reports 323 questions (63.2 vs 61.6 %); `gepa_frames_confirm.py` and its run.log say 324 unused questions. The script drops the one killed question (q529) from the pairing although its docstring says killed counts wrong. | GEPA-3 | PO-I3; `logs/gepa_frames_confirm/run.log` line 1; re-judge log (323 results) | Owner decides n = 323 (as computed) or 324 (missing = wrong); not changed here |
| K32 | A | **The GPT-5 search-policy pilot (SP-3) did not send reasoning effort.** Its launcher loads only the `retry429` hook, so rlm's Azure client dropped `reasoning_effort` (K3): GPT-5 ran at the deployment default. Its reference (the stored BCP-23 run) may have run at high; the re-run pairs of VAR-2 ran at medium. | SP-3, VAR-2 | `agents/document_repl/configs/SP-3.env`; registry SP-3 | State it next to the SP-3 result |
| K33 | A | **The re-run pairs of VAR-2 are not identical configurations.** The eight pairs agree only on effort (medium) and served model; e.g. one pair compares the published 512-token ColBERT index with the 8,192-token index. A first attempt that required identical configurations at high effort found no pair. | VAR-2 (the comparison point of SP-3) | RV-I1; `studies/run_variation/original/` | Call the ranges "re-run and configuration variation" |
| K34 | A | **The search-policy proxy changed during SP-1 and SP-2.** The version that served SP-2 and the test200 slice of SP-1 (2026-10-10) predates cross-request batching and is not preserved; which file state served each later slice is not recorded. Batched scoring can change scores slightly (padding). | SP-1, SP-2 | SP-I1; `studies/search_policy/original/RLM/logs/restart_1013/bcp_rerank_proxy.v3.py` (10-13 backup) | Provenance gap; reproduction uses the preserved version (`RERANK_XBATCH=0` for the 10-10 slices) |
| K35 | A | **Resolved 2026-11-11: `Failure_Analysis.xlsx` replaced by the 2026-10-31 state.** Was: The original repository rebuilt it on 2026-10-31 with the completed Qwen3.6 snippet rows (BCP-15, BCP-17, BCP-19 on all 830); the workbook here has BCP-19 partial and no BCP-15/17. `scripts/reporting/` matches the workbook here; the later scripts are in `agents/shared/original/RLM_2026-11-11/`. | Failure analysis of the Qwen3.6 snippet rows | agents/shared/CHANGES_FROM_ORIGINAL.md F-A | Done (owner, 2026-11-11): the 2026-10-31 workbook, sanitised with `sanitize_failure_analysis.py` (its 3,503 formula values evaluated by `failure_analysis_eval.py` equal its cached values); `scripts/reporting/` now carries the 2026-10-31 logic; the earlier workbook is in git history |
| K36 | B | **The 17.8 % "discordance between two identical 830-question runs" is an extrapolation.** It comes from a replay of the 112 questions one DSPy.RLM run retrieved gold for but answered wrong (47 recovered), assumed symmetric: 2 × 0.4196 × 176 / 830. It is an upper bound (replay four days later) and concerns DSPy.RLM, not the snippet REPL agent. | AGENT_COMPARISON.md, research_overview.md, FINAL_RESEARCH_AUDIT.md (corrected) | original repository `docs/rlm_results_summary.xlsx`, sheet Retrieval §27 (§12: the only identical repeats, 3 × 50 questions) | Corrected in this repository's documents; the original HANDOFF.md §6 and GEPA_ASSESSMENT.md keep the old wording |
| K37 | B | **`GEPA_ASSESSMENT.md` lists two "measured prompt effects" on OBLIQ that are not prompt effects.** Congress 0.139 → 0.190 keeps the prompt and changes retriever and model; Twitter 0.095 → 0.126 keeps the prompt and changes the model (`rlm_results_summary.xlsx`, sheets Congress and Twitter). | The motivation of GEPA-4 | original repository `docs/GEPA_ASSESSMENT.md` §3 | Do not cite them as prompt effects |
| K38 | B | **VAR-2 had no saved script.** The computation was run inline in a Claude Code session (2026-10-27) and is recovered byte for byte from the transcript (`studies/run_variation/original/TRANSCRIPT/`, with session, line, tool_use id and hash). | VAR-2 | `studies/run_variation/original/MANIFEST.json` | None: provenance recorded |
| K39 | B | **Several runs had no launcher file.** SP-1, SP-2, SP-6 (dev100), SP-7, TWQ-1 (dev100), TWQ-2, GEPA-2's confirmation, GEPA-3's confirmation and GEPA-4 runs 1-2 were started by hand; their launchers are reconstructed from run logs, record configs and session transcripts, and say so. GEPA-1 ran a frozen copy of its launcher that differs from `run_gepa_bcp.sh` (PO-I2). | the runs named | SP-I18, PO-I2, PO-I9 | None: documented in each launcher |
| K40 | B | **Code changed under running experiments.** `gepa_bcp.py` (the split runner) was edited on 2026-10-12: GEPA-1, GEPA-2 and SP-1/SP-2's first slices ran on the earlier version (preserved, 2 lines differ, only the `queries` field of graded records). `gepa_wide.py` and `gepa_night.py` were edited during GEPA-4, and GEPA-4 ran unpreserved September/October versions of the snippet REPL harness (opt-in additions only). The DSPy ablation ran code with a result-queue fix that `agents/dspy_rlm/src/` deliberately does not carry. | GEPA-1/2/4, SP-1/2, DSA-3/5 | F-B (agents/document_repl), PO-I5, PO-I6, F-F (agents/dspy_rlm) | Second snapshots preserved; DSA launchers run the snapshot |
| K41 | C | **The FRAMES single-shot script reads documents through one file handle from up to 16 threads without a lock.** A read can start at another thread's offset and return a wrong article into a prompt; no crash is logged, and the records cannot show whether it happened. | FB-1 (bm25_4, oracle prompts), so ERR-1's oracle bucket | FB-I1 | Re-run with one worker or a per-thread handle to check |
| K42 | C | **Comparison scripts drop unfinished runs.** `compare_bcp_830.py` drops questions the treatment never finished, and `compare_bcp_split.py` drops killed runs, although their docstrings say these count wrong. The reported SP-1/SP-2 numbers come from the gpt-4.1 re-judge, which counts them wrong. | the Qwen3.6-judged (superseded) SP comparisons | SP-I10, SP-I11 | None for the reported numbers |
| K43 | C | **The DSA sample cannot be redrawn.** The script that drew it reads a per-question file from a Claude Code scratchpad that no longer exists; the sample itself is preserved. | DSA-3, DSA-5 | F-G (agents/dspy_rlm) | None |
| K45 | A | **Resolved 2026-11-22: `Experiments.xlsx` replaced by the 2026-10-31 state.** Was: `results/original_baselines/Experiments.xlsx` was also the 2026-10-30 state. It showed BCP-15 (Qwen3.6 snippet REPL, dense) as partial (0.380, 276/830) and BCP-17 (ColBERT) as empty, and the README, registry and research overview called them "running" / "queued". The original repository's `docs/Experiments.xlsx` has their final values (0.3675, 0.6012). `Failure_Analysis.xlsx` (since K35) already covers both rows. | BCP-15, BCP-17 and every document that reads the results table | original repository `docs/Experiments.xlsx`, BrowseComp+ sheet | Done (owner's request, 2026-11-22): the 2026-10-31 workbook, reworded with `neutral_wording.py` as before (`--check`: numbers identical to the original); a cell-by-cell diff against the 2026-10-30 file changes only rows 15 and 17 of the BrowseComp+ sheet and its footnote A37; `recompute_experiments.py` (now also reading BCP-17) reproduces both rows from their `graded_gpt41.json`; trajectories packed (`trajectories/snippet_repl/BCP-15/`, `BCP-17/`); registry, READMEs and generated documents updated; the earlier workbook is in git history |
| K44 | C | **Two GPT-5 re-judge values differ from the recorded verdicts of the same runs.** Re-judging the stored GPT-5 document REPL answers with gpt-4.1 (answers cut at 1,500 characters, a pipeline check) gives 0.8639 for the run reported as 0.8747 (BCP-23) and 0.8301 for 0.8313 (BCP-21). The results table keeps the runs' own gpt-4.1 verdicts. | BCP-21, BCP-23 (unchanged) | `results/follow_up/verdicts/bcp/summary.json` rows 11, 13 | None: the cut answers explain part; reported to the owner as E4 |

## Corrections made during the review

- **Rejected claim: Document REPL OBLIQ runs called the sub-LLM.** An agent's draft said model code called the sub-LLM
  functions in 37 CON-11 and 17 TWI-11 runs. A direct scan of every trajectory found no call (0/254, 0/281, 0/151).
  The claim was withdrawn, and F2 now records the correction.
- **Corrected registry notes: TWI-11 and CON-11 (2026-11-22).** Their notes still said the model called the sub-LLM
  functions on 17 / 37 questions. A scan of model-written code in every MAT-11, TWI-11 and CON-11 trajectory finds 0 calls
  of `llm_query` / `rlm_query`; the 33 / 17 / 37 questions call the non-existent `read_docs` (K10). Notes corrected.
- **Corrected time limit of the Qwen3.6 document REPL rows (2026-11-22).** `AGENT_COMPARISON.md` said "runaway kill
  only". The configs pass `--max-timeout 5400`; rlm (`core/rlm.py`, `_check_timeout`) checks it at the start of each
  turn, and the launchers kill the process at 6,300 s. In the archived runs with these settings (SP-1, SP-2, SP-4),
  questions stopped by the limit ended between 5,400 and 6,300 s. Template and document corrected.
- **Corrected registry entry: BCP-21's launcher.** The `final830_*` scripts were first listed as BCP-21's launcher.
  They belong to a different run (Design G, a different served model). BCP-21's launchers are not preserved.
