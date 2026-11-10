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
| K7 | **Unfinished questions are scored wrong** (time limit, watchdog, API error after retries). | BCP-19 (243), BCP-27 (121), FR-9 (73), BCP-29 (53), BCP-31 (18), BCP-13 (9), FR-13 (6), BCP-25 (4) | R3, S2 | Documented convention; the Qwen3.6 rows measure accuracy within the time budget |
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

## Corrections made during the review

- **Rejected claim: Document REPL OBLIQ runs called the sub-LLM.** An agent's draft said model code called the sub-LLM
  functions in 37 CON-11 and 17 TWI-11 runs. A direct scan of every trajectory found no call (0/254, 0/281, 0/151).
  The claim was withdrawn, and F2 now records the correction.
- **Corrected registry entry: BCP-21's launcher.** The `final830_*` scripts were first listed as BCP-21's launcher.
  They belong to a different run (Design G, a different served model). BCP-21's launchers are not preserved.
