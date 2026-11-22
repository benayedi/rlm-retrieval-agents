# Final research audit

Date: 2026-10-30. This is the handover document for the restructured research repository. It states what was done,
how it was checked, what the evidence shows, and what remains unverified.

## 1. Executive summary

The research compares three recursive-language-model (RLM) agents on retrieval-heavy benchmarks: **DSPy.RLM**,
**Snippet REPL agent** and **Document REPL agent**, across retrievers and models, on BrowseComp-Plus, FRAMES and OBLIQ-Bench.

This repository turns the original working repository into a reviewable research artifact. Four things were done:

- **Code.** The code that produced every result was preserved byte for byte. A readable copy was built next to it,
  with identical behaviour.
- **Results.** Every number in the results table was recomputed independently from the per-question records, and all
  of them match.
- **Review.** The three implementations were reviewed against their method descriptions. The review records 30
  issues with their effect on the results, and fixes none of them.
- **Documentation.** Every experiment in the results table has its own document, and the test suite has 446 tests.

**No experiment was re-run.** The refactored code was shown to be equivalent at source level (identical syntax trees,
prompts, tool specifications, cache keys and records under mocks). It was **not** shown to be equivalent at the level
of benchmark results.

Main caveats a reader of the results must know:
- Snippet REPL vs document REPL pairs differ in search and turn budgets (K2).
- Some GPT-5 rows did not receive the reasoning effort their labels state (K3).
- Two document REPL rows ran on an unpreserved library build (K5).
- The Recall column mixes five definitions (K4).
- DSPy.RLM was never run under the final protocol.

## 2. Original codebase

The original was a research working repository at commit `7685538` on branch `obliq-dataset-experiment`, with
336 uncommitted paths. It contained:
- 216 scripts, most of them one-off chains and probes;
- 8 example drivers and the small `rag/` package (DSPy.RLM);
- 431 GB of logs, tracked in a 1.4 TB `.git`;
- 89 GB of data;
- three results workbooks.

The snippet and document REPL agent loops come from an external clone of the RLM library (`rlm`, github.com/alexzhang13/rlm
@ 854e688, MIT), loaded through `PYTHONPATH`. The document REPL harness lived in a separate directory that is not a git
repository.

Configuration lived in environment variables, in shell wrappers and, for the document REPL harness, in source files patched
in place. There were no tests.

Full inventory: [INITIAL_CODEBASE_INVENTORY.md](INITIAL_CODEBASE_INVENTORY.md); plan and deviations:
[MIGRATION_PLAN.md](MIGRATION_PLAN.md).

**The original repository was not modified.** All work happened in a new directory with a new git history. A final
check found no file changed in the original repository, the document REPL directory or the library clone since the work
began. The only exceptions were log files written by servers that were running at the time.

## 3. Final repository architecture

| Directory | Purpose |
|---|---|
| `agents/<agent>/src/` | the readable, runnable code |
| `agents/<agent>/original/` | the exact files that produced the results, with a SHA-256 manifest; never edited |
| `agents/<agent>/experiments/`, `configs/` | launchers, and one `<ID>.env` per results row holding the exact settings |
| `agents/<agent>/tests/`, `tests/` | lightweight tests (mocks only) |
| `agents/shared/rlm_shared/` | paths, benchmark loaders, reference metrics, grading |
| `benchmarks/` | retrieval servers, index builders, benchmark sources and licences |
| `experiments/registry.yaml` | one entry per experiment |
| `results/` | the results workbooks (BrowseComp-Plus text removed) and generated figures |
| `trajectories/` | per-question records and full trajectories of 17 rows, as byte-exact Git LFS archives with SHA-256 manifests; BrowseComp-Plus archives encrypted with the benchmark's canary key ([README](../trajectories/README.md)) |
| `scripts/` | provenance (snapshots, leak scan, wording), audit (recompute results), docs (generators), reporting |
| `docs/` | overview, architecture, one document per experiment, verification reports |

Details: [architecture/OVERVIEW.md](architecture/OVERVIEW.md); refactor rules: [architecture/REFACTOR_RULES.md](architecture/REFACTOR_RULES.md).

## 4. The three agents

- **[DSPy.RLM](../agents/dspy_rlm/README.md)** is built on `dspy.RLM`. Its typed signatures carry task-specific
  strategy text. Its tools return 500-character snippets from a top-10 search, it reads through an Azure sub-LM, and it
  ends on SUBMIT or at the iteration cap. It was evaluated in an earlier campaign, with a different judge and budgets,
  and is not part of the final table.
- **[Snippet REPL agent](../agents/snippet_repl/README.md)** runs the RLM authors' loop and prompt, adapted for retrieval
  by runtime patches. Search returns 5 results by default, each the first 512 tokens of the server's text (servers cut
  text to 2,000 characters). Documents are read through `llm_query` sub-LLM calls. A format gate checks the final
  answer, and history compaction is applied.
- **[Document REPL agent](../agents/document_repl/README.md)** runs the same library with its own harness. Search puts
  whole documents into a `DOCS` variable, with a model-chosen k of at most 50. The model reads slices itself. The
  harness caps searches at 10 on the client side and turns at 30, then forces an answer. The sub-LLM functions remain
  installed even though the prompt forbids them.

Comparison of methods: [research_overview.md](research_overview.md).

## 5. Benchmarks

| Benchmark | Role | Notes |
|---|---|---|
| BrowseComp-Plus (830 q, 100,195 docs) | main QA benchmark: hard multi-constraint questions over a fixed corpus | distributed encrypted; no plain text in this repository |
| FRAMES (824 q, Wikipedia) | multi-hop QA where answers sit inside long articles | BM25 index over Wikipedia 20230601.en |
| OBLIQ-Bench (Math 151, Twitter 281, Congress 254) | ranking tasks with oblique relevance | WildChat and Writing were not run |
| HoVer | used only in a GEPA side study with none of the three agents | not part of the results |

Sources, indexes, servers and licence status: [benchmarks/README.md](../benchmarks/README.md).

## 6. Complete experiment inventory

The 22 rows of [Experiments.xlsx](../results/original_baselines/Experiments.xlsx) that this project produced. Each has
its own document, listed in [experiments/README.md](experiments/README.md). Implementation, configuration, launcher and
original output paths are in each document and in [registry.yaml](../experiments/registry.yaml).

| ID | Agent | Benchmark | Model | Retriever | Status | Verification |
|---|---|---|---|---|---|---|
| BCP-9 | Snippet REPL | BrowseComp-Plus | GPT-5 | Qwen3-Embed-8B | complete | recomputed ✓; effort correction (K3); harness version not preserved (K8) |
| BCP-11 | Snippet REPL | BrowseComp-Plus | GPT-5 | Reason-ModernColBERT | complete | recomputed ✓; K3, K8 |
| BCP-13 | Snippet REPL | BrowseComp-Plus | GPT-5 | BM25 | complete | recomputed ✓; dense-retriever sentence in the prompt (K26) |
| BCP-15 | Snippet REPL | BrowseComp-Plus | Qwen3.6 | Qwen3-Embed-8B | complete (running on 2026-10-30; migrated 2026-11-22, K45) | recomputed ✓ (2026-11-22); 190 timeouts (K7) |
| BCP-17 | Snippet REPL | BrowseComp-Plus | Qwen3.6 | Reason-ModernColBERT | complete (queued on 2026-10-30; migrated 2026-11-22, K45) | recomputed ✓ (2026-11-22); 68 timeouts (K7) |
| BCP-19 | Snippet REPL | BrowseComp-Plus | Qwen3.6 | BM25 | complete | recomputed ✓; 243 timeouts (K7); K26 |
| BCP-21 | Document REPL | BrowseComp-Plus | GPT-5 | Qwen3-Embed-8B | complete | recomputed from the trace ✓; launchers lost, library build unpreserved (K5) |
| BCP-23 | Document REPL | BrowseComp-Plus | GPT-5 | Reason-ModernColBERT (8,192) | complete | recomputed from the trace ✓; K5 (up to 19 searches logged) |
| BCP-25 | Document REPL | BrowseComp-Plus | GPT-5 | BM25 | complete | recomputed ✓; effort not sent (footnoted) |
| BCP-27 | Document REPL | BrowseComp-Plus | Qwen3.6 | Qwen3-Embed-8B | complete | recomputed ✓ |
| BCP-29 | Document REPL | BrowseComp-Plus | Qwen3.6 | Reason-ModernColBERT | complete | recomputed ✓ |
| BCP-31 | Document REPL | BrowseComp-Plus | Qwen3.6 | BM25 | complete | recomputed ✓ |
| FR-7 | Snippet REPL | FRAMES | GPT-5 | BM25 | complete | recomputed ✓ |
| FR-9 | Snippet REPL | FRAMES | Qwen3.6 | BM25 | complete | recomputed ✓; 73 timeouts |
| FR-11 | Document REPL | FRAMES | GPT-5 | BM25 | complete | recomputed ✓ |
| FR-13 | Document REPL | FRAMES | Qwen3.6 | BM25 | complete | recomputed ✓ |
| MAT-9 / TWI-9 / CON-9 | Snippet REPL | OBLIQ | GPT-5.2 | Gemini-Embedding-2 | complete | recomputed ✓; format gate on (K6) |
| MAT-11 / TWI-11 / CON-11 | Document REPL | OBLIQ | GPT-5.2 | Gemini-Embedding-2 | complete | recomputed ✓; calls to a missing `read_docs` tool (K10) |

DSPy.RLM's earlier-campaign results are summarised in its README, labelled as not comparable.

## 7. Code review

Each implementation was reviewed against its method description, point by point:
- [REVIEW_snippet_repl.md](verification/REVIEW_snippet_repl.md)
- [REVIEW_document_repl.md](verification/REVIEW_document_repl.md)
- [REVIEW_dspy_rlm.md](verification/REVIEW_dspy_rlm.md)

All findings are consolidated in [KNOWN_ISSUES.md](verification/KNOWN_ISSUES.md), 30 entries:
- **8 of severity A.** They change how a reported number must be read.
- **9 of severity B.** A documentation or methodology statement is wrong, but the numbers are not affected.
- **13 of severity C.** Latent bugs or hygiene problems, with no effect on any reported number.

The most consequential:
- **K1, K2.** Document REPL retrieves a model-chosen number of whole documents (1–50; mean 24.5 on BrowseComp-Plus BM25),
  and snippet REPL vs document REPL pairs differ in search caps, turn limits and time limits. A pair compares complete packages,
  not reading strategies.
- **K3.** The RLM library's Azure client drops `reasoning_effort`. BCP-9 and BCP-11 ran at the deployment default
  (≈ medium), not the labelled high / medium. BCP-25's footnote already said so; the experiment documents now correct
  BCP-9 and BCP-11.
- **K5.** BCP-21 and BCP-23 ran on an unpreserved patched library build, and only a summary trace of them survives.
- **K6.** The OBLIQ snippet REPL rows ran with the BrowseComp answer-format gate on. It rejected at least one submission on
  140/151, 279/281 and 237/254 questions.

**Fixes.** None was applied, because fixing would change the code that produced the results. Each issue lists a
proposed action.

**Review corrections.** Two claims made during the review were withdrawn after a direct check:
- that document REPL OBLIQ runs called the sub-LLM (0 calls found; the calls were to a missing `read_docs` tool);
- that the snippet length has no character cap (every preserved server cuts text to 2,000 characters).

## 8. Algorithmic verification

**Demonstrated by execution, with mocks and without models:**
- **Source equivalence.**
  - The DSPy.RLM package has the same syntax tree as the original, docstrings excluded.
  - The snippet REPL agent builds identical prompts, tool specs, task texts, message lists, records and `RLM` arguments in
    nine configurations covering every final row. A one-word mutation made 8 tests fail.
  - The document REPL agent has identical prompts and BM25 rewrites.
  - The three result-cache keys recompute to the recorded values.
- **Metrics.** Binary NDCG@k and Recall@k match hand-computed values, and the experiments' own implementation matches
  an independent reference on 30 cases. Gold recall, majority vote and accuracy follow the stated definitions.
- **Behaviour under fakes.** Covered: the document REPL search cap and k override, DOCS merging, the snippet REPL format gate and
  answer extraction, DSPy.RLM search budgets, judge vote parsing and retry, and server text cutting.

**Inspected only, not demonstrated:**
- the agents' behaviour with real models;
- the judge's correctness on real answers;
- index contents;
- the retrieval servers with real models (4 tests skip for that reason).

## 9. Testing results

| Run | Tests | Passed | Failed | Skipped |
|---|---|---|---|---|
| with the RLM library importable | 446 | 442 | 0 | 4 (need a GPU model, a FAISS index or Vertex AI) |
| without it | 446 | 417 | 0 | 29 |

- **Also run:** `compileall` and `bash -n` on all 30 launchers, both clean.
- **Not run:** linting, type checking and coverage, because those tools are not installed and none was added.

Details: [TESTING_REPORT.md](verification/TESTING_REPORT.md).

## 10. Existing experimental results

The results are those recorded before this work, in [Experiments.xlsx](../results/original_baselines/Experiments.xlsx)
and [Failure_Analysis.xlsx](../results/original_baselines/Failure_Analysis.xlsx). The comparison, with what it does and
does not establish, is in [AGENT_COMPARISON.md](experiments/AGENT_COMPARISON.md).

What the evidence shows:
- **Retriever and model matter most.** ColBERT beats BM25 in both harnesses (p ≤ 4e-24), and GPT-5 beats Qwen3.6 in
  every pair.
- **Snippet REPL vs document REPL depends on the retriever** (GPT-5, BrowseComp-Plus). Document REPL wins with the dense retriever
  (p = 0.012), the two tie with ColBERT, and snippet REPL wins with BM25 (p ≈ 2e-5).
- **On FRAMES, document REPL wins with GPT-5** (p ≈ 0.001).
- **OBLIQ differences are small.**

**Wording changes in the published copies.** The published workbook copies differ from the originals in one way only:
their text notes describe the document REPL runs neutrally (`scripts/provenance/neutral_wording.py`). A check confirms
every number and formula is identical to the original.

**Redaction.** `Failure_Analysis.xlsx` additionally has BrowseComp-Plus questions and answers removed.

## 11. Regression and preservation assessment

- **Originals unchanged.** `original/` holds 104 source files with SHA-256 manifests;
  `scripts/provenance/preserve_originals.py --check` passes.
- **Every difference recorded.** Each difference between `src/` and `original/` is listed in the agent's
  `CHANGES_FROM_ORIGINAL.md` and falls under the refactor rules: renames, import wiring, parametrised paths and
  credentials, documentation, launchers.
- **Tests compare the two.** Syntax-tree identity, string identity of every model-visible text, cache-key identity, and
  record identity under mocks.
- **Results recomputed.** Every complete row of Experiments.xlsx was recomputed from the per-question records:
  accuracy, recall under the row's own definition, searches, and OBLIQ NDCG@10 / Recall@10 all match
  ([RESULT_INTEGRITY.md](verification/RESULT_INTEGRITY.md)).

**Original experimental results were preserved and inspected. The refactored implementation was reviewed through
source-level comparison, configuration inspection, and lightweight testing. Full experimental equivalence was not
independently re-established because no new full experiments were executed.** Re-running would also not reproduce
per-question outcomes exactly: sampling is unseeded. (An earlier version of this sentence said a repeat of a full run
disagreed on 17.8 % of questions; that figure is an extrapolation from a 112-question replay of one DSPy.RLM run, not a
repeat of a full run: KNOWN_ISSUES K36.)

## 12. Reproducibility

The outputs of the original runs can be inspected without re-running anything: `git lfs pull`, then
`python scripts/provenance/unpack_trajectories.py <ID> OUT_DIR` restores and hash-checks a row's records and trajectories.
Not included:
- BCP-31 (left out for now);
- BCP-21 and BCP-23 (not available);
- BCP-9 and BCP-11 trajectories (deleted before this work; their records are included);
- BCP-15 and BCP-17 (unfinished).

To re-run any row:
1. Install `requirements.txt`, which pins the versions that ran and the RLM library at 854e688.
2. Fill in `configs/paths.env`.
3. Build or obtain the index and start the server named in the row's config ([benchmarks/README.md](../benchmarks/README.md)).
4. Run the launcher with the row's `configs/<ID>.env`. The experiment document gives the exact command.
5. Grade with `agents/shared/rlm_shared/grading/`.

These steps were **not executed**.

Known obstacles to exact reproduction:
- BCP-21 and BCP-23: unpreserved library build and launchers.
- BCP-9 and BCP-11: unpreserved harness state and four lost launchers.
- The dense :8001 server code is not preserved.
- Azure deployments change over time.
- No seeds are set.

## 13. Code quality improvements

- **One layout per agent.** Each agent has its own `src/`, `experiments/`, `configs/`, `tests/` and README,
  with intuitive file names; rename tables are in each change record.
- **No machine-specific values in code.** Absolute paths, credential files and personal deployment names were removed
  from all non-original code. They are now environment variables documented in `configs/paths.example.env`, and
  launchers stop with a message naming any missing variable.
- **Exact configurations.** Every results row has a config file holding its exact settings. Values that cannot be
  determined are marked UNKNOWN rather than guessed.
- **Explicit imports.** Imports are wired explicitly. The old `examples.*` / `sys.path` web is replaced by `rlm_shared`
  and package imports.
- **Single, tested definitions.** Reference metrics and a results audit were added; experiment documents and the
  comparison are generated from the evidence.
- **Deliberately not done.** Consolidating duplicated code (K23) and removing dead code. Both would change the code
  that produced the results, so they belong in a later, re-validated change.

## 14. Remaining issues

The full list is in [KNOWN_ISSUES.md](verification/KNOWN_ISSUES.md). The open items that matter most:

1. **Matched-budget comparison.** A matched-budget re-run is needed before any causal snippet-vs-document-REPL claim (K2).
2. **Effort labels.** Correct the BCP-9/11 effort footnote in any paper table (K3).
3. **BCP-21/23 provenance.** Obtain their run records and library build, or mark them non-reproducible (K5).
4. **Recall column.** Footnote the column, or restate it under one definition (K4).
5. **OBLIQ snippet REPL re-run.** Re-run with the format gate off to measure its effect (K6).
6. **Unfinished rows.** BCP-15 (running) and BCP-17 (queued) need documenting and auditing once they finish. Their
   configs and documents already exist. *Done 2026-11-22:* both finished (2026-10-30 / 2026-10-31); the 2026-10-31
   workbook, their trajectories and the audit were migrated (KNOWN_ISSUES K45).
7. **Licences.** Benchmark redistribution licences are undetermined. No licence was chosen for this repository, and
   `CITATION.cff` was not created because author and publication metadata were not available. Both must be decided
   before any public release. *Done 2026-11-22:* the repository is licensed under Apache-2.0 (`LICENSE`, owner's
   decision); benchmark licences and `CITATION.cff` are still open.
8. **Personal names in `original/`.** The verbatim originals still contain personal Azure deployment names and
   absolute home paths. Byte-identical preservation requires them; review them before making the repository public.

## 15. Installation and execution

See the [README](../README.md#installation). In short: Python 3.12; `pip install -r requirements.txt`; `python -m
pytest`. Running experiments needs model endpoints, retrieval servers and data, as described above.

## 16. GitHub repository

**https://github.com/benayedi/rlm-retrieval-agents**: **private**, created 2026-10-30 with the owner's approval. The
initial commit `443a26f` (315 files) was pushed to `main`.

Before the push:
- the remote branch head was checked to equal the local commit;
- the BrowseComp-Plus leak scan and the secrets scan passed on every committed file.

The repository is not public; making it public needs the owner's explicit approval.

## 17. Supervisor access

`Torky24` was invited as a collaborator with **write** permission (invitation id 340570807, created
2026-10-30T12:45:03Z). Verified through the GitHub API (`GET /repos/benayedi/rlm-retrieval-agents/invitations`): the
invitation exists and has not expired.

**It is pending.** GitHub grants access only once the invitee accepts. At the time of writing the collaborator list
contains only the owner. Torky24 can accept at https://github.com/benayedi/rlm-retrieval-agents/invitations or from
the GitHub notification.

## 18. Final verification summary

| | Status |
|---|---|
| Completed | inventory; preservation of originals; refactor of all three agents and shared code; 22 experiment documents; agent READMEs and reviews; comparison; testing; results audit; this audit |
| Reviewed | all source used by the 22 rows, the DSPy.RLM package and drivers, graders, retrieval servers, reporting scripts |
| Tested | 446 tests, 0 failures; static compile and shell checks |
| Independently verified | every headline number of the 20 complete rows (recomputed); metric implementations (reference + hand-computed); source equivalence of the refactor (AST, prompts, keys, records); absence of BrowseComp-Plus plain text; absence of secrets |
| Original evidence preserved | 104 source files with SHA-256 manifests; both workbooks (numbers identical to the originals); per-question records and trajectories of 17 rows as byte-exact archives (`trajectories/`, each member hash-verified after packing); the rest left on the original machine |
| Not verified | behaviour with real models; judge correctness; benchmark-level equivalence of the refactored code; BCP-21/23 exact code; the two unfinished Qwen3.6 rows |

## Addendum (2026-11-11): migration of the follow-up studies

Migrated: the search-policy and GEPA studies, the FRAMES search-cap run, the DSPy.RLM ablation, the GPT-5 re-run
variation, the FRAMES error analysis and baselines, and their controls (19 IDs; [studies/README.md](../studies/README.md),
[FOLLOW_UP_STUDIES.md](experiments/FOLLOW_UP_STUDIES.md)). Same rules as the first migration:

- **Preserved:** every file that ran, byte-exact with SHA-256 manifests (`studies/*/original/`, second snapshots
  `agents/*/original/RLM_2026-11-11/`); `preserve_originals.py --check` covers all of them. VAR-2's code, which was only
  run inline, is recovered from its session transcript.
- **Refactored, not changed:** readable copies in `src/`/`analysis/`; no bug fixed (issues SP-I*, PO-I*, RV-I*, EA-I*,
  FB-I* and K31-K44 in [KNOWN_ISSUES.md](verification/KNOWN_ISSUES.md)). Where a run had no launcher file, its
  launcher is reconstructed and marked so.
- **Tested offline:** each readable copy that can be exercised without models was run next to its original on synthetic
  inputs and gives identical output; cache keys and prompt ids of the original runs are reproduced from the code.
- **Records:** 17 archives in `trajectories/` (BrowseComp-Plus ones encrypted), each verified byte-identical to its
  originals; GEPA optimisation-time rollouts are left out.
- **Numbers:** not recomputed. Each is copied with its source into `results/follow_up/numbers.yaml`; accuracies are
  gpt-4.1 verdicts, Qwen3.6-judged values are marked superseded or Qwen3.6-only.
- **Not verified:** that re-running reproduces any number (no experiment was run); which proxy version served each
  SP-1 slice; whether reasoning effort reached the model in DSA-3/5.
