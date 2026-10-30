# Scientific code review: Snippet RLM

Reviewed code: `algorithms/snippet_rlm/src/` (refactored), `algorithms/snippet_rlm/original/RLM/` (the code that
produced the results), `algorithms/snippet_rlm/original/RLM_git_7685538/` (the committed September harness), the
launchers in `algorithms/snippet_rlm/experiments/`, and the `rlm` library checkout at `854e688`. Issue ids **K*n*** refer
to [KNOWN_ISSUES.md](KNOWN_ISSUES.md), and **S*n*** to
[algorithms/snippet_rlm/CHANGES_FROM_ORIGINAL.md](../../algorithms/snippet_rlm/CHANGES_FROM_ORIGINAL.md). Findings
not yet in KNOWN_ISSUES.md are numbered **N*n***.

## 1. Scope

In scope:
- the method as implemented: prompts, tools, the agent loop, stopping rules and the record each question produces;
- the three runners (BrowseComp-Plus, FRAMES, OBLIQ);
- the launchers of the 11 workbook rows (BCP-9, -11, -13, -15, -17, -19, FR-7, FR-9, MAT-9, TWI-9, CON-9);
- the parts of `rlm` 854e688 the method relies on: the loop, compaction, the default answer, usage merging and the
  Azure client.

Out of scope:
- the retrieval servers and indexes (`benchmarks/`);
- the graders (`shared/rlm_shared/grading/`, reviewed with the shared code);
- the Full-Doc and DSPy algorithms;
- the correctness of the workbook numbers, which is covered in [RESULT_INTEGRITY.md](RESULT_INTEGRITY.md).

No experiment, model call, server or network call was made for this review.

## 2. Specification vs implementation

The specification is the method description used in this repository's documents: the experiment documents
(`docs/experiments/snippet_rlm/*.md` §3), the comparison table in
[ALGORITHM_COMPARISON.md](../experiments/ALGORITHM_COMPARISON.md), and the harness's own docstring.

| # | Specification | What the code does | Verdict |
|---|---|---|---|
| 1 | The model works in a persistent Python REPL, one turn at a time | `rlm` `RLM.completion`: per turn, one LM call, then every ```` ```repl ```` block in the response is executed in a `LocalREPL` whose variables persist (`core/rlm.py:361-462`) | Matches |
| 2 | "Search returns 5 results" | `search_index(query, top_k=DEFAULT_TOP_K)`, with `DEFAULT_TOP_K = OFFICIAL_TOP_K` (5 for BCP/FRAMES, 20 for OBLIQ). It is a default, not a cap (`harness.py`, `make_tools`); overrides are counted. In the original logs, BCP-13 and FR-7 had 0 overrides; BCP-19 had overrides in 9 questions (48 calls, up to `top_k=10`) | Matches as a default; "5" is not enforced (N4) |
| 3 | "Each result is the document head cut to min(2,000 chars, 512 tokens)" | `_truncate`: with `SEARCH_SNIPPET_TOKENS=512`, the first 512 Qwen3-0.6B tokens of the server's `text`, with **no character cap**; the 500-character cut applies only when the token setting is unset. "~2,000 characters" appears only in the tool description shown to the model. Whether `text` is the document head depends on the server (outside scope; the harness comments say it is) | **Differs**: 512 tokens only (N4) |
| 4 | Search results go into a variable; the model sees them only if it prints them | The tool returns a Python list into the REPL; only stdout is returned to the model (`format_iteration`), trimmed to 60,000 chars per block | Matches |
| 5 | The model can open a whole document | `get_document(doc_id)` returns the server's `/document` JSON uncut | Matches |
| 6 | Long text is read by a sub-LLM (`llm_query`) | `llm_query`/`_batched` are plain completions at depth 1 on the root deployment (no `SUB_LM_MODEL` in any row), with `sub_sampling_args={"reasoning_effort": "medium"}`. On Azure the effort is sent only with the `azure_sampling`/`run5d` hook (K3). The prompt tells the model to pass fetched documents to `llm_query` rather than print them | Matches; effort caveat K3 |
| 7 | A format gate on the final answer | `_AnswerDict.__setitem__` is patched: `ready=True` is refused unless the content matches `Explanation\s*:` and `Exact Answer\s*:`, for at most 3 refusals per question. The rule does not fit OBLIQ, where it stayed on (K6). It can be bypassed (N1) | Matches the description; see K6, N1 |
| 8 | History compaction | `--compaction` uses the library mechanism: summarise when `count_tokens(history) ≥ 0.85 × get_context_limit(model_name)`. The limit comes from a name-substring table: 272k for names containing `gpt-5`, 128k for `qwen36-35b-a3b` (matches the key `qwen3`) | Matches; Qwen threshold N3 |
| 9 | "A run ends when the model sets a final answer or the turn limit is reached" | (a) `answer["ready"]` passes the gate → return; (b) `max_iterations` reached → one extra LM call, after an **assistant-role** message "Please provide a final answer ...", returned **ungated**; (c) exception → `predicted="ERROR"`, wrong; (d) external `timeout`/watchdog → no record, counted wrong (K7). No `max_timeout`/`max_budget`/`max_errors` are passed | Matches; (b)-(d) need stating |
| 10 | Snippet RLM uses the authors' loop and prompt, adapted for retrieval | 7 prompt patches, each asserted on its anchor text, plus the turn-0 and size lines removed. The adapted system prompt is 3,779 chars for the BCP settings (measured). With `ORCHESTRATOR=0` the authors' orchestrator addendum is dropped in every final row | Matches; the prompt is substantially rewritten, as the harness itself states |
| 11 | The prompt states facts about the retriever | The "dense semantic retriever ... quotation marks have NO effect" line is on by default. FRAMES replaces it with a BM25 sentence; the BrowseComp-Plus BM25 rows (BCP-13, BCP-19) keep the dense sentence | **Differs** for BCP-13/19 (N2) |
| 12 | Budgets per row (turns, search cap, time limit) | As listed in `configs/*.env` and README §8. GPT-5 and Qwen rows differ (S7) | Matches the configs; comparability K2 |
| 13 | Reported turns and tokens per question | The record's `iterations`/`input_tokens` are wrong, because the library merges usage dicts by model name (`lm_handler.py:237-251`). The workbook uses trajectories and `usage.jsonl` instead (K11) | Record fields unusable; workbook unaffected |

## 3. Checks performed

**Equivalence of the refactored code** (`tests/test_snippet_rlm_equivalence.py`, using `tests/_probe.py`).
- Original and refactored entry points are imported in separate interpreters under identical environments, with a fake
  retriever (`requests.post`) and a fake agent. The fake replaces the library's own `RLM.__init__`/`completion` at the
  bottom of the patch chain, so every runner and harness patch still runs.
- 9 scenarios: harness defaults; the run5d switches; all other prompt switches; BCP-13; BCP-15/17/19; FR-7; FR-9;
  TWI-9; MAT-9 prompt files plus the remaining OBLIQ switches.
- Compared items, all identical:
  - system prompt (both module copies), addendum, task text, tool specs, the full first message list, turn-0 message;
  - the search-cap refusal;
  - parsing of 10 synthetic turns (fence tolerance, prose-final answer);
  - the format gate sequence;
  - the `run_one` record, the `RLM(...)` kwargs (including injected timeouts and sampling args) and the retriever
    requests.
- A mutation check made 8 tests fail when one word of a tool description was changed.
- Static tests: every non-docstring string constant of every src file equals the original's, apart from 6 listed
  wiring strings; hooks and `prompt_modules.py` are AST-identical apart from the docstring.
- Result: 78 passed (see [TESTING_REPORT.md](TESTING_REPORT.md)).

**Recall definition.** `gold_recall = |G ∩ R| / |G|` (0 when G is empty), and `evid_recall` likewise with E, where:
- R is every `doc_id` in every result list `search_index` returned, whether or not the model printed it, plus every
  id passed to a successful `get_document`;
- searches refused by a search cap add nothing.

So it measures what the tools returned, not what the model read. Per benchmark:
- FRAMES: E = G (gold articles);
- OBLIQ: G = E = qrels.

Tested on a synthetic record: G = {d1,d3,d5,d7}, E = {d2,d4}, R = {d1,d2,d9,d3} gives 0.5 / 0.5. The workbook's
Recall columns use several aggregations over this field (K4, R1).

**Format gate.** Tested on synthetic submissions:
- three malformed submissions are refused, with notices `[SUBMISSION REJECTED 1/3]`..`3/3`;
- the fourth is accepted and fires the library callback;
- a well-formed one is accepted immediately.

The counter is reset in `run_one`, so it is per question, not per process. The comment in `run_obliq.py` that says
otherwise is wrong (S3). On OBLIQ the gate's requirement contradicts the ranking contract. The original logs show at
least one rejection in 140/151, 279/281 and 237/254 questions (K6).

**Termination and stopping conditions.** Read in `rlm` `core/rlm.py`; the probe tests exercise the first two (gate and
parsing):
- the loop checks `final_answer` after all blocks of a turn have run;
- `_default_answer` adds one ungated call at the turn limit (the workbook footnote counts it as one extra turn: "a run
  that reaches its turn cap adds one forced final-answer turn");
- compaction is checked before each turn;
- the Qwen opt-ins change what counts as code: `QWEN_FENCE_TOLERANT` runs mislabelled fences only if they parse and do
  something; `QWEN_PROSE_FINAL` turns a code-less turn with a non-empty `Exact Answer:` into a submission. Both are
  tested on synthetic turns.
- Search caps return an error hit (`doc_id ""`) and are stated in the system prompt.
- External limits are launcher `timeout`s (7,200 s / 3,600 s) and the BCP-13 watchdog; they produce no record (K7).

**Provenance checks**, read-only on the original logs:
- the BM25 BCP-13 trajectories contain the "dense semantic retriever" sentence (N2);
- the September records (`rlmfix24/31/32`) lack the `child_*` fields of commit 7685538 (K8);
- the OBLIQ `all.json` files hold 2 × n rows (K19);
- the MAT-9 log's prompt-file sizes (1,571 / 932 chars) equal the copies in `configs/obliq/math/`.

## 4. Findings

Already in KNOWN_ISSUES.md:

| K id | Finding (snippet-specific evidence) |
|---|---|
| K2 | Snippet ↔ Full-Doc pairs differ in budgets; GPT-5 and Qwen Snippet rows also differ among themselves (S7) |
| K3 | Azure drops `reasoning_effort` without the hook; BCP-9/11 ran at the deployment default (S4) |
| K4 | Recall aggregation differs across rows (the per-question definition above is uniform within this harness) |
| K6 | OBLIQ rows ran with the BrowseComp format gate on (S3) |
| K7 | Unfinished questions scored wrong; BCP-19 243, FR-9 73, BCP-13 9 (S2) |
| K8 | The harness version behind BCP-9/11 is not preserved; launchers partly lost (S11) |
| K11 | `iterations`/`input_tokens` record fields wrong (S1) |
| K13 | `bcp_snip.sh` header contradicts its `--compaction` (S8) |
| K15 | Implicit `.env` supplied the in-run judge model (S9) |
| K17 | `heuristic_match` accepts an empty prediction in the in-run field (S12) |
| K19 | OBLIQ `all.json` duplicates every question (S10) |
| K23 | Duplicated runner code (S5) |
| K24 | Watchdog prices every model at GPT-5 rates (S13) |

New in this review (proposed for KNOWN_ISSUES.md; none changes a reported number by itself):

| Id | Severity | Finding | Evidence |
|---|---|---|---|
| N1 | C | **The format gate can be bypassed.** It patches only `_AnswerDict.__setitem__`. If the model rebinds `answer = {"content": ..., "ready": True}`, `LocalREPL._restore_scaffold` captures the content directly (`rlm` `environments/local_repl.py:527-538`). (`answer.update(...)`, by contrast, never triggers capture at all: the submission is silently ignored.) The turn-limit default answer is also ungated (spec row 9) | Code reading; how often this happened in the runs was not measured |
| N2 | B | **The BrowseComp-Plus BM25 rows (BCP-13, BCP-19) told the model "The index is a dense semantic retriever: it matches meaning, not literal strings"**, because `RETRIEVER_LINE` defaults to on and `bcp_snip*.sh` does not set it. FRAMES rows swap in a BM25 sentence. This is part of the evaluated prompt for those rows | `harness.py` `_add_retriever_line`; sentence found in a BCP-13 trajectory |
| N3 | C | **Compaction threshold for Qwen3.6 is computed against 128k tokens**, not the served model's context: the library matches the served name `qwen36-35b-a3b` to the table key `qwen3` (128k). Compaction triggers at about 108.8k tokens (by tiktoken or 4 chars/token). For GPT-5 the limit (272k) depends on the deployment name containing `gpt-5` | `rlm` `utils/token_utils.py` |
| N4 | B | *(Corrected in the consolidated review: every preserved search server cuts `text` to 2,000 chars before the harness cuts to 512 tokens, so the effective length is min(2,000 chars, 512 tokens); only the unpreserved :8001 server is unverified. The `top_k` point stands. See KNOWN_ISSUES K27.)* **The harness applies no character cap of its own, only 512 tokens**, and `top_k=5` is a default the model can raise. ALGORITHM_COMPARISON.md's table says "5 results, each ... cut to min(2,000 chars, 512 tokens)" | `harness.py` `_truncate`, `make_tools`; override counts above |

## 5. Independently demonstrated vs only inspected

**Demonstrated by execution** (mocks, no model):
- the refactored code builds byte-identical prompts, tool specs, task texts and message lists, and produces identical
  records, `RLM` kwargs, gate behaviour and code-block parsing to the original, in 9 configurations covering every
  final row's switches;
- the recall formula on a synthetic record;
- the format gate's 3-refusal behaviour;
- the fence-tolerance and prose-final parsing;
- the search-cap refusal text;
- that the snippet tokenizer is imported from `dspy_rlm.agent`;
- that all launchers and configs parse, and that configs name existing launchers.

**Only inspected** (code reading, or read-only looks at original logs):
- the library loop, compaction, `_default_answer`, usage merging and the Azure client (not executed against a model);
- N1-N3;
- the launcher environments (compared line by line with the originals, not run);
- the hooks' runtime effect (AST-identical to the originals; not loaded against Azure);
- that the original runs actually used these settings (from launch lines, logs and record fields);
- the retrieval servers' behaviour (out of scope).

## 6. Statement on experimental equivalence

**Full experimental equivalence was not re-established.** No question was re-run with the refactored code, so it is
not shown that `src/` reproduces any workbook number. The evidence is source- and behaviour-level, with mocked models
and retrievers: given the same environment, the refactored code sends the model the same text, exposes the same tools,
applies the same rules and writes the same records as the original code.

Re-running would not give identical numbers in any case, because the models sample. The September rows (BCP-9,
BCP-11) were produced by a harness version that is not preserved (K8), and the Qwen BrowseComp-Plus rows were still
running or queued when this snapshot was taken.
