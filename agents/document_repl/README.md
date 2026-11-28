# Document REPL agent

A recursive-language-model (RLM) research agent in which every search puts **whole documents** into a Python REPL
variable, and the model reads them itself with code. This folder holds the runnable harness (`src/`), the launchers
that produced the document REPL rows of the results workbook (`experiments/`), one configuration file per row
(`configs/`), mock-only tests (`tests/`) and the exact original files (`original/`, never edited).

Everything below describes behaviour checked against the code in `src/` and the `rlm` library it runs on
(github.com/alexzhang13/rlm at commit 854e688). Where a statement comes from run records instead of code, the
section says so.

## 1. Overview

The document REPL agent answers one question per process. The root model gets a REPL and a `search` tool. Each search
returns the complete text of the top-k documents and also merges them into a dictionary `DOCS` that persists across
turns. The model never sees a document unless its own code prints part of it. The harness caps the number of
searches (10) and turns (30). After the last turn it makes one more model call that forces a final answer. A run writes
one JSON record per question: the answer, the docids retrieved, client- and server-side search counts, the number of
iterations, token usage, wall time, and optionally the full trajectory.

It was evaluated on BrowseComp-Plus (830 questions), FRAMES (824) and three OBLIQ-Bench tasks (Math, Twitter,
Congress), with GPT-5, GPT-5.2 and Qwen3.6-35B-A3B and with dense, multi-vector and BM25 retrievers: 11 rows of
[Experiments.xlsx](../../results/original_baselines/Experiments.xlsx).

## 2. Research motivation

A standard search agent sees each hit as a short snippet in its message history, and evidence deep inside a long page
stays invisible unless the agent fetches the page. An RLM lets the model keep material in program variables and
inspect it with code instead of reading it all in context. The document REPL agent pushes that to the retrieval step: the
search tool hands over the whole document, and the model decides with Python what to read. The research question is
whether this reading strategy, compared with the [snippet REPL agent](../snippet_repl/) on the same benchmark, model and
retriever, changes accuracy, recall, searches and cost.

## 3. The problem addressed

Multi-hop questions over a fixed corpus, where the answer rests on several documents and often on text far from the
part of a page that a retriever indexed. On BrowseComp-Plus and FRAMES the output is a short exact answer graded by
an LLM judge. On OBLIQ the output is a ranked list of document ids, scored by NDCG@10 against qrels. Three
sub-problems are addressed in the harness itself:
- getting full documents to the model without flooding its context;
- bounding retrieval cost with a hard search cap;
- avoiding answer strings that appear in no source (the grounding gate).

## 4. Methodology

**REPL loop.** `run_one.py` builds an `rlm.RLM` (`environment="local"`, `depth=0`, `max_depth=1`) whose REPL contains
the tools of `tools.ToolBox.custom_tools()`: `search`, `search_many`, `grep_docs`, `unread_docs`, and the variables
`DOCS` and `SEARCH_LOG`. The final rows use the solo system prompt (`--solo-prompt`, no mention of sub-calls) and
the lean prologue (`--lean-prologue`), without the library's orchestrator addendum (`--no-orchestrator`). The
library's turn loop then runs: the model writes ```` ```repl ```` blocks, and the library executes them and appends the
output. The model submits by setting `answer["content"]` and `answer["ready"] = True`.

**Search into DOCS.** In design B (every final row), `search(query, k)` sends `mode: "full"` to the retrieval server.
Each hit is `{docid, score, text, n_chars, embedded_span, snippet_span}`, where `text` is the complete document. Hits
are merged into `DOCS[docid]`; a later snippet-only hit never overwrites a full text already held. The docid joins
`retrieved_docids` once, and the call is appended to `SEARCH_LOG` with its query and k. The model chooses k per call:
any value is clamped to 1..`MAX_K` = 50, and `--default-k` (5) applies only when it passes none. On OBLIQ,
`--fixed-k --default-k 20` pins k to 20 and the model's k is ignored. `search_many` runs several queries in one turn;
each one counts as a search. `grep_docs` (regex over DOCS) and `unread_docs` (docids never accessed) cost no search.

**Print cap.** Only `print` output reaches the model. The library truncates the output of each code block to 20,000
characters before it enters the next prompt (`rlm/utils/parsing.py`, `format_iteration`). With
`SUP_MAX_BLOCKS=5` (the Qwen3.6 rows) at most five code blocks per turn are executed.

**Search cap.** `--max-searches 10` is enforced inside `ToolBox.search`, in the same process as the model's code: the
11th call is not sent to the server. It returns `[]`, prints `[search refused] ...`, and increments `search_refused`.
A call that fails at the server still uses one unit of the cap. The retrieval server counts searches per run tag
independently; `judge.py` flags a run whose server count exceeds the cap.

**Turns and the forced answer.** `--max-iterations 30`. If no answer has been submitted after 30 turns, the library
appends "Please provide a final answer to the user's question based on the information provided." and makes one
more call (`RLM._default_answer`). That call is logged as an iteration, so a capped run records `n_iterations` = 31. On
OBLIQ the `cap_ranking` startup hook replaces that message with a request for a plain ranking. A run that hits
`--max-timeout` ends with status `error` and an empty answer.

**Seed context** (`--seed-context`, every final row). Before turn 1 the harness calls `search(question)`. This counts
as search 1 of 10, and its hits are both in `DOCS` and passed as the REPL's `context`.

**Grounding gate** (`--grounding-gate`, every BrowseComp-Plus and FRAMES row; off on OBLIQ). When the model sets
`answer["ready"]`, the harness takes the text after `Exact Answer:` and checks whether it occurs in the text of any
document in `DOCS`. The check is case-insensitive and never looks at the gold answer. If the text is not found, the
harness prints a notice and un-sets `ready`, once per run. The second submission is accepted as is.

**Sandbox.** In the REPL, `eval`, `exec`, `compile`, `globals`, `locals` and `input` are `None` (library default).
`run_one.harden_sandbox` adds three restrictions:
- `__import__` refuses a list of modules (network, subprocess, `importlib`, `pathlib`, `rlm`, `tools`, `openai`, `dotenv`, ...);
- `open` refuses every path;
- hostname resolution is restricted, process-wide, to the model endpoint hosts and loopback.

After the run, `scan_trajectory` flags model code containing introspection, network or filesystem tokens, and
`judge.py` scores flagged runs as incorrect.

**Sub-LLM functions are still installed.** `--no-delegation` is passed as an environment keyword argument that the
stock library stores and never reads, so `llm_query`, `llm_query_batched`, `rlm_query` and `rlm_query_batched` remain
callable in every row that ran on the stock library. Only the prompt asks the model to read by itself. The tool
description also advertises a `read_docs` function that is not installed (see §13).

## 5. Formulation

```
inputs: question q, model M, server S, cap C = 10, turns T = 30, MAX_K = 50, fallback k0 (5; 20 fixed on OBLIQ)
state:  H (message history), DOCS: docid -> record, n = 0 (searches), refused = 0

search(query, k):
    if C > 0 and n >= C:  refused += 1; print(refusal); return []
    n += 1
    k' = k0                      if k is None or k is fixed
       = clip(int(k), 1, MAX_K)  otherwise
    R = S.search(query, k', mode=full, tag)          # S counts under tag independently
    for r in R: DOCS[r.docid] = merge(DOCS[r.docid], r)   # full text is never replaced by a snippet
    return R

run(q):
    if seed_context: context = search(q)             # search 1 of C
    H = system_prompt(tools) + prologue + query_template(q)
    for t in 1..T:
        y_t = M(H);  blocks = repl_blocks(y_t)[:max_blocks]
        outputs = [truncate(exec(b), 20_000) for b in blocks]
        H += [y_t, outputs]
        if answer.ready:
            if gate and not warned and exact_answer(answer) not in any DOCS text: warned = True; answer.ready = False
            else: return answer.content                # n_iterations = t
    return M(H + "Please provide a final answer ...")  # n_iterations = T + 1
```

## 6. Architecture

```
 experiments/*.sh, configs/<ROW>.env
          |
          v
 run_experiment.py ─┐   split_runner.py (+ bcp_qwen_dense.py, bcp_bm25_split.py)   xargs loops in launchers
                    └──────────────┬──────────────────────────────────────────────────────────┘
                                   v  one subprocess per question
                  run_bm25.py ──> run_one.py ──> rlm.RLM (RLMLIB_PATH) ──> model endpoint (Azure / vLLM)
              (BM25 text rewrite)   │  prompts.py, fence_tolerant.py, rep_guard.py
                                    v
                               tools.ToolBox ──HTTP──> retrieval_server.py (dense) | retrieval_server_colbert.py
                                    │                   | BM25 server (benchmarks/retrieval) | obliq_adapter.py
                                    v
                         <RUNS_DIR>/.../<qid>.json ──> judge.py / grade_local.py / score_obliq.py
```

| Module | Role |
|---|---|
| `run_one.py` | per-question harness: argument parsing, sandbox, grounding gate, RLM construction, run record |
| `tools.py` | `ToolBox`: search (cap, k handling, DOCS merge), `search_many`, `grep_docs`, `unread_docs`, tool descriptions |
| `prompts.py` | query template, context note, prologue clauses, solo system prompt, `prologue_id()`, `GEPA_PROMPTS` override hook |
| `fence_tolerant.py`, `rep_guard.py` | opt-in Qwen switches. `fence_tolerant.py` runs mislabelled code fences and caps the blocks per turn: on in every Qwen3.6 row (`SUP_FENCE_TOLERANT=1`, `SUP_MAX_BLOCKS=5`). `rep_guard.py` stops repetition loops: off in every row (no config sets `SUP_REP_GUARD`) |
| `run_bm25.py` | runs `run_one.py` with the retriever sentences rewritten for BM25 (FRAMES or BrowseComp-Plus) |
| `run_experiment.py` | batch driver: sample/ids, one worker per question, resume, provenance, timeout stubs |
| `split_runner.py` | GEPA prompt optimiser; its `--seed-split` mode and `run_batch()` ran the Qwen3.6 BrowseComp-Plus rows with a result cache |
| `bcp_qwen_dense.py`, `bcp_bm25_split.py` | the BCP-27 and BCP-31 drivers built on `split_runner.py` |
| `retrieval_server.py`, `encoder.py` | dense Qwen3-Embedding-8B server (exact cosine) |
| `eval_first_stage.py` | evidence recall@k and hit@k of the raw question as one query (dense, BM25, fusion); source of the recall figures in the search description, see [results/first_stage/](../../results/first_stage/README.md) |
| `retrieval_server_colbert.py`, `hybrid.py` | Reason-ModernColBERT (PLAID) server; `hybrid.rrf` for an optional BM25 arm |
| `obliq_adapter.py` | translates this harness's wire format onto an OBLIQ search server |
| `prepare_data.py`, `build_bm25.py` | decrypt BrowseComp-Plus into `FULLDOC_ROOT/data/prepared`; tuned BM25 index (not used by any row) |
| `judge.py`, `grade_local.py`, `score_obliq.py`, `provenance.py` | gpt-4.1 judge, local Qwen grader, OBLIQ NDCG scorer, run provenance |

The harness was built by in-place patch scripts that are not part of `src/`; [src/PATCH_HISTORY.md](src/PATCH_HISTORY.md)
lists what each one changed and which of those changes are present. [CHANGES_FROM_ORIGINAL.md](CHANGES_FROM_ORIGINAL.md)
lists every difference between `src/` and `original/`.

## 7. Execution workflow

1. Prepare data under `FULLDOC_ROOT` (`prepare_data.py` for BrowseComp-Plus; FRAMES and OBLIQ query files are passed
   with `QUERIES_FILE`).
2. Start the retrieval server for the row (port and command in [benchmarks/README.md](../../benchmarks/README.md)) and,
   for Qwen3.6, the vLLM model server(s) on :8030/:8031.
3. Run the launcher. It sources `configs/paths.env`, sets credentials from `AZ_ENV` and `PYTHONPATH`
   (`<hook dir>:$RLMLIB_PATH`, or `$RLMLIB_PATH` alone), then starts one `run_one.py` process per question with
   `cwd = FULLDOC_ROOT`. Existing result files are skipped, so a run can be resumed.
4. Each process writes `<out>/<qid>.json`; GPT launchers also write `<qid>.log`; `split_runner.py` stores graded
   summaries in `<GEPA_OUT>/cache.json` under `<cache key>|<qid>`.
5. Grade: `judge.py` (gpt-4.1, BrowseComp-Plus template, votes, cache) or `score_obliq.py` (NDCG). The numbers in
   §11 are the workbook's gpt-4.1 grades; see each experiment document for its grading.

## 8. Parameters

"Final rows" are the 11 rows in `configs/`. Values are from the run records' `config` header, the launchers, or
`_provenance.json`; BCP-21 values come from run summaries only.

| Flag / env | Default | Final rows |
|---|---|---|
| `--design` | `A` | `B` (whole documents) in all rows |
| `--default-k` / `--fixed-k` | 5 / off | 5 as fallback with model-chosen k ≤ 50 (BCP, FRAMES); `--fixed-k --default-k 20` (OBLIQ) |
| `--max-searches` | 0 (no cap) | 10 (BCP-21: not recorded, max observed 10) |
| `--max-iterations` | 30 | 30 |
| `--max-timeout` (s) | 1800 | 2700 (BCP-25, FR-11), 5400 (BCP-23, Qwen3.6 rows), 3600 (OBLIQ), BCP-21 unknown |
| `--lm-timeout` (s) | 600 | 1800 (Qwen3.6 rows, OBLIQ); 600 otherwise |
| `--reasoning-effort` / `--sub-reasoning-effort` | high / high | high (GPT-5 BCP-23/25, FR-11; GPT-5.2 OBLIQ), medium (BCP-21, Qwen3.6). Whether it is sent: §13, K3 |
| `--seed-context`, `--compaction` (`--compaction-threshold-pct 0.85`) | off | on |
| `--grounding-gate` | off | on (BCP, FRAMES); off (OBLIQ) |
| `--solo-prompt`, `--lean-prologue`, `--no-delegation`, `--no-orchestrator` | off | on |
| `--max-concurrent-subcalls` | 4 | 4 |
| `--save-trajectory` | off | on (BCP-23, Qwen3.6 rows, OBLIQ); off (BCP-25, FR-11) |
| `--backend` | `azure_openai` | `openai` (vLLM) for Qwen3.6 |
| `SUP_FENCE_TOLERANT` / `SUP_MAX_BLOCKS` / `SUP_PRESENCE_PENALTY` | unset | `1` / `5` / `1.5` for Qwen3.6; unset otherwise |
| `SUP_REP_GUARD` | unset | unset |
| `BM25_CORPUS` (with `run_bm25.py`) | FRAMES wording | `bcp` for BCP-25/31; unset for FR-11/13 |
| `QUERIES_FILE` | `<FULLDOC_ROOT>/data/prepared/queries.jsonl` | FRAMES and OBLIQ query files |
| `GEPA_PROMPTS` | unset | OBLIQ prompt files (`obliq_fd_gemini/prompts_<task>.json`); the seed blocks in `split_runner.py` rollouts |
| startup hooks (`PYTHONPATH`) | none | `retry429` (BCP-25); `run5d` = retry429 + azure_sampling + cap_ranking (FR-11, OBLIQ, with `OBLIQ_CAP_RANKING=1`); none (Qwen3.6) |
| `MAX_K` (constant) | 50 | 50 |

## 9. Benchmark integrations

- **BrowseComp-Plus.** `prepare_data.py` decrypts the Tevatron dataset into
  `FULLDOC_ROOT/data/prepared/{queries.jsonl, corpus.jsonl, qrel_*.json}`. `run_one.load_query` reads only the question
  text, never the answer.

  | Retriever | Server |
  |---|---|
  | dense | `retrieval_server.py` (:8765) |
  | ColBERT | `retrieval_server_colbert.py`, 8,192-token index (:8766) |
  | BM25 | `benchmarks/retrieval/frames_search_server.py --dataset bcp --protocol sup` (:8043), driven through `run_bm25.py` |

- **FRAMES.** `QUERIES_FILE=$FRAMES_DATA_DIR/frames_queries.jsonl`. The retriever is BM25 over Wikipedia,
  `frames_search_server.py --protocol sup` (:8041), always through `run_bm25.py`.
- **OBLIQ-Bench.** `run_experiment.py` with `QUERIES_FILE` and `GEPA_PROMPTS` set to the task's prompt file.
  - Retrieval: Gemini-Embedding-2. Math goes through `obliq_math_search.py` on :8014; Twitter and Congress go through
    `obliq_adapter.py` on :8768 and :8769.
  - Settings: k fixed at 20, no grounding gate, and the `cap_ranking` hook.
  - Scoring: `score_obliq.py` computes NDCG with `agents/shared/rlm_shared/obliq_data.py`, the scorer the snippet REPL agent uses.

All servers speak the "sup" wire format: every search carries a per-run tag, and `/tag_counts` and `/tag_docids`
return the server's own record (see [benchmarks/README.md](../../benchmarks/README.md)).

## 10. Associated experiments

- [BCP-21: BrowseComp-Plus, GPT-5, Qwen3-Embedding-8B](../../docs/experiments/document_repl/BCP-21_browsecomp-plus-gpt-5-qwen3-embedding-8b.md)
- [BCP-23: BrowseComp-Plus, GPT-5, Reason-ModernColBERT (8,192-token index)](../../docs/experiments/document_repl/BCP-23_browsecomp-plus-gpt-5-reason-moderncolbert-over-an-8-192-token-document-index.md)
  (also present under a truncated file name:
  [BCP-23 …over-an-8](../../docs/experiments/document_repl/BCP-23_browsecomp-plus-gpt-5-reason-moderncolbert-over-an-8-192-token-document-index.md))
- [BCP-25: BrowseComp-Plus, GPT-5, BM25](../../docs/experiments/document_repl/BCP-25_browsecomp-plus-gpt-5-bm25.md)
- [BCP-27: BrowseComp-Plus, Qwen3.6, Qwen3-Embedding-8B](../../docs/experiments/document_repl/BCP-27_browsecomp-plus-qwen3-6-35b-a3b-fp8-qwen3-embedding-8b.md)
- [BCP-29: BrowseComp-Plus, Qwen3.6, Reason-ModernColBERT](../../docs/experiments/document_repl/BCP-29_browsecomp-plus-qwen3-6-35b-a3b-fp8-reason-moderncolbert.md)
- [BCP-31: BrowseComp-Plus, Qwen3.6, BM25](../../docs/experiments/document_repl/BCP-31_browsecomp-plus-qwen3-6-35b-a3b-fp8-bm25.md)
- [FR-11: FRAMES, GPT-5, BM25](../../docs/experiments/document_repl/FR-11_frames-gpt-5-bm25.md)
- [FR-13: FRAMES, Qwen3.6, BM25](../../docs/experiments/document_repl/FR-13_frames-qwen3-6-35b-a3b-fp8-bm25.md)
- [MAT-11: OBLIQ Math analogues, GPT-5.2, Gemini-Embedding-2](../../docs/experiments/document_repl/MAT-11_obliq-bench-math-analogues-gpt-5-2-gemini-embedding-2.md)
- [TWI-11: OBLIQ Twitter descriptive, GPT-5.2, Gemini-Embedding-2](../../docs/experiments/document_repl/TWI-11_obliq-bench-twitter-conflict-descriptive-gpt-5-2-gemini-embedding-2.md)
- [CON-11: OBLIQ Congress tip-of-the-tongue, GPT-5.2, Gemini-Embedding-2](../../docs/experiments/document_repl/CON-11_obliq-bench-congress-hearings-tip-of-the-tongue-gpt-5-2-gemini-embedding-2.md)

### Follow-up runs

- **FRC-1**: FRAMES with search cap 25 / 50 turns (FR-13 with only the caps raised): `experiments/frames_qwen_cap25_test200.sh`,
  `frames_qwen_cap25_rest.sh`, `configs/FRC-1.env`, [FRC-1](../../docs/experiments/document_repl/FRC-1.md).
- **TWQ-2**: OBLIQ-Twitter with Qwen3.6 and plain dense search, the control of SP-7: `experiments/obliq_twitter_qwen36.sh`,
  `configs/TWQ-2.env`, [TWQ-2](../../docs/experiments/document_repl/TWQ-2.md).
- The search-policy and GEPA studies run this agent unchanged ([studies/](../../studies/README.md)); their agent-side
  configs are `configs/SP-{1,2,3,4,7}.env`, and `src/split_runner.py` is also the BrowseComp-Plus GEPA driver.
- The Qwen3.6 server settings (262,144-token context, 16,384-token output cap) are in `benchmarks/models/serve_qwen36.sh`.

## 11. Existing results

Copied from [Experiments.xlsx](../../results/original_baselines/Experiments.xlsx) (document REPL agent rows; "Iterations"
counts root turns including the forced final-answer turn). The model labels are the workbook's. Comparison with the
other agents: [AGENT_COMPARISON.md](../../docs/experiments/AGENT_COMPARISON.md).

**BrowseComp-Plus (830)**

| Row | Model (reasoning) | Retriever | Accuracy | Recall | Searches / q | Iterations / q | Time / q | Uncached cost / q ($) | Input / output tokens / q |
|---|---|---|---|---|---|---|---|---|---|
| BCP-21 | GPT 5 (medium)* | Qwen3-Embedding-8B | 0.8313 | 0.7619 | 7.9 | 11.2 | 300.2 s | 0.4395 | 165,685 / 23,236 |
| BCP-23 | GPT 5 (high) | Reason-ModernColBERT | 0.8747 | 0.7875 | 8.12 | 10.8 | 482.5 s | 0.5847 | 166,432 / 37,666 |
| BCP-25 | GPT 5 (medium)* | BM25 (k1 0.9, b 0.4) | 0.6 | 0.4973 | 9.2 | 18.2 | 567 s | 0.9266 | 390,854 / 43,804 |
| BCP-27 | Qwen3.6 (medium) | Qwen3-Embedding-8B | 0.4422 | 0.3973 | 8.66 | 22.4 | 2439 s | local GPU | 392,252 / 36,558 |
| BCP-29 | Qwen3.6 (medium) | Reason-ModernColBERT | 0.5795 | 0.5123 | 8.6 | 20.9 | 1829 s | local GPU | 335,994 / 30,906 |
| BCP-31 | Qwen3.6 (medium) | BM25 (k1 0.9, b 0.4) | 0.2241 | 0.1806 | 9.7 | 27.3 | 1388 s | local GPU | 617,614 / 58,153 |

\* Workbook footnote: these two GPT-5 rows ran at medium effort; BCP-25 requested high, but the Azure client dropped
the setting.

**FRAMES (824)**

| Row | Model (reasoning) | Retriever | Accuracy | Recall | Searches / q | Iterations / q | Time / q | Cached / uncached cost / q ($) | Input / output tokens / q |
|---|---|---|---|---|---|---|---|---|---|
| FR-11 | GPT 5 (high) | BM25 (k1 0.9, b 0.4) | 0.807 | 0.7039 | 6.17 | 10.4 | 333 s | 0.3537 / 0.4161 | 107,126 / 28,214 |
| FR-13 | Qwen3.6 (medium) | BM25 (k1 0.9, b 0.4) | 0.6104 | 0.5981 | 7.6 | 20.2 | 394 s | local GPU | 225,873 / 19,312 |

**OBLIQ-Bench** (GPT 5.2 (high), Gemini-Embedding-2; NDCG shown as gold / pooled qrels, Congress gold only)

| Row | Task | Recall@10 | NDCG@10 | NDCG@50 | Searches / q | Iterations / q | Time / q | Cached / uncached cost / q ($) | Input / output tokens / q |
|---|---|---|---|---|---|---|---|---|---|
| MAT-11 | Math (151) | 0.1361 | 0.1590 / 0.2024 | 0.1487 / 0.1879 | 10 | 15.4 | 494 s | 0.6188 / 0.7358 | 178,744 / 30,214 |
| TWI-11 | Twitter (281) | 0.0951 | 0.0950 / 0.1800 | 0.1028 / 0.1814 | 10 | 8.5 | 179 s | 0.2298 / 0.271 | 74,073 / 10,096 |
| CON-11 | Congress (254) | 0.1496 | 0.1332 | 0.1332 | 10 | 12.4 | 391 s | 0.5264 / 0.6271 | 168,368 / 23,744 |

The WildChat and Writing OBLIQ rows of the workbook are marked "to run" for this agent.

## 12. Computational characteristics

- **One process per question.** The library's `LocalREPL` swaps process-global `sys.stdout` and the working
  directory, so concurrent questions need separate interpreters. Launchers run 5 to 20 questions in parallel.
- **Context grows with what the model prints.** Inputs per question are 74k to 618k tokens in §11. They are largest
  with BM25, where more searches and turns are used. The 20,000-character cap per block bounds each output, not the
  number of turns.
- **Wall time.** GPT-5/GPT-5.2: 3 to 10 min per question. Qwen3.6 on local vLLM: 6.6 min (FRAMES) to 40.6 min
  (BrowseComp-Plus dense) per question, under the 5,400 s harness limit.
- **Retrieval servers hold everything in memory.**
  - The dense server loads the 100,195 × 4,096 corpus matrix (on CPU by default), the full corpus text and the
    Qwen3-Embedding-8B query encoder; its docstring gives 15 GiB for the encoder.
  - The ColBERT server loads the PLAID index on the GPU in a separate pylate environment (`PYLATE_PYTHON`).
- **Records are large.** A record with `--save-trajectory` holds every turn. The original BM25 launcher notes 0.3 to
  2.3 GB of trajectory per question with whole BM25 pages, and BCP-25 and FR-11 ran without it.

## 13. Limitations

Each item has an id in [KNOWN_ISSUES.md](../../docs/verification/KNOWN_ISSUES.md).

- **K1:** k is chosen by the model, up to 50 documents per search, not 5. Only OBLIQ fixes k (20).
- **K2:** a document REPL row and its snippet REPL counterpart differ in search cap, turn limit, time limit and ColBERT index
  length, not only in reading strategy.
- **K3:** the stock Azure client drops `reasoning_effort`. BCP-25 ran at the deployment default; BCP-21/23 are
  unverified. On Azure, only the `run5d` hook (FR-11, OBLIQ) sends it; the OpenAI-compatible client of the Qwen3.6
  rows forwards it.
- **K5:** BCP-21 and BCP-23 ran on a patched `rlm` build that was not preserved. BCP-21's launchers are lost.
  BCP-23 logged up to 19 searches against a cap of 10.
- **K7:** a question that does not finish (time limit, API error) is scored wrong. The partial answer is not kept.
- **K9:** "no sub-LLM" is enforced by the prompt only; the functions stay installed. A scan of the OBLIQ trajectories
  found no call to them. The BrowseComp-Plus and FRAMES trajectories were not scanned.
- **K10:** the tool text describes things that are not there. It advertises `read_docs` (not installed) and
  `llm_query` for reading, and describes a dense single-vector retriever even for the ColBERT rows. In the OBLIQ
  trajectories the model's code calls `read_docs` in 33 of 151 MAT-11, 17 of 281 TWI-11 and 37 of 254 CON-11 runs,
  and each of those runs records a `NameError` for it.
- **K12:** `n_iterations` includes the forced final-answer call (31 at the cap).
- **K21:**
  - `served_models` is always `None` on the stock library;
  - `prompt_id` does not hash the solo prompt;
  - `run_bm25.py` patches `json.dump` process-wide;
  - temporary prompt files are not deleted;
  - failed searches count against the cap.
- **K22:** `assert_served_model.py`, `report.py` and `budget.py` were not preserved, so the BCP-23 launchers stop at
  the `--expect-served-model` gate.

The cap, the sandbox and the gate all run in the same process as the model's code. Their integrity rests on the
server-side counts and the trajectory scan, not on isolation.

## 14. Installation

1. Python 3.12 and the pinned packages in [requirements.txt](../../requirements.txt) (agent harness: `openai`,
   `python-dotenv`, `requests`; servers: `torch`, `transformers`, `bm25s`, `PyStemmer`, `numpy`, `pyarrow`; ColBERT:
   `pylate` in its own environment; GEPA mode: `gepa`).
2. Check out the `rlm` library at commit 854e688 and put it on `PYTHONPATH` through `RLMLIB_PATH`. The launchers
   do this; it is not pip-installed.
3. Copy `configs/paths.example.env` (repository root) to `configs/paths.env` and fill it in:
   - `RLMLIB_PATH`, `FULLDOC_ROOT`, `RUNS_DIR`, `PYTHON`, `HF_HOME`;
   - `FRAMES_DATA_DIR`, `BCP_BM25_INDEX_DIR`;
   - `AZ_ENV` (a file with `AZ_EP` and `AZ_KEY`), `AZURE_DEPLOYMENT_GPT5`, `AZURE_DEPLOYMENT_GPT52`;
   - this agent also reads `RLM_DATA_DIR` and `HOOKS_DIR`, plus `AZURE2_API_VERSION` for the BCP-23 launchers
     (see [CHANGES_FROM_ORIGINAL.md](CHANGES_FROM_ORIGINAL.md)).
4. Data: run `python src/prepare_data.py` after placing the BrowseComp-Plus parquet files under
   `FULLDOC_ROOT/data/`. Then build or obtain the indexes and models listed in
   [benchmarks/README.md](../../benchmarks/README.md).

## 15. How to run

**Every command in this section launches real model calls** (Azure OpenAI or a local vLLM server) and needs a
running retrieval server. A full 830-question GPT-5 run costs on the order of the per-question costs in §11 times
830. Nothing here is mocked.

```bash
set -a; . configs/paths.env; set +a          # from the repository root
FD=$PWD/agents/document_repl
cd "$FULLDOC_ROOT"
# one question, Azure gpt-5, dense server on :8765 (credentials exported from $AZ_ENV)
PYTHONPATH=$RLMLIB_PATH $PYTHON $FD/src/run_one.py --query-id 1 --out $RUNS_DIR/demo/1.json \
  --design B --server http://127.0.0.1:8765 --root-model gpt-5 --sub-model gpt-5 \
  --root-deployment $AZURE_DEPLOYMENT_GPT5 --sub-deployment $AZURE_DEPLOYMENT_GPT5 \
  --max-iterations 30 --max-searches 10 --seed-context --compaction --grounding-gate \
  --lean-prologue --no-delegation --solo-prompt --no-orchestrator --save-trajectory
# a batch, resumable
PYTHONPATH=$RLMLIB_PATH $PYTHON $FD/src/run_experiment.py --name demo --n 10 --design B ...same flags...
# grade (gpt-4.1 on Azure, AZURE2_* credentials)
$PYTHON $FD/src/judge.py --run-dir $RUNS_DIR/demo --votes 3
```

Note that `run_one.py` reads Azure credentials from `AZURE2_ENDPOINT` / `AZURE2_API_KEY` / `AZURE2_API_VERSION` and
otherwise falls back to the library's `AZURE_OPENAI_*` variables. The launchers set the latter from `AZ_ENV`.

## 16. Reproducing the existing experiments

Each row has a configuration file in `configs/<ROW>.env`. Its header gives the model, the effort requested and the
effort actually sent, the retriever, the caps and the original output location. Its variables give the exact
arguments and environment. Launchers by row:

| Row | Launcher |
|---|---|
| BCP-25 | `experiments/bcp_gpt5_bm25.sh` |
| BCP-27 | `python src/bcp_qwen_dense.py <workers>` |
| BCP-29 | `python src/split_runner.py --seed-split <split>` with the env in `configs/BCP-29.env` |
| BCP-31 | `python src/bcp_bm25_split.py`, plus `BCP_IDS=... BCP_OUT=... BCP_SERVER=...` for the remaining slices |
| FR-11 | `experiments/frames_gpt5.sh <out> <workers> 0 824` |
| FR-13 | `BM25_PROMPT=1 experiments/frames_qwen.sh` |
| MAT-11, TWI-11, CON-11 | `experiments/obliq_fd.sh fd <task> <ids.json> <out> 5` |
| BCP-23 | `experiments/bcp_gpt5_final830/launch_high*.sh` |

The Qwen3.6 drivers reuse their cache: `split_runner.py` computes the same cache keys as the original runs
(e925f0f6c7, ee5ea32a1c, 11d0c5bc6b), so pointing `GEPA_OUT` at an original cache skips finished questions.

LLM sampling is not deterministic, so a re-run reproduces the configuration, not the per-question outputs. Two rows
cannot be reproduced exactly from this repository:
- **BCP-21:** its launchers were not preserved (`bcp_gpt5_final830/final830_*.sh` is a different run).
- **BCP-23:** its launchers abort at `--expect-served-model`, because the helper script is missing.

Both also ran on an unpreserved patched `rlm` build (K5). On the stock library the same flags leave the sub-LLM
functions installed and do not send the effort unless the `azure_sampling` hook is loaded. To match BCP-25 exactly,
keep its `retry429`-only hook path: the row was produced without effort being sent.

## 17. Tests

```bash
PYTHONPATH=shared:$RLMLIB_PATH .venv/bin/python -m pytest agents/document_repl/tests -q
```

Mocks only: no network, model or retrieval server. With `rlm` importable, 89 tests pass; without it, 82 pass and 7
are skipped. They check:
- the search cap of 10, the refusal counter and `search_many` against a fake server;
- k handling: model-chosen and clamped to 1..50, ignored under `--fixed-k` and design A;
- DOCS merging, docid de-duplication and unread tracking;
- every prompt string, prologue and tool description equal to `original/`;
- the BM25 rewrite equal to the original wrapper's output;
- `prompt_id` equal to the run records' values;
- the three cache keys;
- `run_one.main()` with a fake agent that drives the real logger and the library's forced-answer call
  (`n_iterations` = 31 at the cap, `served_models` = None, partial answer dropped on timeout);
- `no_delegation` ignored by `LocalREPL`, and the adapter's snippet mode;
- `py_compile` of `src/`, `bash -n` of the launchers, every config parsing, and launcher flags equal to the originals.
