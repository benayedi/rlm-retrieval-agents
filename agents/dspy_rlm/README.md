# DSPy.RLM

A retrieval agent built on `dspy.RLM` (DSPy 3.3.1). The language model answers a question by writing Python in a
sandboxed REPL. In that REPL it can call a search tool, a document fetcher and a sub-LM. It was evaluated on
BrowseComp-Plus and OBLIQ-Bench in an **earlier campaign**. It has **no row in this repository's final results table**.

Everything below describes `src/` as it is. `src/` is identical in behaviour to `original/`; only names, imports,
paths and comments changed (see [CHANGES_FROM_ORIGINAL.md](CHANGES_FROM_ORIGINAL.md)). Statements about the code
were checked against the source and, where marked, by the tests in `tests/`. Statements about `dspy.RLM` refer to
the installed DSPy 3.3.1 (`dspy/predict/rlm.py`).

## 1. Overview

| | |
|---|---|
| Agent loop | `dspy.RLM`: one LM call per REPL turn writes code; the code runs in a Deno/Pyodide sandbox; the answer is `SUBMIT(answer=...)` |
| Tools given to the model | `search_index(query, top_k)`, `get_document(doc_id)` (both HTTP calls to a retrieval server); `llm_query` / `llm_query_batched` (sub-LM, added by dspy); `rlm_query` (a nested RLM, two OBLIQ WildChat variants only) |
| Prompt | a `dspy.Signature` per task: shared tool/rule text + a task "strategy" block (`src/dspy_rlm/signatures.py`, 35+ variants) |
| Benchmarks | BrowseComp-Plus (exact answer, LLM-judged accuracy); OBLIQ-Bench Math, Twitter, Congress, WildChat, Writing (ranked doc ids, NDCG@k) |
| Drivers | `src/run_browsecomp.py`, `src/run_obliq.py` |
| Launchers | `experiments/browsecomp.sh` (illustrative), `experiments/obliq_shard.sh` |
| Status | earlier campaign only; no final-table row; no per-experiment configs |

## 2. Research motivation

The Recursive Language Model (RLM) idea is to let a model keep large material outside its own context: the material
sits in REPL variables, and the model inspects it with code and with calls to a sub-model. `dspy.RLM` implements this
loop. This agent asks whether that loop, given only a retrieval tool, can do deep-research question answering
(BrowseComp-Plus) and oblique retrieval (OBLIQ-Bench). The project used it to measure how much the prompt ("strategy"
text), the reasoning effort, the root model and the retriever each contribute. The original code's comments record
these measurements, for example `search_index` in `agent.py` and the signature variants v2–v20 in `signatures.py`.

## 3. The problem addressed

- **BrowseComp-Plus**: 830 multi-constraint questions over a fixed corpus of about 100k web documents. Each question
  has one short answer, and the answer is in a few "gold" documents. The agent must find those documents through
  search and read them.
- **OBLIQ-Bench**: queries whose relevant documents share no surface terms with the query (analogues, tip-of-tongue
  passages, stance descriptions). The output is a ranked list of document ids, scored by NDCG@k against qrels.

In both, the agent works under a fixed budget of REPL turns and searches.

## 4. Methodology

### dspy.RLM (installed DSPy 3.3.1)
`dspy.RLM(signature, tools, max_iters, sub_lm, verbose[, max_output_chars])` runs this loop, read from `forward()`:
1. Each iteration makes one root-LM call (`generate_action`). Its inputs are the signature's instructions, dspy's own
   action instructions, the tool docs, the variable metadata, the REPL history so far and `iteration = "i/max_iters"`.
   The model returns `reasoning` and `code`.
2. The code runs in `PythonInterpreter`, which is Pyodide inside a Deno subprocess. Tools are callable from the code.
   Their calls run in the host Python process.
3. Printed output is appended to the history, cut to `max_output_chars` (default 10,000). The whole history goes
   into the next turn's prompt.
4. `SUBMIT(answer=...)` ends the run with that answer. If `max_iters` iterations pass without SUBMIT, dspy makes one
   more LM call (`extract`) to pull an answer from the history.
5. `llm_query`/`llm_query_batched` call the `sub_lm`. dspy caps them at `max_llm_calls = 50` per run; the harness
   does not change this. A call beyond the cap raises `RuntimeError` inside the tool.

### Signatures and strategy text
`make_rlm_signature(strategy, answer_desc, ...)` builds a signature with one input field (`question`) and one output
field (`answer`). Its instructions are `_REPL_TOOLS + _REPL_RULES + strategy + _ORCHESTRATION`:
- `_REPL_TOOLS`: a description of the tools.
- `_REPL_RULES`: rules such as "every turn must execute code" and "never loop over search_index".
- `_ORCHESTRATION`: tells the model to push long-context work into `llm_query`.

The strategy block is the task-specific method. For example, BrowseComp v1 says: search for named entities rather
than clue phrasing, verify every clue of a promising document in one `llm_query`, and commit after turn 12 of 16.
Some variants are built differently:
- v3 (`browsecomp_minimal_budget_sig`) writes the real `max_iters` and `max_search` into the prompt. Its commit turn
  is `round(0.75·max_iters)`, which `BROWSECOMP_COMMIT_TURN` can override.
- v9 (`BrowseCompSubtractiveSignature`) is a hand-written short prompt that does not use `make_rlm_signature`.

All prompt text is identical to the original; `tests/test_dspy_rlm_signatures.py` checks this.

### Tools (BrowseComp-Plus, `dspy_rlm.agent.build_rlm_agent`)
- `search_index(query, top_k=default_top_k)` POSTs `/search` through `RemoteRetriever`. Before the request, it applies
  these steps in order:
  1. `MAX_TOP_K` clamp (off by default).
  2. Per-turn cap `MAX_SEARCH_PER_TURN` (off by default). Searches less than 5 s apart count as one REPL turn; over
     the cap, the tool returns a one-item "limit reached" stub.
  3. Hard budget: once `max_search_calls` searches have been made, every call returns a "SEARCH DISABLED … call
     SUBMIT" stub, and the retriever is not called.
  4. Per-question cache keyed on the exact query string.

  Snippets are then cut to `SEARCH_SNIPPET_LEN=500` characters plus "…", or to `SEARCH_SNIPPET_TOKENS` Qwen3-0.6B
  tokens when that is set. If the server is unreachable, the process exits with `sys.exit(1)`. The cap, stub, cache
  and clamp behaviour is tested in `tests/test_dspy_rlm_agent.py`.
- `get_document(doc_id)` POSTs `/document`. If the text is at most `GETDOC_INLINE_CHARS=6000` characters, it is also
  printed into the REPL output, between `--- BEGIN/END ---` markers. The tool's docstring, which the model sees, is
  generated from that cap. `DISABLE_GET_DOCUMENT` removes the tool; `MINIMAL_TOOL_DOCS` replaces both docstrings with
  BrowseComp-Plus one-liners.

### Sub-LM
- **BrowseComp-Plus**: an Azure deployment. It is `SUB_LM_MODEL`, falling back to `AZURE_OPENAI_JUDGE_MODEL`, then
  `AZURE_OPENAI_MODEL`, then `gpt-5.4-mini`, with 8000 max tokens, a 120 s timeout and optional reasoning effort. It
  is always Azure, even when the root LM is local (issue D3). Calls are counted in `RunMetrics.llm_query_calls`.
- **OBLIQ**: `AZURE_OPENAI_JUDGE_MODEL`, falling back to `AZURE_OPENAI_MODEL`, then `gpt-5.4-mini`. It uses 4000
  tokens and a 60 s timeout, or 16000 tokens and 300 s with `SUB_LM_REASONING_EFFORT`. With `LOCAL_SUB_LM=1` it uses
  the local root model instead.
- **Root LM** (`build_lm`): Azure `azure/<AZURE_OPENAI_MODEL>` with temperature 1.0, 16000 max tokens and a 120 s
  timeout. If `LOCAL_LLM_BASE_URL` is set, it is an OpenAI-compatible local server instead
  (`openai/<LOCAL_LLM_MODEL>`, reasoning effort medium).

### Termination
A question ends in one of five ways:
1. The model calls SUBMIT.
2. `max_iters` iterations pass, then dspy's extract call produces the answer.
3. An exception occurs; the harness records the answer "ERROR".
4. The optional per-question timeout fires; the harness records "TIMEOUT". For BrowseComp the worker is a killed
   subprocess; for OBLIQ the thread is abandoned (issue D8).
5. The retrieval server is unreachable, and the process exits.

The search budget does not end the run. After the budget is spent, each further search returns the stub that asks
the model to SUBMIT.

## 5. Pseudocode

```
for each question q:                                   # sequential within one process
    metrics ← RunMetrics()
    tools ← [search_index, get_document]               # closures over retriever + metrics + budgets
    rlm ← dspy.RLM(signature, tools, max_iters, sub_lm=CountedAzureLM)
    history ← []
    for i in 1..max_iters:
        reasoning, code ← RootLM(instructions(signature), tool_docs, history, "i/max_iters")
        out ← Sandbox.execute(code)                    # may call search_index / get_document / llm_query / SUBMIT
        if out is SUBMIT(answer): break with answer
        history.append(reasoning, code, truncate(out, max_output_chars))
    else:
        answer ← RootLM.extract(history)               # dspy fallback after max_iters
    BrowseComp+: correct ← judge_match(q, answer, gold)        # substring shortcut, else Azure LLM judge
    OBLIQ:       ranking ← doc ids parsed from answer;  NDCG@k / Recall@k vs qrels
                 (also on trajectory order and on answer + trajectory tail)
    write per-question record; checkpoint

search_index(query, top_k):
    top_k ← min(top_k, MAX_TOP_K)                      if MAX_TOP_K set
    if per-turn count > MAX_SEARCH_PER_TURN:  return [limit stub]
    if metrics.search_calls ≥ max_search_calls: return [SEARCH DISABLED stub]
    if query in cache: return cache[query]
    results ← POST /search {query, top_k[, index]};  cut snippets;  count; cache
```

## 6. Architecture

```
 experiments/*.sh ──► src/run_browsecomp.py ───────────────┐      src/run_obliq.py ◄── experiments/obliq_shard.sh
   (paths.env,         │ Tevatron/browsecomp-plus (HF)      │        │ OBLIQ_BENCH_ROOT/<task>/<subset>/queries+qrels
    PYTHONPATH)        │ rlm_shared.browsecomp_data         │        │ own loaders + NDCG (dup. of rlm_shared.obliq_data)
                       ▼                                    │        ▼
              dspy_rlm.agent.run_question                   │   _build_agent_with_sig / run_obliq_question
              dspy_rlm.agent.build_rlm_agent                │   (own search_index/get_document closures)
                       │  signature ◄── dspy_rlm.signatures ┴──────── signature (SUBSET_SIGNATURES)
                       ▼
              dspy.RLM ── root LM (build_lm: Azure or LOCAL_LLM_BASE_URL)
                │   └──── sub_lm (llm_query) ── Azure deployment (OBLIQ: optionally local)
                ▼
          Deno + Pyodide sandbox ── tool calls ──► host Python closures
                                                     │
                                                     ▼
                                       dspy_rlm.tools.RemoteRetriever
                                       POST /search, /document; GET /health
                                                     │
                                                     ▼
                         retrieval server at EMBEDDING_SERVER_URL (benchmarks/retrieval/: search_server.py,
                         multi_search_server.py, fusion_proxy.py; protocol in ../../benchmarks/README.md)
```

| File | Role |
|---|---|
| `src/dspy_rlm/agent.py` | `RunMetrics`, `build_lm`, `build_rlm_agent` (tools, sub-LM, budgets), `run_question`, LM-time profiling |
| `src/dspy_rlm/signatures.py` | all prompts: BrowseComp v1–v20, OBLIQ Congress/Math/Twitter/WildChat/Writing variants, `ReaderSignature` |
| `src/dspy_rlm/tools.py` | `RemoteRetriever` HTTP client (`FaissRetriever` alias) |
| `src/run_browsecomp.py` | BrowseComp-Plus driver: dataset, optional subprocess timeout, judge, JSON/TXT output |
| `src/run_obliq.py` | OBLIQ driver: its own tool closures, optional nested reader RLM, NDCG/recall, JSON/TXT output |
| `agents/shared/rlm_shared/` | `browsecomp_data` (decrypt, judge), `obliq_score_advice`, `obliq_data` (shard mapping), `paths` |

## 7. Workflow

1. Start a retrieval server over the benchmark corpus that speaks `/search`, `/document` and `/health`. See
   [benchmarks/README.md](../../benchmarks/README.md).
2. Fill in `configs/paths.env` (repository level) and export the Azure and retrieval variables
   (`configs/example.env`).
3. Run a driver directly or through a launcher. For each question:
   1. Check `/health`, retrying 6 times at 60 s intervals.
   2. Build a fresh agent and run it.
   3. Score the answer and print a one-line summary.
4. Outputs, with the `--out` prefix:
   - BrowseComp: `<out>.json` (list of per-question records), `<out>.txt` (summary) and `<out>_checkpoint.json`
     (every 5 questions).
   - OBLIQ: `<out>.json` (header with averages and `results`), `<out>.txt` and `<out>_checkpoint.json` (after every
     question).
   - `obliq_shard.sh` writes `OUT_s<i>.{json,txt,log}` per shard. It does not merge them.

## 8. Parameters

CLI flags. The defaults are those of the code.

| Flag | `run_browsecomp.py` | `run_obliq.py` | Meaning |
|---|---|---|---|
| `--start`, `--end` | 0, 50 | 0, 0 (= all) | question range (0-based, end exclusive) |
| `--indices` | "" | "" | comma-separated indices; overrides the range |
| `--out` | `logs/run` | `logs/obliq_run` | output prefix |
| `--max-iters` | 16 | 12 (help says 16, D7) | `dspy.RLM` `max_iters` |
| `--max-search` | 35 | 8 | hard `search_index` budget per question |
| `--top-k` | 10 | 20 | default `top_k` of `search_index` |
| `--timeout` | `$BROWSECOMP_TIMEOUT` or 0 | `$OBLIQ_TIMEOUT` or 0 | per-question limit in s (0 = none) |
| `--sig` | v1 (choices v1–v11, v13–v17, v20) | — | BrowseComp signature |
| `--task`, `--subset` | — | required | OBLIQ task and subset (subset selects the signature; `wildchat-v7` fails, D6) |
| `--pool` | — | off | use `qrels_pool.tsv` |
| `--model` | — | "" | overrides `AZURE_OPENAI_MODEL` |
| `--max-depth` | 5 | — | not used (dead, D12) |
| `--verbose` | off | off | dspy.RLM verbose logging |

The main environment variables are listed below. The full list, with defaults, is in
[configs/example.env](configs/example.env).

| Variable | Default | Effect |
|---|---|---|
| `EMBEDDING_SERVER_URL` | required | retrieval server |
| `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY` | required | root LM (unless local), sub-LM, BrowseComp judge |
| `AZURE_OPENAI_MODEL` | `gpt-5.4-mini` | root-LM deployment |
| `AZURE_OPENAI_REASONING_EFFORT` | unset | root-LM effort (the earlier campaign varied low/medium/high) |
| `LOCAL_LLM_BASE_URL`, `LOCAL_LLM_MODEL` | unset, `qwen3.8-27b-bf16` | local OpenAI-compatible root LM |
| `SUB_LM_MODEL`, `SUB_LM_MAX_TOKENS`, `SUB_LM_TIMEOUT` | see §4, 8000, 120 | BrowseComp sub-LM |
| `SEARCH_SNIPPET_LEN` / `SEARCH_SNIPPET_TOKENS` | 500 / unset | snippet length shown to the agent |
| `GETDOC_INLINE_CHARS` | 6000 | documents up to this size are printed into context (BrowseComp) |
| `MAX_TOP_K`, `MAX_SEARCH_PER_TURN` | unset | top-k clamp, per-turn search cap (BrowseComp) |
| `RLM_MAX_OUTPUT_CHARS` | unset (dspy 10,000) | per-turn stdout cap |
| `OBLIQ_INDEX` | unset | index name sent to a multi-index server |
| `DSPY_FORMAT_FIX` | unset | OBLIQ: patches dspy.RLM with a reasoning/code cue and 3 parse attempts per turn |
| `OBLIQ_RELATIVE_SCORES` | unset | OBLIQ: replaces the fixed score cutoffs in prompts |

## 9. Benchmark integrations

### BrowseComp-Plus (`src/run_browsecomp.py`)
- **Data**: `datasets.load_dataset("Tevatron/browsecomp-plus", split="test")`. The rows are decrypted by
  `rlm_shared.browsecomp_data` (question, answer, gold and evidence doc ids).
- **Retrieval**: any server that speaks the protocol. In the earlier campaign this was Qwen3-Embedding-8B FAISS
  IVF-SQ8, or Reason-ModernColBERT behind `fusion_proxy.py`.
- **Scoring**: `judge_match(question, predicted, gold)`. A normalised two-way substring test on the declared final
  answer comes first (D2/K17). Otherwise one call to the Azure judge (`AZURE_OPENAI_JUDGE_MODEL`, falling back to
  `AZURE_OPENAI_MODEL`, then `gpt-5.4-mini`) with the official BrowseComp grader prompt; on error it falls back to
  the heuristic. This is the in-run `correct` field.
- **Records**: also `gold_recall` and `evid_recall`, the fractions of gold and evidence docs seen in searches or
  fetches, plus latencies, LM/retrieval/REPL time split, tokens, iterations, and search and fetch counts.

### OBLIQ-Bench (`src/run_obliq.py`)
- **Data**: `$OBLIQ_BENCH_ROOT/<task>/<subset>/queries+qrels/{queries.jsonl, qrels.tsv[, qrels_pool.tsv,
  per_query_excluded_ids.json]}`. Ablation subsets such as `congress-v5` map to the base data directory
  (`SUBSET_DATA`).
- **Tools**: the driver's own closures. They differ from BrowseComp:
  - no per-turn cap, `MAX_TOP_K` or token snippets;
  - 6 retries 60 s apart on connection errors;
  - `get_document` does not print the text;
  - per-query excluded ids (analogue tasks) are filtered after top-k (D10).

  `wildchat` and `wildchat-v6` drop `get_document` and add `rlm_query`, a nested `dspy.RLM(ReaderSignature,
  [get_document], max_iters=12)`.
- **Scoring**: three rankings per question, each with NDCG@k and Recall@k for k = 1, 3, 5, 10, 20, 50:
  - trajectory (first-retrieval order of every searched doc);
  - answer (doc ids parsed from the submitted text by per-corpus regexes);
  - tail-fill (the answer ranking followed by the rest of the trajectory).

## 10. Associated experiments

DSPy.RLM has **no row in the final results table**
([../../results/original_baselines/Experiments.xlsx](../../results/original_baselines/Experiments.xlsx); verified: no
sheet mentions it). It therefore has **no per-experiment documents** in `docs/experiments/` and no per-experiment
configs. `experiments/browsecomp.sh` only writes out the driver defaults. `experiments/obliq_shard.sh` is the earlier
campaign's launcher for running the OBLIQ harness on a local model (§11). Neither reproduces a final-table row.

## 11. Existing results (earlier campaign, not comparable with the final table)

These numbers come from the **original repository's** `docs/rlm_results_summary.xlsx`, sheet *Experiments*. That
file is **not part of this repository**; it was read read-only for this summary. They come from an earlier campaign
with a different protocol: different indexes, models, budgets and question handling.

**BrowseComp-Plus judging.**
- The in-run judge was **gpt-5.4-mini**. It came from an implicit `.env`; see K15.
- The workbook later re-graded these rows with gpt-4.1 (3 votes). For two rows that re-grade is an **estimate**,
  because some answers were never saved and were counted at the old judge's rate.

**OBLIQ scoring.** OBLIQ rows are NDCG@10 against gold qrels; no LLM judge is involved. The workbook's NDCG@10 column
is described as the agent's answer ranking in the Math sheet, but one WildChat row [80] notes that its value is
trajectory order, so the column is not uniformly defined.

**Do not compare any of these numbers with `results/original_baselines/Experiments.xlsx`.**

Best DSPy.RLM row per benchmark (workbook row numbers in brackets):

| Benchmark | Best DSPy.RLM row | Score | Notes in the workbook |
|---|---|---|---|
| BrowseComp-Plus (830 q) | "minimal" signature + Reason-ModernColBERT alone, doc_len 4096; gpt-5, effort medium, 16 iters / 35 searches / top-k 10 [8] | accuracy **0.7807** (gpt-4.1 re-grade); 0.7952 by gpt-5.4-mini in run | same config at doc_len 512: 0.7771 (0.788) [7]. Dense Qwen3-Embed-8B rows: v1 low 0.7096 (0.69) [3]; minimal low 0.6699 (estimate) [4]; minimal medium 0.712 (estimate) [5] |
| OBLIQ Math (151 q) | meta-program fingerprint agent, gpt-5.6-terra, effort medium [27] | NDCG@10 **0.1721** | dense Qwen3-Embed-8B FAISS IVF-SQ8 |
| OBLIQ Twitter (281 q) | implicit-stance agent, gpt-5.6-terra, effort medium [50] | NDCG@10 **0.1256** | |
| OBLIQ Congress (254 q) | contextual witness search + Reason-ModernColBERT alone, gpt-5.2, effort low [71] | NDCG@10 **0.1898** | requires `fusion_proxy --chunk-parent`; best dense row 0.1816 (gpt-5.6-terra medium) [68] |
| OBLIQ WildChat (40 q) | nested scorer with score-sort (outer + inner RLM), gpt-5, effort low [77] | NDCG@10 **0.0504** | |
| OBLIQ Writing | no complete run | 0.0556 on **71 of 512** questions [84] | partial: "server aborted" |

**Invalid and partial rows, as stated in the workbook:**
- [66] Congress, gpt-5.2 high: **"INVALID — DO NOT CITE"**. 225 of 254 questions hit the 300 s cap before the agent
  acted and were scored 0.
- [78] WildChat, gpt-5.2 nested scorer: **partial** (23/40 questions, run killed).
- [84] Writing: **partial** (71/512, server aborted).
- [4], [5] BrowseComp minimal low and medium: the gpt-4.1 accuracy is an **estimate**. 535 and 775 answers were
  re-judged; the rest are counted at the gpt-5.4-mini rate.
- [70] (Congress fusion) warns: "never quote partial congress numbers".

The workbook also has Qwen3.6-35B-A3B DSPy rows (Math 0.093 [35], Twitter 0.0468 [56]), which ran the same harness
on a local model. `obliq_shard.sh`'s header describes such an arm, but the workbook does not name the launcher.

## 12. Computational characteristics

These figures come from the code. The workbook figures are for the earlier campaign.

- **LM calls per question**:
  - up to `max_iters` root calls, plus one `extract` call if the model never SUBMITs;
  - up to 50 sub-LM calls (the dspy default `max_llm_calls`);
  - for the OBLIQ `rlm_query` variants, each nested reader adds up to 12 root calls of its own.
- **Prompt growth**: the whole REPL history is re-sent every turn, so input tokens grow with turns and with printed
  output. The comments in `agent.py` attribute most of the token cost to turns that batch many searches.
- **Retrieval**: at most `--max-search` searches, plus document fetches. Each is one HTTP request with a 60 s timeout
  (`RETRIEVER_TIMEOUT`).
- **Process model**: questions run sequentially within one process, and parallelism comes from separate processes
  (`obliq_shard.sh` default W = 6). Each `rlm(...)` call creates a fresh Deno/Pyodide interpreter and shuts it down at the end. The Qwen3-0.6B
  tokenizer is loaded only when `SEARCH_SNIPPET_TOKENS` is set.
- **Profiling**: each record splits latency into time blocked on the LM API, retrieval time, and the remainder (REPL
  plus overhead).
- **Workbook, earlier campaign**:
  - BrowseComp v1 dense averaged 8.7 iterations, 11.2 searches and 1.87 fetches per question, with a mean latency of
    288.9 s (125.8 s excluding questions over 600 s).
  - OBLIQ rows report about 48–428 s per question.

## 13. Limitations

See [../../docs/verification/KNOWN_ISSUES.md](../../docs/verification/KNOWN_ISSUES.md), in particular:
- **K17**: `heuristic_match` counts an empty prediction, or any substring of the gold, as correct.
- **K20**: the DSPy.RLM harness defects.
- K15: the in-run judge was taken from an implicit `.env`.
- K23: duplicated code.

The detailed list, with file:line for both trees, is in [CHANGES_FROM_ORIGINAL.md](CHANGES_FROM_ORIGINAL.md):

| ID | Issue |
|---|---|
| D1 | with `--timeout > 0`, signatures v13–v20 silently fall back to v1 |
| D2 | `heuristic_match`: an empty prediction (or any substring of the gold) is correct, and the LLM judge is skipped |
| D3 | a local root LM still needs Azure for the sub-LM (BrowseComp always; OBLIQ unless `LOCAL_SUB_LM=1`) |
| D4 | OBLIQ (and BrowseComp TXT) records log `AZURE_OPENAI_MODEL` even for local-model runs |
| D5 | the OBLIQ `judge_model` field's default differs from the sub-LM actually used |
| D6 | `--subset wildchat-v7` is accepted but raises `KeyError` |
| D7 | `--max-iters` help/default mismatch; the BrowseComp TXT header hard-codes "25 / 5" |
| D8 | an OBLIQ timeout leaves the worker thread (and its timing patch) running |
| D9 | BrowseComp timeout worker: join-before-get can turn a large result into TIMEOUT, or hang on a dead worker |
| D10 | OBLIQ excluded-id filter runs after top-k, so fewer than k results come back |
| D11 | OBLIQ loaders and metrics duplicate `rlm_shared.obliq_data`, with small differences |
| D12 | dead code: `max_depth`, `delegation_calls`, `fetched_docs`/`trajectory` (BrowseComp), `make_embedding_fn`, the undefined `FaissRetriever` annotation, the missing `update_baseline_summary.py`, unused imports |
| D13 | prompts promise "≤2000-char" snippets; the agent receives 500 characters by default |

Method-level limitations:
- The BrowseComp v1 prompt says "after turn 12 out of 16" whatever `--max-iters` is; only v3 states the real budget.
- `load_dotenv()` reads the nearest `.env` above the source files, so credentials and models can come from an
  implicit file (K15).
- Results depend on dspy's own action and extract prompts, which belong to the installed DSPy version (3.3.1 here).
  The earlier campaign's dspy version is not recorded in this repository.

## 14. Installation

1. Python 3.11 or later, with `dspy==3.3.1`, `requests`, `python-dotenv`, `datasets` (BrowseComp data) and `openai`
   (judge). Add `transformers` only for `SEARCH_SNIPPET_TOKENS`. The tests were run in such an environment (Python
   3.12).
2. **Deno**, which `dspy.RLM`'s default `PythonInterpreter` uses to run Pyodide:
   `curl -fsSL https://deno.land/install.sh | sh`. This installs to `~/.deno/bin`; the launchers add
   `${DENO_INSTALL:-$HOME/.deno}/bin` to `PATH`. dspy looks for the `deno` Python package's binary first, then
   `deno` on `PATH`.
3. On first use, Deno fetches `npm:pyodide@0.29.4` (the version imported by dspy's `runner.js`), so the first run
   needs network access or a warm Deno cache.
4. Copy `configs/paths.example.env` (repository root) to `configs/paths.env` and set `PYTHON`, `HF_HOME` and, for
   OBLIQ, `OBLIQ_BENCH_ROOT`. Set the variables in [configs/example.env](configs/example.env) in your shell.
5. Run a retrieval server from `benchmarks/retrieval/` over the right corpus; see
   [benchmarks/README.md](../../benchmarks/README.md).

## 15. How to run

**These commands make real model calls: Azure (paid) or a local server, plus the judge.** Nothing in this
repository's tests does.

```bash
# BrowseComp-Plus (illustrative defaults; any extra flag overrides them)
EMBEDDING_SERVER_URL=http://localhost:8010 AZURE_OPENAI_MODEL=<deployment> \
  agents/dspy_rlm/experiments/browsecomp.sh --start 0 --end 50 --out runs/dspy_rlm/bcp_q0_q50

# or the driver directly
PYTHONPATH=shared:agents/dspy_rlm/src python agents/dspy_rlm/src/run_browsecomp.py \
  --start 0 --end 10 --sig v2 --out runs/dspy_rlm/bcp_v2_q0_q10

# OBLIQ, one process
PYTHONPATH=shared:agents/dspy_rlm/src python agents/dspy_rlm/src/run_obliq.py \
  --task tip-of-tongue --subset congress --start 0 --end 30 --out runs/dspy_rlm/congress_q0_q30

# OBLIQ, sharded over a list of query ids (local model by default; see the script header)
OBLIQ_TASK=descriptive OBLIQ_SUBSET=twitter \
  agents/dspy_rlm/experiments/obliq_shard.sh dev_ids.json runs/dspy_rlm/twitter_dev 6
```

The drivers also add `src/` and `agents/shared/` to `sys.path` themselves, so `PYTHONPATH` is optional when they are run as
scripts.

## 16. Reproduction

- **Final table**: there is nothing to reproduce. DSPy.RLM has no row.
- **Earlier-campaign rows**: a rerun would need the same corpus index, root model, effort, budgets and signature as
  the workbook row (columns *Strategy / Version*, *Model*, *Embedding Model / Index*, *Max Iters*, *Max Search*,
  *Top-K*, *Reasoning Effort*). The obstacles:
  - Some indexes are not built by this repository. The OBLIQ dense rows used a FAISS IVF-SQ8 index built through
    `embed_server /encode`, and the :8001 dense server is not preserved (K22).
  - The workbook rows do not name the `--sig` flag.
  - Models such as gpt-5.6-terra are deployment-specific.
  - The in-run judge differs from this repository's gpt-4.1 grading.
- **Code identity**: `src/` reproduces the original code's behaviour exactly. The tests compare the ASTs and the
  prompts with `original/`.
- **No experiment was re-run for this repository.**

## 17. Tests

```bash
.venv/bin/python -m pytest agents/dspy_rlm/tests -q
```

98 tests. All are offline: Azure settings are dummies, `requests.post` is monkeypatched, and `dspy.RLM` is either a
recorder or built without being called.

| File | Tests | What it checks |
|---|---|---|
| `test_dspy_rlm_signatures.py` | 49 | every signature class constructs (fields, instructions, own name, builds a `dspy.RLM`); class set, instructions, field descriptions and all prompt constants equal the original `rag.signatures`; `make_rlm_signature` and `browsecomp_minimal_budget_sig` (several budgets, `BROWSECOMP_COMMIT_TURN`) match the original |
| `test_dspy_rlm_agent.py` | 18 | `build_rlm_agent` wiring (`max_iters`, tools, signature, sub-LM model choice, `DISABLE_GET_DOCUMENT`, `RLM_MAX_OUTPUT_CHARS`, docstrings), one real `dspy.RLM` construction; `search_index` truncation, cache, `MAX_TOP_K`, per-turn cap, hard `max_search_calls` stub, server-down exit; `get_document` inline print; the same call sequence on the original `rag.agent` gives identical results; `run_question` with a fake RLM |
| `test_dspy_rlm_static.py` | 14 | `py_compile` of every `src/` file; `bash -n` of both launchers; AST of `src/dspy_rlm/*.py` equals `original/RLM/rag/*.py` without module docstrings; driver ASTs equal the originals after undoing only the allowed renames |
| `test_dspy_rlm_tools.py` | 9 | `RemoteRetriever` URLs, payloads (with and without `index` / `OBLIQ_INDEX`), timeouts; byte-identical requests to the original |
| `test_dspy_rlm_drivers.py` | 8 | signature maps, shared-module wiring, `OBLIQ_BENCH_ROOT` loaders on a temp tree, metrics and id parser; pins D1, D2, D6 |
