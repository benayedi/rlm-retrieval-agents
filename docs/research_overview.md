# Research overview

## Motivation

A Recursive Language Model (RLM) does not answer from one long prompt. The model writes Python, one turn at a time, in
a persistent REPL that holds its tools and data, and it sees only what its code prints. Instead of stuffing retrieved
documents into the context, it can search, filter, read parts of documents, call a sub-model on a piece of text, and
keep intermediate results in variables.

This project asks how such agents should **retrieve and read** when the answer is spread over several documents in a
large corpus.

## Research questions

1. **Reading strategy.** Should an RLM see short snippets and read documents through a sub-LLM (Snippet RLM)? Or should
   it get whole documents and read them itself (Full-Doc RLM)? Does the answer depend on the retriever and the
   benchmark?
2. **Retriever.** How much do the first-stage retriever and its index matter under an agentic loop? The candidates are
   lexical BM25, an 8B single-vector dense embedder (Qwen3-Embedding-8B), a 149M late-interaction model
   (Reason-ModernColBERT) and Gemini-Embedding-2.
3. **Model.** How much does a frontier API model (GPT-5 / GPT-5.2) gain over an open-weight local model (Qwen3.6-35B-A3B)
   under the same harness?
4. **Where agents fail.** Do failures come from never retrieving the evidence, from seeing only part of it, from
   misreading it, or from running out of time or turns?

## The three approaches

| | DSPy.RLM | Snippet RLM | Full-Doc RLM |
|---|---|---|---|
| Agent loop | `dspy.RLM` (DSPy 3.3.1) | the RLM authors' library (`rlm` @ 854e688) with runtime patches | the same library, separate harness |
| Prompting | typed DSPy signatures with task-specific strategy text | the authors' RLM system prompt, adapted task text and tool docs | its own solo system prompt |
| A search returns | top-k snippets (500 chars) | 5 snippets: min(2,000 chars, 512 tokens) of each document's head | whole documents into a `DOCS` variable; the model chooses k (≤ 50) |
| Reading | `get_document` inline, or an Azure sub-LM | `get_document`, plus `llm_query` sub-LLM calls on text | the model reads slices itself (print cap 20,000 chars per block) |
| Budgets (final rows) | 16 iterations, 35 searches (earlier campaign) | 100 / 50 / 12 turns; no cap or 25 searches | 30 turns + forced answer; 10 searches |
| In the final table | **no** (earlier campaign only) | yes | yes |

Details: [algorithms/dspy_rlm](../algorithms/dspy_rlm/README.md), [algorithms/snippet_rlm](../algorithms/snippet_rlm/README.md),
[algorithms/full_doc_rlm](../algorithms/full_doc_rlm/README.md).

## Benchmarks

| Benchmark | Task | Size | Metric |
|---|---|---|---|
| BrowseComp-Plus | multi-constraint questions with one short answer, over a fixed web corpus | 830 questions, 100,195 documents | accuracy (LLM judge), gold-document recall |
| FRAMES | multi-hop factual questions over Wikipedia | 824 questions | accuracy (LLM judge), gold-article recall |
| OBLIQ-Bench | rank documents relevant in an oblique way: Math analogues, Twitter conflict stance, Congress tip-of-the-tongue | 151 / 281 / 254 queries | NDCG@10, Recall@10 |

Sources, indexes and licence notes: [benchmarks/README.md](../benchmarks/README.md).

## Experimental methodology

Each experiment is one row of [Experiments.xlsx](../results/original_baselines/Experiments.xlsx): one algorithm × model ×
retriever on one benchmark, run over every question.

- **Execution.** Each question runs in its own process with a time limit. The run writes a per-question record
  (answer, retrieved document ids, tool counts, turns, time, tokens) and the full trajectory.
- **Judging (QA benchmarks).** Correctness is judged after the run by gpt-4.1 with the official BrowseComp-Plus grader
  template, 3 votes, majority. A question without a completed answer counts as wrong.
- **Scoring (OBLIQ).** OBLIQ is scored by NDCG@10 against the gold qrels (binary relevance), on the ranking parsed from
  the answer text.
- **Seeds.** None are set. Every row is a single run.

## Findings

The full, sourced statement is in [ALGORITHM_COMPARISON.md](experiments/ALGORITHM_COMPARISON.md). In short:

- **Retriever and model dominate.** Reason-ModernColBERT > BM25, and GPT-5 > Qwen3.6, in every paired test (p < 1e-15
  for the model).
- **Snippet vs Full-Doc depends on the retriever.** With GPT-5 on BrowseComp-Plus, Full-Doc wins with the dense
  retriever, the two tie with ColBERT, and Snippet wins with BM25. On FRAMES with GPT-5, Full-Doc wins. These pairs
  also differ in search and turn budgets, so they compare complete packages.
- **Retrieval limits the best runs.** On BrowseComp-Plus, with every gold document retrieved, the GPT-5 Full-Doc runs
  are right 95–96% of the time (84% on FRAMES). Most failures are partial or missing evidence (failure analysis:
  [Failure_Analysis.xlsx](../results/original_baselines/Failure_Analysis.xlsx)).
- **OBLIQ is hard for every configuration.** NDCG@10 stays at 0.09–0.19, near or below the OBLIQ paper's multi-hop
  agent.

## Limitations

- One run per cell, sampled without seeds. An earlier repeat of a whole run disagreed on 17.8% of questions.
- Paired rows differ in several settings at once (see [KNOWN_ISSUES.md](verification/KNOWN_ISSUES.md) K2).
- Some effort labels do not match what reached the model (K3).
- Two Full-Doc rows ran on an unpreserved library build (K5).
- DSPy.RLM was not evaluated under the final protocol.
- Two Qwen3.6 Snippet rows were still running or queued when this repository was prepared.
