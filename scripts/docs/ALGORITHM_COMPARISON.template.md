# Algorithm comparison

How the three algorithms compare on the benchmarks where comparable results exist. `scripts/docs/make_comparison.py`
reads every number in the tables and figures from
[Experiments.xlsx](../../results/original_baselines/Experiments.xlsx) or
[Failure_Analysis.xlsx](../../results/original_baselines/Failure_Analysis.xlsx). To change a number, regenerate the
file; do not edit numbers here. The workbook numbers were recomputed independently from the per-question records:
see [RESULT_INTEGRITY.md](../verification/RESULT_INTEGRITY.md).

## What is compared, and what is not

**Snippet RLM vs Full-Doc RLM** is the comparison the final table was built for. Each pair shares the benchmark, the
model and the retriever. Scoring is also shared:
- BrowseComp-Plus and FRAMES: the same judge (gpt-4.1, 3 votes);
- OBLIQ: NDCG@10 on gold qrels.

**DSPy.RLM is not in this comparison.** It was evaluated in an earlier campaign (see
[its README](../../algorithms/dspy_rlm/README.md)) under a different protocol:
- a different judge (gpt-5.4-mini);
- a different retrieval setup (FAISS IVF index, top-10);
- different budgets (16 iterations, 35 searches);
- no FRAMES run.

Its rows were removed from the final table on 2026-10-28 and never re-run under the final protocol. Placing its
numbers next to these would compare protocols, not algorithms, so this document does not rank it.

**The two harnesses differ in more than their reading strategy.** Each pair of rows changes the harness, and with it
every setting below. A difference between two rows is the effect of the whole package, not of snippet-vs-full-document
reading alone.

| Setting | Snippet RLM | Full-Doc RLM |
|---|---|---|
| What a search returns | 5 results by default (the model may ask for more; BCP-19 did in 9 questions), each the first 512 tokens of the text the server returns, which servers cut to 2,000 chars | whole documents into `DOCS`, with k chosen by the model (≤ 50; mean 24.5 on BrowseComp-Plus BM25, 13.2 on FRAMES); fixed k = 20 on OBLIQ |
| Search cap | none (GPT-5 BrowseComp-Plus, OBLIQ); 25 (FRAMES, all Qwen3.6 rows) | 10, enforced in the client. The patched build behind BCP-23 logged up to 19 |
| Turn limit | 100 (GPT-5 BrowseComp-Plus); 50 (FRAMES, Qwen3.6 BrowseComp-Plus); 12 (OBLIQ) | 30, then a forced answer |
| Sub-LLM | `llm_query` is used for reading | the functions stay installed and only the prompt says not to use them (no call found in the OBLIQ trajectories) |
| Reasoning effort actually sent | BCP-9, BCP-11: deployment default (≈ medium), see [K3](../verification/KNOWN_ISSUES.md); BCP-13, FRAMES, OBLIQ: as labelled | BCP-21, BCP-23: unverified (unpreserved patched build); BCP-25: default (≈ medium); FRAMES, OBLIQ: as labelled |
| ColBERT index | 512-token documents | 8,192-token documents |
| Per-question time limit (Qwen3.6) | 1 h (BrowseComp-Plus), 2 h (FRAMES) | runaway kill only |

\* The "Recall" column uses five definitions across rows: gold vs evidence documents, and different denominators.
It is comparable only within one definition; see [RESULT_INTEGRITY.md §R1](../verification/RESULT_INTEGRITY.md).

## BrowseComp-Plus (830 questions)

{{FIG_BCP}}

{{TABLE_BCP}}

## FRAMES (824 questions)

{{FIG_FRAMES}}

{{TABLE_FRAMES}}

## OBLIQ-Bench (Math 151, Twitter 281, Congress 254 queries)

{{FIG_OBLIQ}}

{{TABLE_OBLIQ}}

NDCG is shown as G / P: against the gold qrels / against the pooled qrels. Congress has gold qrels only. WildChat and
Writing were not run, because no index was built for them.

## Paired tests on the same questions

Exact McNemar test on the questions that one row answers correctly and the other does not, from the Pairwise sheet of
Failure_Analysis.xlsx.

OBLIQ pairs are not tested here: NDCG is continuous, and the paired t statistics quoted in the original notes were not
recomputed for this repository.

{{TABLE_PAIRS}}

## What the evidence establishes

Each finding below is measured, and cites the pair that shows it.

1. **The retriever matters more than the harness.**
   - Within each harness, Reason-ModernColBERT beats BM25 by a wide margin: GPT-5 Full-Doc 255 vs 27 questions
     (p ≈ 1e-47); GPT-5 Snippet 196 vs 44 (p ≈ 4e-24).
   - Under the Snippet RLM, ColBERT also beats the 8B dense retriever: 100 vs 47 (p ≈ 1.5e-5).
2. **The model matters more than the harness.** GPT-5 beats Qwen3.6-35B-A3B in all six tested pairs, across both
   benchmarks and both harnesses (p < 1e-15 in each).
3. **On BrowseComp-Plus with GPT-5, the better harness depends on the retriever.**
   - Dense retriever: Full-Doc is ahead (88 vs 57, p = 0.012).
   - ColBERT: the two cannot be distinguished (65 vs 51, p = 0.23).
   - BM25: Snippet is ahead (136 vs 74, p ≈ 2e-5).
4. **On FRAMES with GPT-5, Full-Doc is ahead** (74 vs 38, p ≈ 0.001). With Qwen3.6 the two cannot be distinguished
   (81 vs 89, p = 0.59).
5. **On OBLIQ the differences are small.**
   - Math and Twitter: NDCG@10 within 0.01 of each other, in opposite directions.
   - Congress: the Snippet RLM is higher (0.188 vs 0.133).
   - Against the OBLIQ paper's multi-hop agent: all rows are below it on Math (0.161) and around it on Twitter (0.141)
     and Congress (0.183).

## What it does not establish

- **Why a harness wins.** Each harness win in finding 3 comes with a budget difference:
  - With BM25, the Snippet RLM made 60.9 searches per question against Full-Doc's cap of 10.
  - With the dense retriever, the reasoning efforts differ in a way that is only partly known (BCP-9 ran at the
    default effort; BCP-21's effort is unverified).

  Isolating the reading strategy needs matched budgets: a Full-Doc run without the search cap, or a Snippet run with it.
- **An overall ranking of the three algorithms.** DSPy.RLM was not run under this protocol, and the Snippet vs
  Full-Doc result changes sign with the retriever.
- **Run-to-run stability.**
  - Each cell is a single run, and the agents sample without seeds.
  - In the earlier campaign, two identical 830-question Snippet RLM runs disagreed on 17.8% of questions. That is a
    ±2.9-point no-change band on accuracy.
  - Between unpaired rows, differences smaller than that should not be read as effects.
- **The Qwen3.6 Snippet rows on BrowseComp-Plus with dense and ColBERT retrieval.** On 2026-10-30 they were still
  running or queued.
- **Like-for-like costs and times.** Times depend on how many runs shared a server, and the Qwen3.6 rows ran on local
  GPUs with no API cost.
