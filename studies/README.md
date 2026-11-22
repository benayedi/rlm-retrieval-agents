# Studies

A **study** changes something *around* an agent and leaves the agent itself unchanged: what each search returns (the
search-policy proxy), or the text of its prompt (GEPA). A run that only changes one setting of an agent (a search cap,
a model, the library) is not a study: it stays under that agent (`agents/<agent>/configs/<ID>.env`, launchers in
`agents/<agent>/experiments/`).

Each study folder follows the agents' layout and the same rules ([REFACTOR_RULES.md](../docs/architecture/REFACTOR_RULES.md)):

| Folder | What |
|---|---|
| `original/` | byte-exact copies of the code that ran, with `MANIFEST.json` (source path, SHA-256, size); never edited. `scripts/provenance/preserve_originals.py --check` verifies every manifest under `studies/*/original/` and `agents/*/original/` |
| `src/`, `analysis/` | readable copies with identical behaviour; every change is listed in `CHANGES_FROM_ORIGINAL.md` |
| `experiments/` | launchers. Where a run had no launcher file, the launcher is reconstructed from run logs, record configs and session transcripts, and says so |
| `tests/` | offline tests (mocks only) |
| `CHANGES_FROM_ORIGINAL.md` | renames, per-file diff summary, issues found and not fixed |

Results: the per-question records are in [`trajectories/<study>/<ID>/`](../trajectories/README.md) (Git LFS; BrowseComp-Plus
archives encrypted), the reported verdicts in [`results/follow_up/verdicts/`](../results/follow_up/verdicts/), every
number with its source in [`results/follow_up/numbers.yaml`](../results/follow_up/numbers.yaml) and
[`studies.md`](../results/follow_up/studies.md), the facts of every run in
[`experiments/registry.yaml`](../experiments/registry.yaml) (section `follow_up_studies`). The overview is
[docs/experiments/FOLLOW_UP_STUDIES.md](../docs/experiments/FOLLOW_UP_STUDIES.md).

## Index

| ID | Study / agent | What | Questions | Control | Judge of the reported numbers |
|---|---|---|---|---|---|
| [SP-1](../docs/experiments/search_policy/SP-1.md) | search_policy | BrowseComp-Plus, Qwen3.6, ColBERT, novelty + reranker | 830 | BCP-29 | gpt-4.1 |
| [SP-2](../docs/experiments/search_policy/SP-2.md) | search_policy | reranker only | 197 (test200) | BCP-29 (test200) | gpt-4.1 |
| [SP-3](../docs/experiments/search_policy/SP-3.md) | search_policy | GPT-5, novelty + reranker, pilot | 100 | BCP-23 (stored answers) | gpt-4.1 |
| [SP-4](../docs/experiments/search_policy/SP-4.md) | search_policy | FRAMES, Qwen3.6, search cap 25 + novelty + reranker | 824 | FRC-1 | gpt-4.1 |
| [SP-6](../docs/experiments/search_policy/SP-6.md) | search_policy | OBLIQ-Twitter, snippet REPL agent, Qwen3.6, novelty + reranker | 281 | TWQ-1 | none (NDCG@10) |
| [SP-7](../docs/experiments/search_policy/SP-7.md) | search_policy | OBLIQ-Twitter, document REPL agent, Qwen3.6, novelty + reranker | 281 | TWQ-2 | none (NDCG@10) |
| [GEPA-1](../docs/experiments/prompt_optimisation/GEPA-1.md) | prompt_optimisation | BrowseComp-Plus GEPA run 1 (proportional selection set) | 100 (D_pareto) | the seed prompt | Qwen3.6 (optimisation-time) |
| [GEPA-2](../docs/experiments/prompt_optimisation/GEPA-2.md) | prompt_optimisation | BrowseComp-Plus GEPA run 2 + test200 confirmation (winner d911ebbe23) | 199 | BCP-29 (test200) | gpt-4.1 |
| [GEPA-3](../docs/experiments/prompt_optimisation/GEPA-3.md) | prompt_optimisation | FRAMES GEPA + confirmation on unused questions (8ba345b481) | 323 (see K31) | FR-13 | gpt-4.1 |
| [GEPA-4](../docs/experiments/prompt_optimisation/GEPA-4.md) | prompt_optimisation | OBLIQ-Twitter GEPA, four runs, Qwen3.8-27B | 100 (D_pareto) | the seed prompt | none (NDCG@10) |
| [GEPA-5](../docs/experiments/prompt_optimisation/GEPA-5.md) | prompt_optimisation | HoVer sanity check of the GEPA pipeline | 300 (test) | the default instructions | none (exact metric) |
| [VAR-2](../docs/experiments/run_variation/VAR-2.md) | run_variation | re-run variation of the GPT-5 document REPL agent (eight pairs) | 50-350 per pair | — | gpt-4.1 (recorded) |
| [ERR-1](../docs/experiments/error_analysis/ERR-1.md) | error_analysis | FRAMES failure taxonomy (search cap 10 and 25) | 824 | — | **Qwen3.6 only** |
| [FB-1](../docs/experiments/frames_baselines/FB-1.md) | frames_baselines | FRAMES single-shot baselines (closed book, BM25 top 4, oracle) | 824 | — | **Qwen3.6 only** |
| [FRC-1](../docs/experiments/document_repl/FRC-1.md) | agents/document_repl | FRAMES, search cap 25 / 50 turns | 824 | FR-13 | gpt-4.1 |
| [TWQ-1](../docs/experiments/snippet_repl/TWQ-1.md) | agents/snippet_repl | OBLIQ-Twitter, Qwen3.6, plain dense search (control of SP-6) | 281 | — | none (NDCG@10) |
| [TWQ-2](../docs/experiments/document_repl/TWQ-2.md) | agents/document_repl | OBLIQ-Twitter, Qwen3.6, plain dense search (control of SP-7) | 281 | — | none (NDCG@10) |
| [DSA-3](../docs/experiments/dspy_rlm/DSA-3.md) | agents/dspy_rlm | DSPy.RLM ablation arm 3: caps lifted, prompt matched | 100 | BCP-9 | gpt-4.1 |
| [DSA-5](../docs/experiments/dspy_rlm/DSA-5.md) | agents/dspy_rlm | DSPy.RLM ablation arm 5: models and search output matched too | 100 | BCP-9 | gpt-4.1 |

Controls named here (BCP-9, BCP-23, BCP-29, FR-13) are rows of the results table; their records are already in
`trajectories/`. The seed arm of GEPA-3 has no run of its own: its answers are FR-13's.

## Not migrated

By the owner's decision of 2026-11-11 (only what the thesis uses): SP-5 and GEPA-3t (FRAMES test200 arms with Qwen3.6
verdicts only); VAR-1 (the 112-question replay and the 3 × 50 identical runs of the DSPy.RLM fusion campaign; see
KNOWN_ISSUES K36), the Qwen3.8-27B discordance test (its second part crashed), fusion; the abandoned 10-question GPT-5
control of SP-3; DSPy.RLM ablation arm 4 (stopped after 7 questions); `gepa_obliq_dspy.py` (never run) and the
hand-written H1 prompt test on OBLIQ-Twitter; the GPT-5.2 OBLIQ-Twitter reranker run (no matched control); the matched
DSPy.RLM vs snippet REPL comparison on OBLIQ Math; GEPA's optimisation-time rollouts. Earlier exclusions: the OBLIQ
multi-hop re-implementation, the OBLIQ retriever comparisons, the FRAMES multi-step pipeline re-implementation.

## Names

Run names, `original/` files and the original repository's documents keep the research-time names: "full-doc RLM" /
"supervisor harness" = the **document REPL agent**, "snippet RLM" / "official RLM" / `run_obliq_mine.py` = the
**snippet REPL agent**, "10 searches" / `MAX_SEARCHES` / `s25` = the **search cap**, `rr` / "reranker proxy" = the
search policy. `scripts/docs/naming.py` maps them.
