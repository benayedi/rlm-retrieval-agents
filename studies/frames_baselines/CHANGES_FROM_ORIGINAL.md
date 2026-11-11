# FRAMES single-shot baselines (FB-1): changes from the original code

`original/` holds the exact files (`original/MANIFEST.json`); `src/` is the same code made runnable from this
repository under [REFACTOR_RULES.md](../../docs/architecture/REFACTOR_RULES.md). No logic was changed.

**Judge.** The official BrowseComp grader prompt run on the local Qwen3.6 with thinking off: every verdict of FB-1 is
**Qwen3.6 judge only**. FB-1 is migrated because the FRAMES failure taxonomy (ERR-1) reads its oracle, closed-book
and BM25 top-4 results; the BM25 top-2 condition is not used by the thesis and its records are not migrated.

| Original | New |
|---|---|
| `original/RLM/scripts/frames_eval.py` | `src/single_shot.py` |
| `original/RLM/logs/frames/run_rest.sh` (the second invocation: oracle on 100, then oracle, bm25_4, bm25_2) | not copied: provenance of how the conditions were run (`--conds oracle --limit 100 --workers 12`, then `--conds oracle,bm25_4,bm25_2 --workers 16`); the closed-book invocation is not preserved |
| (new) | `tests/test_single_shot.py` |

## Configuration
`FRAMES_DATA_DIR` (`test.tsv`, `corpus.jsonl`, `corpus_meta.npz`, `bm25_index/`), `LOGS_ROOT` (outputs
`frames/llm_<cond>.jsonl`, `llm_summary.json`, `coverage_recall.json`, `bm25_top100.npy`, as in the original),
`ORIGINAL_REPO_ROOT` (its `.env` is loaded, as in the original). The reader and the judge are the local vLLM server on
`:8030` (unchanged; `benchmarks/models/serve_qwen36.sh`).

## Per-file diff summary
Module docstring (rule 4); `ROOT/DATA/LOGS` replaced by `FRAMES_DATA_DIR` and `LOGS_ROOT/frames` (rule 3); in
`run_llm` the `sys.path.insert(f"{ROOT}/examples")` + `from browsecomp_data import ...` become
`from rlm_shared.browsecomp_data import ...` (rule 2; the module is the refactored copy of the same file, with the
grader prompt unchanged) and `load_dotenv(f"{ROOT}/.env")` reads `ORIGINAL_REPO_ROOT/.env` (rule 3).

## Issues found, not fixed
- **FB-I1. Concurrent reads share one file handle.** `src/single_shot.py:79,86` (original 62, 69): `Corpus` opens
  `corpus.jsonl` once and `doc()` does `seek` + `readline` on that handle from up to `--workers` threads (the
  original runs used 12 and 16) without a lock. A read can start at another thread's offset: it then raises a JSON
  error (none is in the original `logs/frames/eval.log`) or, if it lands on a line start, returns a different article
  silently, which would put a wrong article into a bm25_2 / bm25_4 / oracle prompt. The records keep the doc ids but
  not the prompt text, so whether it happened cannot be checked. Pinned by `tests/test_single_shot.py`.
- **FB-I2. The judge sees only the extracted answer line.** `src/single_shot.py` `one()` passes
  `f"Exact Answer: {ans}"` to the grader, where `ans` is the last non-placeholder "Exact Answer:" line after `</think>`
  or the last 300 characters; the agents' judges see the whole response. Differences of judging context between FB-1
  and the agent rows.
- **FB-I3. Errors count wrong.** A condition whose three completion attempts all fail is recorded with
  `correct: False` and its error, without judging (consistent with "missing = wrong").

## How verified
`py_compile`; `tests/test_single_shot.py` (synthetic tree, fake OpenAI client, one worker: the readable copy and the
original write identical records, summary and coverage file; FB-I1 pinned); `preserve_originals.py --check`.
