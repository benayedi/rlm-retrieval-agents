# Testing report

Date: 2026-10-30. Framework: **pytest 9.2** in a virtual environment that inherits the experiment environment's packages
(Python 3.12, dspy 3.3.1, openpyxl, datasets) and adds only pytest.

The suite is lightweight by design:
- No test calls an LLM, a judge, an embedding model or a retrieval server.
- No test downloads data or uses a GPU.
- External services are replaced by mocks or fakes: the OpenAI/Azure clients, `requests`/`urllib`, `datasets.load_dataset`.

Tests that would need a model or an index are skipped and give the reason.

## How to run

```bash
python -m pytest                                    # all tests (testpaths in pyproject.toml)
python -m pytest algorithms/snippet_rlm/tests       # one algorithm
RLMLIB_PATH=/path/to/rlm PYTHONPATH=/path/to/rlm python -m pytest   # also run the tests that need the rlm library
```

`rlm` is the RLM authors' library (github.com/alexzhang13/rlm, commit 854e688). It is not pip-installable at that commit.
Without it on the path, the tests that import it skip themselves.

## Results

| Run | Tests | Passed | Failed | Skipped |
|---|---|---|---|---|
| with `rlm` importable | 446 | **442** | **0** | 4 |
| without `rlm` | 446 | 417 | 0 | 29 (25 need `rlm`) |

The raw JUnit XML is in `data/pytest_with_rlm.xml` and `data/pytest_without_rlm.xml`.

| Suite | File | What it checks | Passed (with rlm) |
|---|---|---|---|
| shared | `tests/test_metrics.py` | Reference metrics against hand-computed NDCG / recall / majority / accuracy. The experiments' own `obliq_data.ndcg_at` / `recall_at` against the reference on 30 cases. Doc-id extraction per task, and the duplicate-id edge case | 37 |
| shared | `tests/test_shared_grading.py` | Judge-reply parsing, 3-vote majority and the retry flow with a fake client, the content-filter path (pins the known K16 behaviour), FRAMES verdict parsing, recall denominators | 29 |
| shared | `tests/test_shared_retrieval.py` | Search-server request handling (2,000-char text cut, top-k clamping, id mapping) through fake indexes; `py_compile` / `bash -n` of every server and builder | 92 (+4 skipped) |
| Snippet RLM | `test_snippet_rlm_equivalence.py` | Original vs refactored code in **separate interpreters** with mocked models, on 9 scenarios. Checked equal: system and user prompts, tool specs, task text, message sequence, format gate, code-block parsing, `run_one` records and `RLM` constructor arguments. A mutation check during the refactor (one word changed in a tool description) made 8 tests fail | 10 |
| Snippet RLM | `test_snippet_rlm_behaviour.py` | Recall computation on a synthetic record, the format gate, final-answer extraction, the prose-final path | 9 |
| Snippet RLM | `test_snippet_rlm_static.py` | Every config parses and names an existing launcher; `bash -n`; `py_compile`; no absolute machine paths | 59 |
| Full-Doc RLM | `test_prompts_identical.py` | System prompt, prologue, query template and BM25 rewrites identical to the original | 6 |
| Full-Doc RLM | `test_cache_keys.py` | The three result-cache keys (`e925f0f6c7`, `ee5ea32a1c`, `11d0c5bc6b`) recompute to the values the original runs recorded | 3 |
| Full-Doc RLM | `test_toolbox.py` | Search cap (10) and refusal counter, model-chosen k (≤ 50) vs fixed k, `DOCS` merge and dedup, against a fake retrieval server | 14 |
| Full-Doc RLM | `test_run_one_record.py` | Record building; `n_iterations` = trajectory length including the forced-answer call | 6 |
| Full-Doc RLM | `test_documented_issues.py` | Pins the behaviour behind K9 and K12, so a future fix is a visible change | 3 |
| Full-Doc RLM | `test_static.py` | Configs, launchers, `py_compile`, `bash -n`, no machine paths | 57 |
| shared | `tests/test_trajectories.py` | Every trajectory manifest is consistent (unique paths, sizes, encrypted ⇔ BrowseComp-Plus, parts under the LFS 2 GB limit); two archives (BCP-25 encrypted, FR-11 plain) round-trip through `unpack_trajectories.py` with every file's SHA-256 matching. Skips when LFS content is not pulled | 19 |
| DSPy.RLM | `test_dspy_rlm_signatures.py` | Every signature class constructs; signature set, prompt constants and `make_rlm_signature` / budget-signature output equal the original `rag` package's | 49 |
| DSPy.RLM | `test_dspy_rlm_agent.py` | `build_rlm_agent` wiring (max_iters, tools, signature, sub-LM incl. the Azure requirement), real `dspy.RLM` construction, tool docstrings; `search_index` truncation, caching, per-turn cap, hard `max_search_calls` stub, top-k clamp; search sequence equals the original's; `run_question` with a fake RLM | 18 |
| DSPy.RLM | `test_dspy_rlm_tools.py` | `RemoteRetriever` request payloads (index, top-k, timeout) with `requests` mocked, equal to the original's | 9 |
| DSPy.RLM | `test_dspy_rlm_drivers.py` | Driver signature maps (pins the v13–v20 worker-map gap, D1), shared grading / score-advice imports, OBLIQ root from env, OBLIQ metrics and id parser, pins `heuristic_match` on empty predictions (K17) | 8 |
| DSPy.RLM | `test_dspy_rlm_static.py` | `py_compile`, `bash -n`; package code AST-identical to the original `rag` package (docstrings excluded); drivers differ only in import wiring and the OBLIQ root | 14 |

### Skipped tests (with rlm importable)

| Test | Reason |
|---|---|
| 3 server-import tests in `test_shared_retrieval.py` | `search_server.py` and `multi_search_server.py` load Qwen3-Embedding-8B and FAISS indexes at import, and `gemini/search_server.py` builds a Vertex AI client. Their text-cut logic is checked statically instead |
| 1 BM25 server protocol test | `--protocol sup` loads the Qwen3-0.6B tokenizer from the Full-Doc data directory at import |

## Static checks

| Check | Result |
|---|---|
| `python -m compileall` over `algorithms/*/src`, `shared/`, `scripts/`, `benchmarks/`, `tests/` | passes |
| `bash -n` over all 30 shell launchers and helpers | passes |
| `scripts/provenance/preserve_originals.py --check` (originals unchanged; SHA-256) | all files match |
| `scripts/provenance/bcp_leak_scan.py` over the repository (no BrowseComp-Plus plain text) | clean; see [FINAL_RESEARCH_AUDIT.md](../FINAL_RESEARCH_AUDIT.md) |
| Linting (ruff / pyflakes) | **not run**: no linter is installed in the experiment environment, and none was installed for this task |
| Type checking (mypy) | **not run**, for the same reason; the code has few type hints |
| Coverage | **not measured** (coverage.py not installed) |

## Deliberately not executed

- **Any experiment, or any part of one.** That covers the agent loops against a real model, the judges against Azure,
  the retrieval servers with real indexes and the index builders. This was a task restriction, and the experiments are
  expensive: hundreds of model calls and hours per row.
- **End-to-end equivalence with the original results.** Equivalence is established at source level (AST equality,
  identical prompts, tool specs, cache keys and records under mocks). It is **not** established at the level of
  benchmark results: that would need re-running the experiments, and even then sampling without seeds would give
  different per-question outcomes.

## Limitations

- The mocks check that the code sends and parses what the original does; they cannot check model behaviour.
- The tests that pin known issues (K9, K12, K16) assert the current, imperfect behaviour on purpose. When an issue is
  fixed, its test must change with it.
- Without `rlm` importable, 25 tests skip, including most of the Snippet RLM equivalence checks. Run the suite with
  `RLMLIB_PATH` set to get the full verification.
