# Contributing

## Understand the project first

1. Read [docs/research_overview.md](docs/research_overview.md) and [docs/architecture/OVERVIEW.md](docs/architecture/OVERVIEW.md).
2. Read the README of the algorithm you will touch (`algorithms/<algorithm>/README.md`) and its
   `CHANGES_FROM_ORIGINAL.md`.
3. Check [docs/verification/KNOWN_ISSUES.md](docs/verification/KNOWN_ISSUES.md) before relying on a result or a code path.

## Set up

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp configs/paths.example.env configs/paths.env    # fill in; never commit it (git-ignored)
python -m pytest                                  # must pass before and after your change
```

Set `RLMLIB_PATH` (or `pip install` the `rlms` line of `requirements.txt`) to also run the tests that need the RLM library.

## Ground rules

- **`original/` is frozen.** It is the code that produced the published numbers. `python
  scripts/provenance/preserve_originals.py --check` must keep passing.
- **Behaviour changes are explicit.** A change to a prompt, tool spec, budget, default or scoring rule must:
  - be its own commit;
  - update the tests that pin the old behaviour;
  - be recorded in `CHANGES_FROM_ORIGINAL.md`;
  - invalidate the affected rows until they are re-run.
  Never mix it with refactoring.
- **No machine paths or secrets in code.** Add a variable to `shared/rlm_shared/paths.py` and
  `configs/paths.example.env` instead.
- **No BrowseComp-Plus plain text.** Questions, answers and model outputs on BrowseComp-Plus must not be committed. Run
  `python scripts/provenance/bcp_leak_scan.py <paths>` before committing anything derived from runs.
- **Numbers come from evidence files, never from typing.** Regenerate documents with `scripts/docs/*.py`.

## Run an algorithm

Every results row has `algorithms/<algorithm>/configs/<ID>.env`. Launch it with the algorithm's launcher (the experiment
document of the row gives the exact command). Outputs go to `$RUNS_DIR`. Grade with
`shared/rlm_shared/grading/`.

## Add an experiment

1. **Write the config.** Create `algorithms/<algorithm>/configs/<NEW-ID>.env` with every environment variable and
   argument the run uses. Use a header comment like the existing ones.
2. **Run it.** Run it, grade it, and add the row to `results/original_baselines/Experiments.xlsx`. Keep the column
   conventions; state the recall definition.
3. **Register it.** Add an entry to `experiments/registry.yaml`.
4. **Audit and regenerate.** Extend `scripts/audit/recompute_experiments.py` with a reader for the new outputs, run
   it, then run `python scripts/docs/make_experiment_docs.py` and `python scripts/docs/make_comparison.py`.

## Add a benchmark

1. **Loader.** Add a loader to `shared/rlm_shared/` (questions, gold, scoring) with tests in `tests/`.
2. **Retrieval.** Build an index and serve it through the `/search` + `/document` wire format
   (`benchmarks/retrieval/`), so every algorithm can use it unchanged.
3. **Adapter.** Add a thin runner per algorithm that adapts task text and answer parsing; see `run_frames.py` /
   `run_obliq.py` in the Snippet RLM.
4. **Document it.** Record the benchmark's source and licence in `benchmarks/README.md`.

## Tests

Tests must not call models, judges or servers. Use fakes: see `tests/test_shared_grading.py` (fake OpenAI client) and
`algorithms/full_doc_rlm/tests/test_toolbox.py` (fake retrieval server). A test that needs a model or an index must
skip itself with a reason.

## Pull requests

State in the description:
- what changed;
- whether behaviour changed, and which rows are affected;
- the test result;
- that the leak scan and the originals check pass.
