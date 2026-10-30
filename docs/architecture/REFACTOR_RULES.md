# Refactor rules (binding for every file in `algorithms/*/src/`)

The code in `algorithms/<algorithm>/original/` is the exact code that produced the results in
`results/original_baselines/Experiments.xlsx`. It is never edited (`scripts/provenance/preserve_originals.py --check`
verifies it). `src/` is a readable, runnable copy of it. The research logic in `src/` must stay identical.

## Allowed changes (and only these)

1. **File and module names.** Rename to intuitive names. Every rename is listed in the algorithm's
   `CHANGES_FROM_ORIGINAL.md`, with original → new.
2. **Import wiring.** Replace `sys.path.insert(<absolute path>)`, `examples.*` imports and `spec_from_file_location`
   paths with imports that work from the new layout:
   - Shared code is in `shared/rlm_shared/`: `paths.py`, `browsecomp_data.py`, `obliq_data.py`,
     `obliq_score_advice.py`.
   - The `rlm` library comes from `RLMLIB_PATH` on `PYTHONPATH`.
   - Compute repo-relative locations from `__file__`; never hard-code an absolute path.
3. **Machine-specific values.**
   - Absolute paths, credential-file paths and Azure deployment names become environment variables from
     `configs/paths.example.env`: `rlm_shared.paths.path("NAME")` in Python, `${NAME:?}` in shell.
   - If a default existed only to point at the original machine, remove it. Python reads the variable when it is used;
     shell fails with a clear message.
   - Keep the original value in a comment only when it is not personal (no person's name, no `/tmp/claude-*`).
4. **Documentation.**
   - A module docstring at the top of every file: what it does, which original file it comes from, and how it is run.
   - Short comments that explain non-obvious research logic.
   - Do not reword prompt strings, tool docstrings or any text the model sees: that is research logic.
5. **Launchers.** Put shell wrappers in `algorithms/<algorithm>/experiments/`. They source `configs/paths.env`, set
   `PYTHONPATH` to the new layout, and run `src/`. Their arguments and environment settings must be identical to the
   original wrapper.

## Forbidden

- Any change to behaviour: prompts, tool specs, defaults, caps, budgets, thresholds, the order of operations, retry
  logic, scoring, file formats written, or env-var names that select behaviour (`ORCHESTRATOR`, `QWEN_*`, `SUP_*`,
  `OBLIQ_*`, …).
- Fixing bugs. Report them in `CHANGES_FROM_ORIGINAL.md` under "Issues found, not fixed", with file:line and a
  description, even if the fix is obvious.
- Deleting code, including dead code. Mark it with a comment instead.
- Running experiments, model servers, network calls, GPU jobs, or anything that calls an LLM API.

## Required checks before finishing

- `python -m py_compile` on every `.py` in `src/`; `bash -n` on every `.sh`.
- `diff -u original/<file> src/<file>` for every file. Every hunk must fall under rules 1–5. List any hunk that does
  not in `CHANGES_FROM_ORIGINAL.md`.
- `pytest algorithms/<algorithm>/tests -q` passes, using mocks only.
