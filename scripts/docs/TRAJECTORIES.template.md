# Trajectories and per-question records

This folder holds the per-question run outputs of every Experiments.xlsx row that are still available: final records,
full trajectories (every turn's prompt, response, code and REPL output), grades and run logs, exactly as the runs
wrote them, and of the follow-up studies ([FOLLOW_UP_STUDIES.md](../docs/experiments/FOLLOW_UP_STUDIES.md)). There is
one folder per row or study ID: `<agent>/<ID>/` or `<study>/<ID>/`. For GEPA, only held-out evaluations and run metadata
are archived, not the optimisation-time rollouts. Each folder holds:

- `<ID>.tar.zst` (or `.tar.zst.enc`, possibly split into `.partNN`): the archive, stored with **Git LFS**;
- `MANIFEST.json`: every file in the archive with its original path, size and SHA-256, plus the archive's own hashes.

The archives are byte-exact. `scripts/provenance/pack_trajectories.py --verify` decompressed every archive and checked
every file against its manifest. Nothing was rewritten or redacted.

## Getting and opening them

```bash
git lfs install
git lfs pull --include "trajectories/snippet_repl/FR-7/*"                     # one row (or omit --include for all)
python scripts/provenance/unpack_trajectories.py FR-7 /tmp/fr7                 # extract and verify
python scripts/provenance/unpack_trajectories.py BCP-13 --list                 # list without extracting
```

A plain `git clone` downloads only small LFS pointer files for the archives. Pull only the rows you need: every
download counts against the repository owner's LFS bandwidth.

## BrowseComp-Plus archives are encrypted

BrowseComp-Plus distributes its questions and answers encrypted, with a canary asking that they never appear online
in plain text. Its trajectories contain both in plain text, so the `BCP-*` archives and the BrowseComp-Plus follow-up archives (SP-1, SP-2, SP-3, DSA-3, DSA-5,
GEPA-1, GEPA-2, VAR-2) are encrypted:
- AES-256-CTR via `openssl enc`, PBKDF2 with 200,000 iterations and a random salt;
- the passphrase is the benchmark's own public canary string.

This is the same protection the dataset uses: it keeps the text out of plain view online, and anyone who has the
dataset can undo it. `unpack_trajectories.py` decrypts automatically. **Keep the extracted BrowseComp-Plus files off
the web.** FRAMES and OBLIQ archives are compressed only.

## What is in each row

{{TABLE}}

Not included:
- **BCP-31** (Document REPL agent, Qwen3.6, BM25): left out for now at the owner's request. Its trajectories are 206 GB
  raw (≈ 1.5 GB compressed), because each turn's record repeats a snapshot of every REPL variable, including the
  full text of every retrieved document.
- **BCP-21, BCP-23**: no run records or trajectories exist on the machine that holds this data. Only the per-question
  summary trace in the original results page survives (see their experiment documents).
- **BCP-9, BCP-11**: the trajectories (`logs/rlm_traj32`, `logs/rlm_traj24`) were deleted from the original
  machine some time between 2026-09-22 and 2026-10-29, before this repository was prepared. The archives hold every
  question's final record, the gpt-4.1 verdicts, and the per-question turn and token counts computed from the
  trajectories while they existed (`repo/scripts/workbook/tokmaps/`).
- **BCP-15, BCP-17**: still running / queued on 2026-10-30; to be added when they finish.

## File formats

**Snippet REPL agent**: in `logs/run5d/full/<run>/` (run5d rows) or `logs/rlmfix*_q*.json` (September rows):
- `q<i>.json` (FRAMES: `q<iii>.json`): the question's record. Fields: `predicted` (final answer); `correct`, the
  harness's interim judge (the reported numbers use `graded_gpt41.json`); `gold_recall`, `evid_recall`,
  `search_calls`, `get_document_calls`, `llm_query_calls`, `unique_docs`, `latency_s`, `error`. The `iterations`
  and `input_tokens` fields are unreliable (docs/verification/KNOWN_ISSUES.md K11).
- `traj/rlm_<timestamp>_<id>.jsonl`: the trajectory. Line 1 is `metadata` (model, backend, turn limit). Each further
  line is one turn: `prompt` (the full message list sent), `response`, `code_blocks` (code plus `result` with
  `stdout`, `stderr` and a snapshot of the REPL variables), and `final_answer`. A question's trajectory is matched by
  its first user message. When a question was retried, the latest file counts.
- `graded_gpt41.json`: the gpt-4.1 verdict per question, plus recall and searches. `_judge_cache_gpt41.json`: cached
  votes. `usage.jsonl`: one line per Azure call (tokens). `q*.log`: stdout of each question. `<run>.out`: the
  launcher log.
- OBLIQ: `all.json` holds one record per question. Every question appears twice: the shard merge also picked up the
  checkpoint files (K19). Fields as above plus `query_id`, `ndcg@k`, `recall@k`, `n_parsed`.

**Document REPL agent**: one JSON record per question (`<qid>.json`, or `rollouts/<cache key>/<qid>.json` for the Qwen3.6
rows):
- **Run metadata:** `query_id`, `design`, `config` (every flag of the run), `status` (`completed` / `error` /
  `empty_output`), `result[0].output` (the final answer), `usage`, `n_iterations` (includes the forced final-answer
  call: 31 at the cap).
- **Trajectory:** `trajectory.iterations[]`, with the same structure as above. Most of the size is the REPL variable
  snapshot after each block, which includes `DOCS`, the full retrieved documents.
- **Retrieval:** `tool_call_counts`, `server_counts`, `retrieved_docids`, `server_docids`, `search_log`,
  `search_refused`.
- **Run checks:** `sandbox_flags`, `wall_time_s`.

`cache.json` (Qwen3.6 rows) holds a per-question summary keyed by `<cache key>|<qid>`, where the cache key
identifies the prompt and settings. `graded_gpt41.json` and `logs/bcp_rejudge_gpt41/<n>.json` hold the gpt-4.1
verdicts used in Experiments.xlsx.

Paths inside the archives keep their original layout:
- `logs/...` is relative to the original repository's `logs/`;
- `fulldoc/...` is relative to the document REPL agent directory (OBLIQ document REPL runs);
- `repo/...` is relative to the original repository root.

The records name the Azure endpoint and deployments that served the runs. They contain no API keys: every candidate
file was checked against the actual key values before packing.
