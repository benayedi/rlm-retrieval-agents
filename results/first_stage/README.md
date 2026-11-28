# First-stage retrieval: where the search description's recall figures come from

The document REPL agent's search description (`agents/document_repl/src/tools.py`, carried into the paper's
replication package) tells the model: "the top-5 hold about 15% of the relevant documents (at least one of them 48%
of the time), the top-30 about 33% (74%), the top-50 about 39% (80%)". The patch that added them
(`agents/document_repl/original/FULLDOC/src/apply_design_g_patch.py`) names their source as `runs/first_stage_eval.json`
of the original research repository; that file was not preserved. This folder re-measures them.

Run on 2026-11-28 with [`agents/document_repl/src/eval_first_stage.py`](../../agents/document_repl/src/eval_first_stage.py)
(a port of the original script), against `retrieval_server.py` (Qwen3-Embedding-8B, exact cosine, index on CPU) on
one A100; 137 s; no model API calls. Each question, unchanged, is one query; scored against the BrowseComp-Plus
evidence labels (`qrel_evidence.json`, sha256 prefix db8c851d0888b3d2). Development questions: the 150 ids of the
replication package's `results/per_question/fd_dense_high150.csv`; the other 680 are the held-out questions.

| Dense, recall@k % (hit@k %) | top 5 | top 30 | top 50 |
|---|---|---|---|
| search description | 15 (48) | 33 (74) | 39 (80) |
| all 830 questions | 14.91 (48.2) | 32.74 (73.6) | 38.71 (79.9) |
| 150 development questions | 15.56 (49.3) | 33.84 (75.3) | 39.89 (80.7) |
| 680 held-out questions | 14.77 (47.9) | 32.49 (73.2) | 38.45 (79.7) |

Only the 830-question measurement rounds to all six figures, so they were measured on all 830 questions, held-out
questions included. On the development questions alone each figure differs by about one percentage point.
The query-length figures in the same description (6-12 versus 13-20 words) come from agent-issued queries
(`eval_first_stage_agentq.py` of the original repository) and are not re-measured here.

Files: `first_stage_eval.json` (all systems, k = 5, 10, 20, 30, 50, three subsets; BM25 and reciprocal-rank-fusion rows
as in the original script), `first_stage_eval.log` (console output). No question or document text.
