# Question-id splits of the prompt-optimisation runs

Byte-exact copies of the split files in `data/` of the original repository (`MANIFEST.json`: source path, SHA-256,
size, modification time). They hold question ids and counts only; no question, answer or document text. The
directory layout mirrors the original `data/`, so the code reads `splits/<same path>` where the original read
`data/<same path>`. Three files carry the original machine's input path in a metadata field (`source`,
`_meta.seed_run`), as written by the split builders.

| File | What it is | Used by |
|---|---|---|
| `bcp_gepa_splits/split.json` | BrowseComp-Plus, stratified on the Qwen3.8-27B 830-question outcomes (`src/make_bcp_splits.py`, seed 20261006): pareto 100, feedback 200, test200 200, test_rest 330 | GEPA-1 (D_pareto, D_feedback); GEPA-2 confirmation and BCP-29 / SP-1/SP-2 controls (test200, test_rest) |
| `bcp_gepa_splits/split_summary.txt` | accuracy, gold recall, wall time and strata of each split (written with `split.json`) | documentation |
| `bcp_gepa_splits/split_enriched_q36.json` | run 2's failure-enriched D_pareto: 70 Qwen3.6 seed failures + 30 full-recall guards, from pareto + feedback (`src/make_bcp_enriched_split.py`, seed 20261008); test200 / test_rest unchanged | GEPA-2 |
| `bcp_full830/split_rest300.json` | `split.json` plus `rest300` = pareto + feedback (the 300 questions not in test200 / test_rest) | SP-1 (the merged pareto + feedback reranker run, `logs/bcp_rest300_rr`) |
| `frames/gepa_splits/split.json` | FRAMES, stratified on the FR-13 seed run's outcomes (`src/gepa_frames.py --make-splits`): feedback 200, pareto 100, test 200; q0-99 never in test | GEPA-3; its confirmation runs the 324 ids in none of the three splits (computed, no file) |
| `obliq_splits/twitter_dev.json` | OBLIQ-Twitter dev100 (D_feedback of the GEPA-4 runs) | GEPA-4 runs 1-4 |
| `obliq_splits/twitter_heldout.json` | the other 181 questions; each run draws its D_pareto from them (runs 3-4 also write `test_split.json`, the 81 never used) | GEPA-4 runs 1-4 |
| `obliq_splits/twitter_pareto100.json`, `twitter_pareto_gepa.json`, `twitter_smoke3.json` | a 100-question pareto list, a second 100-question list, and the three smoke-test questions of the campaign | GEPA-4 chains (smoke tests, earlier pareto choices) |

The split builders (`src/make_bcp_splits.py`, `src/make_bcp_enriched_split.py`, `src/gepa_frames.py --make-splits`)
write into this directory, as the originals wrote into `data/`; re-running one rewrites the committed copy, and git
shows any difference. `tests/test_splits.py` checks the manifest and the properties above, and (with the original
grades available) that `make_bcp_splits.py`'s draws reproduce `split.json`.
