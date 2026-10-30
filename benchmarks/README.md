# Benchmarks and retrieval

Three benchmarks are used in `results/original_baselines/Experiments.xlsx`: BrowseComp-Plus, FRAMES and OBLIQ-Bench.
This file says where each one comes from, how the code loads its questions and corpus, which index and server every
filled Experiments.xlsx row used, and how to build those indexes. The servers and index builders are in
`benchmarks/retrieval/`. They are refactored copies of the original scripts. Only paths and import wiring changed; see
`shared/CHANGES_FROM_ORIGINAL.md`.

Paths are environment variables from `configs/paths.example.env`. Copy that file to `configs/paths.env`, fill it in,
and load it with `set -a; . configs/paths.env; set +a`. Nothing in this directory is run by the tests. They exercise
request handling in-process (`tests/test_shared_retrieval.py`).

No benchmark data, corpus or index is part of this repository (see `.gitignore`).

---

## Wire formats

Every retriever the agents call speaks one of two HTTP formats.

### "rlm" format (Snippet RLM, DSPy.RLM, every server in `benchmarks/retrieval/`)

| request | body | response |
|---|---|---|
| `POST /search` | `{"query": str, "top_k": int[, "index": str][, "tag": str]}` | `{"results": [{"doc_id": str, "score": float, "text": str}, ...]}` |
| `POST /document` | `{"doc_id": str[, "index": str]}` | `{"doc_id": str, "text": str}`, or `{"error": ..., "doc_id": ..., "text": ""}` |
| `GET /health` | — | `{"status": "ok", ...}` |

- **Text is cut to 2,000 characters on the server**, in `/search` only (`text[:2000]` in `search_server.py`,
  `multi_search_server.py`, `gemini/search_server.py` and `frames_search_server.py --protocol rlm`).
  `fusion_proxy.py` applies the same cap to the text it fetches for documents only its multi-vector arm returned
  (`PROXY_TEXT_CAP`, default 2000). `/document` returns the whole text.
- `index` selects an index on multi-index servers. If it is missing, the server's default index is used.
- `tag` (`<query_id>-<uuid>`) is required by `obliq_math_search.py`. It needs the query id to drop that query's own
  problem from the results. The other servers ignore it.
- `frames_search_server.py` clamps `top_k` to 1–100 (default 5) and drops hits with a BM25 score of 0. The FAISS
  servers pass `top_k` through unchanged.
- `/encode` (`{"inputs": [str]}` → `{"embeddings": [[float]]}`) exists on the Qwen3 and Gemini servers. The OBLIQ
  index builder uses it.

### "sup" format (Full-Doc RLM)

This is the format of `algorithms/full_doc_rlm/src/retrieval_server.py`. `frames_search_server.py --protocol sup` and
`obliq_math_search.py` also serve it. `POST /search {"query", "k", "mode": snippet|full|ids, "snippet_max_tokens",
"tag"}` → `{"results": [{"docid", "score", "snippet"} | {"docid", "score", "text", "n_chars", "embedded_span",
"snippet_span"}]}`. There are also `POST /tag_counts {"tag"}`, `POST /tag_docids {"tag"}` and `GET /health`. A search
without a tag is refused with HTTP 400. The Full-Doc harness uses the tag to count searches and retrieved documents
per question on the server side.

---

## BrowseComp-Plus (830 questions, 100,195 documents)

**Source.**
- Questions: the Hugging Face dataset `Tevatron/browsecomp-plus`, split `test`.
- Corpus: `Tevatron/browsecomp-plus-corpus`, split `train`.

Both names are taken from the code (`rlm_shared/grading/bcp_judge.py`, `benchmarks/retrieval/build_bm25_index.py`,
`reindex_rmcb_8192.py`).

**Licence:** not determined. No file read for this refactor states one.

**The questions and answers are distributed encrypted and must not be published in plain text.** Each test row holds
`query` and `answer`, and the docids in `gold_docs` / `evidence_docs`, as base64 of the plain text XOR-ed with the
SHA-256 of a canary string. `shared/rlm_shared/browsecomp_data.py` decrypts them in memory with `get_question`,
`get_gold_answer`, `get_gold_doc_ids` and `get_evidence_doc_ids`. Never write decrypted questions or answers into this
repository, a log you publish, or a results file. `scripts/provenance/bcp_leak_scan.py` checks for leaks.

**Loading.**
- Snippet RLM and the graders: `datasets.load_dataset("Tevatron/browsecomp-plus", split="test")`, then decryption.
  The question index `q_idx` is the row position in that split.
- Full-Doc RLM: reads a decrypted local copy, `$FULLDOC_ROOT/data/prepared/{corpus.jsonl,queries.jsonl,
  qrel_evidence.json}`, written by `algorithms/full_doc_rlm/src/prepare_data.py`. The BM25 index below is built from
  that `corpus.jsonl` (`{"docid", "text"}` per line).

**Index and server of each Experiments.xlsx row** (BrowseComp+ sheet):

| rows | harness | retrieval signal | index | server and port |
|---|---|---|---|---|
| 9, 15 | Snippet RLM | Qwen3-Embedding-8B | FAISS IVF-SQ8 over the corpus embeddings | `embed_server_new.py` on :8001. **Not among the copied sources.** The original repository's `docs/server_commands.md` launches it with `--faiss-index ~/faiss_cache/_cached_ivfsq8.faiss --max-length 8192 --nprobe 128`. It speaks the rlm format, like `search_server.py` |
| 11, 17 | Snippet RLM | Reason-ModernColBERT | PLAID index `bc_reason_mcb`, **document length 512** | `fusion_proxy.py --mv lightonai/Reason-ModernColBERT:bc_reason_mcb --mv-only --dense-url http://localhost:8001` on :8024. Scores are 1/rank. `/document` text comes from :8001 |
| 13, 19 | Snippet RLM | BM25 (k1 0.9, b 0.4) | `$BCP_BM25_INDEX_DIR` (`bcp_build_bm25_default.py`) | `frames_search_server.py --dataset bcp --protocol rlm` on :8044 |
| 21, 27 | Full-Doc RLM | Qwen3-Embedding-8B | exact cosine over the Tevatron embeddings, `$FULLDOC_ROOT/data/indexes/qwen3-embedding-8b` | `algorithms/full_doc_rlm/src/retrieval_server.py` on :8765 (sup format) |
| 23, 29 | Full-Doc RLM | Reason-ModernColBERT | PLAID, **document length 8192**, `$FULLDOC_ROOT/data/indexes/lighton/pylate` | `algorithms/full_doc_rlm/src/retrieval_server_colbert.py` on :8766 (sup format) |
| 25, 31 | Full-Doc RLM | BM25 (k1 0.9, b 0.4) | `$BCP_BM25_INDEX_DIR` | `frames_search_server.py --dataset bcp --protocol sup` on :8043 |

Rows 5 and 7 are the paper's own baselines. They are not run here.

Sources for the ports: the run launchers in `algorithms/snippet_rlm/original/RLM/scripts/run5d/bcp_snip*.sh`
(":8001 Qwen3-Embedding-8B FAISS IVFSQ8 | :8044 BM25 k1 0.9 b 0.4 | :8024 Reason-ModernColBERT"),
`algorithms/full_doc_rlm/original/RLM/scripts/bcp_gpt5_bm25.sh` (:8043) and `run5d/bcp_fd_qwen_dense.py` (:8765). The
512-length index behind :8024 is named in the original `docs/HANDOFF.md` ("bc_reason_mcb (512) ... every published
result uses it"). `bc_rmcb_dl8192` is this project's own 8192-length rebuild, made with `build_bc_rmcb_dl8192.sh` and
then `reindex_rmcb_8192.py`. No Experiments.xlsx row is documented as using it in the files read. The Full-Doc
ColBERT rows used the Full-Doc harness's own 8192 index.

**Build commands.**

```bash
set -a; . configs/paths.env; set +a
# BM25, k1 0.9 / b 0.4 (Lucene, bm25s, English stemmer + stopwords). Writes bm25_index/, docids.json and offsets.npy
# to $BCP_BM25_INDEX_DIR. Needs the prepared corpus at $FULLDOC_ROOT/data/prepared/corpus.jsonl.
$PYTHON benchmarks/retrieval/bcp_build_bm25_default.py
# Reason-ModernColBERT, document length 8192, PLAID. Resumable; needs a GPU with >30 GB free. Run it where .pylate_idx/
# should be written.
$PYLATE_PYTHON benchmarks/retrieval/reindex_rmcb_8192.py            # add --start N to resume
# The older one-shot build. Needs scripts/retrieval_eval.py in $RETRIEVAL_EVAL_ROOT, which is not in this repository.
bash benchmarks/retrieval/build_bc_rmcb_dl8192.sh
# BM25 with the bm25s library defaults (k1 1.5, b 0.75) to ~/bm25_index. This is the lexical arm of fusion_proxy.py,
# not an Experiments.xlsx row.
$PYTHON benchmarks/retrieval/build_bm25_index.py
```

The 512-length `bc_reason_mcb` index and the :8001 IVF-SQ8 index were built in an earlier repository. Their build
scripts are not among the copied sources.

**Servers.**

```bash
$PYTHON benchmarks/retrieval/frames_search_server.py --dataset bcp --protocol rlm --port 8044
$PYTHON benchmarks/retrieval/frames_search_server.py --dataset bcp --protocol sup --port 8043
$PYLATE_PYTHON benchmarks/retrieval/fusion_proxy.py --dense-url http://localhost:8001 \
    --mv lightonai/Reason-ModernColBERT:bc_reason_mcb --mv-only --query-len 256 --device cuda --port 8024
```

The fusion-proxy command is the one in the original `scripts/start_mvonly_proxy.sh`, which used port 8021, run from
the directory that holds `.pylate_idx/`.

**Grading:** `shared/rlm_shared/grading/` (`grade_bcp_snippet.py`, `grade_bcp_fulldoc.py`, `rejudge_bcp.py`). The
judge is gpt-4.1 with the official BrowseComp-Plus template, 3 votes and a majority.

---

## FRAMES (824 questions, Wikipedia 2023-06-01, 6.67M articles)

**Source.**
- Questions: `$FRAMES_DATA_DIR/test.tsv` (column `Prompt`) and `$FRAMES_DATA_DIR/frames_rows.json` (one object per
  question with `i`, `question`, `answer`, `gold_ids` and `reasoning_types`). The code read does not record where
  these two files were downloaded from or how `frames_rows.json` was derived. The paper is FRAMES, Krishna et al.,
  arXiv 2409.12941 (cited in `frames_build_bm25.py`).
- Corpus: the TFDS dataset `wikipedia/20230601.en`, 256 TFRecord shards
  (`gs://tfds-data/datasets/wikipedia/20230601.en/1.0.0/`, according to the original `docs/DATASET_BENCHMARKS.md`).
  The shards are placed in `$FRAMES_DATA_DIR/tfds/`.

**Licence:** not determined.

**Loading.**
- `frames_build_bm25.py` parses the TFRecords by hand, without TensorFlow. It writes
  `$FRAMES_DATA_DIR/corpus.jsonl` (`{"title", "text"}`, one article per line) and the bm25s index
  `$FRAMES_DATA_DIR/bm25_index/`.
- A FRAMES document id is the line number in `corpus.jsonl`, and the document text is `"title\ntext"`.
- `frames_search_server.py` also reads `$FRAMES_DATA_DIR/corpus_meta.npz` (`offsets`, the byte offset of each line).
  **The step that writes `corpus_meta.npz` is not among the copied sources.** The array is the byte offset of every
  line of `corpus.jsonl`, in order.
- `shared/rlm_shared/grading/grade_frames.py` reads the gold answers and `gold_ids` from `frames_rows.json`.

**Index and server of each Experiments.xlsx row** (FRAMES sheet). Every row uses BM25 with k1 0.9 / b 0.4 (Lucene)
over the same index:

| rows | harness | server and port |
|---|---|---|
| 7, 9 | Snippet RLM | `frames_search_server.py --protocol rlm` on :8040 |
| 11, 13 | Full-Doc RLM | `frames_search_server.py --protocol sup` on :8041. Snippets are cut with the Qwen3-0.6B tokenizer from `$FULLDOC_ROOT/data/models/Qwen3-0.6B` |

Row 5 is the paper's baseline. The ports come from `algorithms/snippet_rlm/original/RLM/scripts/run5d/frames_snip.sh`
(:8040) and `algorithms/full_doc_rlm/original/RLM/scripts/run5d/frames_fd_gpt5.sh` (:8041).

**Build and serve.**

```bash
set -a; . configs/paths.env; set +a
# shard (parallel tokenisation) -> validate (compares 2 shards with plain bm25s) -> merge
$PYTHON benchmarks/retrieval/frames_build_bm25.py all --workers 48
$PYTHON benchmarks/retrieval/frames_search_server.py --protocol rlm --port 8040
$PYTHON benchmarks/retrieval/frames_search_server.py --protocol sup --port 8041
```

---

## OBLIQ-Bench (Math 151, Twitter 281, Congress 254 questions)

**Source.** The Hugging Face dataset `dianetc/OBLIQ-Bench` (`build_obliq_index.py`), with files
`<task>/<subset>/corpus/corpus.jsonl` (`{"_id", "text"[, "title"]}`) and `<task>/<subset>/queries+qrels/`
(`queries.jsonl`, `qrels.tsv`, `qrels_pool.tsv` where the task has pooled labels, and `per_query_excluded_ids.json`
for Math). Tasks used: `analogues/math` (3,508 documents), `descriptive/twitter` (72,122) and `tip-of-tongue/congress`
(213,650). The corpus sizes are from the Experiments.xlsx sheet headers.

**Licence:** not determined.

**Loading.**
- `shared/rlm_shared/obliq_data.py` reads queries and qrels from `$OBLIQ_BENCH_ROOT/<task>/<subset>/queries+qrels/`
  (`load_queries`, `load_qrels(pool=...)`).
- It parses each ranked answer into document ids (`extract_doc_ids_from_answer`).
- It scores NDCG@k and Recall@k (`ndcg_at`, `recall_at`), which `shared/rlm_shared/grading/obliq_report.py` uses.
- Math excludes each query's own source problem (`per_query_excluded_ids.json`). `obliq_math_search.py` applies the
  exclusion on the server, for both formats.

**Index and server of each Experiments.xlsx row** (OBLIQ sheets, rows 9 = Snippet RLM and 11 = Full-Doc RLM). Every
row uses Gemini-Embedding-2 on Vertex AI over the raw document text, as in the paper (`--doc-format raw`,
`--query-format raw`), with an exact FAISS `IndexFlatIP` on CPU:

| task | index (under `~/indexes/gemini/`) | Snippet RLM endpoint | Full-Doc RLM endpoint |
|---|---|---|---|
| Math | `obliq-ana-math` | `obliq_math_search.py` :8014, in front of the Gemini server :8020 | the same :8014 (sup format) |
| Twitter | `obliq-des-twitter` | `gemini/search_server.py` :8021 | `algorithms/full_doc_rlm/src/obliq_adapter.py` :8768 |
| Congress | `obliq-tot-congress` | `gemini/search_server.py` :8022 | `obliq_adapter.py` :8769 |

The ports come from `algorithms/snippet_rlm/original/RLM/scripts/run5d/obliq_gemini.sh`. Rows 5 and 7 are the paper's
numbers. The WildChat and Writing sheets have no run.

**Build and serve.** Gemini needs a Vertex AI service account. **Configuration requirement:**
`benchmarks/retrieval/gemini/vertex.py` reads the service-account JSON from the environment variable
`NNEXT_PROVIDERS_GCP_VERTEX_SERVICE_ACCOUNT_JSON`. When that variable is unset, it reads the line of the same name in
the `.env`-format file named by `GCP_VERTEX_ENV`. The key is never printed or written to disk. Never commit that file.
`GEMINI_LOCATION` defaults to `eu`.

```bash
set -a; . configs/paths.env; set +a
$GEMINI_PYTHON benchmarks/retrieval/gemini/build_index.py \
    --corpus $OBLIQ_BENCH_ROOT/analogues/math/corpus/corpus.jsonl --out ~/indexes/gemini/obliq-ana-math
$GEMINI_PYTHON benchmarks/retrieval/gemini/search_server.py --port 8020 --index-root ~/indexes/gemini \
    --default-index obliq-ana-math                          # :8021 obliq-des-twitter, :8022 obliq-tot-congress
$PYTHON benchmarks/retrieval/obliq_math_search.py --port 8014 --upstream http://localhost:8020 --index obliq-ana-math
```

The Qwen3-Embedding-8B OBLIQ indexes are not used by any filled Experiments.xlsx row. `build_obliq_index.py` builds
them through a running server's `/encode`, and `search_server.py` / `multi_search_server.py` serve them.

---

## Files in `benchmarks/retrieval/`

| file | original (`scripts/` of the original repository) | role |
|---|---|---|
| `frames_search_server.py` | `frames_search_server.py` | BM25 server for FRAMES and BrowseComp-Plus, rlm or sup format, CPU |
| `fusion_proxy.py` | `fusion_proxy.py` | RRF fusion or multi-vector-only (PLAID) proxy in front of a dense server |
| `search_server.py` | `search_server.py` | single-index Qwen3-Embedding-8B FAISS server |
| `multi_search_server.py` | `multi_search_server.py` | multi-index Qwen3-Embedding-8B FAISS server |
| `obliq_math_search.py` | `obliq_math_search.py` | OBLIQ Math front-end that drops excluded ids |
| `gemini/search_server.py`, `gemini/vertex.py`, `gemini/build_index.py` | `gemini/*` | Gemini-Embedding-2 server, client and index builder |
| `frames_build_bm25.py` | `frames_build_bm25.py` | FRAMES Wikipedia BM25 index |
| `bcp_build_bm25_default.py` | `bcp_build_bm25_default.py` | BrowseComp-Plus BM25 index, k1 0.9 / b 0.4 |
| `build_bm25_index.py` | `build_bm25_index.py` | BrowseComp-Plus BM25 index with the bm25s defaults (fusion arm) |
| `build_obliq_index.py` | `build_obliq_index.py` | OBLIQ FAISS index through `/encode` |
| `reindex_rmcb_8192.py`, `build_bc_rmcb_dl8192.sh` | same names | Reason-ModernColBERT 8192 PLAID index |
