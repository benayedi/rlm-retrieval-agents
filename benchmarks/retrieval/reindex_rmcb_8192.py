"""Refactored copy of scripts/reindex_rmcb_8192.py of the original repository (unchanged code; documentation only, see
agents/shared/CHANGES_FROM_ORIGINAL.md). Needs a GPU with >30 GB free and pylate; run from the directory that holds (or should
hold) .pylate_idx/:  ${PYLATE_PYTHON} <repo>/benchmarks/retrieval/reindex_rmcb_8192.py [--start N]

Rebuild Reason-ModernColBERT at 8192 — RESUMABLE, chunked.

WHY NOT scripts/build_bc_rmcb_dl8192.sh. That path goes through retrieval_eval.py, which makes
exactly ONE add_documents call and checkpoints nothing: the 4096 build's own header records
losing a two-hour encode to a CUDA OOM with 15 GB of partial shards left unusable on disk. A
6-10 hour job with no resume is not worth running when the fix is this small.

Adapted from the document REPL agent's src/reindex_colbert_8192.py, keeping its three fixes:
  1. torch.cuda.empty_cache() BEFORE add_documents, not after. The encoder leaves ~16 GB of
     freed blocks in PyTorch's caching allocator; PLAID's centroid update then moves the whole
     chunk onto the GPU and OOMs.
  2. Checkpoint each chunk's embeddings to .npz BEFORE the index step, so a failed add costs
     the add, not the encode.
  3. use_triton=False. The Triton kernel path dies with a device-side assert on this driver —
     kernel module 595.71.05 against NVML 595.91, the same mismatch that breaks nvidia-smi here.

RESUME: --start N skips the first N documents and APPENDS (override is only set when N == 0).
The per-chunk line prints `done/total`, so on a crash restart with --start <that number>.
"""
import argparse, os, time
import numpy as np
import torch
from datasets import load_dataset
from pylate import indexes, models

ap = argparse.ArgumentParser()
ap.add_argument("--index-folder", default=".pylate_idx")
ap.add_argument("--index-name", default="bc_rmcb_dl8192")
ap.add_argument("--document-length", type=int, default=8192)
ap.add_argument("--query-length", type=int, default=256)   # matches every other index here
ap.add_argument("--chunk", type=int, default=25000, help="documents per encode/add cycle")
ap.add_argument("--batch-size", type=int, default=4)       # 4096 used 8; sequences are 2x longer
ap.add_argument("--nbits", type=int, default=4)
ap.add_argument("--start", type=int, default=0, help="resume: skip the first N documents")
ap.add_argument("--add-chunk", type=int, default=12500,
                help="documents per add_documents() call; caps GPU-resident embeddings")
ap.add_argument("--centroid-batch", type=int, default=512,
                help="rows per centroid-distance batch inside fast_plaid (upstream default 4096)")
ap.add_argument("--rust-batch", type=int, default=2048,
                help="batch_size handed to fast_plaid_rust.update (upstream default 50000)")
a = ap.parse_args()

# Two thresholds, because they answer different questions.
MIN_FREE_START_GB = 30   # nothing of ours is loaded yet, so free == what others left us
MIN_FREE_CHUNK_GB = 8    # mid-build, free EXCLUDES our own resident PLAID structures, which
                         # grow with the index (21.7 GB of "used" at chunk 2 was us, not an
                         # intruder). Only a real intruder -- a vLLM taking 20+ GB -- pushes
                         # free below this, and 8 GB still covers the next chunk's encode.

# ---------------------------------------------------------------------------
# PATCH 2: cap the batch fast_plaid_rust.update() works in.
#
# FastPlaid.update() defaults to batch_size=50_000, and pylate's PLAID.add_documents
# never forwards a value, so the Rust extension always ran at the default. Its
# matmul against the centroid table panicked once the table reached 1,068,588 rows
# (it doubled in a single chunk). Same OOM-in-driver-error's-clothing as before,
# but in Rust, so PATCH 1 -- which only touches the Python outlier loop -- cannot
# reach it. pylate passes keywords only, so setdefault injects ours without
# disturbing anything it does send.
import inspect
import fast_plaid.search.fast_plaid as _fpf

_orig_update = _fpf.FastPlaid.update
assert "batch_size" in inspect.signature(_orig_update).parameters, \
    "FastPlaid.update no longer takes batch_size; re-read the source"

def _update_with_capped_batch(self, *args, **kwargs):
    kwargs.setdefault("batch_size", a.rust_batch)
    return _orig_update(self, *args, **kwargs)

_fpf.FastPlaid.update = _update_with_capped_batch
print(f"patched fast_plaid_rust update batch 50000 -> {a.rust_batch}", flush=True)


# ---------------------------------------------------------------------------
# PATCH: cap fast_plaid's centroid-distance batch.
#
# update_centroids() finds embeddings far from existing centroids with
#     dists_sq = x2 + y2 - 2 * torch.matmul(batch, existing_centroids.T)
# over batches of a hardcoded 4096 rows. That tensor is batch x n_centroids, and
# n_centroids GROWS with every chunk we append: 522,348 after 50k documents, so
# 4096 x 522,348 x 4B = 8.6 GB per temporary, and the expression builds three of
# them. Chunk 2 fit; chunk 3 did not, and the failure surfaced as
#     RuntimeError: NVML_SUCCESS == nvmlInit_v2_() INTERNAL ASSERT FAILED
# rather than a clean OOM, because this host's driver mismatch (kernel module
# 595.71.05 vs NVML 595.91.07) breaks the allocator's out-of-memory reporting
# path. It is an OOM wearing a driver error's clothes, and it would recur on
# every later chunk since the centroid count only rises.
#
# 512 rows cuts each temporary 8x (~1 GB now, ~2 GB at the final centroid count)
# and changes no arithmetic -- the loop is a batched argmin, so batch size
# affects only peak memory, not which embeddings are flagged as outliers.
import fast_plaid.search.update as _fpu

_src = inspect.getsource(_fpu)
_needle = "    batch_size = 4096\n"
assert _src.count(_needle) == 1, (
    "fast_plaid.search.update no longer contains a single 'batch_size = 4096'; "
    "re-read the source before trusting this patch")
_ns = dict(_fpu.__dict__)
exec(compile(_src.replace(_needle, f"    batch_size = {a.centroid_batch}\n"),
             _fpu.__file__ + " [patched]", "exec"), _ns)
_fpu.update_centroids = _ns["update_centroids"]   # process_update resolves this as a global
print(f"patched fast_plaid centroid-distance batch 4096 -> {a.centroid_batch}", flush=True)


def gpu_free_gb():
    free, _ = torch.cuda.mem_get_info(0)
    return free / 1e9

# GATE. The previous attempt encoded for 25 minutes and then asserted, because another
# process took the card mid-build: PyTorch's allocator calls nvmlInit under memory pressure
# and nvmlInit is broken on this host (kernel module 595.71.05 vs NVML 595.91.07, the same
# mismatch that breaks nvidia-smi). Refuse to start rather than discover it later.
_f = gpu_free_gb()
if _f < MIN_FREE_START_GB:
    raise SystemExit(f"ABORT: only {_f:.1f} GB GPU free, need >{MIN_FREE_START_GB}. "
                     f"Something else holds the card; start again when it is free.")
print(f"gate passed: {_f:.1f} GB GPU free", flush=True)

t0 = time.time()
ds = load_dataset("Tevatron/browsecomp-plus-corpus", split="train")
ids, texts = [str(x) for x in ds["docid"]], list(ds["text"])
print(f"corpus: {len(ids):,} documents in {time.time()-t0:.0f}s", flush=True)

model = models.ColBERT(model_name_or_path="lightonai/Reason-ModernColBERT",
                       document_length=a.document_length, query_length=a.query_length,
                       model_kwargs={"torch_dtype": torch.float16}).to("cuda")
print(f"model on cuda, document_length={a.document_length}", flush=True)

index = indexes.PLAID(index_folder=a.index_folder, index_name=a.index_name,
                      override=(a.start == 0), nbits=a.nbits, use_triton=False)
if a.start:
    ids, texts = ids[a.start:], texts[a.start:]
    print(f"resuming: skipping the first {a.start:,} documents already indexed", flush=True)

done, t_start, stopped_early = a.start, time.time(), False
total = a.start + len(ids)
for s in range(0, len(ids), a.chunk):
    # Re-check before each chunk. If another tenant has taken the card, stop at this
    # boundary with a usable --start rather than asserting mid-encode 25 minutes later.
    _f = gpu_free_gb()
    if _f < MIN_FREE_CHUNK_GB:
        print(f"STOPPING CLEANLY: GPU free dropped to {_f:.1f} GB (<{MIN_FREE_CHUNK_GB}). "
              f"Everything up to document {done:,} is indexed. "
              f"RESUME WITH --start {done}", flush=True)
        stopped_early = True
        break
    cid, ctx = ids[s:s+a.chunk], texts[s:s+a.chunk]
    ckpt = os.path.join(a.index_folder, f"{a.index_name}.chunk{done}.npz")
    # fix 3: a checkpoint left behind means the encode succeeded and only the index
    # step failed, so reuse it instead of paying ~23 min to encode the same documents.
    if os.path.exists(ckpt):
        t = time.time()
        with np.load(ckpt) as z:
            emb = [z[str(i)] for i in range(len(cid))]
        t_enc = time.time() - t
        print(f"  reused checkpoint for documents {done:,}-{done+len(cid):,} "
              f"({t_enc:.0f}s, no re-encode)", flush=True)
    else:
        t = time.time()
        emb = model.encode(sentences=ctx, batch_size=a.batch_size, is_query=False, show_progress_bar=False)
        t_enc = time.time() - t
        torch.cuda.empty_cache()                   # fix 1: release BEFORE handing to PLAID
        try:                                        # fix 2: checkpoint before the index step
            np.savez(ckpt, **{str(i): e for i, e in enumerate(emb)})
        except Exception as e:
            print(f"  (checkpoint failed, continuing: {e})", flush=True)
    n_emb = sum(e.shape[0] for e in emb)
    torch.cuda.empty_cache()
    # fix 4: hand PLAID at most --add-chunk documents at a time. add_documents copies
    # every embedding it is given onto the GPU at once, so a 25,000-document call pins
    # ~19 GB in fp16 before update_centroids even allocates its distance matrices.
    t = time.time()
    for j in range(0, len(cid), a.add_chunk):
        index.add_documents(documents_ids=cid[j:j+a.add_chunk],
                            documents_embeddings=emb[j:j+a.add_chunk])
        torch.cuda.empty_cache()
        print(f"    added {done+min(j+a.add_chunk, len(cid)):,} | "
              f"{gpu_free_gb():.1f} GB free", flush=True)
    try: os.remove(ckpt)
    except Exception: pass
    t_add = time.time() - t
    del emb; torch.cuda.empty_cache()
    done += len(cid)
    rate = (done - a.start) / (time.time() - t_start)
    print(f"  {done:,}/{total:,} docs | {n_emb/len(cid):.0f} emb/doc | encode {t_enc:.0f}s "
          f"add {t_add:.0f}s | {rate:.2f} docs/s | ETA {(total-done)/rate/60:.0f}m"
          f" | {gpu_free_gb():.1f} GB free | RESUME WITH --start {done}", flush=True)
# Only claim DONE on true completion -- a clean early stop used to print it too, which
# reads in the log as "the index is finished" when half the corpus is missing.
if stopped_early:
    print(f"STOPPED EARLY after {(time.time()-t_start)/60:.1f} min at document {done:,}/{total:,}"
          f" -- index is INCOMPLETE, resume with --start {done}", flush=True)
else:
    print(f"DONE in {(time.time()-t_start)/60:.1f} min -> {a.index_folder}/{a.index_name}", flush=True)
