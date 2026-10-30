"""Hybrid retrieval + passage selection for the BrowseComp-Plus retrieval server.

Pure functions only (no model, no IO); retrieval_server.py wires them up.

Design D = the common denominator of every leaderboard row that clears 80% at
<12 search calls (openJiuwen, LightOn Console, Mixedbread v3):

  1. first stage over WHOLE documents — dense (existing Qwen3-8B index) fused
     with BM25 by reciprocal rank fusion. BM25 sees the 36% of the corpus whose
     tails the dense index (passage_max_len=4096) never embedded;
  2. window the fused top-k documents (~512-token windows, half-window stride);
  3. cheap lexical pre-selection of windows (BM25 over the windows, CPU, ms);
  4. dense rescoring of only the surviving windows with the resident encoder,
     same query vector, so k=30 documents cost ~64 window encodes not ~1,000;
  5. return the best windows grouped by document with {docid, offset, score,
     text}; the full document text still lands in DOCS for grep / get_span.

Nothing here reads the question's gold answer or the qrels.
"""

from __future__ import annotations

from dataclasses import dataclass

import bm25s
import Stemmer

_STEMMER = Stemmer.Stemmer("english")


# --------------------------------------------------------------------------
# 1. reciprocal rank fusion
# --------------------------------------------------------------------------
def rrf(rankings: list[list[str]], k: int = 60, weights: list[float] | None = None) -> list[tuple[str, float]]:
    """Fuse ranked docid lists. Standard RRF: score = Σ w_i / (k + rank_i)."""
    weights = weights or [1.0] * len(rankings)
    score: dict[str, float] = {}
    for w, ranking in zip(weights, rankings):
        for r, d in enumerate(ranking, 1):
            score[d] = score.get(d, 0.0) + w / (k + r)
    return sorted(score.items(), key=lambda x: -x[1])


# --------------------------------------------------------------------------
# 2. windowing
# --------------------------------------------------------------------------
@dataclass
class Window:
    docid: str
    start: int
    end: int
    text: str


def _snap(text: str, pos: int, lo: int, hi: int) -> int:
    """Move `pos` to the nearest sentence/paragraph boundary within [lo, hi]."""
    best = pos
    for cand in (text.rfind("\n\n", lo, hi), text.rfind(". ", lo, hi), text.rfind("\n", lo, hi)):
        if cand != -1:
            best = cand + (2 if text[cand:cand + 2] in ("\n\n", ". ") else 1)
            break
    return max(lo, min(best, hi))


def windows_for(docid: str, text: str, size: int = 2000, stride: int = 1000, max_windows: int = 200) -> list[Window]:
    """~512-token windows (2000 chars) with half overlap, snapped to boundaries."""
    n = len(text)
    if n <= size:
        return [Window(docid, 0, n, text)]
    out: list[Window] = []
    start = 0
    while start < n and len(out) < max_windows:
        end = min(n, start + size)
        if end < n:
            end = _snap(text, end, start + size // 2, end)
        out.append(Window(docid, start, end, text[start:end]))
        if end >= n:
            break
        nxt = start + stride
        nxt = _snap(text, nxt, start + stride // 2, nxt)
        start = max(nxt, start + 1)
    return out


# --------------------------------------------------------------------------
# 3. lexical pre-selection of windows
# --------------------------------------------------------------------------
def bm25_preselect(query: str, wins: list[Window], top: int = 64) -> list[int]:
    """Indices of the `top` windows by BM25 against the query (in-memory, ms)."""
    if len(wins) <= top:
        return list(range(len(wins)))
    corpus_tok = bm25s.tokenize([w.text for w in wins], stopwords="en", stemmer=_STEMMER, show_progress=False)
    idx = bm25s.BM25(k1=1.2, b=0.75)
    idx.index(corpus_tok, show_progress=False)
    q_tok = bm25s.tokenize([query], stopwords="en", stemmer=_STEMMER, show_progress=False)
    res, _ = idx.retrieve(q_tok, k=min(top, len(wins)), show_progress=False)
    chosen = [int(i) for i in res[0]]
    # Always keep each doc's first window: title/lede carries the entity name and
    # the dense index was computed over the head. Mirrors openJiuwen's chunk-0 backfill.
    firsts = {i for i, w in enumerate(wins) if w.start == 0}
    for i in firsts:
        if i not in chosen:
            chosen.append(i)
    return chosen


# --------------------------------------------------------------------------
# 5. group for return
# --------------------------------------------------------------------------
def group_passages(
    wins: list[Window], scores: list[float], per_doc: int = 3, max_docs: int | None = None
) -> list[dict]:
    """Best windows per document, documents ordered by their best window."""
    by_doc: dict[str, list[tuple[float, Window]]] = {}
    for w, s in zip(wins, scores):
        by_doc.setdefault(w.docid, []).append((s, w))
    docs = []
    for docid, items in by_doc.items():
        items.sort(key=lambda x: -x[0])
        kept = items[:per_doc]
        kept.sort(key=lambda x: x[1].start)  # reading order inside the doc
        docs.append({
            "docid": docid,
            "score": round(float(items[0][0]), 6),
            "passages": [
                {"offset": w.start, "end": w.end, "score": round(float(s), 6), "text": w.text}
                for s, w in kept
            ],
        })
    docs.sort(key=lambda d: -d["score"])
    return docs[:max_docs] if max_docs else docs
