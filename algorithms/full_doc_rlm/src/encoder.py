"""Qwen3-Embedding-8B encoder matching Tevatron's DenseModel encode path.

From original/FULLDOC/src/encoder.py. Imported by retrieval_server.py; not run on its own.

Tevatron encoded the BrowseComp-Plus corpus with:
    --pooling eos --normalize --passage_prefix "" --passage_max_len 4096 --fp16
and serves queries with:
    task_prefix + query, --pooling eos --normalize, max_length 8192

`_pooling` below mirrors tevatron.retriever.modeling.EncoderModel._pooling for
the 'eos'/'last' branch, including the left-padding fast path.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# Import wiring: shared/rlm_shared (configs/paths.example.env variables), found relative to this file.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir, "shared"))
from rlm_shared.paths import path  # noqa: E402

DEFAULT_MODEL = f"{path('FULLDOC_ROOT')}/data/models/Qwen3-Embedding-8B"  # original: /home/extstu_00/suprlm/data/models/...

# Verbatim from BrowseComp-Plus searcher/searchers/faiss_searcher.py --task-prefix default.
QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
)
PASSAGE_PREFIX = ""

QUERY_MAX_LEN = 8192  # faiss_searcher --max-length default
PASSAGE_MAX_LEN = 4096  # tevatron encode --passage_max_len


def _pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """EOS / last-token pooling, matching Tevatron."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_state[:, -1]
    lengths = attention_mask.sum(dim=1) - 1
    idx = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
    return last_hidden_state[idx, lengths]


class Qwen3Encoder:
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        device: str | None = None,
        dtype: torch.dtype = torch.float16,
        append_eos: bool = False,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.append_eos = append_eos

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        # Separate instance for span computation. HF fast tokenizers store
        # truncation/padding settings on the shared Rust object, so calling
        # embedded_char_span(max_length=4096) concurrently with an encode at
        # max_length=8192 lets the two bleed into each other. Own instance,
        # own settings, no lock needed.
        self.span_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._span_cache: dict[str, tuple[int, int]] = {}
        self.model = AutoModel.from_pretrained(
            model_path,
            dtype=dtype if device == "cuda" else torch.float32,
            attn_implementation="sdpa",
        )
        self.model = self.model.to(device).eval()

    def _encode(self, texts: list[str], max_length: int) -> torch.Tensor:
        if self.append_eos:
            eos = self.tokenizer.eos_token or ""
            texts = [t + eos for t in texts]
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            out = self.model(**batch)
            reps = _pooling(out.last_hidden_state, batch["attention_mask"])
            reps = F.normalize(reps, p=2, dim=-1)
        return reps.float().cpu()

    def encode_queries(self, queries: list[str]) -> torch.Tensor:
        return self._encode([QUERY_PREFIX + q for q in queries], QUERY_MAX_LEN)

    def encode_passages(self, passages: list[str]) -> torch.Tensor:
        return self._encode([PASSAGE_PREFIX + p for p in passages], PASSAGE_MAX_LEN)

    def embedded_char_span(self, text: str) -> tuple[int, int]:
        """Character span of `text` that the corpus vector was actually computed over.

        Tevatron truncated passages at PASSAGE_MAX_LEN tokens *including* the
        auto-appended <|endoftext|>, so only PASSAGE_MAX_LEN - 1 text tokens
        were embedded. Returns (0, end_char).
        """
        # Key on the text itself, not hash(text): the cache stores no copy of the
        # text and never verifies it, so a hash collision would silently return
        # another document's span. Keying on the string costs the same hash and
        # makes collisions impossible.
        key = text
        hit = self._span_cache.get(key)
        if hit is not None:
            return hit
        enc = self.span_tokenizer(
            text,
            truncation=True,
            max_length=PASSAGE_MAX_LEN - 1,  # one slot goes to the appended EOS
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        offsets = [tuple(o) for o in enc["offset_mapping"] if tuple(o) != (0, 0)]
        # No offsets means we know nothing about what was embedded; do not claim
        # the whole document.
        span = (0, int(offsets[-1][1])) if offsets else (0, 0)
        if len(self._span_cache) < 200_000:
            self._span_cache[key] = span
        return span
