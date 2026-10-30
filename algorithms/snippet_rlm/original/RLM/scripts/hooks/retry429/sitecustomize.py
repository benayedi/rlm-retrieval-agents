"""Patient retries on Azure 429 (token-rate limit), 2026-10-26. Loaded at interpreter start when this directory is on
PYTHONPATH. The openai client retries a 429 only twice by default (honouring retry-after), after which one RLM call
fails and the whole question is lost: 4 of the first 46 GPT-5 high questions died that way with 24 runs in flight.
This only raises the default max_retries of the openai clients (OPENAI_MAX_RETRIES, default 10); an explicit
max_retries argument still wins. Agent, prompts and models are untouched.
"""
import os

try:
    import openai

    _N = int(os.environ.get("OPENAI_MAX_RETRIES", "10"))
    for _cls in (openai.OpenAI, openai.AzureOpenAI, openai.AsyncOpenAI, openai.AsyncAzureOpenAI):
        _orig = _cls.__init__

        def _init(self, *a, __orig=_orig, **kw):
            kw.setdefault("max_retries", _N)
            __orig(self, *a, **kw)

        _cls.__init__ = _init
except Exception as e:  # noqa: BLE001
    print(f"[retry429] hook not installed: {e}", flush=True)
