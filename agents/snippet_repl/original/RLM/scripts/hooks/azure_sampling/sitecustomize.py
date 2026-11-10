"""Make rlm's AzureOpenAIClient send its sampling_args (2026-10-27). Loaded by Python at startup when this directory is
on PYTHONPATH.

WHY. rlm/clients/azure_openai.py builds every request from model + messages only. The reasoning_effort that run_one.py
and the snippet harness pass as sampling_args reaches the client (BaseLM stores it in self.sampling_args) and is then
dropped, so every Azure run through rlm used the deployment's DEFAULT effort. Measured on this endpoint (same prompt,
3 calls each): gpt-5 with no effort 192-384 reasoning tokens = medium (high 320-512); gpt-5.2 with no effort 0 =
no reasoning (high 21-31). rlm's OpenAIClient does forward sampling_args; this gives the Azure client the same
behaviour, normalised by the same helper. The library file is untouched.

RLM_USAGE_LOG=<file> also appends one JSON line per Azure call: deployment, prompt / cached / completion / reasoning
tokens, pid and time. The harnesses' own token fields are unreliable (the snippet harness overwrites root usage with
the sub-LM's) and none records cached tokens, so this is the source for the cached / uncached cost columns.
"""
import json as _json
import os as _os
import time as _time


def _log_usage(model, response):
    path = _os.environ.get("RLM_USAGE_LOG")
    if not path:
        return
    try:
        u = response.usage
        pd = getattr(u, "prompt_tokens_details", None); cd = getattr(u, "completion_tokens_details", None)
        rec = {"t": round(_time.time(), 1), "pid": _os.getpid(), "model": model, "prompt": u.prompt_tokens,
               "cached": getattr(pd, "cached_tokens", 0) or 0, "completion": u.completion_tokens,
               "reasoning": getattr(cd, "reasoning_tokens", 0) or 0}
        with open(path, "a") as f:
            f.write(_json.dumps(rec) + "\n")
    except Exception:  # noqa: BLE001  logging must never break a run
        pass


try:
    from rlm.clients import azure_openai as _AZ
    from rlm.clients.openai import _normalize_sampling_args as _norm

    def _completion(self, prompt, model=None):
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for Azure OpenAI client.")
        response = self.client.chat.completions.create(model=model, messages=messages,
                                                       **_norm(getattr(self, "sampling_args", {}) or {}))
        self._track_cost(response, model)
        _log_usage(model, response)
        return response.choices[0].message.content

    async def _acompletion(self, prompt, model=None):
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for Azure OpenAI client.")
        response = await self.async_client.chat.completions.create(model=model, messages=messages,
                                                                   **_norm(getattr(self, "sampling_args", {}) or {}))
        self._track_cost(response, model)
        _log_usage(model, response)
        return response.choices[0].message.content

    _AZ.AzureOpenAIClient.completion = _completion
    _AZ.AzureOpenAIClient.acompletion = _acompletion
except Exception as _e:  # noqa: BLE001
    print(f"[azure-sampling] hook not installed: {_e}", flush=True)
