"""Opt-in: stop a generation that has fallen into a repetition loop. SUP_REP_GUARD=1.

Measured 2026-10-11 on BrowseComp+ test_rest (Qwen3.6-35B-A3B, presence_penalty 1.5 already on):
~0.6% of turns degenerate into one token or line repeated to the 16k-token cap
("Feldman, Feldman, ..." / a dinosaur list x330). One such turn takes up to 25 min under load and
produced 3 of the novelty+re-rank arm's extra timeouts. presence_penalty is flat per token, so it
cannot stop a loop that has already started.

HOW. OpenAIClient.completion / acompletion are replaced by streaming versions (vLLM with
continuous usage stats, so token accounting stays exact). Every CHECK_EVERY new chars the last
WINDOW chars are tested; the stream is closed when either
  - token diversity < MIN_DIVERSITY over >= 300 whitespace tokens, or
  - one non-trivial line (>= 4 chars, not just punctuation) occurs >= MAX_LINE_REPEATS times.
The text up to that point is returned with a short note appended, so the agent sees what happened
and continues on the next turn. Legit code and prose never get near these thresholds (checked on
12,819 normal test_rest turns, see RLM/docs/GEPA_BROWSECOMP_PLAN.md). Unset = no effect.
"""
import collections
import os
import re
import types

WINDOW = 6000
CHECK_EVERY = 1500
MIN_DIVERSITY = 0.15
MAX_LINE_REPEATS = 25
NOTE = "\n\n[generation stopped: repetition loop detected]\n"
STOPPED = [0]


def looping(text: str) -> bool:
    tail = text[-WINDOW:]
    toks = re.findall(r"\S+", tail)
    if len(toks) >= 300 and len(set(toks)) / len(toks) < MIN_DIVERSITY:
        return True
    lines = [l.strip() for l in tail.split("\n")]
    lines = [l for l in lines if len(l) >= 4 and re.search(r"\w", l)]
    if lines:
        _, n = collections.Counter(lines).most_common(1)[0]
        if n >= MAX_LINE_REPEATS:
            return True
    return False


def _usage_obj(u):
    return types.SimpleNamespace(usage=u)


def install() -> None:
    if os.environ.get("SUP_REP_GUARD") != "1":
        return
    from rlm.clients import openai as oc

    def _prep(self, prompt, model):
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        model = model or self.model_name
        extra = {}
        if self.client.base_url == oc.DEFAULT_PRIME_INTELLECT_BASE_URL:
            extra["usage"] = {"include": True}
        return messages, model, oc._merge_extra_body(extra, self.sampling_args)

    def completion(self, prompt, model=None):
        messages, model, extra = _prep(self, prompt, model)
        stream = self.client.chat.completions.create(
            model=model, messages=messages, extra_body=extra, stream=True,
            stream_options={"include_usage": True, "continuous_usage_stats": True},
            **oc._normalize_sampling_args(self.sampling_args))
        text, seen, usage, last_check, n = [], [], None, 0, 0
        try:
            for ch in stream:
                if getattr(ch, "usage", None) is not None:
                    usage = ch.usage
                if ch.choices:
                    d = ch.choices[0].delta
                    piece = getattr(d, "content", None) or ""
                    think = getattr(d, "reasoning_content", None) or getattr(d, "reasoning", None) or ""
                    if piece:
                        text.append(piece)
                    if piece or think:      # loops can sit in the thinking too: watch both, return content
                        seen.append(piece or think); n += len(piece or think)
                if n - last_check >= CHECK_EVERY:
                    last_check = n
                    if looping("".join(seen)):
                        STOPPED[0] += 1
                        print(f"[repguard] repetition loop stopped at {n} chars", flush=True)
                        text.append(NOTE)
                        break
        finally:
            stream.close()
        if usage is None:
            raise ValueError("No usage data received. Tracking tokens not possible.")
        self._track_cost(_usage_obj(usage), model)
        return "".join(text)

    async def acompletion(self, prompt, model=None):
        messages, model, extra = _prep(self, prompt, model)
        stream = await self.async_client.chat.completions.create(
            model=model, messages=messages, extra_body=extra, stream=True,
            stream_options={"include_usage": True, "continuous_usage_stats": True},
            **oc._normalize_sampling_args(self.sampling_args))
        text, seen, usage, last_check, n = [], [], None, 0, 0
        try:
            async for ch in stream:
                if getattr(ch, "usage", None) is not None:
                    usage = ch.usage
                if ch.choices:
                    d = ch.choices[0].delta
                    piece = getattr(d, "content", None) or ""
                    think = getattr(d, "reasoning_content", None) or getattr(d, "reasoning", None) or ""
                    if piece:
                        text.append(piece)
                    if piece or think:      # loops can sit in the thinking too: watch both, return content
                        seen.append(piece or think); n += len(piece or think)
                if n - last_check >= CHECK_EVERY:
                    last_check = n
                    if looping("".join(seen)):
                        STOPPED[0] += 1
                        print(f"[repguard] repetition loop stopped at {n} chars", flush=True)
                        text.append(NOTE)
                        break
        finally:
            await stream.close()
        if usage is None:
            raise ValueError("No usage data received. Tracking tokens not possible.")
        self._track_cost(_usage_obj(usage), model)
        return "".join(text)

    oc.OpenAIClient.completion = completion
    oc.OpenAIClient.acompletion = acompletion
