"""
RAG RLM agent built on dspy.RLM with FAISS retrieval and delegation.

Usage:
    from rag import build_lm, build_rlm_agent, FaissRetriever, make_embedding_fn

    lm = build_lm()
    dspy.configure(lm=lm)

    retriever = FaissRetriever(index_pattern, embedding_fn)
    rlm, metrics = build_rlm_agent(retriever)

    result = rlm(question="Who invented the telephone?")
    print(result.answer)
    print(metrics)
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
from dataclasses import dataclass, field

import dspy
import requests as _requests
from dotenv import load_dotenv
from dspy.utils.usage_tracker import track_usage

from .signatures import BrowseCompSignature
from .tools import RemoteRetriever as FaissRetriever

load_dotenv()


@dataclass
class RunMetrics:
    """Per-question metrics accumulated across all REPL turns and delegations."""

    latency_s: float = 0.0
    retrieval_latency_s: float = 0.0  # time spent inside search_index + get_document calls
    # Phase split for profiling. latency_s is wall-clock for the whole question;
    # lm_latency_s is time blocked on the Azure API, retrieval_latency_s is the
    # embedding server + FAISS, and the residual is Deno REPL exec + DSPy overhead.
    # Recorded per LM call so a question with 12 turns attributes all 12.
    lm_latency_s: float = 0.0
    lm_calls: int = 0
    lm_call_latencies: list = field(default_factory=list)
    # Prefill vs decode. Azure reports how much of each prompt was served from its
    # prefix cache; a high ratio means prefill is cheap and the cost is decode, a
    # low one means we are re-paying for the same prefix every turn. Without this
    # the phase split cannot say WHICH half of the API time is addressable.
    cached_input_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    iterations: int = 0
    delegation_calls: int = 0
    search_calls: int = 0
    get_document_calls: int = 0
    llm_query_calls: int = 0
    rlm_query_calls: int = 0
    retrieved_doc_ids: list = field(default_factory=list)
    fetched_docs: dict = field(default_factory=dict)  # doc_id → text for get_document calls
    # The REPL transcript dspy.RLM already returns: the code the agent wrote each
    # turn and what it printed. Without this a finished run reports a score but no
    # evidence of what the strategy actually did.
    trajectory: list = field(default_factory=list)
    _query_cache: dict = field(default_factory=dict)
    _gold_ids: set = field(default_factory=set)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def unique_docs_retrieved(self) -> int:
        return len(set(self.retrieved_doc_ids))

    def gold_recall(self, gold_ids: list[str]) -> float:
        if not gold_ids:
            return 0.0
        retrieved = set(self.retrieved_doc_ids)
        return sum(1 for g in gold_ids if g in retrieved) / len(gold_ids)

    def __repr__(self) -> str:
        return (
            f"RunMetrics(lat={self.latency_s:.1f}s, "
            f"tok={self.input_tokens}/{self.output_tokens}, "
            f"iters={self.iterations}, "
            f"search={self.search_calls}, getdoc={self.get_document_calls}, "
            f"llmq={self.llm_query_calls}, deleg={self.delegation_calls}, "
            f"docs={self.unique_docs_retrieved})"
        )


@contextlib.contextmanager
def time_lm_calls(metrics: "RunMetrics", lm=None):
    """Attribute a question's wall-clock across API / retrieval / REPL.

    dspy.RLM is a black box that alternates Azure calls with Deno REPL execution,
    so total latency alone cannot say which dominates. Wrapping the LM's __call__
    gives the API share; retrieval is timed in the tool wrappers; the residual is
    REPL + DSPy overhead.

    Purely additive: the original method is restored on exit, and if the patch
    cannot be installed we degrade to lm_latency_s == 0 rather than failing a run.
    """
    lm = lm or dspy.settings.lm
    if lm is None:
        yield
        return
    cls = type(lm)
    orig = cls.__call__
    # Patching the class also catches the sub-LM behind llm_query, and dspy retries
    # by re-entering __call__. Charge only the outermost call, or nested calls get
    # billed twice and lm_latency_s can exceed wall-clock.
    depth = [0]

    def timed(self, *a, **kw):
        if depth[0]:
            return orig(self, *a, **kw)
        depth[0] = 1
        t = time.perf_counter()
        try:
            return orig(self, *a, **kw)
        finally:
            depth[0] = 0
            d = time.perf_counter() - t
            metrics.lm_latency_s += d
            metrics.lm_calls += 1
            metrics.lm_call_latencies.append(round(d, 3))
            try:
                usage = (self.history[-1] or {}).get("usage") or {}
                det = usage.get("prompt_tokens_details")
                metrics.cached_input_tokens += int(
                    getattr(det, "cached_tokens", None)
                    or (det or {}).get("cached_tokens", 0) or 0)
            except Exception:
                pass

    try:
        cls.__call__ = timed
    except Exception:
        yield
        return
    try:
        yield
    finally:
        cls.__call__ = orig


_SNIPPET_TOKENIZER = None


def _get_snippet_tokenizer():
    """Qwen/Qwen3-0.6B — the tokenizer the BrowseComp+ harness uses for snippet limits."""
    global _SNIPPET_TOKENIZER
    if _SNIPPET_TOKENIZER is None:
        from transformers import AutoTokenizer
        _SNIPPET_TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    return _SNIPPET_TOKENIZER


def build_lm() -> dspy.LM:
    """Configure the LM. Azure by default; LOCAL_LLM_BASE_URL switches to a local
    OpenAI-compatible server (our vLLM Qwen3.8-27B) with NO other change.

    WHY. The recorded OBLIQ numbers come from THIS harness with THIS signature on gpt-5.2.
    Running the same harness and signature on Qwen is the only comparison that isolates the
    MODEL -- our other two OBLIQ arms changed the harness as well, so they cannot separate
    "different model" from "different scaffold". litellm reaches an OpenAI-compatible
    endpoint as model="openai/<served-name>" plus api_base; no Azure credentials are read.

    UNSET LOCAL_LLM_BASE_URL AND THIS FUNCTION IS UNCHANGED.
    """
    _local = os.environ.get("LOCAL_LLM_BASE_URL", "").strip()
    if _local:
        _name = os.environ.get("LOCAL_LLM_MODEL", "qwen3.8-27b-bf16")
        return dspy.LM(
            model=f"openai/{_name}",
            api_base=_local,
            api_key=os.environ.get("LOCAL_LLM_API_KEY", "EMPTY"),
            temperature=1.0,
            max_tokens=int(os.environ.get("LOCAL_LLM_MAX_TOKENS", "16000")),
            cache=False,
            # Qwen3.8 rejects reasoning_effort="high" with 400; medium is what every local
            # arm in this project uses.
            reasoning_effort=os.environ.get("LOCAL_LLM_EFFORT", "medium"),
            timeout=float(os.environ.get("LOCAL_LM_TIMEOUT", "1800")),
            # Qwen3.6 model card: presence_penalty 1.5 in thinking mode against endless repetition
            # (as OBLIQ_PRESENCE_PENALTY in scripts/run_obliq_mine.py). Unset = not sent.
            **({"presence_penalty": float(os.environ["LOCAL_LLM_PRESENCE_PENALTY"])}
               if os.environ.get("LOCAL_LLM_PRESENCE_PENALTY") else {}),
        )
    api_key = os.environ["AZURE_OPENAI_API_KEY"]
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    model = os.environ.get("AZURE_OPENAI_MODEL", "gpt-5.4-mini")

    kwargs = dict(
        model=f"azure/{model}",
        api_base=endpoint,
        api_key=api_key,
        api_version=api_version,
        temperature=1.0,
        max_tokens=int(os.environ.get("AZURE_OPENAI_MAX_TOKENS", "16000")),
        cache=False,
        timeout=int(os.environ.get("AZURE_OPENAI_TIMEOUT", "120")),
    )
    reasoning_effort = os.environ.get("AZURE_OPENAI_REASONING_EFFORT", "")
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    return dspy.LM(**kwargs)


def build_rlm_agent(
    retriever: FaissRetriever,
    depth: int = 0,
    max_depth: int = 5,
    max_iterations: int = 16,
    max_search_calls: int = 35,
    metrics: RunMetrics | None = None,
    verbose: bool = False,
    default_top_k: int = 10,
    signature=None,
) -> tuple[dspy.RLM, RunMetrics]:
    """
    Build a dspy.RLM agent with FAISS retrieval and delegation tools.

    Args:
        retriever:      Loaded FaissRetriever instance (shared across delegations).
        depth:          Current recursion depth (0 = root agent).
        max_depth:      Maximum delegation depth.
        max_iterations: Max REPL iterations per agent call.
        metrics:        Shared RunMetrics (created fresh if None at depth=0).
        verbose:        Enable dspy.RLM verbose logging.
        default_top_k:  Default number of results returned by search_index.

    Returns:
        (rlm, metrics) — call rlm(question=...) to run.
    """
    if metrics is None:
        metrics = RunMetrics()

    # ------------------------------------------------------------------ #
    # Tools                                                                #
    # ------------------------------------------------------------------ #

    def search_index(query: str, top_k: int = default_top_k) -> list[dict]:
        """Search the BrowseComp+ corpus for relevant passages.

        Returns list of {score: float, doc_id: str, text: str (≤2000 chars)}.
        Search for specific named entities, titles, or proper nouns.
        Tip: scores below 0.30 are usually off-topic — try a different query.
        """
        # HARD CLAMP. --top-k only sets the DEFAULT parameter; the agent overrides it in
        # its own code and does so constantly — 69% of searches in the .7880 run requested
        # 20 results and 1.5% requested 50, against our nominal 10. That matters because
        # dspy.RLM truncates each code block's stdout at max_output_chars (default 10,000,
        # head+tail, middle silently deleted) and a fully-printed result costs ~557 chars at
        # SEARCH_SNIPPET_LEN=500 — so top_k>=18 overflows and the agent loses the middle
        # ranks of its own search. MAX_TOP_K makes the ceiling real.
        _cap = os.environ.get("MAX_TOP_K", "")
        if _cap and top_k > int(_cap):
            top_k = int(_cap)

        # MAX_SEARCH_PER_TURN — cap searches per REPL code block. Default OFF.
        # THE BAND IS THE POINT. Per-question, grouped by the MOST searches issued in any single
        # turn (snipmin, q0-99):
        #     max 1/turn   n=57  acc .877     189k input tok    5.7 searches
        #     2-4/turn     n=17  acc .941     319k input tok   12.0 searches   <- BEST group
        #     5+/turn      n=26  acc .654   1,195k input tok   38.5 searches   <- the bloat
        # So a cap of 1 would destroy the best-performing regime; the cap belongs at 4.
        # (Causation is unproven and probably partly reversed -- the model batches BECAUSE it is
        # stuck -- but 5+ is where cost explodes, and that part is not ambiguous.)
        # NOT A PAPER FEATURE. The RLM paper's "symbolic recursion" is about batching SUB-LM
        # calls: "invoke M on programmatically constructed transformations of P (e.g., inside
        # arbitrarily large loops)", and their own system prompt tells the model to batch
        # llm_query to reduce runtime cost. Batching llm_query keeps content OUT of history;
        # batching SEARCH pours it in. The paper never discusses search batching -- their setting
        # has the prompt in a variable and no retrieval tool.
        # WHY. Hosted MCP cannot batch: all 1,645 of its tool-call bursts were exactly one call,
        # because each tool result is a message and the model must reason before the next call.
        # Our search_index returns a list, so the model writes `for q in queries: search_index(q)`
        # and issues up to 20 in a single block. Measured: only 12% of turns batch, but those
        # turns produce every block over 25,000 chars and 58-63% of the entire token bill.
        # Since repl_history re-serialises every prior print into every later prompt, a batched
        # turn is paid for on all remaining turns -- cost is (accumulated prints) x (turns left).
        # TURN DETECTION. The tool runs host-side and cannot see dspy's iteration counter, but the
        # boundary is unambiguous in time: searches inside one code block land ~110ms apart
        # (verified in logs/snipmin_s0.log: 8 searches at 103-129ms), while searches in different
        # turns are separated by a full LM call, ~30s. A 5s gap therefore separates turns cleanly.
        _spt = os.environ.get("MAX_SEARCH_PER_TURN", "")
        if _spt:
            _n_cap = int(_spt); _now = time.perf_counter()
            _last = getattr(metrics, "_last_search_t", None)
            if _last is None or (_now - _last) > 5.0:
                metrics._turn_search_n = 0          # new turn
            metrics._last_search_t = _now
            metrics._turn_search_n = getattr(metrics, "_turn_search_n", 0) + 1
            if metrics._turn_search_n > _n_cap:
                print(f"    [search REFUSED — per-turn cap {_n_cap}] {query[:60]!r}")
                return [{"score": 0.0, "doc_id": "", "text": (
                    f"[SEARCH-PER-TURN LIMIT ({_n_cap}) REACHED IN THIS CODE BLOCK. Print and read "
                    "the results you already have, reason about them, then search again in your "
                    "NEXT code block. Do not loop over long lists of queries.")}]

        if metrics.search_calls >= max_search_calls:
            print(f"    [search BLOCKED — limit {max_search_calls} reached] {query[:60]!r}")
            return [{"score": 0.0, "doc_id": "", "text": (
                f"[SEARCH DISABLED — you have used all {max_search_calls} search calls. "
                "You must call SUBMIT(answer=...) immediately with your best answer. "
                "Do not call search_index again."
            )}]

        if query in metrics._query_cache:
            print(f"    [search CACHED] {query[:60]!r}")
            return metrics._query_cache[query]

        t0 = time.perf_counter()
        try:
            results = retriever.search_index(query, top_k=top_k)
        except (_requests.exceptions.ConnectionError, _requests.exceptions.Timeout) as e:
            print(f"\n[ABORT] Embedding server unreachable during search: {e}")
            sys.exit(1)
        elapsed = time.perf_counter() - t0
        metrics.retrieved_doc_ids.extend(r["doc_id"] for r in results)
        # SNIPPET TRUNCATION. SEARCH_SNIPPET_TOKENS truncates by TOKENS using the same
        # tokenizer the BrowseComp+ harness uses (Qwen/Qwen3-0.6B, searcher/tools.py:39-41),
        # so "512" here means exactly what --snippet-max-tokens 512 means to LightOn.
        # Falls back to the character-based SEARCH_SNIPPET_LEN when unset, so every existing
        # run reproduces byte-for-byte.
        _tok_cap = os.environ.get("SEARCH_SNIPPET_TOKENS", "")
        if _tok_cap:
            _tk = _get_snippet_tokenizer()
            n = int(_tok_cap)
            for r in results:
                if "text" in r and r["text"]:
                    ids = _tk.encode(r["text"], add_special_tokens=False)
                    if len(ids) > n:
                        r["text"] = _tk.decode(ids[:n], skip_special_tokens=True)
        else:
            snippet_len = int(os.environ.get("SEARCH_SNIPPET_LEN", "500"))
            for r in results:
                if "text" in r and len(r["text"]) > snippet_len:
                    r["text"] = r["text"][:snippet_len] + "…"
        metrics.retrieval_latency_s += elapsed
        metrics.search_calls += 1
        metrics._query_cache[query] = results
        doc_ids = [r["doc_id"] for r in results]
        gold_hits = sum(1 for d in doc_ids if d in metrics._gold_ids)
        hit_str = f"  ★{gold_hits}" if gold_hits else ""
        print(f"    [search #{metrics.search_calls}] {query[:60]!r}  → {len(results)} docs{hit_str}  ({elapsed*1000:.0f}ms)")
        return results

    def get_document(doc_id: str) -> dict:
        """Fetch the full text of a document."""
        t0 = time.perf_counter()
        result = retriever.get_document(doc_id)
        elapsed = time.perf_counter() - t0
        metrics.retrieval_latency_s += elapsed
        metrics.get_document_calls += 1
        metrics.fetched_docs[doc_id] = result.get("text", "")
        metrics.retrieved_doc_ids.append(doc_id)
        text = result.get("text", "") or ""
        # WHERE THE READING HAPPENS. Previously this returned text into a Python variable
        # and nothing more, so the reasoning model saw the document only through whatever
        # question the agent chose to ask llm_query about it. If that framing missed, the
        # answer was unrecoverable — the model never saw the raw text and moved on. That is
        # the delegation boundary behind the 57 questions answered "Unknown" despite the
        # gold document having been retrieved.
        #
        # Gold documents are SHORT: p50 1,297 tokens (~5k chars), and 75.9% of answers sit
        # within the first 512 tokens. So printing documents under the cap puts the
        # majority of answer-bearing text directly into the reasoning model's context —
        # which is what the BrowseComp+ harness does by default, since its tool results are
        # appended to the conversation. Long documents still go through llm_query, because
        # printing a 90k-token outlier would blow the context that makes this work.
        cap = int(os.environ.get("GETDOC_INLINE_CHARS", "6000"))
        inline = len(text) <= cap
        print(f"    [getdoc #{metrics.get_document_calls}] {doc_id}  "
              f"({len(text)} chars, {'INLINE' if inline else 'too long — use llm_query'})"
              f"  ({elapsed*1000:.0f}ms)")
        if inline and text:
            print(f"--- BEGIN {doc_id} ---\n{text}\n--- END {doc_id} ---")
        return result

    # DISABLE_GET_DOCUMENT removes the whole-document reader from the tool list, which is
    # the exact ablation the BrowseComp-Plus paper runs in Table 5 and that the leaderboard's
    # LightOn/Mixedbread pairs differ by. Three groups independently measure the tool as
    # worth +8.07 / +10.48 / +8.19 pp. This system already HAS it (1.93 calls/question) and
    # scores at their NO-get_doc level, which is the whole 8.8pp gap — so the open question
    # is whether our reading contributes anything at all. Default off; existing runs unchanged.
    # THE DOCSTRING IS THE BINDING CONSTRAINT ONCE INLINE PRINTING WORKS (2026-09-06).
    # Measured mid-arm: with GETDOC_INLINE_CHARS=30000, 100% of 58 fetched documents were
    # printed IN FULL into the reasoning model's context (median 4,500 chars, max 21,035) —
    # and llm_query STILL fired 1.84x/question, on text the model already had in front of it.
    # The old docstring ended "Long documents are not printed; summarise those with llm_query",
    # which offers delegation as the reading path even when nothing needs delegating. It is
    # advisory and the model followed it out of habit. So the docstring is now built from the
    # ACTUAL cap: when everything is printed, it does not mention llm_query at all.
    _cap_now = int(os.environ.get("GETDOC_INLINE_CHARS", "6000"))
    if _cap_now > 0:
        get_document.__doc__ = (
            "Fetch a document and PRINT ITS FULL TEXT into your context.\n\n"
            f"Documents up to {_cap_now:,} characters are printed in full — the text is "
            "already in front of you. READ IT YOURSELF. Do NOT pass it to llm_query: the "
            "text is present, delegating it only loses detail and wastes a call.\n\n"
            "Args:\n    doc_id: The doc_id string from a search_index result.\n"
            "Returns:\n    {doc_id: str, text: str} — the same text that was printed."
        )
    else:
        get_document.__doc__ = (
            "Fetch the full text of a document.\n\n"
            "Args:\n    doc_id: The doc_id string from a search_index result.\n"
            "Returns:\n    {doc_id: str, text: str}. The text is NOT printed; pass it to "
            "llm_query to read it."
        )

    # MINIMAL_TOOL_DOCS — LightOn/BrowseComp+ tool descriptions, verbatim except for
    # parameter names. Theirs are in bcp/searcher/searchers/base.py:57-67:
    #     search:       "Perform a search on a knowledge source. Returns top-{k} hits with
    #                    docid, score, and snippet. The snippet contains the document's
    #                    contents (may be truncated based on token limits)."
    #     get_document: "Retrieve a full document by its docid."
    # Nine words for get_document, and NO reading advice at all — in a tool-calling loop the
    # document arrives as a message, so there is no print-or-delegate decision to describe.
    # Ours carry three untested priors this arm removes: "search for specific named entities,
    # titles, or proper nouns", "scores below 0.30 are usually off-topic", and the whole
    # reading policy. Section 32 measured that the docstring has real behavioural pull
    # (llm_query fired 1.84x/question on text already in context purely because the docstring
    # mentioned it), so this is a live variable, not cosmetics.
    # CAVEAT: this is NOT a like-for-like copy. Their nine words work because fetching IS
    # reading on their side. Here the text lands in a Python variable, so removing the
    # "pass it to llm_query" line takes away information the REPL model actually needs.
    # A drop is therefore an expected outcome, not a refutation of the minimal style.
    if os.environ.get("MINIMAL_TOOL_DOCS", "").strip() not in ("", "0", "false", "False"):
        search_index.__doc__ = (
            "Perform a search on a knowledge source. Returns top-"
            f"{default_top_k} hits with doc_id, score, and text. The text contains the "
            "document's contents (may be truncated based on token limits)."
        )
        get_document.__doc__ = "Retrieve a full document by its doc_id."

    tools = [search_index, get_document]
    if os.environ.get("DISABLE_GET_DOCUMENT", "").strip() not in ("", "0", "false", "False"):
        tools = [search_index]
        print("    [get_document DISABLED — search-only ablation]")

    # WHICH MODEL READS THE DOCUMENTS. This resolved to AZURE_OPENAI_JUDGE_MODEL, which
    # .env sets to gpt-5.4-mini — so in every run to date the agent planned with gpt-5 and
    # handed every fetched document to a MINI model to read. Measured on the 830-question
    # .7880 run: 680 of the 694 questions that fetched a document also called llm_query,
    # i.e. essentially all reading was done by the small model. That variable was never
    # deliberately chosen; it fell out of sharing one env var with the grader.
    # SUB_LM_MODEL now separates the two so the reader can be set independently of the
    # judge. Default is unchanged, so existing runs reproduce exactly.
    judge = os.environ.get("SUB_LM_MODEL",
                           os.environ.get("AZURE_OPENAI_JUDGE_MODEL",
                                          os.environ.get("AZURE_OPENAI_MODEL", "gpt-5.4-mini")))
    # The sub-LM behind llm_query is where DOCUMENTS ARE ACTUALLY READ: the agent fetches
    # a document, hands it here, and only the returned summary reaches the reasoning
    # model. It was configured strictly weaker than the model doing the planning —
    # no reasoning_effort at all (the main LM gets it from the environment), a 4k output
    # cap against the main LM's 16k, and a 60s timeout against 120s. That is backwards:
    # recognition is the step the whole benchmark turns on, and 57 of 830 questions were
    # answered "Unknown" DESPITE the gold document having been retrieved.
    _sub_kwargs = dict(
        model=f"azure/{judge}",
        api_base=os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/"),
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        temperature=1.0, cache=False,
        # Overridable so a run can reproduce the pre-2026-09-01 sub-LM exactly
        # (4000/60) and isolate a single variable against the .7880 baseline.
        max_tokens=int(os.environ.get("SUB_LM_MAX_TOKENS", "8000")),
        timeout=int(os.environ.get("SUB_LM_TIMEOUT", "120")),
    )
    _sub_effort = os.environ.get("SUB_LM_REASONING_EFFORT",
                                 os.environ.get("AZURE_OPENAI_REASONING_EFFORT", ""))
    if _sub_effort:
        _sub_kwargs["reasoning_effort"] = _sub_effort
    base_sub_lm = dspy.LM(**_sub_kwargs)
    # Count sub-LM calls by swapping the CLASS, not by assigning __call__ on the
    # instance: Python resolves obj() via the type, so an instance-level __call__
    # is never consulted and the old version silently counted nothing. Every
    # BrowseComp+ run before this fix reports llm_query_calls == 0 regardless of
    # actual usage. obliq_eval_dspy_rlm already used the __class__ approach, which is why
    # its llmq= figures are real.
    class _CountedLM(type(base_sub_lm)):
        def __call__(self, *args, **kwargs):
            metrics.llm_query_calls += 1
            print(f"    [llm_query #{metrics.llm_query_calls}]")
            return super().__call__(*args, **kwargs)

    base_sub_lm.__class__ = _CountedLM

    sig = signature if signature is not None else BrowseCompSignature
    # RLM_MAX_OUTPUT_CHARS caps each REPL block's stdout. dspy's default is 10,000, applied
    # head+tail with the MIDDLE SILENTLY DELETED and no signal to the agent. We never set it.
    # Measured on 10 questions at baseline settings (logs/trunc_probe.log): 15 truncation
    # events across 82 blocks — 18.3% — deleting a median 1,142 and a maximum 10,726 chars,
    # 1.5 times per question. Raising it is the RIGHT way to remove that defect: clamping
    # top_k attacked the same problem by cutting retrieval breadth and cost 9pp (section 33).
    _moc = os.environ.get("RLM_MAX_OUTPUT_CHARS", "")
    _rlm_kwargs = dict(
        signature=sig,
        tools=tools,
        max_iters=max_iterations,   # dspy >=3.3 renamed max_iterations
        verbose=verbose,
        sub_lm=base_sub_lm,
    )
    if _moc:
        _rlm_kwargs["max_output_chars"] = int(_moc)
        print(f"    [max_output_chars = {int(_moc):,} (dspy default is 10,000)]")
    rlm = dspy.RLM(**_rlm_kwargs)

    return rlm, metrics


def run_question(
    retriever: FaissRetriever,
    question: str,
    max_depth: int = 5,
    max_iterations: int = 16,
    max_search_calls: int = 35,
    verbose: bool = False,
    default_top_k: int = 10,
    gold_ids: set | None = None,
    signature=None,
) -> tuple[str, RunMetrics]:
    """
    Convenience wrapper: build a fresh agent, run one question, return (answer, metrics).

    Tracks latency and token usage from the DSPy LM history.
    """
    rlm, metrics = build_rlm_agent(
        retriever=retriever,
        max_depth=max_depth,
        max_iterations=max_iterations,
        max_search_calls=max_search_calls,
        verbose=verbose,
        default_top_k=default_top_k,
        signature=signature,
    )
    if gold_ids:
        metrics._gold_ids = set(gold_ids)

    with track_usage() as tracker, time_lm_calls(metrics, getattr(rlm, "lm", None)):
        t0 = time.time()
        try:
            result = rlm(question=question)
        finally:
            metrics.latency_s = time.time() - t0

    for usage in tracker.get_total_tokens().values():
        metrics.input_tokens += usage.get("prompt_tokens", 0)
        metrics.output_tokens += usage.get("completion_tokens", 0)

    # Count REPL iterations from trajectory
    trajectory = getattr(result, "trajectory", None) or {}
    # dspy.RLM stores trajectory as dict with iteration keys
    if isinstance(trajectory, dict):
        metrics.iterations = len([k for k in trajectory if k.startswith("iteration_")])
    elif hasattr(trajectory, "__len__"):
        metrics.iterations = len(trajectory)

    answer = getattr(result, "answer", str(result))
    return str(answer), metrics
