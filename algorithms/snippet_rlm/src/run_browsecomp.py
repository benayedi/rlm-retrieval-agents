"""PUBLICATION COPY (refactor header). Snippet RLM runner for BrowseComp+. Original file: scripts/run_bcp_snippet.py.
Changed: import wiring only (harness = examples/browsecomp_eval_official_rlm.py is imported as `harness`, the prompt
modules file is src/prompt_modules.py); every opt-in below is byte-identical. Run by ../experiments/bcp_snip.sh and
bcp_snip2.sh: `python src/run_browsecomp.py --start Q --end Q+1 --max-iterations N --max-depth 1 [--compaction] --out P`.

The official/snippet RLM harness (examples/browsecomp_eval_official_rlm.py) on BrowseComp+ with three opt-in patches
for a local Qwen3.6-35B-A3B (2026-10-27), copied from scripts/run_obliq_mine.py, where they were measured:
  LOCAL_LM_TIMEOUT        per-LM-call timeout (default 1800 s; the library's 300 s killed long questions as 'ERROR')
  QWEN_PRESENCE_PENALTY   presence_penalty for root and sub calls (Qwen3.6 model card: 1.5 in thinking mode)
  QWEN_FENCE_TOLERANT=1   run a code block labelled ```python or bare ``` when no ```repl block exists (Qwen3.6
                          mislabels ~40% of turns); executed only if it parses and does something
  BCP_SEARCH_CAP=N        hard budget of N search_index calls, as FRAMES_SEARCH_CAP in examples/frames_eval_rlm.py
                          (same refusal text and system-prompt sentence); Qwen3.6 ran 100-300 searches per question
  QWEN_REPL_CUE=1         FRAMES's repl-fence cue sentence in TURN_RULE (2026-10-28, see below)
  QWEN_MAX_TOKENS=N       output-token cap per model call, root and sub-LLM (2026-10-28: 8192; Qwen3.6 spent whole 16k-token
                          turns listing candidates with no code, 12-13 min each)
  QWEN_PROSE_FINAL=1      a code-less turn with an "Exact Answer:" line is submitted as the answer (2026-10-28, see below)
Unset, nothing changes: GPT runs go through this file byte-identical to calling the harness directly.
Arguments are the harness's own (--start/--end/--max-iterations/--max-depth/--out).
"""
import ast
import os
import re
import sys

# Refactor (import wiring): the original inserted the original repo root so that `examples.*` resolved. Here the
# harness sits next to this file (src/) and the shared data/grading modules in <repo>/shared (package rlm_shared).
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(_SRC_DIR))), "shared"))
sys.path.insert(0, _SRC_DIR)
import rlm as _rlm  # noqa: E402

_LM_TIMEOUT = float(os.environ.get("LOCAL_LM_TIMEOUT", "1800"))
_orig_rlm_init = _rlm.RLM.__init__


def _rlm_init_patched(self, *a, **kw):
    bk = kw.get("backend_kwargs")
    if isinstance(bk, dict) and "timeout" not in bk and os.environ.get("LOCAL_LLM_BASE_URL"):
        kw["backend_kwargs"] = {**bk, "timeout": _LM_TIMEOUT}
    if os.environ.get("QWEN_MAX_TOKENS"):   # 2026-10-28: output cap per call, root and sub (vLLM default here: 16384)
        mt = int(os.environ["QWEN_MAX_TOKENS"])
        for key in ("sampling_args", "sub_sampling_args"):
            kw[key] = {**(kw.get(key) or {}), "max_tokens": mt}
    if os.environ.get("QWEN_PRESENCE_PENALTY"):
        pp = float(os.environ["QWEN_PRESENCE_PENALTY"])
        for key in ("sampling_args", "sub_sampling_args"):
            kw[key] = {**(kw.get(key) or {}), "presence_penalty": pp}
    return _orig_rlm_init(self, *a, **kw)


_rlm.RLM.__init__ = _rlm_init_patched

if os.environ.get("QWEN_FENCE_TOLERANT", "") == "1":
    import rlm.core.rlm as _RC  # the call site binds find_code_blocks in ITS namespace

    _orig_fcb = _RC.find_code_blocks
    _FENCE_RE = re.compile(r"```([A-Za-z0-9_+-]*)[ \t]*\n(.*?)\n```", re.DOTALL)
    _DOES_SOMETHING = (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Call,
                       ast.Import, ast.ImportFrom, ast.For, ast.While, ast.If)

    def _tolerant_find_code_blocks(text):
        blocks = _orig_fcb(text) if text else []
        if blocks or not text:
            return blocks
        out = []
        for lang, body in _FENCE_RE.findall(text):
            b = body.strip()
            if lang.lower() == "repl" or not b:
                continue
            try:
                tree = ast.parse(b)
            except SyntaxError:
                continue
            if any(isinstance(n, _DOES_SOMETHING) for n in ast.walk(tree)):
                out.append(b)
        return out

    _RC.find_code_blocks = _tolerant_find_code_blocks
    print("[fence] tolerant code-block parsing ON (```repl still preferred)", flush=True)

import harness as H  # noqa: E402  (refactor: was `from examples import browsecomp_eval_official_rlm as H`)

# --- 2026-10-28: two more Qwen3.6 opt-ins, after the first BrowseComp+ run (bcp_snip_qwen_bm25, no cue): a third of all
# turns were dead turns, and 45% of all turns came after the search budget was spent -- the model wrote
# "Explanation: ... Exact Answer: X" as prose with no ```repl block, the harness never saw answer["ready"], and the same
# answer was repeated for up to 40 turns of 1-5 min each until the 2 h kill (32% of questions, no result). With the FRAMES
# cue on, the FRAMES snippet run had 12% dead turns and 1.1 turns after exhaustion.
#   QWEN_REPL_CUE=1     the FRAMES_REPL_CUE sentence (examples/frames_eval_rlm.py) appended to TURN_RULE through
#                       scripts/official_prompt_modules.py; applied after the harness's own prompt patches, before the cap
#   QWEN_PROSE_FINAL=1  a turn with NO code block whose text has an "Exact Answer:" line is taken as the submission: the
#                       response (from its last "Explanation:" on, after any </think>) is assigned to answer["content"]
#                       with answer["ready"] = True, as the model should have done inside ```repl```. The value after
#                       "Exact Answer:" must be non-empty: on smoke q7 the first turn was the bare template
#                       ("Exact Answer: \nConfidence: 0%") and was submitted as the answer
if os.environ.get("QWEN_REPL_CUE", "") == "1":
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("_opm", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                              "prompt_modules.py"))  # refactor: renamed file
    _OPM = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_OPM)
    _OPM.assert_roundtrip()
    _blocks = _OPM.split()
    _blocks["TURN_RULE"] = _blocks["TURN_RULE"].rstrip() + (
        " The fence label matters: write ```repl, never ```python and never a bare ```. "
        "A turn whose code is not in a ```repl``` block is discarded and its work is "
        "lost -- including the final turn, where answer[\"content\"] and "
        "answer[\"ready\"] = True must be assigned inside ```repl```.")
    _OPM.apply(_blocks)
    print("[cue] repl-fence cue appended to TURN_RULE", flush=True)

if os.environ.get("QWEN_PROSE_FINAL", "") == "1":
    import rlm.core.rlm as _RC3

    _prev_fcb = _RC3.find_code_blocks
    _EXACT_RE = re.compile(r"Exact Answer\s*:[ \t]*\S", re.I)   # a non-empty value: Qwen3.6 also writes the empty template

    def _prose_final_find_code_blocks(text):
        blocks = _prev_fcb(text) if text else []
        if blocks or not text or not _EXACT_RE.search(text):
            return blocks
        body = text.rsplit("</think>", 1)[-1]
        i = body.lower().rfind("explanation:")
        content = body[i:].strip() if i >= 0 else "Explanation: (given above)\n" + body.strip()
        print("[prose-final] no code block but an 'Exact Answer:' line: submitting the prose answer", flush=True)
        return [f"answer[\"content\"] = {content!r}\nanswer[\"ready\"] = True"]

    _RC3.find_code_blocks = _prose_final_find_code_blocks
    print("[prose-final] prose 'Exact Answer:' turns are submitted as the answer", flush=True)

SEARCH_CAP = int(os.environ.get("BCP_SEARCH_CAP", "0") or 0)
if SEARCH_CAP:
    import rlm.core.rlm as _core2
    import rlm.utils.prompts as _PR2

    _orig_make_tools = H.make_tools

    def _capped_make_tools(state):
        search_index, get_document = _orig_make_tools(state)

        def capped_search_index(query: str, top_k: int = H.DEFAULT_TOP_K) -> list[dict]:
            if state["search_calls"] >= SEARCH_CAP:
                state["search_refused"] = state.get("search_refused", 0) + 1
                return [{"doc_id": "", "score": 0.0, "text": (
                    f"ERROR: search budget exhausted ({SEARCH_CAP} of {SEARCH_CAP} calls used). No results. "
                    "Answer from the documents and snippets you already have.")}]
            return search_index(query, top_k)

        capped_search_index.__doc__ = search_index.__doc__
        capped_search_index.__name__ = "search_index"
        return capped_search_index, get_document

    H.make_tools = _capped_make_tools
    _marker = "If a search returns only documents you have already seen"
    assert _marker in _core2.RLM_SYSTEM_PROMPT, "search-cap anchor not found"
    _core2.RLM_SYSTEM_PROMPT = _core2.RLM_SYSTEM_PROMPT.replace(_marker, (
        f"You have a hard budget of {SEARCH_CAP} search_index calls for this question; calls beyond that "
        "are refused and return an error instead of results, so plan the queries you spend them on. " + _marker), 1)
    _PR2.RLM_SYSTEM_PROMPT = _core2.RLM_SYSTEM_PROMPT
    print(f"[cap] search_index capped at {SEARCH_CAP} calls", flush=True)

if __name__ == "__main__":
    H.main()
