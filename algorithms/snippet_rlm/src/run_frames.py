"""PUBLICATION COPY (refactor header). Snippet RLM runner for FRAMES. Original file: examples/frames_eval_rlm.py.
Changed: import wiring (`harness`, `rlm_shared.browsecomp_data`, src/prompt_modules.py) and the FRAMES question file,
which is read from $FRAMES_DATA_DIR/frames_rows.json (configs/paths.example.env) instead of a hard-coded path.
Run by ../experiments/frames_snip.sh: `python src/run_frames.py --start Q --end Q+1 --max-iterations 50 --compaction
--out P`.

Our RLM harness (examples/browsecomp_eval_official_rlm.py) on FRAMES, over our BM25 index of the
paper's June 2023 Wikipedia (scripts/frames_search_server.py).

Nothing in the harness is changed: this swaps only the dataset accessors and the judge, then calls
its run_one. Set EMBEDDING_SERVER_URL to the BM25 server and the usual LOCAL_LLM_* / LIGHTON_* env
as for BrowseComp+ (see scripts/run_frames_rlm.sh).

The judge is the official BrowseComp grader prompt (the harness's own), run on the local model
because .env has no Azure key on this box -- the same judge as scripts/frames_eval.py, so the RLM
and the single-shot baselines are scored alike. Responses are stored for a gpt-5.4-mini re-judge.
"""
import argparse, json, os, re, sys, time
from pathlib import Path

# Refactor (import wiring): the original inserted the original repo root so that `examples.*` resolved. Here the
# harness sits next to this file (src/) and the shared data/grading modules in <repo>/shared (package rlm_shared).
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(_SRC_DIR))), "shared"))
sys.path.insert(0, _SRC_DIR)
import harness as H  # noqa: E402  (refactor: was `import examples.browsecomp_eval_official_rlm as H`)
from rlm_shared.browsecomp_data import _JUDGE_PROMPT, _final_answer_span, heuristic_match  # noqa: E402


def _rows_path() -> str:
    """Refactor: was the module constant ROWS = "<original repo>/data/frames/frames_rows.json"; now
    $FRAMES_DATA_DIR/frames_rows.json, read when main() loads the questions."""
    from rlm_shared.paths import path
    return str(path("FRAMES_DATA_DIR") / "frames_rows.json")

# --- Qwen3.6 format fixes, copied from scripts/run_obliq_mine.py (docs/HANDOFF.md Part I §F) -------
# rlm/utils/parsing.py runs only ```repl blocks. Qwen3.6-35B-A3B labels ~40% of turns ```python
# and answers ~18% in prose; on the FRAMES smoke test it looped on a prose "Exact Answer:" for
# 25 minutes. Both switches are NO-OPs unless set; do not set them for Qwen3.8-27B (100% ```repl).
#   FRAMES_REPL_CUE=1        one sentence appended to TURN_RULE naming the fence (the fix that
#                            took Qwen3.6 dead turns 57% -> 4% on OBLIQ)
#   FRAMES_FENCE_TOLERANT=1  run a mislabelled fence only if it parses as Python and does something
if os.environ.get("FRAMES_FENCE_TOLERANT", "") == "1":
    import ast as _ast
    import rlm.core.rlm as _RC  # the call site binds find_code_blocks in ITS namespace

    _orig_fcb = _RC.find_code_blocks
    _FENCE_RE = re.compile(r"```([A-Za-z0-9_+-]*)[ \t]*\n(.*?)\n```", re.DOTALL)
    _DOES_SOMETHING = (_ast.Assign, _ast.AnnAssign, _ast.AugAssign, _ast.Call,
                       _ast.Import, _ast.ImportFrom, _ast.For, _ast.While, _ast.If)

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
                tree = _ast.parse(b)
            except SyntaxError:
                continue
            if any(isinstance(n, _DOES_SOMETHING) for n in _ast.walk(tree)):
                out.append(b)
        return out

    _RC.find_code_blocks = _tolerant_find_code_blocks
    print("[fence] tolerant code-block parsing ON (```repl still preferred)", flush=True)

# --- BM25 retriever line --------------------------------------------------------------------
# The harness's RETRIEVER_LINE (default on) tells the agent the index is "a dense semantic
# retriever ... describe the thing in your own words", true of BrowseComp+'s ColBERT, false of the
# FRAMES BM25 index. Run with RETRIEVER_LINE=0 FRAMES_BM25_LINE=1 to state the BM25 fact instead,
# at the same place; the wording is shared with scripts/frames_sup_run_one.py (Full-Doc harness).
_BM25_LINE = (
    "The retriever is BM25 keyword search over English Wikipedia (one article per document, the "
    "whole article indexed): it scores documents by the words of your query, after stemming and "
    "stopword removal, not by their meaning, and quotation marks are ignored (a quoted phrase is "
    "not matched verbatim)."
)
if os.environ.get("FRAMES_BM25_LINE", "") == "1":
    import rlm.core.rlm as _core
    import rlm.utils.prompts as _PR

    assert "dense semantic retriever" not in _core.RLM_SYSTEM_PROMPT, "set RETRIEVER_LINE=0 with FRAMES_BM25_LINE=1"
    _marker = "If a search returns only documents you have already seen"
    assert _marker in _core.RLM_SYSTEM_PROMPT, "BM25-line anchor not found"
    _core.RLM_SYSTEM_PROMPT = _core.RLM_SYSTEM_PROMPT.replace(_marker, _BM25_LINE + " " + _marker, 1)
    _PR.RLM_SYSTEM_PROMPT = _core.RLM_SYSTEM_PROMPT
    print("[bm25] BM25 retriever line in system prompt", flush=True)

# --- search cap ------------------------------------------------------------------------------
# The harness has no search budget (it ran away to 50 searches/question, max 1,081, on FRAMES).
# FRAMES_SEARCH_CAP=N gives it the full-doc RLM's rule: after N calls search_index returns an error
# instead of results, and the system prompt states the budget in the full-doc RLM's words.
SEARCH_CAP = int(os.environ.get("FRAMES_SEARCH_CAP", "0") or 0)
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

if os.environ.get("FRAMES_REPL_CUE", "") == "1":
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("_opm", os.path.join(  # refactor: was <repo>/scripts/official_prompt_modules.py
        os.path.dirname(os.path.abspath(__file__)), "prompt_modules.py"))
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

# --- LM timeout ------------------------------------------------------------------------------
# run_frames_rlm.sh sets LOCAL_LM_TIMEOUT=1800 (as the Full-Doc RLM harness's --lm-timeout 1800),
# but neither the harness nor rlm reads it: every call ran on rlm's DEFAULT_TIMEOUT of 300 s, and
# long calls on a full context died with APITimeoutError (FRAMES q7). Honour it here.
if os.environ.get("LOCAL_LM_TIMEOUT"):
    import rlm.clients.base_lm as _BL

    _lm_timeout = float(os.environ["LOCAL_LM_TIMEOUT"])
    _orig_bl_init = _BL.BaseLM.__init__

    def _bl_init(self, *a, **kw):
        kw.setdefault("timeout", _lm_timeout)
        _orig_bl_init(self, *a, **kw)

    _BL.BaseLM.__init__ = _bl_init

H.get_question = lambda r: r["question"]
H.get_gold_answer = lambda r: r["answer"]
H.get_gold_doc_ids = lambda r: set(r["gold_ids"])
H.get_evidence_doc_ids = lambda r: set(r["gold_ids"])


def judge_match(question: str, response: str, gold: str) -> bool:
    if heuristic_match(gold, _final_answer_span(response)):
        return True
    from openai import OpenAI
    cli = OpenAI(base_url=os.environ.get("LOCAL_JUDGE_BASE_URL") or os.environ["LOCAL_LLM_BASE_URL"],
                 api_key="EMPTY", timeout=600)
    model = os.environ.get("LOCAL_JUDGE_MODEL", os.environ.get("LOCAL_LLM_MODEL"))
    for _ in range(3):
        try:
            c = cli.chat.completions.create(
                model=model, temperature=0, max_tokens=512,
                messages=[{"role": "user", "content": _JUDGE_PROMPT.format(
                    question=question, response=response, correct_answer=gold)}],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            m = re.search(r"correct:\s*\**\s*(yes|no)", c.choices[0].message.content or "", re.I)
            if m:
                return m.group(1).lower() == "yes"
        except Exception as exc:  # noqa: BLE001
            print(f"  [judge error: {exc}]", flush=True)
            time.sleep(5)
    return heuristic_match(gold, _final_answer_span(response))


H.judge_match = judge_match


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=1)
    p.add_argument("--max-iterations", type=int, default=100)
    p.add_argument("--max-depth", type=int, default=1)
    p.add_argument("--compaction", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    rows = json.load(open(_rows_path()))
    results = []
    for q in range(args.start, args.end):
        rec = H.run_one(rows[q], q, args)
        rec["reasoning_types"] = rows[q]["reasoning_types"]
        results.append(rec)
    Path(f"{args.out}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
