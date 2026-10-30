"""Opt-in: accept a code block the model labelled with the wrong fence. SUP_FENCE_TOLERANT=1.

Ported unchanged from RLM/scripts/run_obliq_mine.py (OBLIQ_FENCE_TOLERANT), where it was
validated 7/7 on unit cases and rescued 67% of Qwen3.6's real dead turns. rlm/utils/parsing.py
matches ONLY ```repl; Qwen3.6-35B-A3B labels ~40% of its turns ```python or a bare ``` -- it
writes correct code and loses the turn on the label.

SAFETY: a non-repl fence is executed only if it parses as Python AND does something
(assignment / call / import / loop / if). That excludes the answer template the model also
emits in bare fences ("Explanation: ... Exact Answer: ..."), which must never be run.
```repl always wins when present, so the default path is untouched. Unset = no effect.
"""
import ast
import os
import re

RESCUED = [0]


def _install_block_cap(RC) -> None:
    """SUP_MAX_BLOCKS=N: execute at most N code blocks per turn (opt-in).

    Every block in a turn is executed and each block's output is truncated at 20k chars, but
    the TOTAL is not capped. Measured 2026-10-07 on BrowseComp+: Qwen3.8-27B emits exactly 1
    block per turn (230/231 turns, one turn 2); Qwen3.6-35B-A3B emits 1-5 normally but loops
    ("I'll run. I'll execute.") into turns of 30, 41, 56, 126, 136, 169 blocks -- one such turn
    wrote 921k chars into the next prompt and overflowed even a 262k context. A cap of 5 never
    binds on normal behaviour of either model.
    """
    cap = int(os.environ.get("SUP_MAX_BLOCKS", "0") or 0)
    if cap <= 0:
        return
    inner = RC.find_code_blocks

    def capped(text):
        blocks = inner(text)
        if len(blocks) > cap:
            print(f"[blockcap] {len(blocks)} code blocks in one turn -> executing the first {cap}",
                  flush=True)
            return blocks[:cap]
        return blocks

    RC.find_code_blocks = capped
    print(f"[blockcap] at most {cap} code blocks per turn", flush=True)


def install() -> bool:
    import rlm.core.rlm as RC  # the call site binds find_code_blocks in ITS namespace
    if os.environ.get("SUP_FENCE_TOLERANT", "") != "1":
        _install_block_cap(RC)
        return False

    orig = RC.find_code_blocks
    fence = re.compile(r"```([A-Za-z0-9_+-]*)[ \t]*\n(.*?)\n```", re.DOTALL)
    does = (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Call, ast.Import, ast.ImportFrom,
            ast.For, ast.While, ast.If)

    def tolerant(text):
        blocks = orig(text) if text else []
        if blocks or not text:
            return blocks
        out = []
        for lang, body in fence.findall(text):
            b = body.strip()
            if lang.lower() == "repl" or not b:
                continue
            try:
                tree = ast.parse(b)
            except SyntaxError:
                continue
            if any(isinstance(n, does) for n in ast.walk(tree)):
                out.append(b)
        if out:
            RESCUED[0] += 1
        return out

    RC.find_code_blocks = tolerant
    print("[fence] tolerant code-block parsing ON (```repl still preferred)", flush=True)
    _install_block_cap(RC)   # wraps the tolerant parser, so the cap counts rescued blocks too
    return True
