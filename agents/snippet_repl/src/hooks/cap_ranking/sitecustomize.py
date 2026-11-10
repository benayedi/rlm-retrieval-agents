"""PUBLICATION COPY (refactor header). Original file: scripts/hooks/cap_ranking/sitecustomize.py; code unchanged. Python runs it at interpreter
start when src/hooks/cap_ranking is the FIRST directory on PYTHONPATH that holds a sitecustomize.py (the launchers
in ../experiments/ put src/hooks/run5d there).

Turn-cap fallback that asks for the ranking (OBLIQ Math document REPL arm C, 2026-10-22). Loaded by Python at startup
only when this directory is on PYTHONPATH, and only active with OBLIQ_CAP_RANKING=1.

WHY. When the RLM runs out of turns without answer["ready"], rlm/core/rlm.py:_default_answer makes one more call with
"Please provide a final answer to the user's question based on the information provided." On OBLIQ Math with Qwen3.6,
16 of 151 document REPL runs hit the 12-turn cap and that call came back as more REPL code, so the run submitted no doc ids
and scored 0. This swaps only that one fallback message for one asking for the ranking in the task's format. The
library file is untouched; every other turn, prompt and cap is unchanged.
"""
import os

if os.environ.get("OBLIQ_CAP_RANKING") == "1":
    try:
        import rlm.core.rlm as _R

        _MSG = ("You are out of turns: no more code will run. Reply now with your final ranking and nothing else: "
                "one docid per line, strongest to weakest, using the doc_ids of documents you retrieved. Put every "
                "plausible candidate you saw, best first. No code, no prose, no numbering.")
        _orig = _R.RLM._default_answer

        def _default_answer(self, message_history, lm_handler):
            hist = message_history + [{"role": "user", "content": _MSG}]
            response = lm_handler.completion(hist)
            if self.logger:
                self.logger.log(_R.RLMIteration(prompt=hist, response=response, final_answer=response, code_blocks=[]))
            return response

        _R.RLM._default_answer = _default_answer
    except Exception as e:  # noqa: BLE001
        print(f"[cap-ranking] hook not installed: {e}", flush=True)
