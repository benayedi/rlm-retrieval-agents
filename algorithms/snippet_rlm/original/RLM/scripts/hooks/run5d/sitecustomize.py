"""Startup hooks for the 2026-10-27 runs, all at once (Python imports only the FIRST sitecustomize.py on sys.path):
retry429 (openai max_retries 10), azure_sampling (rlm's Azure client sends reasoning_effort), cap_ranking (turn-cap
fallback asks for a ranking; active only with OBLIQ_CAP_RANKING=1), and balance_local (2026-10-28, below)."""
import os

_H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _name in ("retry429", "azure_sampling", "cap_ranking"):
    _p = os.path.join(_H, _name, "sitecustomize.py")
    exec(compile(open(_p).read(), _p, "exec"), {"__name__": f"sitecustomize_{_name}", "__file__": _p})

# balance_local (2026-10-28): the snippet launchers assign a question to :8030 or :8031 by question parity. With two
# runs drawing from one id sequence, the slower server's questions pile up (seen 09:25: 22 running at 94% KV cache on
# :8031 vs 8 at 36% on :8030). At process start, if LOCAL_LLM_BASE_URL names one of the two, re-point it to the server
# with fewer running + waiting requests. Same model on both; only the placement changes.
try:
    import re as _re
    import urllib.request as _ur

    _u = os.environ.get("LOCAL_LLM_BASE_URL", "")
    _m = _re.fullmatch(r"http://localhost:(803[01])/v1/?", _u)
    if _m:
        def _load(port):
            txt = _ur.urlopen(f"http://localhost:{port}/metrics", timeout=5).read().decode()
            return sum(float(v) for v in _re.findall(r"^vllm:num_requests_(?:running|waiting)\{[^}]*\} ([0-9.]+)", txt, _re.M))
        _loads = {p: _load(p) for p in (8030, 8031)}
        _best = min(_loads, key=_loads.get)
        if _best != int(_m.group(1)):
            os.environ["LOCAL_LLM_BASE_URL"] = f"http://localhost:{_best}/v1"
        print(f"[balance] running+waiting {_loads} -> :{_best}", flush=True)
except Exception as _e:  # noqa: BLE001
    print(f"[balance] hook skipped: {_e}", flush=True)
