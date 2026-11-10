"""PUBLICATION COPY (refactor header). Original file: scripts/hooks/retry429_cap_ranking/sitecustomize.py; code unchanged. Python runs it at interpreter
start when src/hooks/retry429_cap_ranking is the FIRST directory on PYTHONPATH that holds a sitecustomize.py (the launchers
in ../experiments/ put src/hooks/run5d there).

Both startup hooks at once (2026-10-27): Python imports only the FIRST sitecustomize.py on sys.path, so putting the
retry429 and cap_ranking directories side by side on PYTHONPATH silently loads just one. This runs both."""
import os

_H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _name in ("retry429", "cap_ranking"):
    _p = os.path.join(_H, _name, "sitecustomize.py")
    exec(compile(open(_p).read(), _p, "exec"), {"__name__": f"sitecustomize_{_name}", "__file__": _p})
