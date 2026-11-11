"""Recover rollouts that run_one wrote under suprlm/logs/... because GEPA_OUT was relative.

Refactored copy of original/RLM/scripts/bcp_recover_misfiled.py
  (studies/prompt_optimisation/CHANGES_FROM_ORIGINAL.md).
Run: one-off repair of 2026-10-08 (experiments/gepa2_bcp_run2_continue.sh); MOVES files, as the original did.
Changed: import wiring and the misfiled-rollout location.
Names: 'supervisor harness' / 'full-doc RLM' in the text below is the document REPL agent (agents/document_repl);
  'official RLM harness' / 'our harness' / run_obliq_mine.py is the snippet REPL agent (agents/snippet_repl,
  src/run_obliq.py).

Each misfiled <key>/<qid>.json is moved into RLM/logs/gepa_bcp/rollouts/<key>/ and graded with the
SAME grader the driver uses (gepa_bcp._grade -> grade_local.judge on the served model), replacing the
"killed" placeholder in the cache. Run with the run's env (GEPA_BCP_MODEL, SUP_*), cwd RLM.
"""
import glob, json, os, shutil, sys
# Import wiring and paths (refactor, rules 2-3). Original: hard-coded locations of the original repository (R), the
# document REPL agent working directory (SUP) and the interpreter. Code now comes from this repository (computed from
# __file__), data and run outputs from configs/paths.env (rlm_shared.paths). The original logs/ directory is RUNS_DIR.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, os.pardir))
DOC_SRC = os.path.join(_REPO, "agents", "document_repl", "src")    # split_runner.py (was scripts/gepa_bcp.py), run_bm25.py
SPLITS = os.path.join(_REPO, "studies", "prompt_optimisation", "splits")   # original: <original repo>/data
sys.path.insert(0, os.path.join(_REPO, "agents", "shared"))
from rlm_shared.paths import path  # noqa: E402
RUNS = str(path("RUNS_DIR"))                                         # original: f"{R}/logs"
sys.argv = ["x"]; sys.path.insert(0, DOC_SRC)   # original: <original repo>/scripts
os.environ.pop("GEPA_PROMPTS", None)
import split_runner as G   # original: import gepa_bcp as G
SRC = f"{path('FULLDOC_ROOT')}/logs/gepa_bcp/rollouts"   # original: <document REPL working dir>/logs/gepa_bcp/rollouts
G._cache_load()
n = fixed = 0
for f in sorted(glob.glob(f"{SRC}/*/*.json")):
    key = os.path.basename(os.path.dirname(f)); q = os.path.basename(f)[:-5]
    dst = os.path.join(G.OUT, "rollouts", key, f"{q}.json"); os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        raw = json.load(open(f))
    except Exception:
        continue          # still being written by an in-flight worker
    shutil.move(f, dst); n += 1
    old = G._CACHE.get(f"{key}|{q}") or {}
    if old.get("status") in (None, "killed"):
        G._CACHE[f"{key}|{q}"] = G._grade(q, raw); fixed += 1
G._cache_save()
print(f"[recover] moved {n} rollouts, graded {fixed} into the cache (replacing killed/missing)")
