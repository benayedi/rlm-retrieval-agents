"""Recover rollouts that run_one wrote under suprlm/logs/... because GEPA_OUT was relative.

Each misfiled <key>/<qid>.json is moved into RLM/logs/gepa_bcp/rollouts/<key>/ and graded with the
SAME grader the driver uses (gepa_bcp._grade -> grade_local.judge on the served model), replacing the
"killed" placeholder in the cache. Run with the run's env (GEPA_BCP_MODEL, SUP_*), cwd RLM.
"""
import glob, json, os, shutil, sys
sys.argv = ["x"]; sys.path.insert(0, "/home/extstu_00/RLM/scripts")
os.environ.pop("GEPA_PROMPTS", None)
import gepa_bcp as G
SRC = "/home/extstu_00/suprlm/logs/gepa_bcp/rollouts"
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
