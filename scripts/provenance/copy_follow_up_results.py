"""Copy, byte for byte, the small result files of the follow-up studies into the repository, with a MANIFEST.json per
destination (source path, SHA-256, size, modification time). The large per-question records go to trajectories/
(scripts/provenance/pack_trajectories.py); this script handles only the files that are read directly:

  results/follow_up/verdicts/     gpt-4.1 verdicts (question id -> bool, plus numeric fields; no question or answer text)
  studies/error_analysis/results/   ERR-1 failure buckets (FRAMES; Qwen3.6 judge only)
  studies/frames_baselines/results/ FB-1 single-shot records used by ERR-1 (FRAMES; Qwen3.6 judge only)

FRAMES is a public benchmark (Apache-2.0); its questions and answers may appear in plain text, as in the FRAMES
archives of trajectories/. Every BrowseComp-Plus file copied here holds only ids, booleans and numbers; run
scripts/provenance/bcp_leak_scan.py on the destinations after copying.

usage: python scripts/provenance/copy_follow_up_results.py [--check]
  LOGS_ROOT: logs/ of the original repository (copying only).  --check re-hashes the copies against the manifests.
"""
import argparse, datetime, hashlib, json, os, shutil, sys
from pathlib import Path

ART = Path(__file__).resolve().parents[2]
L = Path(os.environ.get("LOGS_ROOT", "/home/extstu_00/RLM/logs"))

_FRAMES_GPT41 = ["sup_q36_bm25f", "sup_q36_bm25f_s25", "sup_q36_bm25f_s25_rest0", "sup_q36_bm25f_s25_rest1",
                 "sup_q36_s25_rr_g0", "sup_q36_s25_rr_g1"]
SETS = {
    "results/follow_up/verdicts": (
        # BrowseComp-Plus, gpt-4.1 (logs/bcp_rejudge_gpt41/; rows as in the original repository's rlm_results_summary.xlsx)
        [(f"bcp_rejudge_gpt41/{f}", f"bcp/{f}") for f in (
            "16.json",                 # BCP-29: control of SP-1, SP-2, GEPA-2 (row 16)
            "17.json",                 # SP-1: novelty + rerank, all 830 (row 17)
            "rq4_rerank_only.json",    # SP-2 (2026-11-10)
            "rq4_gepa_winner.json",    # GEPA-2 test200 (2026-11-10)
            "rq4_summary.json",
            "pilot_rr.json",           # SP-3 reranker arm, 100 questions
            "5.json",                  # the original DSPy dense run (row 5): reference of DSA-3 / DSA-5
            "summary.json")]           # the 2026-10-27 re-judge of every row (counts and accuracies)
        # FRAMES, gpt-4.1 (graded_gpt41.json of each run directory)
        + [(f"frames/{d}/graded_gpt41.json", f"frames/{d}/graded_gpt41.json") for d in _FRAMES_GPT41]
        + [("gepa_frames_confirm/rollouts/8ba345b481/graded_gpt41.json", "frames/gepa_frames_confirm_8ba345b481/graded_gpt41.json"),
           ("frames/rejudge_gpt41_rq4_2026-11-10.log", "frames/rejudge_gpt41_rq4_2026-11-10.log")]
        # DSPy.RLM ablation: per-arm report (gpt-4.1)
        + [(f"dspy_ablation/arm{a}/report.json", f"dspy_ablation/arm{a}_report.json") for a in (3, 5)]),
    "studies/error_analysis/results": [(f"frames/{f}", f) for f in ("failure_taxonomy.json", "failure_taxonomy_s25.json")],
    "studies/frames_baselines/results": [(f"frames/{f}", f) for f in (
        "llm_oracle.jsonl", "llm_closed.jsonl", "llm_bm25_4.jsonl", "llm_summary.json")],
}


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def copy():
    for dest, items in SETS.items():
        d = ART / dest
        man = {"created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "files": []}
        for rel, out_rel in items:
            src, out = L / rel, d / out_rel
            if not src.exists():
                sys.exit(f"missing source: {src}")
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)
            assert sha256(out) == sha256(src)
            man["files"].append(dict(copy=out_rel, source=f"LOGS_ROOT:{rel}", sha256=sha256(src), bytes=src.stat().st_size,
                                     mtime=datetime.datetime.fromtimestamp(src.stat().st_mtime).isoformat(timespec="seconds")))
        (d / "MANIFEST.json").write_text(json.dumps(man, indent=1))
        print(f"{dest}: {len(man['files'])} files")


def check():
    bad = 0
    for dest in SETS:
        d = ART / dest
        for f in json.loads((d / "MANIFEST.json").read_text())["files"]:
            if not (d / f["copy"]).exists() or sha256(d / f["copy"]) != f["sha256"]:
                print("DIFFERS:", dest, f["copy"]); bad += 1
    print("all copies match their manifests" if not bad else f"{bad} files differ")
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    sys.exit(1 if check() else 0) if a.check else copy()
