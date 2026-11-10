"""Copy, byte for byte, the source files that produced the results in Experiments.xlsx into
agents/<agent>/original/ (and agents/shared/original/), and write a MANIFEST.json per destination with each file's
origin path, SHA-256, size and modification time.

These snapshots are the reference for every refactor: agents/*/src/ is compared against them
(tests/test_preservation.py), and they are never edited.

usage: python scripts/provenance/preserve_originals.py [--check]
  --check   recompute hashes of the copies and compare with MANIFEST.json (no copying)

Source roots default to the machine the experiments ran on; override with RLM_SRC_ROOT, FULLDOC_SRC_ROOT.
"""
import argparse, datetime, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

ART = Path(__file__).resolve().parents[2]
RLM = Path(os.environ.get("RLM_SRC_ROOT", "/home/extstu_00/RLM"))
FD = Path(os.environ.get("FULLDOC_SRC_ROOT", "/home/extstu_00/suprlm"))

# destination -> list of (source root, relative path). Paths are files; a trailing "/" copies a directory's *.py/*.sh.
SETS = {
    "agents/snippet_repl/original": [(RLM, p) for p in [
        "examples/browsecomp_eval_official_rlm.py", "examples/browsecomp_data.py", "examples/frames_eval_rlm.py",
        "examples/obliq_data.py", "scripts/run_bcp_snippet.py", "scripts/run_obliq_mine.py",
        "scripts/official_prompt_modules.py", "scripts/run_obliq_shard.sh",
        "scripts/hooks/retry429/sitecustomize.py", "scripts/hooks/azure_sampling/sitecustomize.py",
        "scripts/hooks/run5d/sitecustomize.py", "scripts/hooks/cap_ranking/sitecustomize.py",
        "scripts/hooks/retry429_cap_ranking/sitecustomize.py",
        "scripts/run5d/bcp_snip.sh", "scripts/run5d/bcp_snip2.sh", "scripts/run5d/bcp_snip_gpt5_bm25_finish.sh",
        "scripts/run5d/frames_snip.sh", "scripts/run5d/obliq_gemini.sh", "scripts/run5d/qwen_bcp_snip_queue2.sh",
        "scripts/run5d/question_watchdog.sh", "scripts/run5d/budget_guard.sh", "scripts/run5d/night_finisher.sh",
        "scripts/run_qwen_full.sh", "scripts/run_no_orchestrator.sh"]],
    "agents/document_repl/original": [(FD, f"src/{p}") for p in [
        "run_one.py", "tools.py", "prompts.py", "fence_tolerant.py", "rep_guard.py", "obliq_adapter.py",
        "retrieval_server.py", "retrieval_server_colbert.py", "encoder.py", "hybrid.py", "build_bm25.py",
        "prepare_data.py", "run_experiment.py", "judge.py", "grade_local.py", "score_obliq.py", "provenance.py",
        "final830_run.sh", "final830_complete.sh", "launch_high400.sh", "launch_high100b.sh", "launch_high100c.sh",
        "launch_high_rest230.sh",
        "apply_answer_check_patch.py", "apply_cap_patch.py", "apply_design_d_patch.py", "apply_design_e_patch.py",
        "apply_design_f_patch.py", "apply_design_g_patch.py", "apply_design_m_patch.py", "apply_hybrid_patch.py",
        "apply_rlm_side_patch.py"]] + [(RLM, p) for p in [
        "scripts/bcp_gpt5_sup.sh", "scripts/bcp_gpt5_bm25.sh", "scripts/frames_sup_run_one.py",
        "scripts/run_frames_sup.sh", "scripts/run5d/frames_fd_gpt5.sh", "scripts/run5d/bcp_fd_qwen_dense.py",
        "scripts/gepa_bcp.py", "scripts/bcp_bm25_test.py"]],
    "agents/dspy_rlm/original": [(RLM, p) for p in [
        "rag/__init__.py", "rag/agent.py", "rag/signatures.py", "rag/tools.py",
        "examples/browsecomp_eval_dspy_rlm.py", "examples/obliq_eval_dspy_rlm.py", "scripts/run_obliq_dspy_shard.sh",
        "pyproject.toml"]],
    "agents/shared/original": [(RLM, p) for p in [
        # retrieval servers and index builders
        "examples/obliq_score_advice.py", "scripts/fusion_proxy.py", "scripts/frames_search_server.py", "scripts/multi_search_server.py",
        "scripts/search_server.py", "scripts/obliq_math_search.py", "scripts/gemini/search_server.py",
        "scripts/gemini/vertex.py", "scripts/gemini/build_index.py", "scripts/frames_build_bm25.py",
        "scripts/bcp_build_bm25_default.py", "scripts/build_bm25_index.py", "scripts/build_obliq_index.py",
        "scripts/build_bc_rmcb_dl8192.sh", "scripts/reindex_rmcb_8192.py",
        # grading and scoring
        "scripts/bcp_judge.py", "scripts/run5d/bcp_snip_grade.py", "scripts/run5d/bcp_fd_grade.py",
        "scripts/bcp_rejudge_gpt41.py", "scripts/frames_judge_gpt41.py", "scripts/run5d/obliq_report.py",
        # reporting
        "scripts/run5d/iteration_counts.py", "scripts/run5d/token_columns.py",
        "scripts/failure_analysis_extract.py", "scripts/failure_analysis_build.py", "scripts/failure_analysis_findings.py",
        "scripts/failure_analysis_extras.py", "scripts/failure_analysis_turns.py",
        "scripts/failure_analysis_turns_merge.py", "scripts/failure_analysis_turns_sheet.py",
        "scripts/failure_analysis_eval.py", "scripts/failure_analysis_inject.py"]],
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_rev(root):
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except Exception:
        return None


def copy_all():
    for dest, items in SETS.items():
        d = ART / dest
        man = {"created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "files": []}
        for root, rel in items:
            src = root / rel
            if not src.exists():
                sys.exit(f"missing source file: {src}")
            label = "RLM" if root == RLM else "FULLDOC"
            out = d / label / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)
            man["files"].append(dict(copy=str(out.relative_to(d)), source=f"{label}:{rel}", sha256=sha256(src),
                                     bytes=src.stat().st_size,
                                     mtime=datetime.datetime.fromtimestamp(src.stat().st_mtime).isoformat(timespec="seconds")))
        man["source_roots"] = {"RLM": {"git_head": git_rev(RLM), "note": "working tree, including uncommitted changes"},
                               "FULLDOC": {"git_head": None, "note": "not a git repository; files as on disk"}}
        (d / "MANIFEST.json").write_text(json.dumps(man, indent=1))
        print(f"{dest}: {len(man['files'])} files")


def check():
    bad = 0
    for dest in SETS:
        d = ART / dest
        for f in json.loads((d / "MANIFEST.json").read_text())["files"]:
            if sha256(d / f["copy"]) != f["sha256"]:
                print("MODIFIED:", dest, f["copy"]); bad += 1
    print("all copies match their manifest" if not bad else f"{bad} files differ")
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    sys.exit(1 if check() else 0) if a.check else copy_all()
