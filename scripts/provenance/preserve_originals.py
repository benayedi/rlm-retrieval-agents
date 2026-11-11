"""Copy, byte for byte, the source files that produced the results into `original/` snapshot directories, and write a
MANIFEST.json per snapshot with each file's origin, SHA-256, size and modification time. Check them later.

Snapshots are never edited. Each snapshot directory holds its own MANIFEST.json, so a later snapshot of a file that
changed is added as a new directory next to the earlier one (for example agents/dspy_rlm/original/RLM_2026-11-11/
next to agents/dspy_rlm/original/RLM/), and the earlier manifest stays untouched.

  2026-10-30  first migration: agents/<agent>/original/ (MANIFEST.json at original/, files under RLM/ and FULLDOC/)
  2026-11-11  follow-up studies: studies/<study>/original/ (same layout) and second snapshots
              agents/<agent>/original/RLM_2026-11-11/ (own MANIFEST.json) of files that changed after 2026-10-30 or
              were not copied then

usage: python scripts/provenance/preserve_originals.py --check          verify every manifest under agents/*/original
                                                                          and studies/*/original (no copying)
       python scripts/provenance/preserve_originals.py --copy DEST ...  create the snapshot DEST (a key of SETS);
                                                                          refuses if DEST already has a MANIFEST.json
       python scripts/provenance/preserve_originals.py --list           the snapshot destinations defined below

Source roots (copying only): RLM_SRC_ROOT (the original research repository), FULLDOC_SRC_ROOT (the document REPL
agent working directory), CLAUDE_TRANSCRIPTS_ROOT (Claude Code session transcripts, for code that was run inline and
never saved as a file). They default to the machine the experiments ran on, as in the 2026-10-30 version.

Transcript extracts. VAR-2's computation was never saved as a script: it was run as a `python - <<'EOF'` heredoc in a
Claude Code session. A TRANSCRIPT entry names the session file, the 1-based line of the JSON record and the tool_use id;
the heredoc body between the `<<'EOF'` line and the closing `EOF` is written out unchanged (plus nothing else), and the
manifest records how it was extracted.
"""
import argparse, datetime, glob, hashlib, json, os, re, shutil, subprocess, sys
from pathlib import Path

ART = Path(__file__).resolve().parents[2]
RLM = Path(os.environ.get("RLM_SRC_ROOT", "/home/extstu_00/RLM"))
FD = Path(os.environ.get("FULLDOC_SRC_ROOT", "/home/extstu_00/suprlm"))
TRX = Path(os.environ.get("CLAUDE_TRANSCRIPTS_ROOT", "/home/extstu_00/.claude/projects"))
ROOTS = {"RLM": RLM, "FULLDOC": FD, "TRANSCRIPT": TRX}

# ---------------------------------------------------------------- 2026-10-30 (first migration; copied, never again)
# destination -> list of (source root, relative path). Kept verbatim as the record of what the first snapshot holds.
SETS_2026_10_30 = {
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

# ---------------------------------------------------------------- 2026-11-11 (follow-up studies)
_R = lambda *ps: [(RLM, p) for p in ps]  # noqa: E731

SETS_2026_11_11 = {
    # search policy: novelty filter + Qwen3-Reranker-4B proxy, its offline replay, comparisons, run chains
    "studies/search_policy/original": _R(
        "scripts/bcp_rerank_proxy.py",
        "logs/restart_1013/bcp_rerank_proxy.v3.py",          # backup taken 2026-10-13 09:17, before WIRE=mine was added
        "scripts/bcp_policy_replay.py", "scripts/bcp_policy_cache.py", "scripts/bcp_policy_eval.py",
        "scripts/bcp_novelty_eval.py", "scripts/bcp_rerank_eval.py", "scripts/frames_rerank_eval.py",
        "scripts/bcp_policy_compare.py", "scripts/bcp_full830_compare.py", "scripts/frames_rr_compare.py",
        "scripts/bcp_pilot_grade.py", "scripts/obliq_rr_compare.py", "scripts/obliq_mine_rr_compare.py",
        "scripts/obliq_q36_next_report.py",
        "scripts/frames_s25_rr_chain.sh",
        "logs/restart_1013/restart_bcp_rr_servers.sh", "logs/bcp_full830_watch.sh",
        "logs/obliq_rr/overnight.sh", "logs/obliq_mine_rr/watch.sh"),
    # prompt optimisation: GEPA on BrowseComp-Plus (driver = agents/document_repl/src/split_runner.py), FRAMES,
    # OBLIQ-Twitter, HoVer
    "studies/prompt_optimisation/original": _R(
        "scripts/gepa_frames.py", "scripts/gepa_frames_confirm.py", "scripts/bcp_confirm.py",
        "scripts/bcp_gepa_split.py", "scripts/bcp_enrich_split.py", "scripts/validate_bcp.py",
        "scripts/bcp_trace_health.py", "scripts/bcp_search_ndcg.py", "scripts/bcp_recover_misfiled.py",
        "scripts/run_gepa_bcp.sh", "scripts/run_gepa_bcp2.sh", "scripts/frames_gepa_chain.sh",
        "scripts/frames_gepa_chain_1gpu.sh",
        "logs/run_gepa_bcp_frozen.sh", "logs/run_gepa_bcp2_frozen.sh", "logs/gepa_bcp2/continue_after_fix.sh",
        "logs/gepa_bcp/restart_after_c36.sh",
        # OBLIQ-Twitter campaign (four runs) and the chains that launched them
        "scripts/gepa_official_obliq.py", "scripts/gepa_official_obliq_v2.py", "scripts/gepa_paper.py",
        "scripts/gepa_night.py", "scripts/gepa_night.py.bak-leak", "scripts/gepa_wide.py", "scripts/validate_wide.py",
        "scripts/trace_check.py",
        "scripts/night_chain.sh", "scripts/overnight2.sh", "scripts/overnight3.sh", "scripts/night5.sh",
        "scripts/night6.sh", "scripts/night7_newmodel.sh", "scripts/night8_gepa27_finish.sh",
        "scripts/on_gepa_finish.sh", "scripts/watchdog.sh",
        "logs/night7/chain_frozen.sh", "logs/night7/watchdog_frozen.sh", "logs/night8_frozen.sh",
        # HoVer sanity check
        "scripts/hover_baseline.py", "scripts/hover_build_index.py", "scripts/hover_gepa.py",
        "scripts/hover_make_splits.py", "scripts/hover_retriever.py", "scripts/run_hover_gepa.sh",
        "logs/run_hover_gepa_frozen.sh"),
    # VAR-2: inline code of Claude Code session 2a1318b5 (2026-10-27 02:05 UTC), run over docs/supervisor_artifact.html
    "studies/run_variation/original": [
        (TRX, "TRANSCRIPT:-home-extstu-00-RLM/2a1318b5-524a-4d81-9aba-3918a191383c.jsonl:3606:var2_flip_rates_draft.py"),
        (TRX, "TRANSCRIPT:-home-extstu-00-RLM/2a1318b5-524a-4d81-9aba-3918a191383c.jsonl:3616:var2_flip_rates.py")],
    "studies/error_analysis/original": _R("scripts/frames_failure_taxonomy.py", "scripts/frames_s25_failure_drill.py"),
    "studies/frames_baselines/original": _R("scripts/frames_eval.py", "logs/frames/run_rest.sh"),
    # second snapshots: files changed after 2026-10-30, or not copied then
    "agents/dspy_rlm/original/RLM_2026-11-11": _R(
        "rag/__init__.py", "rag/agent.py", "rag/signatures.py", "rag/tools.py",
        "examples/browsecomp_eval_dspy_rlm.py", "examples/browsecomp_data.py",
        "scripts/run5d/dspy_ablation_arm3.sh", "scripts/run5d/dspy_ablation_arm5.sh",
        "scripts/run5d/dspy_ablation_report.py", "scripts/run5d/dspy_ablation_strata.py",
        "scripts/run5d/dspy_vs_snip.py"),
    "agents/shared/original/RLM_2026-11-11": _R(
        "scripts/bcp_rejudge_gpt41_rq4.py", "scripts/serve_qwen36_moe.sh",
        "scripts/failure_analysis_extract.py", "scripts/failure_analysis_build.py",
        "scripts/failure_analysis_findings.py", "scripts/failure_analysis_extras.py",
        "scripts/failure_analysis_turns_merge.py", "scripts/failure_analysis_turns_sheet.py"),
    "agents/document_repl/original/RLM_2026-11-11": _R(
        "logs/gepa_bcp.py.bak_1012",                         # gepa_bcp.py before its 2026-10-12 20:28 edit
        "scripts/frames_s25_full.sh", "scripts/frames_s25_chain.sh", "scripts/frames_s25_full_report.py"),
    "agents/snippet_repl/original/RLM_2026-11-11": _R(
        "scripts/obliq_best_chain.sh", "scripts/obliq_best_full.py"),
}
SETS = {**SETS_2026_10_30, **SETS_2026_11_11}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256(p) -> str:
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


def git_state(root, rel):
    """'committed' / 'modified' / 'untracked' / None (not a git repository)."""
    try:
        r = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", rel], capture_output=True,
                           text=True, timeout=10)
        if r.returncode:
            return None
        s = r.stdout.strip()
        return "committed" if not s else ("untracked" if s.startswith("??") else "modified")
    except Exception:
        return None


def extract_heredoc(session_rel: str, line: int):
    """The `python - <<'EOF'` body of the Bash tool_use on 1-based line `line` of a transcript file."""
    rec = json.loads((TRX / session_rel).read_text(errors="strict").splitlines()[line - 1])
    for c in rec["message"]["content"]:
        if c.get("type") == "tool_use" and c.get("name") == "Bash":
            cmd = c["input"]["command"]
            m = re.search(r"<<'EOF'\n(.*?)\nEOF\s*$", cmd, re.S)
            if not m:
                sys.exit(f"no <<'EOF' heredoc in {session_rel}:{line}")
            body = (m.group(1) + "\n").encode()
            meta = {"tool_use_id": c.get("id"), "timestamp": rec.get("timestamp"),
                    "command_sha256": sha256_bytes(cmd.encode()),
                    "extraction": "heredoc body between the <<'EOF' line and the closing EOF, plus one newline"}
            return body, meta
    sys.exit(f"no Bash tool_use on {session_rel}:{line}")


def copy_set(dest: str):
    d = ART / dest
    if (d / "MANIFEST.json").exists():
        sys.exit(f"{dest}: MANIFEST.json exists; snapshots are never rewritten (add a new snapshot directory instead)")
    if dest in SETS_2026_10_30:
        sys.exit(f"{dest}: the 2026-10-30 snapshot is closed")
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    man = {"created": now, "files": []}
    for root, rel in SETS[dest]:
        label = next(k for k, v in ROOTS.items() if v == root)
        if rel.startswith("TRANSCRIPT:"):
            _, session_rel, line, name = rel.split(":", 3)
            body, meta = extract_heredoc(session_rel, int(line))
            out = d / "TRANSCRIPT" / name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(body)
            man["files"].append(dict(copy=str(out.relative_to(d)), source=f"TRANSCRIPT:{session_rel}:{line}",
                                     sha256=sha256_bytes(body), bytes=len(body), **meta))
            continue
        src = root / rel
        if not src.exists():
            sys.exit(f"missing source file: {src}")
        out = d / label / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
        if sha256(out) != sha256(src):
            sys.exit(f"copy differs from source: {src}")
        man["files"].append(dict(copy=str(out.relative_to(d)), source=f"{label}:{rel}", sha256=sha256(src),
                                 bytes=src.stat().st_size, git_state=git_state(root, rel) if label == "RLM" else None,
                                 mtime=datetime.datetime.fromtimestamp(src.stat().st_mtime).isoformat(timespec="seconds")))
    man["source_roots"] = {"RLM": {"git_head": git_rev(RLM), "note": "working tree, including uncommitted changes"},
                           "FULLDOC": {"git_head": None, "note": "not a git repository; files as on disk"},
                           "TRANSCRIPT": {"note": "Claude Code session transcripts (JSON lines), read only"}}
    (d / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    print(f"{dest}: {len(man['files'])} files")


def manifests():
    pats = ["agents/*/original/**/MANIFEST.json", "studies/*/original/**/MANIFEST.json"]
    return sorted({Path(p) for pat in pats for p in glob.glob(str(ART / pat), recursive=True)})


def check():
    bad, covered = 0, set()
    for m in manifests():
        d = m.parent
        files = json.loads(m.read_text())["files"]
        for f in files:
            p = d / f["copy"]
            covered.add(p.resolve())
            if not p.exists():
                print("MISSING:", p.relative_to(ART)); bad += 1
            elif sha256(p) != f["sha256"]:
                print("MODIFIED:", p.relative_to(ART)); bad += 1
        print(f"  {m.relative_to(ART)}: {len(files)} files")
    # files under original/ that no manifest lists (reported, not an error: the first migration's
    # agents/snippet_repl/original/RLM_git_7685538/ predates per-snapshot manifests)
    for orig in glob.glob(str(ART / "agents/*/original")) + glob.glob(str(ART / "studies/*/original")):
        for p in Path(orig).rglob("*"):
            if p.is_file() and p.name != "MANIFEST.json" and p.resolve() not in covered:
                print("  not in any manifest:", p.relative_to(ART))
    print("all copies match their manifests" if not bad else f"{bad} files differ or are missing")
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--copy", nargs="+", metavar="DEST")
    g.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, v in SETS.items():
            print(f"{k:50s} {len(v):3d} files  {'(2026-10-30, closed)' if k in SETS_2026_10_30 else ''}")
    elif a.copy:
        for dest in a.copy:
            copy_set(dest)
    else:
        sys.exit(1 if check() else 0)
