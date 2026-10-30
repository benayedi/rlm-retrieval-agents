"""Graders and report scripts shared by the three algorithms (refactored from scripts/ of the original repository).

  bcp_judge          library: gpt-4.1 judge for BrowseComp-Plus and FRAMES (3 votes, majority, verdict cache)
  grade_bcp_snippet  script: grade a Snippet-RLM BrowseComp-Plus run directory      (orig. scripts/run5d/bcp_snip_grade.py)
  grade_bcp_fulldoc  script: grade a Full-Doc-RLM BrowseComp-Plus run directory     (orig. scripts/run5d/bcp_fd_grade.py)
  rejudge_bcp        script: re-judge stored answers of earlier BrowseComp-Plus runs (orig. scripts/bcp_rejudge_gpt41.py)
  grade_frames       script: grade FRAMES run directories of either harness          (orig. scripts/frames_judge_gpt41.py)
  obliq_report       script: NDCG / recall of OBLIQ-Bench runs, no LLM               (orig. scripts/run5d/obliq_report.py)

This package __init__ imports nothing, so `import rlm_shared.grading` has no side effects. bcp_judge has import-time
side effects (AZ_ENV file, dataset load, client construction) and the grade_* / rejudge / report modules are scripts
whose module-level code runs on import, exactly as in the original. See shared/CHANGES_FROM_ORIGINAL.md.
"""
