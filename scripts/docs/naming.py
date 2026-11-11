"""Agent names as in the paper (github.com/neulandAI/program-state-agents).

The workbooks in results/original_baselines/ and the reporting scripts that wrote them keep the names used during the
research ("Snippet RLM", "Full-Doc RLM"). The document generators pass their output through `paper_names`, so the
documents use the paper's names while the evidence files stay unchanged:

| In the workbooks and run names | In the paper and in these documents |
|---|---|
| Full-Doc RLM, full-doc, `fd_*` runs | document REPL agent |
| Snippet RLM, Snippet, `snip_*` runs | snippet REPL agent |

Follow-up studies (2026-11-11): the run names, logs and `original/` files of the original repository keep their names;
`RUN_NAMES` below maps them to the study IDs and the paper's terms. It is documentation (used by tests and readers); the
generators do not rewrite these names.
"""
import re

# a name starts a sentence, a table cell, a heading, a list item or an SVG/HTML text element
_LEAD = re.compile(r"(^|[.!?:|>]\s*|\*\*)[\s\"'`(\[*_#|-]*$")


def _cap_sub(pattern, low, text):
    def rep(m):
        before = text[max(0, m.start() - 80):m.start()].split("\n")[-1]
        return (low[0].upper() + low[1:]) if _LEAD.search(before) else low
    return re.sub(pattern, rep, text)


def paper_names(text):
    """Rewrite research-time agent names to the paper's names (idempotent)."""
    text = text.replace("Full-Doc-RLM", "document-REPL-agent").replace("Snippet-RLM", "snippet-REPL-agent")
    text = _cap_sub(r"\b[Ff]ull-[Dd]oc RLM\b", "document REPL agent", text)
    text = _cap_sub(r"\bSnippet RLM\b", "snippet REPL agent", text)
    text = _cap_sub(r"\b(?:Full-Doc|Full-doc|full-doc)\b", "document REPL", text)
    text = _cap_sub(r"\bSnippet and document REPL\b", "snippet and document REPL", text)
    # bare "Snippet" as the agent's label; not "Snippet length", "Snippets", the "Snippet" settings row
    text = re.sub(r"(?m)^(\| Snippet) (\| (?:min\(|server text))", r"\1\0\2", text)
    text = _cap_sub(r"\bSnippet\b(?! REPL|s\b| length|\0|-run)", "snippet REPL", text)
    text = text.replace("\0", " ").replace("Snippet-run", "Snippet REPL run")
    return text


# research-time name (run directory, script, phrase) -> (study ID or term in the paper / this repository)
RUN_NAMES = {
    # agents and harness phrases
    "supervisor harness": "document REPL agent (agents/document_repl)",
    "suprlm": "the document REPL agent's working directory (FULLDOC_ROOT)",
    "full-doc RLM": "document REPL agent",
    "official RLM harness": "snippet REPL agent (agents/snippet_repl)",
    "run_obliq_mine.py": "snippet REPL agent on OBLIQ-Bench (agents/snippet_repl/src/run_obliq.py)",
    "dspy RLM": "DSPy.RLM (agents/dspy_rlm)",
    "10 searches / 25 searches, MAX_SEARCHES, s25": "search cap 10 / search cap 25",
    "rr, reranker proxy, search policy, bcp_rerank_proxy.py": "the search-policy proxy (studies/search_policy/src/rerank_proxy.py)",
    "gepa_bcp.py": "agents/document_repl/src/split_runner.py",
    # BrowseComp-Plus runs
    "bcp_policy_e2e, bcp_testrest_rr, bcp_rest300_rr, bcp_pareto_rr, bcp_feedback_rr": "SP-1",
    "bcp_policy_e2e_rerankonly": "SP-2",
    "bcp_gpt5_rr/rr": "SP-3",
    "gepa_bcp": "GEPA-1",
    "gepa_bcp2 (winner d911ebbe23, test200)": "GEPA-2",
    "gepa_bcp2/bcp_testrest_ctrl/gepa_bcp rollouts ee5ea32a1c": "BCP-29 (control of SP-1, SP-2, GEPA-2)",
    "dspy_ablation/arm3, arm5": "DSA-3, DSA-5",
    "supervisor_artifact.html pairs": "VAR-2",
    # FRAMES runs
    "frames/sup_q36_bm25f": "FR-13 (control of FRC-1; seed of GEPA-3)",
    "frames/sup_q36_bm25f_s25, _s25_rest0, _s25_rest1": "FRC-1",
    "frames/sup_q36_s25_rr_g0, _g1": "SP-4",
    "gepa_frames, gepa_frames_confirm (8ba345b481)": "GEPA-3",
    "frames/failure_taxonomy*.json": "ERR-1",
    "frames/llm_closed, llm_bm25_4, llm_oracle": "FB-1",
    # OBLIQ-Twitter runs
    "obliq_mine_rr ctrl_dev100 + snip_rest181": "TWQ-1",
    "obliq_mine_rr rr_dev100 + rr_rest181": "SP-6",
    "suprlm/runs/obliq_q36_ctrl": "TWQ-2",
    "suprlm/runs/obliq_q36_rr": "SP-7",
    "gepa_official, gepa_official_v2, gepa_paper, gepa_wide": "GEPA-4 runs 1-4",
    "hover_gepa": "GEPA-5",
}
