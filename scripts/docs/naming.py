"""Agent names as in the paper (github.com/neulandAI/program-state-agents).

The workbooks in results/original_baselines/ and the reporting scripts that wrote them keep the names used during the
research ("Snippet RLM", "Full-Doc RLM"). The document generators pass their output through `paper_names`, so the
documents use the paper's names while the evidence files stay unchanged:

| In the workbooks and run names | In the paper and in these documents |
|---|---|
| Full-Doc RLM, full-doc, `fd_*` runs | document REPL agent |
| Snippet RLM, Snippet, `snip_*` runs | snippet REPL agent |
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
