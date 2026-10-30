"""Staged patch — Design M: the same agent, tool surface and harness, with the multi-vector retriever.

Nothing about the agent changes. The only edits are to the paragraph of the `search` description that
states how the index behaves, because the Design G text describes a single-vector dense index and would
be false of a late-interaction one. Every replacement sentence is a measured fact about the new index:

  * matching: token-level late interaction rather than one vector per document
  * window: LightOn's prebuilt index embedded the first 512 tokens of each document
            (doclens from the index itself: max 512, mean 424, median 457)
  * recall by k: measured on all 830 questions with runs/first_stage_colbert.json, not carried over
                 from the dense index

Run with no sweep in flight, after apply_design_g_patch.py. Idempotent-guarded.
"""
import ast, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATS = f"{ROOT}/runs/first_stage_colbert.json"
if not os.path.exists(STATS):
    sys.exit("measure the retriever first: .venv/bin/python src/eval_first_stage_colbert.py 8766")
S = json.load(open(STATS))
ev, hit = S["evidence"], S["evidence_hit"]

P = f"{ROOT}/src/tools.py"
s = open(P).read()
assert 'design == "M"' not in s, "Design M already applied"

# ---- the dense retriever contract, as written for Design G -------------------------------------
DENSE_CONTRACT = '''            search_desc += (
                " The retriever is a dense semantic index (one embedding per document): it matches "
                "the MEANING of a full natural-language description, not keywords. Measured on this "
                "corpus, queries of 6-12 words recover about 10% of the relevant documents in their "
                "top-5; queries of 13-20 words about 15%; a complete question-length description "
                "about 15% with far higher chance of at least one hit. Write each query as a "
                "descriptive sentence or two that states the entity type and the constraints you "
                "are matching, rather than a short keyword string."
            )'''
assert DENSE_CONTRACT in s

MULTI_CONTRACT = '''            search_desc += (
                " The retriever is a multi-vector late-interaction index: every token of your query is "
                "scored against every token of each document and the best matches are summed, so exact "
                "names, numbers, titles and distinctive phrases carry weight of their own, alongside "
                "overall meaning. The index embedded the first 512 tokens of each document (roughly the "
                "first 2,000 characters); 'embedded_span' marks that region, and matches are driven by "
                "it even though the tool returns the document in full."
            ) if self.design == "M" else (
                " The retriever is a dense semantic index (one embedding per document): it matches "
                "the MEANING of a full natural-language description, not keywords. Measured on this "
                "corpus, queries of 6-12 words recover about 10% of the relevant documents in their "
                "top-5; queries of 13-20 words about 15%; a complete question-length description "
                "about 15% with far higher chance of at least one hit. Write each query as a "
                "descriptive sentence or two that states the entity type and the constraints you "
                "are matching, rather than a short keyword string."
            )'''
s = s.replace(DENSE_CONTRACT, MULTI_CONTRACT, 1)

# ---- recall-by-k facts: measured on whichever index is in use -----------------------------------
DENSE_K = '''            search_desc += (
                " One call costs one call whatever k is: hits are ranked, so k=30 returns the same "
                "first five plus ranks 6-30. Measured on this corpus with question-length queries: "
                "the top-5 hold about 15% of the relevant documents (at least one of them 48% of the "
                "time), the top-30 about 33% (74%), the top-50 about 39% (80%). Hits at ranks you have "
                "not read are still in DOCS, where "'''
assert DENSE_K in s
MULTI_K = ('''            search_desc += (
                " One call costs one call whatever k is: hits are ranked, so k=30 returns the same "
                "first five plus ranks 6-30. Measured on this corpus with question-length queries: "
                "the top-5 hold about %.0f%% of the relevant documents (at least one of them %.0f%% of the "
                "time), the top-30 about %.0f%% (%.0f%%), the top-50 about %.0f%% (%.0f%%). Hits at ranks you have "
                "not read are still in DOCS, where "''' % (
    ev["recall@5"], hit["hit@5"], ev["recall@30"], hit["hit@30"], ev["recall@50"], hit["hit@50"]))
s = s.replace(DENSE_K, MULTI_K, 1)

# ---- design M reuses the Design B/G tool surface -------------------------------------------------
s = s.replace('        elif self.design == "D":', '        elif self.design == "D":', 1)
s = s.replace('''        if self.design == "B":''', '''        if self.design in ("B", "M"):''', 1)

open(P, "w").write(s)
ast.parse(s)
print("tools.py: Design M retriever contract + measured recall-by-k")
print(f"  recall@5 {ev['recall@5']}  @30 {ev['recall@30']}  @50 {ev['recall@50']}")
print(f"  hit@5    {hit['hit@5']}  @30 {hit['hit@30']}  @50 {hit['hit@50']}")

for f, marker in ((f"{ROOT}/src/run_one.py", None), (f"{ROOT}/src/run_experiment.py", None)):
    t = open(f).read()
    if '"A", "B", "D"' in t:
        t = t.replace('"A", "B", "D"', '"A", "B", "D", "M"')
        open(f, "w").write(t); ast.parse(t)
        print(os.path.basename(f), "accepts --design M")

P2 = f"{ROOT}/src/prompts.py"
t = open(P2).read()
if "DESIGN_M" not in t:
    t += '\nDESIGN_M = "multi-vector-retriever-contract"\n'
    t = t.replace("globals().get('DESIGN_G', '')", "globals().get('DESIGN_G', '') + (globals().get('DESIGN_M', '') if __import__('os').environ.get('RLM_DESIGN') == 'M' else '')")
    open(P2, "w").write(t); ast.parse(t)
    print("prompts.py: DESIGN_M marker (prompt_id differs only when RLM_DESIGN=M)")
