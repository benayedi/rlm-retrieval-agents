"""Search-policy study: extract every logged BrowseComp-Plus search of the document REPL agent into a replay file.

From original/RLM/scripts/bcp_policy_replay.py; changed: import wiring, locations from configs/paths.env, this
paragraph (../CHANGES_FROM_ORIGINAL.md). First step of the offline replay (policy_replay -> policy_cache ->
policy_eval / novelty_eval / rerank_eval). Reads the rollouts of the Qwen3.8-27B 830-question run
(<FULLDOC_ROOT>/runs/qwen27b_colbert8192_*) and of the two BrowseComp-Plus GEPA runs (<LOGS_ROOT>/gepa_bcp*/rollouts);
writes <RLM_DATA_DIR>/bcp_policy/replay.jsonl. test200 is excluded (GEPA_SPLIT). Reads BrowseComp-Plus from the
Hugging Face hub (datasets).
  python analysis/policy_replay.py

Original docstring follows.

Replay dataset for optimising the SEARCH POLICY (code) offline on BrowseComp-Plus.

WHY. Prompt-GEPA on BrowseComp+ tied on held-out test200 (2026-10-10). The agent's errors are
retrieval errors (Qwen3.6: 100% correct with all gold found, 0% with none), and every prompt
evaluation cost a ~15-min rollout. A search policy -- the function between the agent's query and
the documents it gets back -- can be scored OFFLINE on logged agent queries in seconds, which is
the regime where GEPA worked (HoVer, 2026-10-10: +19 pts on val in 35 min).

This script extracts every logged search: (qid, question, query, k, returned docids) from the
supervisor-harness rollouts (Qwen3.8-27B 830-run + all Qwen3.6 GEPA rollouts), attaches gold and
evidence docids, and assigns a question-level split. test200 questions are EXCLUDED entirely:
they are reserved for the final end-to-end check.

Output: data/bcp_policy/replay.jsonl  (+ summary stats on stdout)
"""
import glob, json, os, random, sys, collections

# Import wiring: agents/shared/rlm_shared is located from this file; locations come from configs/paths.env
# (rlm_shared.paths.path). Original: absolute paths of the machine the experiments ran on.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir, "agents", "shared"))
from rlm_shared.paths import path  # noqa: E402
REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir))
from rlm_shared.browsecomp_data import get_question, get_gold_doc_ids, get_evidence_doc_ids  # noqa: E402
# BrowseComp-Plus question splits (D_pareto / D_feedback / test200 / test_rest): GEPA_SPLIT, default the copy in
# studies/prompt_optimisation/splits/ (original: <original repo>/data/bcp_gepa_splits/split.json).
SPLIT_F = os.environ.get("GEPA_SPLIT") or os.path.join(REPO, "studies", "prompt_optimisation", "splits", "bcp_gepa_splits", "split.json")
POLICY_DIR = os.path.join(str(path("RLM_DATA_DIR")), "bcp_policy")  # original: <original repo>/data/bcp_policy

SOURCES = {
    "q27b": [f"{path('FULLDOC_ROOT')}/runs/qwen27b_colbert8192_first20/*.json",   # original: <home>/suprlm/runs/...
             f"{path('FULLDOC_ROOT')}/runs/qwen27b_colbert8192_rest810/*.json"],
    "q36": [f"{path('LOGS_ROOT')}/gepa_bcp/rollouts/*/*.json", f"{path('LOGS_ROOT')}/gepa_bcp2/rollouts/*/*.json"],  # original: f"{R}/logs/..."
}


def main():
    from datasets import load_dataset
    rows = {str(r["query_id"]): r for r in load_dataset("Tevatron/browsecomp-plus", split="test")}
    test200 = set(json.load(open(SPLIT_F))["test200"])
    pool = sorted(set(rows) - test200, key=int)
    rng = random.Random(20261010)
    shuffled = pool[:]; rng.shuffle(shuffled)
    nval = len(shuffled) // 4
    split_of = {q: ("val" if i < nval else "train") for i, q in enumerate(shuffled)}
    os.makedirs(POLICY_DIR, exist_ok=True)
    out = open(f"{POLICY_DIR}/replay.jsonl", "w")
    st = collections.Counter(); seen = set()
    for src, pats in SOURCES.items():
        for pat in pats:
            for f in glob.glob(pat):
                if os.path.basename(f).startswith("_"):
                    continue
                try:
                    r = json.load(open(f))
                except Exception:
                    continue
                q = str(r.get("query_id") or os.path.basename(f)[:-5])
                if q in test200 or q not in rows:
                    st["skipped_test200_or_unknown"] += 1; continue
                gold = sorted(map(str, get_gold_doc_ids(rows[q]))); evid = sorted(map(str, get_evidence_doc_ids(rows[q])))
                for i, s in enumerate(r.get("search_log") or []):
                    query = str(s.get("query") or "").strip()
                    if not query:
                        continue
                    key = (q, query)
                    dup = key in seen; seen.add(key)
                    rec = {"qid": q, "split": split_of[q], "source": src, "hop": i, "seed_search": i == 0,
                           "query": query, "k": s.get("k"), "returned": [str(x) for x in (s.get("docids") or [])],
                           "gold": gold, "evidence": evid, "dup": dup}
                    out.write(json.dumps(rec) + "\n")
                    st["searches"] += 1; st[f"split_{rec['split']}"] += 1; st[f"src_{src}"] += 1
                    st["unique"] += 0 if dup else 1
                    hit = set(rec["returned"]) & set(gold)
                    st["with_gold_hit"] += bool(hit)
    out.close()
    for k in sorted(st):
        print(f"{k:28s} {st[k]}")
    print(f"questions in pool {len(pool)} (train {len(pool)-nval} / val {nval}); test200 excluded")


if __name__ == "__main__":
    main()
