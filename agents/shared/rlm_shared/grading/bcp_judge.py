"""BrowseComp-Plus gpt-4.1 judge (library module of the graders in this package).

Origin: scripts/bcp_judge.py of the original repository (agents/shared/original/RLM/scripts/bcp_judge.py). Changed only:
import wiring, the AZ_ENV path read through rlm_shared.paths, the judge deployment default read from
AZURE_DEPLOYMENT_JUDGE, and this documentation. See agents/shared/CHANGES_FROM_ORIGINAL.md.

Imported, not run:  from rlm_shared.grading.bcp_judge import grade_outputs   (AZ_ENV must name the Azure key file).

DEPENDENCY ON THE DOCUMENT REPL HARNESS. GRADER_TEMPLATE, OFFICIAL_TEMPLATE and parse_judge are imported from the document REPL
RLM's judge module, agents/document_repl/src/judge.py (original: FULLDOC/src/judge.py, see
agents/document_repl/original/FULLDOC/src/judge.py). Its directory is put on sys.path from this file's location,
and the module is imported under its plain name `judge`, as in the original. Importing judge.py runs its own import-time
code (in the original: load_dotenv(<FULLDOC_ROOT>/.env); see that module's CHANGES_FROM_ORIGINAL.md).

SIDE EFFECTS AT IMPORT (kept from the original, documented rather than changed): importing this module reads the
AZ_ENV file, downloads/loads the Tevatron/browsecomp-plus test split with datasets.load_dataset and decrypts every
question and answer into GT, and builds the AzureOpenAI client CLI. No request is sent until grade_outputs is called.
Tests stub datasets.load_dataset and openai before importing it.

Original docstring follows.

BrowseComp-Plus judge on our Azure key, shared by scripts/bcp_rejudge_gpt41.py, scripts/bcp_pilot_grade.py and
scripts/bcp_bm25_gpt5_grade.py (2026-10-27). Judge model JUDGE_MODEL, default gpt-4.1 (deployment gpt-4.1, served
gpt-4.1-2025-04-14), the BrowseComp-Plus paper's judge and the one the document REPL GPT-5 runs were graded with.

The document REPL agent's judge.py (suprlm/src/judge.py) cannot run as-is here (its data/prepared/queries.jsonl has no answers), so its
templates and parse_judge are imported and applied the same way: 3 votes, majority; an answer with no completed status
or empty output is wrong without judging. Gold answers are decrypted from the dataset. Template JUDGE_TEMPLATE, default
"official" (OFFICIAL_TEMPLATE, the BrowseComp-Plus repo's grader, which allows string variants of the gold answer):
the document REPL GPT-5 runs were graded with it ("official BrowseComp-Plus judge template at three votes"); with the stricter
"paper" GRADER_TEMPLATE, gpt-4.1 failed 39 of the 726 correct answers of the BCP-23 run (row 11 of the earlier workbook).
A question whose votes all fail (API error or unparsable reply) is retried, never silently scored wrong or cached. A
prompt blocked by Azure's content filter cannot be judged by gpt-4.1 at all: it returns "filtered" and the caller
decides (scripts/bcp_rejudge_gpt41.py keeps the old verdict and counts it).
Verdicts are cached on (template, judge model, question, normalised answer text).
"""
import hashlib, json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Refactor (import wiring): the original put <home>/suprlm/src (the document REPL harness, for judge.py) and the
# original repository root (for examples.browsecomp_data) on sys.path. Here: <repo>/agents/document_repl/src and
# <repo>/agents/shared, both computed from this file (<repo>/agents/shared/rlm_shared/grading/bcp_judge.py).
_REPO = Path(__file__).resolve().parents[4]
FULLDOC_SRC = _REPO / "agents" / "document_repl" / "src"
sys.path.insert(0, str(FULLDOC_SRC)); sys.path.insert(0, str(_REPO / "agents" / "shared"))
import openai  # noqa: E402
from judge import GRADER_TEMPLATE, OFFICIAL_TEMPLATE, parse_judge  # noqa: E402
from datasets import load_dataset  # noqa: E402
from rlm_shared.browsecomp_data import get_question, get_gold_answer  # noqa: E402
from rlm_shared.paths import path, setting  # noqa: E402

_env = {}
for _l in open(path("AZ_ENV")):
    _k, _, _v = _l.strip().partition("=")
    _env[_k] = _v
GT = {str(r["query_id"]): (get_question(r), get_gold_answer(r)) for r in load_dataset("Tevatron/browsecomp-plus", split="test")}
CLI = openai.AzureOpenAI(azure_endpoint=_env["AZ_EP"] + "/", api_key=_env["AZ_KEY"], api_version="2024-12-01-preview",
                         timeout=180, max_retries=10)
_LK = threading.Lock()
# Refactor: the default was the literal deployment name "gpt-4.1"; it is now AZURE_DEPLOYMENT_JUDGE (gpt-4.1 in
# configs/paths.example.env). JUDGE_MODEL, when set (even to ""), still wins exactly as before. MODEL is part of the
# verdict-cache key, so a deployment with another name does not reuse caches written under "gpt-4.1".
MODEL = os.environ["JUDGE_MODEL"] if "JUDGE_MODEL" in os.environ else setting("AZURE_DEPLOYMENT_JUDGE")
TEMPLATE = os.environ.get("JUDGE_TEMPLATE", "official")
TMPL = OFFICIAL_TEMPLATE if TEMPLATE == "official" else GRADER_TEMPLATE


def _grade_one(q, out, cache):
    if not out.strip():
        return q, False
    key = hashlib.sha256(f"{TEMPLATE}|{MODEL}|{q}|{' '.join(out.lower().split())}".encode()).hexdigest()[:24]
    with _LK:
        if key in cache:
            return q, cache[key]
    prompt = TMPL.format(question=GT[q][0], response=out, correct_answer=GT[q][1])
    # 3 votes, each retried up to 3 times on an exception or an unparsable reply; majority of the parsed votes.
    # A content-filter error on any attempt returns "filtered" at once (callers decide what that means).
    votes = []
    for _ in range(3):
        for _try in range(3):
            try:
                r = CLI.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}],
                                                max_completion_tokens=2048)
                v = parse_judge(r.choices[0].message.content or "")
                if not v.get("parse_error"):
                    votes.append(v.get("correct") == "yes"); break
            except Exception as e:  # noqa: BLE001
                if "content management policy" in str(e) or "content_filter" in str(e):
                    return q, "filtered"
                print(f"[judge] {q}: {type(e).__name__} {str(e)[:200]}", file=sys.stderr, flush=True)
    if not votes:
        return q, None
    ok = sum(votes) * 2 > len(votes)
    with _LK:
        cache[key] = ok
    return q, ok


def grade_outputs(outputs, cache_path, workers=32):
    """outputs: {query_id: answer text ('' = no answer)} -> {query_id: correct}"""
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    with ThreadPoolExecutor(workers) as ex:
        res = dict(ex.map(lambda kv: _grade_one(kv[0], kv[1], cache), outputs.items()))
    for _round in range(5):                      # questions whose votes all failed: retry, slower
        bad = [q for q, v in res.items() if v is None]
        if not bad:
            break
        print(f"[judge] retrying {len(bad)} questions with no vote", file=sys.stderr, flush=True)
        with ThreadPoolExecutor(4) as ex:
            res.update(ex.map(lambda q: _grade_one(q, outputs[q], cache), bad))
    bad = [q for q, v in res.items() if v is None]
    if bad:
        json.dump(cache, open(cache_path, "w"))
        raise RuntimeError(f"no judge vote for {len(bad)} questions: {bad[:10]}")
    json.dump(cache, open(cache_path, "w"))
    return res


def run_outputs(run_dir):
    """{query_id: answer text} for a run directory of run_one.py results; non-completed runs count as ''."""
    o = {}
    for f in os.listdir(run_dir):
        if f.endswith(".json") and not f.startswith("_"):
            x = json.load(open(f"{run_dir}/{f}"))
            out = ((x.get("result") or [{}])[0] or {}).get("output", "") or ""
            o[str(x["query_id"])] = out if x.get("status") == "completed" else ""
    return o
