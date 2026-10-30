"""YOUR official RLM harness, driven on OBLIQ-Bench instead of BrowseComp+.

NOT A FORK. examples/browsecomp_eval_official_rlm.py is untouched; this rebinds the four
row accessors it uses (get_question / get_gold_answer / get_gold_doc_ids /
get_evidence_doc_ids), the task suffix and the scorer, then calls its own main(). Every
prompt patch, tool, cap and instrumentation stays exactly as in the .8446/.7940 runs, so a
difference between this and the supervisor port is a difference of SCAFFOLD, not of plumbing.

THE ONE STRUCTURAL CHANGE. BrowseComp+ wants an exact answer string graded by an LLM judge;
OBLIQ wants an ORDERED LIST OF DOCUMENT IDS graded by NDCG@10 against qrels. So TASK_SUFFIX
is replaced (not edited) and judge_match becomes an NDCG computation. The harness's own
`gold_recall` field keeps working unchanged -- it is |gold n retrieved|/|gold|, which is
meaningful for ranking too -- and the NDCG is stashed per question for the summary.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples import browsecomp_data as BD  # noqa: E402
from examples import obliq_data as OD  # noqa: E402

TASK = os.environ.get("OBLIQ_TASK", "descriptive")
SUBSET = os.environ.get("OBLIQ_SUBSET", "twitter")
POOL = os.environ.get("OBLIQ_QRELS_POOL", "0") == "1"

_QRELS = OD.load_qrels(TASK, SUBSET, pool=POOL)
NDCG: dict[str, dict] = {}

# ── data layer ────────────────────────────────────────────────────────────────────────────
BD.get_question = lambda row: row["text"]
BD.get_gold_answer = lambda row: ""               # no gold string in a ranking task
BD.get_gold_doc_ids = lambda row: set(_QRELS.get(row["_id"], set()))
BD.get_evidence_doc_ids = lambda row: set(_QRELS.get(row["_id"], set()))


def _score(question, response, gold):
    """Stands in for judge_match. Returns a bool so the harness's bookkeeping still works,
    but the number that matters is the NDCG stashed alongside it."""
    s = OD.score_answer(response or "", set(gold_ids_current[0]), SUBSET)
    NDCG[qid_current[0]] = s
    return s["ndcg@10"] > 0.0


BD.judge_match = _score

_TASK = """

Rank documents from the corpus that satisfy the request.

The request describes a LATENT PATTERN. A document can satisfy it without sharing any of
its words, so judge each candidate against what the request MEANS, not how it is phrased.

Before searching, identify the implicit stance: what belief is implied WITHOUT being stated,
what rhetorical move signals it (irony, mock-praise, "so much for X", selective
juxtaposition, conspicuous omission, sarcastic agreement), and what event, topic or actor is
the context.

Each hop, search a DIFFERENT angle -- vary the rhetorical device, the topic framing, the
implied target. If the top score is below 0.20 the angle is wrong: switch immediately.

Set answer["content"] to a newline-separated list of doc_ids ranked STRONGEST TO WEAKEST,
then answer["ready"] = True. Output ONLY doc_ids, one per line, no prose and no numbering.
Order matters: the score is NDCG@10, so a satisfying document at rank 1 is worth more than
the same document at rank 10. Submit more than 10 if you have them -- only the top 10 are
scored, so extra candidates cost nothing.
"""

# ── PER-LM-CALL TIMEOUT ───────────────────────────────────────────────────────────────────
# rlm/clients/base_lm.py sets DEFAULT_TIMEOUT = 300.0 and this harness never overrides it.
# On OBLIQ that is nowhere near enough: questions run 1300-2400s, so the call raises
# APITimeoutError, the harness records predicted='ERROR' with 0 tokens, and the question is
# SCORED ZERO. Measured before this fix: 6/20 on the sweep and 24/48 on dev100 died this way
# -- including questions that had already reached gold_recall 0.50. Those are infrastructure
# failures being read as model failures, exactly the trap GEPA_ASSESSMENT s4 warns about.
#
# Injected into backend_kwargs, which get_client() forwards to OpenAIClient(**backend_kwargs)
# -> BaseLM(timeout=...). The library and the harness are both left unedited.
import rlm as _rlm  # noqa: E402

_LM_TIMEOUT = float(os.environ.get("LOCAL_LM_TIMEOUT", "1800"))
_orig_rlm_init = _rlm.RLM.__init__


def _rlm_init_with_timeout(self, *a, **kw):
    bk = kw.get("backend_kwargs")
    if isinstance(bk, dict) and "timeout" not in bk:
        bk = dict(bk)
        bk["timeout"] = _LM_TIMEOUT
        kw["backend_kwargs"] = bk
    # Qwen3.6 model card: presence_penalty=1.5 in thinking mode "to reduce endless repetitions";
    # vLLM cannot set it server-side. Opt-in (2026-10-13); unset = unchanged.
    if os.environ.get("OBLIQ_PRESENCE_PENALTY"):
        pp = float(os.environ["OBLIQ_PRESENCE_PENALTY"])
        for key in ("sampling_args", "sub_sampling_args"):
            kw[key] = {**(kw.get(key) or {}), "presence_penalty": pp}
    return _orig_rlm_init(self, *a, **kw)


_rlm.RLM.__init__ = _rlm_init_with_timeout
print(f"[lm-timeout] per-call ceiling {_LM_TIMEOUT:.0f}s (library default 300s)", flush=True)

# ── REPLACE THE STALE BROWSECOMP+ STRATEGY BLOCK ──────────────────────────────────────────
# The system prompt this harness builds (8 monkeypatches over the rlm library's
# RLM_SYSTEM_PROMPT) still carries BrowseComp+ search advice, because that is what the
# patches were written for. Materialised and split by scripts/official_prompt_modules.py it
# is six paragraphs, of which exactly one -- STRATEGY, 817 chars -- carries task knowledge
# rather than environment mechanics. On OBLIQ its advice is wrong in a specific way: it tells
# the agent to "commit to the best answer your evidence supports", but there is no single
# best answer here, there is an ORDERED LIST OF TEN scored by NDCG@10.
#
# So the arm was running OBLIQ instructions in the user message (TASK_SUFFIX) and
# BrowseComp+ instructions in the system prompt, contradicting each other. This swaps that
# one block for the OBLIQ analogue of dspy's _OBLIQ_TWITTER_STRATEGY, and freezes the other
# five. The split asserts byte-identical roundtrip before touching anything.
import importlib.util as _ilu  # noqa: E402

_mod_spec = _ilu.spec_from_file_location(
    "_opm", os.path.join(os.path.dirname(os.path.abspath(__file__)), "official_prompt_modules.py"))
_OPM = _ilu.module_from_spec(_mod_spec)
_mod_spec.loader.exec_module(_OPM)

OBLIQ_STRATEGY = (
    "As a general strategy, work out what the request MEANS before you search. It describes a "
    "LATENT PATTERN -- a stance, framing or rhetorical move -- so a document can satisfy it "
    "while sharing none of its words, and a document can share its words while satisfying "
    "nothing. Name the implied belief, the rhetorical device that signals it (irony, "
    "mock-praise, \"so much for X\", selective juxtaposition, conspicuous omission, sarcastic "
    "agreement) and the event or actor in view. The index is a dense semantic retriever: it "
    "matches meaning, not literal strings, and quotation marks have NO effect. Each hop, "
    "search a DIFFERENT angle -- vary the device, the framing, the implied target -- and if "
    "the top score is below 0.20 the angle is wrong, so switch rather than reword. You are "
    "not looking for one answer: you are building an ORDERED LIST, scored by NDCG@10, so keep "
    "every plausible candidate and spend your effort deciding which of them instantiate the "
    "pattern most strongly."
)

from examples import browsecomp_eval_official_rlm as H  # noqa: E402

# --- OBLIQ_FENCE_TOLERANT: accept a code block the model labelled with the wrong fence ---
# rlm/utils/parsing.py matches ONLY ```repl. Qwen3.8-27B obeys that 100% of the time, but
# Qwen3.6-35B-A3B labels 40% of ALL its turns ```python or a bare ``` -- it writes correct
# code and loses the turn on the label. Measured over 344 turns: 40% ok, 40% wrong fence,
# 18% prose, 3% truncated. That is a convention mismatch, not an ability gap, and scoring
# the model through it measures the fence, not the model.
#
# SAFETY: a non-repl fence is executed only if it parses as Python AND actually does
# something (assignment / call / import). That excludes the answer template the model also
# emits in bare fences ("Explanation: ...", a bare list of doc_ids), which must never be run.
# ```repl always wins when present, so the default path is untouched.
_FENCE_RESCUED = [0]
if os.environ.get("OBLIQ_FENCE_TOLERANT", "") == "1":
    import ast as _ast
    import re as _re2
    import rlm.core.rlm as _RC  # the call site binds find_code_blocks in ITS namespace

    _orig_fcb = _RC.find_code_blocks
    _FENCE_RE = _re2.compile(r"```([A-Za-z0-9_+-]*)[ \t]*\n(.*?)\n```", _re2.DOTALL)
    _DOES_SOMETHING = (_ast.Assign, _ast.AnnAssign, _ast.AugAssign, _ast.Call,
                       _ast.Import, _ast.ImportFrom, _ast.For, _ast.While, _ast.If)

    def _tolerant_find_code_blocks(text):
        blocks = _orig_fcb(text) if text else []
        if blocks or not text:
            return blocks
        out = []
        for lang, body in _FENCE_RE.findall(text):
            if lang.lower() == "repl":
                continue
            b = body.strip()
            if not b:
                continue
            try:
                tree = _ast.parse(b)
            except SyntaxError:
                continue
            if not any(isinstance(n, _DOES_SOMETHING) for n in _ast.walk(tree)):
                continue
            out.append(b)
        if out:
            _FENCE_RESCUED[0] += 1
        return out

    _RC.find_code_blocks = _tolerant_find_code_blocks
    print("[fence] tolerant code-block parsing ON (```repl still preferred)", flush=True)

# OBLIQ_STRATEGY_FILE lets an optimiser supply the STRATEGY block. This is the official
# harness's equivalent of the GEPA_PROMPTS hook on the supervisor's: GEPA writes a candidate
# to a file, points the env var at it, and every worker subprocess picks it up at import.
# Only STRATEGY is replaceable; the other five blocks are frozen, so a reflection cannot
# quietly rewrite the REPL mechanics and call it a strategy win.
_strategy_file = os.environ.get("OBLIQ_STRATEGY_FILE", "")
if _strategy_file:
    with open(_strategy_file, encoding="utf-8") as _sf:
        OBLIQ_STRATEGY = _sf.read().strip()

# OBLIQ_RELATIVE_SCORES=1 (2026-10-27, Gemini runs): no fixed score cutoffs in the strategy or the task contract.
from examples.obliq_score_advice import ENABLED as _REL_SCORES, relative as _relative  # noqa: E402
if _REL_SCORES:
    OBLIQ_STRATEGY = _relative(OBLIQ_STRATEGY)
    _TASK = _relative(_TASK)
    print("[prompt] score cutoffs replaced by relative-relevance advice (OBLIQ_RELATIVE_SCORES=1)", flush=True)

# GPTOSS_FINAL_CUE=1 appends one sentence to TURN_RULE. REQUIRED for gpt-oss-20b, a NO-OP
# for Qwen, so leave it unset on Qwen runs and the prompt is byte-identical to the baseline.
#
# WHY. gpt-oss uses OpenAI "harmony" channels: it reasons on an `analysis` channel and
# answers on a `final` channel. Given a system message it frequently stops after analysis,
# so message.content is None, rlm/clients/openai.py:118 returns that None unchanged, and
# rlm/utils/parsing.py:18 does re.finditer(pattern, None) -> TypeError.
#
# MEASURED dead-turn rate (no usable ```python``` block), 12 calls per cell:
#                                          low    medium   high
#   system + user  (unpatched)             50%     67%     50%
#   developer + user                       42%     58%     33%
#   single merged user message              0%      0%     25%
#   system + FINAL-channel cue  <- this     0%      0%     58%
# The cue is chosen over merging because it leaves the system/user structure the rlm
# library builds untouched. NOTE reasoning_effort=high degrades EVERY shape -- more
# analysis means more chances to stop before the final channel -- so do not raise it.
_FINAL_CUE = (" Put the python block in your FINAL response channel, never in your "
              "analysis: a turn whose code never reaches the final channel is a lost turn.")

if os.environ.get("OBLIQ_KEEP_BROWSECOMP_STRATEGY", "") != "1":
    _OPM.assert_roundtrip()
    _blocks = _OPM.split()
    _blocks["STRATEGY"] = OBLIQ_STRATEGY
    # OBLIQ_REPL_CUE=1 names the fence LABEL explicitly. TURN_RULE already says "never reply
    # with prose alone", and the 27B obeys it 100% of the time, so this is aimed at the
    # residual 18% of Qwen3.6 turns that answer in prose -- the part fence tolerance cannot
    # rescue, because there is no code to run.
    if os.environ.get("OBLIQ_REPL_CUE", "") == "1":
        _blocks["TURN_RULE"] = _blocks["TURN_RULE"].rstrip() + (
            " The fence label matters: write ```repl, never ```python and never a bare ```. "
            "A turn whose code is not in a ```repl``` block is discarded and its work is "
            "lost -- including the final turn, where answer[\"content\"] and "
            "answer[\"ready\"] = True must be assigned inside ```repl```.")
        print("[cue] repl-fence cue appended to TURN_RULE", flush=True)
    if os.environ.get("GPTOSS_FINAL_CUE", "") == "1":
        _blocks["TURN_RULE"] = _blocks["TURN_RULE"].rstrip() + _FINAL_CUE
        print(f"[cue] harmony final-channel cue appended to TURN_RULE "
              f"(+{len(_FINAL_CUE)} chars)", flush=True)
    _OPM.apply(_blocks)
    print(f"[strategy] BrowseComp+ block swapped for OBLIQ ({len(OBLIQ_STRATEGY)} chars)", flush=True)


H.get_question = BD.get_question
H.get_gold_answer = BD.get_gold_answer
H.get_gold_doc_ids = BD.get_gold_doc_ids
H.get_evidence_doc_ids = BD.get_evidence_doc_ids
H.judge_match = _score
# --- OBLIQ_TASK_FILE: swap the answer contract for an A/B arm -------------------
# The baseline contract ends with "Submit more than 10 if you have them -- only the top 10
# are scored, so extra candidates cost nothing." Measured consequence: the agent submits
# 22.6 doc_ids and 22% of the gold it DID find sits below rank 10, scoring nothing.
# Perfect ordering of that same submitted list would score .2280 against the actual .1302.
# This hook lets an arm replace the contract without touching the baseline file.
_TASK_FILE = os.environ.get("OBLIQ_TASK_FILE")
if _TASK_FILE:
    _TASK = open(_TASK_FILE).read()
    if _REL_SCORES:
        _TASK = _relative(_TASK)
    print(f"[task] answer contract overridden from {_TASK_FILE} ({len(_TASK)} chars)", flush=True)
H.TASK_SUFFIX = _TASK

# --- OBLIQ_NO_FORMAT_GATE=1 (2026-10-22): turn off the BrowseComp+ submission contract ---------------
# examples/browsecomp_eval_official_rlm.py rejects any answer without an "Explanation:" and an "Exact Answer:"
# line (up to MAX_REJECTS times). OBLIQ's contract is the opposite -- doc_ids only, one per line -- so a correct
# submission is bounced: on OBLIQ Math 136/151 snippet-RLM questions were rejected at least once, and the agent
# then rewrote its ranking as prose (in one smoke run cutting a 100+ id list to 10). The rejection counter is
# also a module global, so it stops enforcing after 3 rejections per shard process. MAX_REJECTS is read at call
# time, so 0 disables the gate. Unset = unchanged.
if os.environ.get("OBLIQ_NO_FORMAT_GATE") == "1":
    H.MAX_REJECTS = 0
    print("[gate] BrowseComp+ Explanation/Exact Answer submission gate OFF", flush=True)

qid_current = [None]
gold_ids_current = [set()]
tag_current = [""]

# OBLIQ_TAG_SEARCH=1 (2026-10-13): add {"tag": "<qid>-<uuid>"} to every /search so a search-policy
# proxy (scripts/bcp_rerank_proxy.py WIRE=mine) knows the question (rerank context, novelty).
# multi_search_server ignores unknown fields; unset = requests unchanged.
if os.environ.get("OBLIQ_TAG_SEARCH") == "1":
    import types as _types
    import uuid as _uuid
    _real_requests = H.requests

    def _tagged_post(url, *a, json=None, **kw):
        if url.endswith("/search") and isinstance(json, dict):
            json = dict(json, tag=tag_current[0])
        return _real_requests.post(url, *a, json=json, **kw)

    H.requests = _types.SimpleNamespace(post=_tagged_post, get=_real_requests.get)
    print("[tag] /search requests carry the question tag", flush=True)

_orig_run_one = H.run_one


def run_one(row, q_idx, args):
    qid_current[0] = row["_id"]
    if os.environ.get("OBLIQ_TAG_SEARCH") == "1":
        tag_current[0] = f"{row['_id']}-{_uuid.uuid4().hex[:8]}"
    gold_ids_current[0] = set(_QRELS.get(row["_id"], set()))
    rec = _orig_run_one(row, q_idx, args)
    rec["query_id"] = row["_id"]
    rec.update({k: v for k, v in (NDCG.get(row["_id"]) or {}).items()
                if k.startswith(("ndcg@", "recall@")) or k == "n_parsed"})
    print(f"    [{row['_id']}] ndcg@10={rec.get('ndcg@10', 0):.4f} "
          f"recall@10={rec.get('recall@10', 0):.4f} parsed={rec.get('n_parsed', 0)}", flush=True)
    return rec


H.run_one = run_one


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=20)
    p.add_argument("--max-iterations", type=int, default=100)
    p.add_argument("--max-depth", type=int, default=1)
    p.add_argument("--compaction", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--ids-file", default=None,
                   help="json list of query ids (the dev/held-out split); overrides --start/--end")
    args = p.parse_args()

    rows = OD.load_queries(TASK, SUBSET)
    if args.ids_file:
        want = set(json.load(open(args.ids_file)))
        rows = [r for r in rows if r["_id"] in want]
        sel = list(enumerate(rows))
    else:
        sel = [(i, rows[i]) for i in range(args.start, min(args.end, len(rows)))]
    # Checkpoint after every question and resume from it: a reboot on 2026-10-13 lost a whole
    # in-progress run because results were only written at the end.
    from pathlib import Path
    ckpt = Path(f"{args.out}_checkpoint.json")
    results = json.loads(ckpt.read_text()) if ckpt.exists() else []
    done = {r.get("_id") or r.get("query_id") for r in results}
    for i, row in sel:
        if row["_id"] in done:
            continue
        results.append(run_one(row, i, args))
        ckpt.write_text(json.dumps(results, indent=2))
    Path(f"{args.out}.json").write_text(json.dumps(results, indent=2))
    n = len(results)
    if n:
        print(f"\n{args.out}: n={n}  "
              f"NDCG@10={sum(r.get('ndcg@10', 0) for r in results)/n:.4f}  "
              f"recall@10={sum(r.get('recall@10', 0) for r in results)/n:.4f}")


if __name__ == "__main__":
    main()
