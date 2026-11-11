"""Search-policy study, FRAMES: offline test of Qwen3-Reranker-4B over BM25's top 30 (the offline evidence behind SP-4).

From original/RLM/scripts/frames_rerank_eval.py; changed: import wiring, locations from configs/paths.env, this
paragraph. Replays the agent's searches of the FR-13 run (<LOGS_ROOT>/frames/sup_q36_bm25f) on the feedback and
pareto questions of studies/prompt_optimisation/splits/frames/gepa_splits/split.json, against the FRAMES BM25 server on :8041 (unchanged
URL). Writes --out (default <RUNS_DIR>/frames/rerank_eval.json; original <original repo>/logs/frames/; the directory
must exist). Needs torch, transformers and a GPU.
  python analysis/frames_rerank_eval.py --n 400

Original docstring follows.

Offline test on FRAMES: does Qwen3-Reranker over BM25's top-DEPTH put more gold articles in the
agent's top-k? (FRAMES counterpart of scripts/bcp_rerank_eval.py, which found +8.7 hit@k on
BrowseComp+ and preceded a +11 point end-to-end gain.)

Searches replayed: the agent's own searches in the 824-question seed run (logs/frames/sup_q36_bm25f,
supervisor harness, facts-only prompt), from FEEDBACK + PARETO questions only -- the test split is
kept for the end-to-end comparison. For each search: BM25 top-DEPTH for the agent's query from the
:8041 server (untagged, so nothing is counted), then, at the agent's own k:
  bm25                 production order
  rr(q+query) head     re-ranked against "<question>\\nCurrent search: <query>", doc = first
                       MAX_DOC_TOK tokens (what scripts/bcp_rerank_proxy.py does)
  rr(q+query) window   same, doc = title + the WINDOW-char window with most question/query terms
                       (Wikipedia articles are long; the needed fact is often far from the head)
NEW-gold variants count only gold articles this run had not been shown before that search, with
and without novelty (dropping already-seen articles before taking the top-k).

  CUDA_VISIBLE_DEVICES=0 python scripts/frames_rerank_eval.py --n 400
"""
import argparse, collections, json, os, random, re, statistics as st, sys, time, urllib.request

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import wiring: agents/shared/rlm_shared is located from this file; locations come from configs/paths.env
# (rlm_shared.paths.path). Original: absolute paths of the machine the experiments ran on.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir, "agents", "shared"))
from rlm_shared.paths import path  # noqa: E402
REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir))
SEED_RUN = f"{path('LOGS_ROOT')}/frames/sup_q36_bm25f"  # original: f"{R}/logs/frames/sup_q36_bm25f"
PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
          "Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INSTR = "Given a web search query for a multi-hop research question, judge whether this document contains evidence needed to answer it"
STOP = set("the a an of in on at to for and or is was were be by with from as that this which who what when where how "
           "his her its their they he she it are has had have not but into than then also about after before".split())


def bm25_top(query: str, depth: int) -> list[tuple[str, str]]:
    req = urllib.request.Request("http://127.0.0.1:8041/search", method="POST",
                                 data=json.dumps({"query": query, "k": depth, "mode": "full", "untagged_ok": True}).encode(),
                                 headers={"Content-Type": "application/json"})
    return [(h["docid"], h["text"]) for h in json.load(urllib.request.urlopen(req, timeout=300))["results"]]


def terms(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2 and w not in STOP}


def best_window(text: str, qterms: set[str], window: int) -> str:
    title, _, body = text.partition("\n")
    if len(body) <= window:
        return text
    best, best_s = body[:window], -1
    for i in range(0, len(body) - window // 2, window // 2):
        w = body[i:i + window]
        s = len(qterms & terms(w))
        if s > best_s:
            best, best_s = w, s
    return f"{title}\n{best}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-Reranker-4B")
    ap.add_argument("--depth", type=int, default=30)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--max-doc-tok", type=int, default=1024)
    ap.add_argument("--window", type=int, default=4000, help="chars (~1,000 tokens)")
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=f"{path('RUNS_DIR')}/frames/rerank_eval.json")
    a = ap.parse_args()

    rows = {str(r["i"]): r for r in json.load(open(f"{path('FRAMES_DATA_DIR')}/frames_rows.json"))}
    sp = json.load(open(os.path.join(REPO, "studies", "prompt_optimisation", "splits", "frames", "gepa_splits", "split.json")))  # original: <original repo>/data/frames/gepa_splits/split.json
    qids = sp["feedback"] + sp["pareto"]
    recs = []
    for q in qids:
        try:
            raw = json.load(open(f"{SEED_RUN}/{int(q):03d}.json"))
        except Exception:
            continue
        seen: set[str] = set()
        for i, s in enumerate(raw.get("search_log") or []):
            if not isinstance(s, dict) or not s.get("query"):
                continue
            recs.append({"qid": q, "query": s["query"], "k": int(s.get("k") or 5), "seed": i == 0,
                         "seen": sorted(seen), "gold": set(map(str, rows[q]["gold_ids"]))})
            seen |= set(map(str, s.get("docids") or []))
    random.Random(0).shuffle(recs)
    recs = recs[: a.n]
    print(f"{len(recs)} searches from {len({r['qid'] for r in recs})} feedback/pareto questions", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(a.device).eval()
    yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    pre, suf = tok(PREFIX, add_special_tokens=False)["input_ids"], tok(SUFFIX, add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def scores(qtext, docs):
        out = []
        body = tok(f"<Instruct>: {INSTR}\n<Query>: {qtext}\n<Document>: ", add_special_tokens=False)["input_ids"]
        for i in range(0, len(docs), a.batch):
            seqs = [pre + body + tok(d, add_special_tokens=False, truncation=True, max_length=a.max_doc_tok)["input_ids"] + suf
                    for d in docs[i:i + a.batch]]
            enc = tok.pad({"input_ids": seqs}, padding=True, return_tensors="pt").to(a.device)
            lg = model(**enc, logits_to_keep=1).logits[:, -1, :]   # last position only (full logits OOM'd vLLM's GPU once)
            out += torch.log_softmax(torch.stack([lg[:, no], lg[:, yes]], 1).float(), 1)[:, 1].exp().tolist()
        return out

    M = ("bm25", "rr head", "rr window")
    hit = {m: [] for m in M}
    new = {m: [] for m in ("bm25", "novelty", "novelty+rr window")}
    any30 = []
    t0 = time.time()
    for n, r in enumerate(recs, 1):
        cand = bm25_top(r["query"], a.depth)
        ids = [d for d, _ in cand]
        k, g = r["k"], r["gold"]
        question = rows[r["qid"]]["question"]
        qtext = f"{question}\nCurrent search: {r['query']}"
        qterms = terms(question) | terms(r["query"])
        any30.append(bool(g & set(ids)))
        hit["bm25"].append(bool(g & set(ids[:k])))
        sc_head = scores(qtext, [t for _, t in cand])
        wins = [best_window(t, qterms, a.window) for _, t in cand]
        sc_win = scores(qtext, wins)
        rank = lambda sc, pool: [d for _, d in sorted(zip(sc, pool), key=lambda x: -x[0])]
        hit["rr head"].append(bool(g & set(rank(sc_head, ids)[:k])))
        hit["rr window"].append(bool(g & set(rank(sc_win, ids)[:k])))
        if not r["seed"]:
            seen = set(r["seen"]); gn = g - seen
            unseen = [i for i, d in enumerate(ids) if d not in seen]
            new["bm25"].append(bool(gn & set(ids[:k])))
            new["novelty"].append(bool(gn & set([ids[i] for i in unseen][:k])))
            new["novelty+rr window"].append(bool(gn & set(rank([sc_win[i] for i in unseen], [ids[i] for i in unseen])[:k])))
        if n % 25 == 0:
            print(f"{n}/{len(recs)} ({(time.time()-t0)/n:.1f}s/search) any-gold-in-top{a.depth} {st.mean(any30):.3f} | " +
                  " | ".join(f"{m} {st.mean(v):.3f}" for m, v in hit.items()), flush=True)
    res = {"n": len(recs), "depth": a.depth, "model": a.model, "gold_in_topdepth": st.mean(any30),
           "hit@k": {m: st.mean(v) for m, v in hit.items()},
           "new_gold@k (agent-written searches)": {m: st.mean(v) for m, v in new.items()}, "n_new": len(new["bm25"])}
    json.dump(res, open(a.out, "w"), indent=1)
    print("FINAL " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
