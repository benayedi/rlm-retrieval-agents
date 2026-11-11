"""Search-policy study: offline test of Qwen3-Reranker over ColBERT's deep list (does more gold reach the agent's top-k?).

From original/RLM/scripts/bcp_rerank_eval.py; changed: import wiring, locations from configs/paths.env, this
paragraph. The reranker prompt (PREFIX, SUFFIX, INSTR) is the original's and the same as src/rerank_proxy.py's.
Needs torch, transformers and a GPU; reads BrowseComp-Plus and its corpus from the Hugging Face hub and
<RLM_DATA_DIR>/bcp_policy/{lists.jsonl, hist.json}. Note the default --model is the 0.6B reranker; the 4B model
of the proxy is --model Qwen/Qwen3-Reranker-4B.
  python analysis/rerank_eval.py --model Qwen/Qwen3-Reranker-4B --depth 30 --n 300 [--novelty]

Original docstring follows.

Offline test: does re-ranking ColBERT's deep list with Qwen3-Reranker put more gold in the agent's top-k?

Motivation (bcp_policy_eval.py, first 55 cached queries): for 27% of agent searches a gold doc
sits in ColBERT's top-100 but not in the top-k (median k 8) the agent actually receives, while
BM25 fusion adds almost nothing. A re-ranker that scores each candidate against the FULL
QUESTION (not just the agent's short query) could promote those buried gold docs.

For each cached query: candidates = ColBERT top-DEPTH; score each (query-text, doc) with
Qwen3-Reranker (yes/no logit, official prompt); take the top-k. Compares, at the agent's k:
  colbert        production order
  rr(query)      re-ranked against the agent's query
  rr(question)   re-ranked against the full question
  rr(q+query)    re-ranked against question + agent query
Docs are truncated to MAX_DOC_TOK tokens (BrowseComp+ pages are long; the reranker reads the head).
"""
import argparse, json, os, statistics as st, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import wiring: agents/shared/rlm_shared is located from this file; locations come from configs/paths.env
# (rlm_shared.paths.path). Original: absolute paths of the machine the experiments ran on.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir, "agents", "shared"))
from rlm_shared.paths import path  # noqa: E402
REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir))
POLICY_DIR = os.path.join(str(path("RLM_DATA_DIR")), "bcp_policy")  # original: <original repo>/data/bcp_policy

PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
          "Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INSTR = "Given a web search query for a multi-hop research question, judge whether this document contains evidence needed to answer it"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-Reranker-0.6B")
    ap.add_argument("--depth", type=int, default=30)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--split", default="val")
    ap.add_argument("--max-doc-tok", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--novelty", action="store_true", help="also score NEW-gold under novelty (+rerank) using data/bcp_policy/hist.json")
    a = ap.parse_args()
    from datasets import load_dataset
    from rlm_shared.browsecomp_data import get_question   # original: from examples.browsecomp_data
    recs = [json.loads(l) for l in open(f"{POLICY_DIR}/lists.jsonl")]
    recs = [r for r in recs if a.split == "all" or r["split"] == a.split][: a.n]
    qrows = {str(r["query_id"]): r for r in load_dataset("Tevatron/browsecomp-plus", split="test")}
    need = {d for r in recs for d in r["colbert"][: a.depth]}
    corpus = {}
    for row in load_dataset("Tevatron/browsecomp-plus-corpus", split="train"):
        if str(row["docid"]) in need:
            corpus[str(row["docid"])] = row["text"]
    tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(a.device).eval()
    yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    pre, suf = tok(PREFIX, add_special_tokens=False)["input_ids"], tok(SUFFIX, add_special_tokens=False)["input_ids"]

    def doc_ids(text):
        return tok(text, add_special_tokens=False, truncation=True, max_length=a.max_doc_tok)["input_ids"]

    @torch.no_grad()
    def scores(qtext, docs):
        out = []
        for i in range(0, len(docs), a.batch):
            seqs = []
            for d in docs[i:i + a.batch]:
                body = tok(f"<Instruct>: {INSTR}\n<Query>: {qtext}\n<Document>: ", add_special_tokens=False)["input_ids"]
                seqs.append(pre + body + doc_ids(d) + suf)
            enc = tok.pad({"input_ids": seqs}, padding=True, return_tensors="pt").to(a.device)
            logits = model(**enc, logits_to_keep=1).logits[:, -1, :]
            out += torch.log_softmax(torch.stack([logits[:, no], logits[:, yes]], 1).float(), 1)[:, 1].exp().tolist()
        return out

    res = {m: [] for m in ("colbert", "rr(query)", "rr(q+query)")}
    hist = json.load(open(f"{POLICY_DIR}/hist.json")) if a.novelty else {}
    nres = {m: [] for m in ("new: colbert", "new: novelty", "new: novelty+rr(q+query)")}
    t0 = time.time()
    for n, r in enumerate(recs, 1):
        k = max(1, int(r.get("agent_k") or 5)); g = set(r["gold"])
        cand = [d for d in r["colbert"][: a.depth] if d in corpus]
        texts = [corpus[d] for d in cand]
        question = get_question(qrows[r["qid"]])
        res["colbert"].append(bool(g & set(r["colbert"][:k])))
        for m, qtext in (("rr(query)", r["query"]),
                         ("rr(q+query)", f"{question}\nCurrent search: {r['query']}")):
            sc = scores(qtext, texts)
            top = [d for _, d in sorted(zip(sc, cand), key=lambda x: -x[0])][:k]
            res[m].append(bool(g & set(top)))
        if a.novelty and not r.get("seed_search"):
            seen = set(hist.get(f"{r['qid']}\t{r['query']}", [])); gn = g - seen
            unseen = [d for d in cand if d not in seen]
            sc = scores(f"{question}\nCurrent search: {r['query']}", [corpus[d] for d in unseen]) if unseen else []
            nres["new: colbert"].append(bool(gn & set(r["colbert"][:k])))
            nres["new: novelty"].append(bool(gn & set(unseen[:k])))
            nres["new: novelty+rr(q+query)"].append(bool(gn & set([d for _, d in sorted(zip(sc, unseen), key=lambda x: -x[0])][:k])))
        if n % 25 == 0:
            print(f"{n}/{len(recs)} ({(time.time()-t0)/n:.1f}s/query) " +
                  " | ".join(f"{m} {st.mean(v):.3f}" for m, v in res.items()), flush=True)
    print(f"FINAL n={len(recs)} depth={a.depth} model={a.model.split('/')[-1]}: " +
          " | ".join(f"{m} hit@k {st.mean(v):.3f}" for m, v in res.items()), flush=True)
    if a.novelty and nres["new: colbert"]:
        print(f"NEW-GOLD (agent-written, n={len(nres['new: colbert'])}): " +
              " | ".join(f"{m} {st.mean(v):.3f}" for m, v in nres.items()), flush=True)


if __name__ == "__main__":
    main()
