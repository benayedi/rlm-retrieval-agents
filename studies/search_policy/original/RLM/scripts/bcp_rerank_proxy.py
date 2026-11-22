"""Search-policy proxy for the supervisor harness: ColBERT -> (novelty) -> Qwen3-Reranker -> top-k.

WHY (2026-10-10, scripts/bcp_rerank_eval.py, 400 val queries): re-ranking ColBERT's top-30 with
Qwen3-Reranker-4B against the FULL QUESTION + the agent's query puts a gold doc in the agent's
top-k for ~8 points more searches (0.41 -> 0.49 at 300 queries); against the query alone it does
~nothing. Novelty (skip docs this run already received; ~31% of returned docs were repeats)
adds ~2 points of new-gold searches at zero cost.

HOW. The harness posts /search {query, k, mode, tag} with tag = "<query_id>-<uuid>". The proxy:
  1. asks the real retriever (:8766) for the top DEPTH docs, UNTAGGED (so upstream counts are not
     polluted by the docs the agent never sees);
  2. POLICY=novelty+rerank: drops docs already returned to this tag (falls back to them if fewer
     than k unseen remain);
  3. scores each candidate with Qwen3-Reranker against "<question>\\nCurrent search: <query>"
     (question looked up from the query_id in the tag);
  4. returns the top k in the upstream's own format, and tracks per-tag returned docids so
     /tag_docids and /tag_counts report what the AGENT actually received (the harness uses them
     for retrieved_docids and gold recall).
Agent, prompt, caps and model are untouched: only --server changes. POLICY=passthrough is a
byte-for-byte control.

Run:  CUDA_VISIBLE_DEVICES=0 POLICY=novelty+rerank python bcp_rerank_proxy.py --port 8768
"""
import argparse, json, os, sys, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

R = "/home/extstu_00/RLM"
sys.path.insert(0, R)
UP = os.environ.get("UPSTREAM", "http://127.0.0.1:8766")
POLICY = os.environ.get("POLICY", "novelty+rerank")      # passthrough | novelty | rerank | novelty+rerank
DEPTH = int(os.environ.get("DEPTH", "30"))
MODEL = os.environ.get("RERANK_MODEL", "Qwen/Qwen3-Reranker-4B")
MAX_DOC_TOK = int(os.environ.get("MAX_DOC_TOK", "1024"))
BATCH = int(os.environ.get("RERANK_BATCH", "15"))
DEVICE = os.environ.get("RERANK_DEVICE", "cuda")              # cuda:1 = beside ColBERT, away from vLLM's GPU
TOK_BUDGET = int(os.environ.get("RERANK_TOK_BUDGET", "16000"))  # cross-request batching: max padded tokens per forward
PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
          "Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INSTR = os.environ.get("RERANK_INSTR", "Given a web search query for a multi-hop research question, judge whether this document contains evidence needed to answer it")

QUESTIONS, TAGS, LOCK, GPU = {}, {}, threading.Lock(), threading.Lock()
STATS = {"searches": 0, "reranked": 0, "novelty_filtered": 0, "rerank_s": 0.0, "fallback_seen": 0}
M = {}


def load():
    if os.environ.get("QUERIES_FILE"):          # another benchmark (OBLIQ): {"query_id", "query"} per line
        for line in open(os.environ["QUERIES_FILE"], encoding="utf-8"):
            d = json.loads(line); QUESTIONS[str(d["query_id"])] = d["query"]
    else:
        from datasets import load_dataset
        from examples.browsecomp_data import get_question
        for r in load_dataset("Tevatron/browsecomp-plus", split="test"):
            QUESTIONS[str(r["query_id"])] = get_question(r)
    if POLICY in ("rerank", "novelty+rerank"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
        # share GPU 0 with vLLM: hard-cap this process so a reranker spike can never take vLLM's memory
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("RERANK_MEM_FRACTION", "0.12")),
                                                   None if DEVICE == "cuda" else torch.device(DEVICE))
        if DEVICE != "cuda":
            torch.cuda.set_device(DEVICE)
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
        M.update(tok=tok, model=model, torch=torch, yes=tok.convert_tokens_to_ids("yes"), no=tok.convert_tokens_to_ids("no"),
                 pre=tok(PREFIX, add_special_tokens=False)["input_ids"], suf=tok(SUFFIX, add_special_tokens=False)["input_ids"])


def _forward(seqs):
    tok, model, torch = M["tok"], M["model"], M["torch"]
    enc = tok.pad({"input_ids": seqs}, padding=True, return_tensors="pt").to(DEVICE)
    lg = model(**enc, logits_to_keep=1).logits[:, -1, :]   # last position only: full-sequence logits OOM'd GPU 0 and killed vLLM (2026-10-10)
    return torch.log_softmax(torch.stack([lg[:, M["no"]], lg[:, M["yes"]]], 1).float(), 1)[:, 1].exp().tolist()


# Cross-request batching (2026-10-11): under 20 agents the per-search lock made searches queue
# (~20 s/search vs 2.4 s idle). One GPU worker drains every pending (search, doc) pair, sorts them
# by length (less padding) and packs forwards up to TOK_BUDGET padded tokens.
import queue
JOBS = queue.Queue()


def _worker():
    torch = M["torch"]
    while True:
        jobs = [JOBS.get()]
        while True:
            try:
                jobs.append(JOBS.get_nowait())
            except queue.Empty:
                break
        items = [(len(s), j, i, s) for j in jobs for i, s in enumerate(j["seqs"])]
        items.sort(key=lambda x: x[0])
        try:
            with torch.no_grad():
                b = 0
                while b < len(items):
                    e = b + 1
                    while e < len(items) and items[e][0] * (e - b + 1) <= TOK_BUDGET:
                        e += 1
                    for (_, j, i, _), sc in zip(items[b:e], _forward([x[3] for x in items[b:e]])):
                        j["out"][i] = sc
                    b = e
        except Exception as ex:
            for j in jobs:
                j["err"] = ex
        for j in jobs:
            j["done"].set()


def rerank_scores(qtext, texts):
    tok = M["tok"]
    body = tok(f"<Instruct>: {INSTR}\n<Query>: {qtext}\n<Document>: ", add_special_tokens=False)["input_ids"]
    seqs = [M["pre"] + body + tok(t, add_special_tokens=False, truncation=True, max_length=MAX_DOC_TOK)["input_ids"] + M["suf"]
            for t in texts]
    if os.environ.get("RERANK_XBATCH", "1") == "0":       # old path: one search at a time, fixed batches
        with GPU, M["torch"].no_grad():
            return [s for i in range(0, len(seqs), BATCH) for s in _forward(seqs[i:i + BATCH])]
    j = {"seqs": seqs, "out": [None] * len(seqs), "done": threading.Event(), "err": None}
    JOBS.put(j); j["done"].wait()
    if j["err"]:
        raise j["err"]
    return j["out"]


def upstream(path, payload):
    req = urllib.request.Request(UP + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    for a in range(5):
        try:
            return json.load(urllib.request.urlopen(req, timeout=900))
        except Exception:
            if a == 4:
                raise
            time.sleep(1 + 2 * a)


WIRE = os.environ.get("WIRE", "sup")     # sup = supervisor ToolBox API; mine = our RLM harness ({query, top_k} -> doc_id)


def search_mine(req):
    """Our harness's wire format (multi_search_server.py): {query, top_k[, tag]} -> {results: [{doc_id, text, score}]}.
    Translated onto search() so novelty / rerank are byte-identical to the supervisor-harness path."""
    k = max(1, int(req.get("top_k", 5)))
    up = {"query": req["query"], "top_k": k if POLICY == "passthrough" else max(k, DEPTH)}
    if req.get("index"):
        up["index"] = req["index"]
    hits = upstream("/search", up).get("results", [])
    res = [dict(h, docid=str(h["doc_id"])) for h in hits]
    out = search({"k": k, "tag": req.get("tag") or "", "query": req["query"], "mode": "full"}, pre=res)["results"]
    return {"results": [dict({kk: v for kk, v in r.items() if kk != "docid"}, doc_id=r["docid"]) for r in out]}


def search(req, pre=None):
    k = max(1, int(req.get("k", 5))); tag = req.get("tag") or ""; qid = tag.split("-")[0]
    payload = dict(req); payload.pop("tag", None); payload["untagged_ok"] = True
    payload["k"] = k if POLICY == "passthrough" else max(k, DEPTH)
    if POLICY != "passthrough" and payload.get("mode") not in ("full",):
        payload["mode"] = "full"          # the reranker needs text; we re-shape below if snippets were asked
    res = pre if pre is not None else upstream("/search", payload).get("results", [])
    with LOCK:
        seen = set(TAGS.get(tag, {}).get("docids", set()))
    cand = res
    if POLICY in ("novelty", "novelty+rerank"):
        unseen = [r for r in res if str(r["docid"]) not in seen]
        if len(unseen) >= k:
            STATS["novelty_filtered"] += len(res) - len(unseen); cand = unseen
        else:
            STATS["fallback_seen"] += 1
    if POLICY in ("rerank", "novelty+rerank") and cand and qid in QUESTIONS:
        t0 = time.time()
        sc = rerank_scores(f"{QUESTIONS[qid]}\nCurrent search: {req.get('query','')}", [r.get("text", "") for r in cand])
        STATS["rerank_s"] += time.time() - t0; STATS["reranked"] += 1
        cand = [r for _, r in sorted(zip(sc, cand), key=lambda x: -x[0])]
        for r, s in zip(cand, sorted(sc, reverse=True)):
            r["score"] = round(float(s), 6)
    out = cand[:k]
    if req.get("mode") != "full" and POLICY != "passthrough":
        snip = int(req.get("snippet_max_tokens", 512)) * 4
        out = [{"docid": r["docid"], "score": r["score"], "snippet": (r.get("text") or "")[:snip]} for r in out]
    with LOCK:
        t = TAGS.setdefault(tag, {"docids": set(), "search": 0, "k_sum": 0, "hits": 0})
        t["search"] += 1; t["k_sum"] += k; t["hits"] += len(out); t["docids"].update(str(r["docid"]) for r in out)
        STATS["searches"] += 1
    return {"results": out}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path == "/stats":
            up = upstream("/stats", {}) if False else {}
            self._send(200, {"proxy_policy": POLICY, "depth": DEPTH, "model": MODEL, **STATS})
        elif self.path == "/health":
            self._send(200, json.load(urllib.request.urlopen(UP + "/health", timeout=30)))
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path == "/search":
                self._send(200, search_mine(req) if WIRE == "mine" else search(req))
            elif self.path == "/document":
                self._send(200, upstream("/document", req))
            elif self.path == "/tag_docids":
                with LOCK:
                    t = TAGS.get(req.get("tag"))
                self._send(200, {"docids": sorted(t["docids"]) if t else None})
            elif self.path == "/tag_counts":
                with LOCK:
                    t = TAGS.get(req.get("tag"))
                self._send(200, {"counts": {k: t[k] for k in ("search", "k_sum", "hits")} if t else None})
            else:
                self._send(404, {"error": "not found"})
        except Exception as e:
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8768); a = ap.parse_args()
    load()
    if M:
        threading.Thread(target=_worker, daemon=True).start()
    ThreadingHTTPServer.request_queue_size = 128
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"READY policy={POLICY} depth={DEPTH} model={MODEL} upstream={UP} port={a.port}", flush=True)
    srv.serve_forever()
