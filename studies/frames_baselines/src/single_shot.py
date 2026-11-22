"""FB-1: single-shot FRAMES baselines with Qwen3.6-35B-A3B (no agent loop): no retrieval (closed), the top-2 / top-4
BM25 articles in the prompt (bm25_2, bm25_4), all gold articles in the prompt (oracle); plus the gold-article coverage
of the Wikipedia dump and BM25 recall@k of the raw question.

From original/RLM/scripts/frames_eval.py. Changed only: the original machine's paths (FRAMES data and index from
FRAMES_DATA_DIR, outputs to LOGS_ROOT/frames/ as in the original, the original repository's .env from
ORIGINAL_REPO_ROOT), the import of the BrowseComp grader prompt from agents/shared/rlm_shared/browsecomp_data.py, and
this documentation (studies/frames_baselines/CHANGES_FROM_ORIGINAL.md). Judge: the official BrowseComp grader prompt
run on the local Qwen3.6 (:8030) with thinking off, so every verdict here is Qwen3.6-judged only. The thesis uses the
oracle, closed-book and BM25 top-4 files as inputs of the FRAMES failure taxonomy (studies/error_analysis/). Run:
  set -a; . configs/paths.env; set +a
  python studies/frames_baselines/src/single_shot.py [--conds closed,oracle,bm25_4,bm25_2] [--workers 24]

Original docstring follows (paths in it are relative to the original repository).

FRAMES on our BM25 index over the paper's Wikipedia (built by scripts/frames_build_bm25.py).

  coverage   are the gold articles in the dump? (FRAMES links -> dump titles)
  recall     BM25 with the raw question as query: gold-article recall@k
  llm        single-shot reader, Qwen3.6 on :8030, judged like BrowseComp+ (examples/browsecomp_data
             .judge_match, gpt-5.4-mini). Conditions mirror FRAMES paper Table 3:
               closed   no retrieval                         (paper, Gemini-Pro-1.5: 0.408)
               bm25_2   top-2 BM25 articles in the prompt    (0.452)
               bm25_4   top-4 BM25 articles in the prompt    (0.474)
               oracle   all gold articles in the prompt      (0.729)
             closed + oracle are the go/no-go test (drop FRAMES if closed > ~60% or oracle > ~90%).

Outputs go to logs/frames/. Every LLM condition is resumable (one JSON line per question).
"""
import argparse, ast, csv, json, os, re, sys, time, urllib.parse
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# Refactor (paths): was ROOT = "<original repository>"; DATA = f"{ROOT}/data/frames"; LOGS = f"{ROOT}/logs/frames".
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "agents", "shared"))
from rlm_shared.paths import path  # noqa: E402
DATA = str(path("FRAMES_DATA_DIR"))
LOGS = os.path.join(str(path("LOGS_ROOT")), "frames")
INDEX, CORPUS = f"{DATA}/bm25_index", f"{DATA}/corpus.jsonl"
ART_CHARS, TOTAL_CHARS = 100_000, 600_000   # ~25k tokens per article, ~150k per prompt (max_model_len 262k)


def load_frames():
    rows = list(csv.DictReader(open(f"{DATA}/test.tsv"), delimiter="\t"))
    for r in rows:
        # python-list literal in the TSV; a few entries are several URLs joined by ", "
        r["links"] = [u.strip(" ,") for l in ast.literal_eval(r["wiki_links"]) if l
                      for u in re.split(r",\s*(?=https?://)", l) if "/wiki/" in u]
    return rows


def link_title(url):
    """https://en.m.wikipedia.org/wiki/Foo_Bar#Section -> 'Foo Bar'."""
    url = url.strip()
    t = url.split("/wiki/", 1)[-1].split("#", 1)[0].split("?", 1)[0]
    t = urllib.parse.unquote(urllib.parse.unquote(t)).replace("_", " ").strip()  # some are double-encoded
    return t[:1].upper() + t[1:]


def corpus_meta():
    """Titles and byte offsets of corpus.jsonl lines (doc id = line number); cached."""
    cache = f"{DATA}/corpus_meta.npz"
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True); return list(z["titles"]), z["offsets"]
    titles, offs, pos = [], [], 0
    with open(CORPUS, "rb") as f:
        for line in f:
            offs.append(pos); pos += len(line)
            titles.append(json.loads(line)["title"])
    offs = np.asarray(offs, np.int64)
    np.savez(cache, titles=np.asarray(titles, dtype=object), offsets=offs)
    return titles, offs


class Corpus:
    def __init__(self):
        self.titles, self.offs = corpus_meta()
        self.f = open(CORPUS, "rb")
        self.by_title = {t: i for i, t in enumerate(self.titles)}
        self.by_lower = {}
        for i, t in enumerate(self.titles):
            self.by_lower.setdefault(t.lower(), i)

    def doc(self, i):
        self.f.seek(int(self.offs[i])); return json.loads(self.f.readline())

    def find(self, title):
        i = self.by_title.get(title)
        return i if i is not None else self.by_lower.get(title.lower())


def gold_ids(rows, C):
    out, miss = [], []
    for r in rows:
        ids = []
        for l in r["links"]:
            i = C.find(link_title(l))
            (ids.append(i) if i is not None else miss.append(link_title(l)))
        out.append(sorted(set(ids)))
    return out, miss


def bm25_topk(rows, k=100):
    cache = f"{LOGS}/bm25_top{k}.npy"
    if os.path.exists(cache):
        return np.load(cache)
    import bm25s, Stemmer
    t0 = time.time()
    r = bm25s.BM25.load(INDEX, mmap=True)
    print(f"index loaded {time.time()-t0:.0f}s", flush=True)
    qt = bm25s.tokenize([x["Prompt"] for x in rows], stopwords="en", stemmer=Stemmer.Stemmer("english"),
                        show_progress=False)
    docs, _ = r.retrieve(qt, k=k, n_threads=16, show_progress=False)
    print(f"retrieved {len(rows)} queries in {time.time()-t0:.0f}s", flush=True)
    np.save(cache, docs); return docs


def step_coverage_recall(rows, C):
    gold, miss = gold_ids(rows, C)
    n_links = sum(len(set(link_title(l) for l in r["links"])) for r in rows)
    uniq_gold = {link_title(l) for r in rows for l in r["links"]}
    uniq_miss = sorted(set(miss))
    full = sum(len(g) == len(set(link_title(l) for l in r["links"])) for g, r in zip(gold, rows))
    top = bm25_topk(rows)
    rep = {"docs": len(C.titles), "gold_links": n_links, "gold_articles_unique": len(uniq_gold),
           "gold_articles_missing": len(uniq_miss), "questions_all_gold_found": full,
           "missing_examples": uniq_miss[:60]}
    for k in (1, 2, 4, 10, 20, 50, 100):
        rec, anyh, allh = [], [], []
        for g, t in zip(gold, top):
            if not g:
                continue
            hit = len(set(g) & set(t[:k].tolist()))
            rec.append(hit / len(g)); anyh.append(hit > 0); allh.append(hit == len(g))
        rep[f"recall@{k}"] = round(float(np.mean(rec)), 4)
        rep[f"any@{k}"] = round(float(np.mean(anyh)), 4)
        rep[f"all@{k}"] = round(float(np.mean(allh)), 4)
    json.dump(rep, open(f"{LOGS}/coverage_recall.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "missing_examples"}, indent=1), flush=True)
    return gold, top


PROMPT = """Answer the question. {ctx}Think it through, then end your reply with one line of the form
Exact Answer: <your final answer>

Question: {q}"""


def build_prompt(q, docs):
    if not docs:
        return PROMPT.format(ctx="Use your own knowledge. ", q=q)
    per = min(ART_CHARS, TOTAL_CHARS // len(docs))
    ctx = "\n\n".join(f"=== Wikipedia article: {d['title']} ===\n{d['text'][:per]}" for d in docs)
    return PROMPT.format(ctx=f"Use the Wikipedia articles below and your own knowledge.\n\n{ctx}\n\n", q=q)


def run_llm(cond, rows, C, gold, top, workers):
    from openai import OpenAI
    # Refactor (import wiring): was sys.path.insert(0, f"{ROOT}/examples"); load_dotenv(f"{ROOT}/.env"); from browsecomp_data ...
    from dotenv import load_dotenv; load_dotenv(os.path.join(str(path("ORIGINAL_REPO_ROOT")), ".env"))
    from rlm_shared.browsecomp_data import _JUDGE_PROMPT, _final_answer_span, heuristic_match
    cli = OpenAI(base_url="http://localhost:8030/v1", api_key="x", timeout=1800)
    out = f"{LOGS}/llm_{cond}.jsonl"
    done = {json.loads(l)["i"] for l in open(out)} if os.path.exists(out) else set()
    todo = [i for i in range(len(rows)) if i not in done]
    print(f"[{cond}] {len(done)} done, {len(todo)} to go", flush=True)

    def judge_match(q, response, gold):
        """Official BrowseComp grader prompt (as examples/browsecomp_data.judge_match), but run on the
        local Qwen3.6 with thinking off: .env has no Azure credentials on this box. Answers are stored,
        so they can be re-judged with gpt-5.4-mini later."""
        if heuristic_match(gold, _final_answer_span(response)):
            return True
        for attempt in range(3):
            try:
                c = cli.chat.completions.create(model="qwen36-35b-a3b", temperature=0, max_tokens=512,
                    messages=[{"role": "user", "content": _JUDGE_PROMPT.format(question=q, response=response, correct_answer=gold)}],
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                m = re.search(r"correct:\s*\**\s*(yes|no)", c.choices[0].message.content or "", re.I)
                if m:
                    return m.group(1).lower() == "yes"
            except Exception:
                time.sleep(5)
        return False

    def one(i):
        r = rows[i]
        ids = {"closed": [], "bm25_2": top[i][:2].tolist(), "bm25_4": top[i][:4].tolist(), "oracle": gold[i]}[cond]
        if cond == "oracle" and not ids:
            return None
        prompt = build_prompt(r["Prompt"], [C.doc(j) for j in ids])
        t0 = time.time(); err = None; resp = ""
        for attempt in range(3):
            try:
                c = cli.chat.completions.create(model="qwen36-35b-a3b", messages=[{"role": "user", "content": prompt}],
                                                temperature=0.6, top_p=0.95, max_tokens=16384, extra_body={"top_k": 20})
                resp = c.choices[0].message.content or ""; err = None; break
            except Exception as e:
                err = f"{type(e).__name__}: {e}"[:300]; time.sleep(10)
        # :8030 runs without a reasoning parser, so the thinking is in `resp` and echoes the
        # template line; read only after </think>, and take the last non-placeholder answer.
        final = resp.rsplit("</think>", 1)[-1]
        hits = [h.strip().strip('"*. ') for h in re.findall(r"exact answer\s*:\s*(.+)", final, re.I)]
        hits = [h for h in hits if h and not h.startswith("<")]
        ans = hits[-1] if hits else final.strip()[-300:]
        ok = False if err else judge_match(r["Prompt"], f"Exact Answer: {ans}", r["Answer"])
        return {"i": i, "cond": cond, "correct": bool(ok), "answer": ans, "gold": r["Answer"], "doc_ids": ids,
                "prompt_chars": len(prompt), "err": err, "sec": round(time.time() - t0, 1),
                "reasoning_types": r["reasoning_types"]}

    with open(out, "a") as f, ThreadPoolExecutor(workers) as ex:
        for n, res in enumerate(ex.map(one, todo), 1):
            if res:
                f.write(json.dumps(res, ensure_ascii=False) + "\n"); f.flush()
            if n % 50 == 0:
                print(f"[{cond}] {n}/{len(todo)}", flush=True)
    res = [json.loads(l) for l in open(out)]
    acc = np.mean([x["correct"] for x in res])
    print(f"[{cond}] ACCURACY {acc:.4f} on {len(res)} questions ({sum(bool(x['err']) for x in res)} errors)", flush=True)
    return acc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", default="closed,oracle,bm25_4,bm25_2")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    rows = load_frames()
    C = Corpus()
    gold, top = step_coverage_recall(rows, C)
    if a.limit:
        rows = rows[:a.limit]
    summary = {}
    for cond in [c for c in a.conds.split(",") if c]:
        summary[cond] = run_llm(cond, rows, C, gold, top, a.workers)
        json.dump(summary, open(f"{LOGS}/llm_summary.json", "w"), indent=1)
    print("SUMMARY", summary, flush=True)
