"""Score the OBLIQ GPT-5.2 + Gemini-Embedding-2 runs (scripts/run5d/obliq_gemini.sh) in the paper's metrics (2026-10-27):
NDCG@10 and NDCG@50 on gold (and pooled where the task has pooled labels: Math, Twitter), Recall@10/@50/@100 on gold;
plus searches, time, tokens and cost per question. Every run's ranking is re-parsed with examples/obliq_data.py, so
both harnesses are scored by one parser; a question without a result counts 0.
  snip  <dir>/all.json (run_obliq_mine records; rows duplicated by the harness are deduplicated by query_id)
  fd    suprlm/runs/<name>/q*.json (run_one records; usage is exact, per model)
Costs: gpt-5.2 $1.75 / $14 per 1M input / output tokens (OpenAI pricing page, 2026-10-27). Snippet-run tokens are not
reported: the harness's input_tokens field is overwritten by the sub-LM (examples/browsecomp_eval_official_rlm.py).
  python scripts/run5d/obliq_report.py <snip|fd> <math|twitter|congress> <dir_or_runs_name> [ids.json]
"""
import glob, json, statistics as st, sys

sys.path.insert(0, "/home/extstu_00/RLM")
from examples import obliq_data as OD  # noqa: E402

ARM, TASK, SRC = sys.argv[1:4]
T = {"math": ("analogues", "math"), "twitter": ("descriptive", "twitter"), "congress": ("tip-of-tongue", "congress")}[TASK]
G = OD.load_qrels(*T)
POOL = OD.load_qrels(*T, pool=True) if TASK in ("math", "twitter") else None
IDS = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else sorted(G)

rank, srch, wall, tin, tout = {}, {}, {}, {}, {}
if ARM == "snip":
    for x in json.load(open(f"{SRC}/all.json")):
        q = str(x["query_id"])
        rank[q] = OD.extract_doc_ids_from_answer(str(x.get("predicted") or "") if not x.get("error") else "", T[1])
        srch[q] = int(x.get("search_calls") or 0); wall[q] = float(x.get("latency_s") or 0)
else:
    d = SRC if SRC.startswith("/") else f"/home/extstu_00/suprlm/runs/{SRC}"
    for f in glob.glob(f"{d}/q*.json"):
        x = json.load(open(f)); q = str(x["query_id"])
        out = ((x.get("result") or [{}])[0] or {}).get("output", "") if x.get("status") == "completed" else ""
        rank[q] = OD.extract_doc_ids_from_answer(out or "", T[1])
        srch[q] = (x.get("tool_call_counts") or {}).get("search", 0); wall[q] = float(x.get("wall_time_s") or 0)
        u = (x.get("usage") or {}).get("model_usage_summaries") or {}
        tin[q] = sum(m["total_input_tokens"] for m in u.values()); tout[q] = sum(m["total_output_tokens"] for m in u.values())

m = lambda f: st.mean(f(q) for q in IDS)
r = lambda q: rank.get(q, [])
line = (f"{ARM} {TASK}: {len(rank)}/{len(IDS)} results | NDCG@10 G {m(lambda q: OD.ndcg_at(r(q), G[q], 10)):.4f}"
        + (f" P {m(lambda q: OD.ndcg_at(r(q), POOL[q], 10)):.4f}" if POOL else "")
        + f" | NDCG@50 G {m(lambda q: OD.ndcg_at(r(q), G[q], 50)):.4f}"
        + (f" P {m(lambda q: OD.ndcg_at(r(q), POOL[q], 50)):.4f}" if POOL else "")
        + f" | Recall@10 {m(lambda q: OD.recall_at(r(q), G[q], 10)):.4f} @50 {m(lambda q: OD.recall_at(r(q), G[q], 50)):.4f}"
        f" @100 {m(lambda q: OD.recall_at(r(q), G[q], 100)):.4f}")
done = [q for q in IDS if q in rank]
if done:
    line += f" | searches {st.mean(srch[q] for q in done):.2f} | time {st.mean(wall[q] for q in done):.1f} s"
if tin:
    c = [(tin[q] * 1.75 + tout[q] * 14) / 1e6 for q in done if q in tin]
    line += f" | tokens in {st.mean(tin[q] for q in done):,.0f} out {st.mean(tout[q] for q in done):,.0f} | cost ${st.mean(c):.4f}/q"
print(line)
