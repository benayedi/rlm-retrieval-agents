"""Naive baseline: one BM25 search with the claim as query, on the 300 val examples.
Also checks that the numba backend returns the same rankings as bm25s's default (numpy) backend."""
import json, sys, time
sys.path.insert(0, "/home/extstu_00/RLM/scripts")
import hover_retriever as hr
import numpy as np

split = sys.argv[1] if len(sys.argv) > 1 else "val"
ex = [x for x in json.load(open("/home/extstu_00/RLM/data/hover/splits.json")) if x["split"] == split]
s = hr._load()
K = 30
same_ids = same_titles_k = 0; mism = []
rows = []
t_np = t_nb = 0.0
for x in ex:
    tok = s["bm25s"].tokenize(x["claim"], stopwords="en", stemmer=s["stemmer"], show_progress=False)
    t = time.time(); a, sa = s["retriever"].retrieve(tok, k=K, n_threads=1, show_progress=False, backend_selection="numpy"); t_np += time.time() - t
    t = time.time(); b, sb = s["retriever"].retrieve(tok, k=K, n_threads=1, show_progress=False, backend_selection="numba"); t_nb += time.time() - t
    if list(a[0]) == list(b[0]): same_ids += 1
    elif not np.allclose(np.sort(sa[0]), np.sort(sb[0]), atol=1e-5): mism.append(x["id"])
    titles = [s["corpus"][int(d)].partition(" | ")[0] for d in a[0]]
    rows.append((titles, x["gold_titles"]))
print(f"{split}: n={len(ex)}  identical top-{K} id lists numpy vs numba: {same_ids}/{len(ex)}  score-multiset mismatches: {len(mism)} {mism[:5]}")
print(f"mean latency numpy {t_np/len(ex)*1000:.1f} ms, numba {t_nb/len(ex)*1000:.1f} ms")
for k in (7, 21, 30):
    rec = [hr.recall_gold(t[:k], g) for t, g in rows]
    af = [hr.all_found(t[:k], g) for t, g in rows]
    dist = np.bincount([round(r * 3) for r in rec], minlength=4)
    print(f"k={k:2d}: mean recall {np.mean(rec):.4f}  all-found {np.mean(af):.4f} ({sum(af)}/{len(af)})  #gold found 0/1/2/3: {dist.tolist()}")
