"""HoVer splits replicating gepa-artifact (gepa_artifact/benchmarks/hover/hover_data.py + benchmark.py).

Refactored copy of original/RLM/scripts/hover_make_splits.py (studies/prompt_optimisation/CHANGES_FROM_ORIGINAL.md).
Run: python hover/src/make_splits.py (once; writes HOVER_DATA_DIR/splits.json).
Changed: the data directory; `os, sys` added to the import line for the wiring.
Names: 'supervisor harness' / 'full-doc RLM' in the text below is the document REPL agent (agents/document_repl);
  'official RLM harness' / 'our harness' / run_obliq_mine.py is the snippet REPL agent (agents/snippet_repl,
  src/run_obliq.py).

- source: official hover_train_release_v1.1.json (== HF hover-nlp/hover train, whose loader script no longer runs)
- keep examples whose supporting_facts have exactly 3 unique titles (GEPA's count_unique_docs == 3)
- random.Random(0).shuffle
- create_splits: test = [:40%], val = [40%:80%], train = [80%:]
- trim: Random(1).sample(split, n) with a fresh rng per split; train 150, test 300, val 300
"""
import json, os, random, re, string, sys, unicodedata, collections

# Import wiring and paths (refactor, rules 2-3). Original: hard-coded locations of the original repository (R), the
# document REPL agent working directory (SUP) and the interpreter. Code now comes from this repository (computed from
# __file__), data and run outputs from configs/paths.env (rlm_shared.paths). The original logs/ directory is RUNS_DIR.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, os.pardir, os.pardir))
DOC_SRC = os.path.join(_REPO, "agents", "document_repl", "src")    # split_runner.py (was scripts/gepa_bcp.py), run_bm25.py
SPLITS = os.path.join(_REPO, "studies", "prompt_optimisation", "splits")   # original: <original repo>/data
sys.path.insert(0, os.path.join(_REPO, "agents", "shared"))
from rlm_shared.paths import path  # noqa: E402

RAW = f"{path('HOVER_DATA_DIR')}/raw_hover/hover_train_release_v1.1.json"   # original: <original repo>/data/hover/...
CORPUS = f"{path('HOVER_DATA_DIR')}/raw/wiki.abstracts.2017.jsonl"
OUT = f"{path('HOVER_DATA_DIR')}/splits.json"

def normalize_text(s):  # copy of dspy.evaluate.normalize_text
    s = unicodedata.normalize("NFD", s)
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

data = json.load(open(RAW))
print("raw train examples:", len(data))
print("num_hops dist:", collections.Counter(d["num_hops"] for d in data))
uniq3 = [d for d in data if len({k for k, _ in d["supporting_facts"]}) == 3]
hops3 = [d for d in data if d["num_hops"] == 3]
print(f"count_unique_docs==3: {len(uniq3)}  num_hops==3: {len(hops3)}  both: {sum(1 for d in uniq3 if d['num_hops']==3)}")

ds = [dict(uid=d["uid"], claim=d["claim"], label=d["label"], num_hops=d["num_hops"],
           gold_titles=sorted({k for k, _ in d["supporting_facts"]})) for d in uniq3]
random.Random(0).shuffle(ds)
n = len(ds)
test, val, train = ds[: int(0.4 * n)], ds[int(0.4 * n): int(0.8 * n)], ds[int(0.8 * n):]
def trim(x, size):
    rng = random.Random(); rng.seed(1); return rng.sample(x, size)
splits = {"train": trim(train, 150), "val": trim(val, 300), "test": trim(test, 300)}

titles = set(); ntitles = set()
with open(CORPUS) as f:
    for line in f:
        t = json.loads(line)["title"]; titles.add(t); ntitles.add(normalize_text(t))
print("corpus titles:", len(titles), "unique normalized:", len(ntitles))

out = []
for name, xs in splits.items():
    for i, x in enumerate(xs):
        out.append(dict(id=f"{name}-{i}", split=name, uid=x["uid"], claim=x["claim"], label=x["label"],
                        num_hops=x["num_hops"], gold_titles=x["gold_titles"]))
allg = [t for x in out for t in x["gold_titles"]]
ex = sum(t in titles for t in allg); nm = sum(normalize_text(t) in ntitles for t in allg)
print(f"gold titles in sampled splits: {len(allg)}  exact match {ex} ({ex/len(allg):.4f})  normalize_text match {nm} ({nm/len(allg):.4f})")
miss = [t for t in allg if normalize_text(t) not in ntitles]
print("unmatched examples:", miss[:15])
exs = sum(all(normalize_text(t) in ntitles for t in x["gold_titles"]) for x in out)
print(f"examples with all gold titles in corpus: {exs}/{len(out)}")
# full-pool match rate too
allp = [t for x in ds for t in x["gold_titles"]]
print(f"full 3-doc pool: {len(allp)} titles, exact {sum(t in titles for t in allp)/len(allp):.4f}, norm {sum(normalize_text(t) in ntitles for t in allp)/len(allp):.4f}")
for name in splits: print(name, len(splits[name]), collections.Counter(x["label"] for x in splits[name]), collections.Counter(x["num_hops"] for x in splits[name]))
json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
print("wrote", OUT, len(out))
