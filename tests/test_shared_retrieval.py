"""Tests of the retrieval servers in benchmarks/retrieval/, offline: no model, no GPU, no network, no server started.

Request handling is exercised in-process (FastAPI TestClient, or the handler class called directly); the upstream
HTTP calls are monkeypatched. The BM25 server runs on a three-document index built here with bm25s. Servers that need
an embedding model, a FAISS/PLAID index or Vertex AI credentials are skipped with the reason, and their 2,000-character
text cut is checked statically instead.
"""
from __future__ import annotations

import ast
import importlib.util
import io
import json
import py_compile
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RETR = REPO / "benchmarks" / "retrieval"


def load_script(monkeypatch, path: Path, argv: list[str], name: str):
    """Import a server script under a private name with the given command line (its argparse runs at import)."""
    monkeypatch.setattr(sys, "argv", [str(path)] + argv)
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, mod)
    spec.loader.exec_module(mod)
    return mod


class FakeResponse:
    def __init__(self, body, status=200):
        self._body, self.status_code, self.ok = body, status, status < 400

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


# ---------------------------------------------------------------------------------------------------- fusion proxy

@pytest.fixture
def proxy(monkeypatch):
    # --passthrough: no multi-vector arm is loaded (models load only under __main__ anyway)
    return load_script(monkeypatch, RETR / "fusion_proxy.py", ["--dense-url", "http://dense.invalid/", "--passthrough"],
                       "fusion_proxy_under_test")


def test_proxy_collapse_chunks_keeps_best_rank_per_parent(proxy):
    ids = ["A##c3", "A##c1", "B", "C##c0", "B##c2", "D"]
    assert proxy._collapse_chunks(ids) == ["A", "B", "C", "D"]


def test_proxy_fetch_text_is_cut_to_text_cap(monkeypatch, proxy):
    calls = []

    def post(url, json=None, timeout=None):
        calls.append((url, json, timeout))
        return FakeResponse({"doc_id": json["doc_id"], "text": "x" * 5000})

    monkeypatch.setattr(proxy.requests, "post", post)
    assert proxy.TEXT_CAP == 2000
    assert len(proxy._fetch_text("d1", None)) == 2000
    assert calls == [("http://dense.invalid/document", {"doc_id": "d1"}, 60)]
    monkeypatch.setattr(proxy.requests, "post", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    assert proxy._fetch_text("d1", "idx") == ""          # a failed lookup yields empty text, not an error


def test_proxy_passthrough_reads_depth_and_cuts_to_top_k(monkeypatch, proxy):
    sent = []

    def post(url, json=None, timeout=None):
        sent.append(json)
        return FakeResponse({"results": [{"doc_id": f"d{i}", "score": 1.0 - i / 100, "text": "t"}
                                         for i in range(json["top_k"])]})

    monkeypatch.setattr(proxy.requests, "post", post)
    out = proxy.search(proxy.SearchRequest(query="q", top_k=3))
    assert [r["doc_id"] for r in out["results"]] == ["d0", "d1", "d2"]
    assert sent == [{"query": "q", "top_k": 50}]          # depth = max(--depth 50, top_k)
    proxy.search(proxy.SearchRequest(query="q", top_k=70, index="other"))
    assert sent[-1] == {"query": "q", "top_k": 70, "index": "other"}


def test_proxy_rrf_fusion_and_text_hydration(monkeypatch, proxy):
    monkeypatch.setattr(proxy.args, "passthrough", False)
    monkeypatch.setattr(proxy.args, "rrf_k", 60)
    monkeypatch.setattr(proxy.args, "mv_weight", 1.0)
    monkeypatch.setattr(proxy, "_dense_search", lambda q, k, i: [{"doc_id": "a", "text": "A"}, {"doc_id": "b", "text": "B"}])
    monkeypatch.setattr(proxy, "_mv_search", lambda q, k: [["b", "c"]])
    monkeypatch.setattr(proxy, "_fetch_text", lambda d, i: f"fetched-{d}")
    res = proxy.search(proxy.SearchRequest(query="q", top_k=3))["results"]
    # b: 1/62 + 1/61; a: 1/61; c: 1/62
    assert [r["doc_id"] for r in res] == ["b", "a", "c"]
    assert res[0]["score"] == round(1 / 62 + 1 / 61, 6)
    assert [r["text"] for r in res] == ["B", "A", "fetched-c"]   # dense text kept; mv-only hit fetched (and capped)


def test_proxy_mv_only_scores_are_reciprocal_rank(monkeypatch, proxy):
    monkeypatch.setattr(proxy.args, "mv_only", True)
    monkeypatch.setattr(proxy, "_mv_search", lambda q, k: [["x", "y", "z"]])
    monkeypatch.setattr(proxy, "_fetch_text", lambda d, i: d.upper())
    res = proxy.search(proxy.SearchRequest(query="q", top_k=2))["results"]
    assert res == [{"score": 1.0, "doc_id": "x", "text": "X"}, {"score": 0.5, "doc_id": "y", "text": "Y"}]


def test_proxy_requires_an_arm_unless_passthrough(monkeypatch):
    with pytest.raises(SystemExit, match="need at least one --mv"):
        load_script(monkeypatch, RETR / "fusion_proxy.py", ["--dense-url", "http://dense.invalid"], "fusion_proxy_noarm")


# ---------------------------------------------------------------------------------------------------- BM25 server

DOCS = [("Alpha river", "The Nile is the longest river in Africa."),
        ("Beta mountain", "Mount Kilimanjaro is the highest mountain in Africa. " + "filler " * 800),
        ("Gamma author", "Jane Austen wrote Pride and Prejudice.")]


def _bm25_index(dirpath: Path, texts: list[str]):
    import bm25s
    import Stemmer
    r = bm25s.BM25(k1=0.9, b=0.4, method="lucene")
    r.index(bm25s.tokenize(texts, stopwords="en", stemmer=Stemmer.Stemmer("english"), show_progress=False),
            show_progress=False)
    r.save(str(dirpath / "bm25_index"))


def _write_jsonl(path: Path, rows):
    offs = []
    with open(path, "wb") as f:
        for r in rows:
            offs.append(f.tell())
            f.write((json.dumps(r) + "\n").encode())
    return offs


@pytest.fixture
def frames_server(monkeypatch, tmp_path):
    pytest.importorskip("bm25s")
    pytest.importorskip("Stemmer")
    import numpy as np
    from fastapi.testclient import TestClient
    data = tmp_path / "frames"
    data.mkdir()
    offs = _write_jsonl(data / "corpus.jsonl", [{"title": t, "text": x} for t, x in DOCS])
    np.savez(data / "corpus_meta.npz", offsets=np.array(offs, dtype=np.int64))
    _bm25_index(data, [f"{t}\n{x}" for t, x in DOCS])
    monkeypatch.setenv("FRAMES_DATA_DIR", str(data))
    mod = load_script(monkeypatch, RETR / "frames_search_server.py", ["--protocol", "rlm"], "frames_server_under_test")
    return mod, TestClient(mod.app)


def test_bm25_server_search_text_cut_and_ids(frames_server):
    mod, client = frames_server
    assert client.get("/health").json() == {"status": "ok", "docs": 3}
    res = client.post("/search", json={"query": "highest mountain Kilimanjaro", "top_k": 2}).json()["results"]
    assert res[0]["doc_id"] == "1"                          # FRAMES doc_id = line number, as a string
    full = "Beta mountain\n" + DOCS[1][1]
    assert len(full) > 2000 and res[0]["text"] == full[:2000]   # text cut to 2,000 characters server-side
    assert set(res[0]) == {"score", "doc_id", "text"}
    doc = client.post("/document", json={"doc_id": "1"}).json()
    assert doc == {"doc_id": "1", "text": full}             # /document returns the whole text
    assert client.post("/document", json={"doc_id": "7"}).json()["error"] == "doc_id '7' not found"


@pytest.mark.parametrize("top_k, k", [(0, 1), (-5, 1), (3, 3), (500, 100), (None, 5)])
def test_bm25_server_top_k_is_clamped_to_1_100(monkeypatch, frames_server, top_k, k):
    mod, client = frames_server
    seen = []
    monkeypatch.setattr(mod, "bm25", lambda q, kk: seen.append(kk) or [])
    body = {"query": "nile"} if top_k is None else {"query": "nile", "top_k": top_k}
    assert client.post("/search", json=body).json() == {"results": []}
    assert seen == [k]


def test_bm25_server_stopword_only_query_returns_nothing(frames_server):
    mod, client = frames_server
    assert client.post("/search", json={"query": "the and of", "top_k": 2}).json() == {"results": []}


def test_bm25_server_browsecomp_mode_uses_corpus_ids(monkeypatch, tmp_path):
    pytest.importorskip("bm25s")
    import numpy as np
    from fastapi.testclient import TestClient
    idx, full = tmp_path / "bcp_idx", tmp_path / "fulldoc"
    idx.mkdir()
    (full / "data" / "prepared").mkdir(parents=True)
    rows = [{"docid": f"doc-{i}", "text": f"{t}. {x}"} for i, (t, x) in enumerate(DOCS)]
    offs = _write_jsonl(full / "data" / "prepared" / "corpus.jsonl", rows)
    np.save(idx / "offsets.npy", np.array(offs, dtype=np.int64))
    (idx / "docids.json").write_text(json.dumps([r["docid"] for r in rows]))
    _bm25_index(idx, [r["text"] for r in rows])
    monkeypatch.setenv("BCP_BM25_INDEX_DIR", str(idx))
    monkeypatch.setenv("FULLDOC_ROOT", str(full))
    monkeypatch.delenv("FRAMES_DATA_DIR", raising=False)    # not needed with --dataset bcp
    mod = load_script(monkeypatch, RETR / "frames_search_server.py", ["--dataset", "bcp"], "bcp_server_under_test")
    client = TestClient(mod.app)
    res = client.post("/search", json={"query": "Jane Austen Prejudice", "top_k": 1}).json()["results"]
    assert res[0]["doc_id"] == "doc-2" and res[0]["text"] == rows[2]["text"]   # text as stored, no title prefix
    assert client.post("/document", json={"doc_id": "doc-0"}).json()["text"] == rows[0]["text"]


def test_bm25_server_names_the_missing_variable(monkeypatch):
    monkeypatch.delenv("FRAMES_DATA_DIR", raising=False)
    with pytest.raises(RuntimeError, match="FRAMES_DATA_DIR is not set"):
        load_script(monkeypatch, RETR / "frames_search_server.py", [], "frames_server_unset")


@pytest.mark.skip(reason="--protocol sup loads the Qwen3-0.6B tokenizer from $FULLDOC_ROOT/data/models at import "
                         "(a model download/directory); not available offline")
def test_bm25_server_sup_protocol():
    pass


# ---------------------------------------------------------------------------------------------------- OBLIQ Math

@pytest.fixture
def math_search(monkeypatch, tmp_path):
    from rlm_shared import obliq_data as OD
    qr = tmp_path / "analogues" / "math" / "queries+qrels"
    qr.mkdir(parents=True)
    (qr / "per_query_excluded_ids.json").write_text(json.dumps({"q1": ["self"], "q2": ["a", "b"]}))
    monkeypatch.setattr(OD, "OBLIQ_ROOT", tmp_path)
    mod = load_script(monkeypatch, RETR / "obliq_math_search.py", ["--upstream", "http://up.invalid/"],
                      "obliq_math_search_under_test")
    sent = []

    def post(url, json=None, timeout=None):
        sent.append((url, json))
        if url.endswith("/document"):
            return FakeResponse({"doc_id": json["doc_id"], "text": "full"})
        return FakeResponse({"results": [{"doc_id": d, "text": f"text {d}", "score": s}
                                         for d, s in (("self", 0.9), ("x", 0.8), ("y", 0.7), ("z", 0.6))]})

    monkeypatch.setattr(mod.requests, "post", post)
    mod._sent = sent
    return mod


def test_math_search_drops_excluded_ids_and_asks_for_more(math_search):
    hits = math_search.search("query", 2, "q1-uuid")
    assert [h["doc_id"] for h in hits] == ["x", "y"]
    assert math_search._sent[-1] == ("http://up.invalid/search", {"query": "query", "top_k": 3, "index": "obliq-ana-math"})
    assert math_search.STATS["excluded_dropped"] == 1
    assert math_search.TAG_COUNTS["q1-uuid"] == {"search": 1, "k_sum": 2, "hits": 2}
    assert math_search.TAG_DOCIDS["q1-uuid"] == {"x", "y"}


def test_math_search_sup_shape(math_search):
    h = {"doc_id": 7, "text": "abc", "score": 0.1234567}
    assert math_search.sup_shape(h, "ids") == {"docid": "7", "score": 0.123457}
    assert math_search.sup_shape(h, "snippet") == {"docid": "7", "score": 0.123457, "snippet": "abc"}
    assert math_search.sup_shape(h, "full")["embedded_span"] == [0, 3]


def _post(mod, path, body):
    """Call the handler's do_POST without a socket."""
    h = mod.H.__new__(mod.H)
    raw = json.dumps(body).encode()
    h.path, h.headers, h.rfile = path, {"Content-Length": str(len(raw))}, io.BytesIO(raw)
    sent = []
    h._send = lambda code, obj: sent.append((code, obj))
    h.do_POST()
    return sent[0]


def test_math_search_handler_wires(math_search):
    code, body = _post(math_search, "/search", {"query": "q"})
    assert code == 400 and "tag required" in body["error"] and math_search.STATS["refused_untagged"] == 1
    code, body = _post(math_search, "/search", {"query": "q", "top_k": 1, "tag": "q2-1"})       # our wire
    assert code == 200 and body == {"results": [{"doc_id": "self", "text": "text self", "score": 0.9}]}
    code, body = _post(math_search, "/search", {"query": "q", "k": 1, "mode": "ids", "tag": "q1-1"})  # document REPL wire
    assert code == 200 and body == {"results": [{"docid": "x", "score": 0.8}]}
    code, body = _post(math_search, "/document", {"docid": "x"})
    assert code == 200 and math_search._sent[-1][1] == {"doc_id": "x", "index": "obliq-ana-math"}
    assert _post(math_search, "/tag_counts", {"tag": "q1-1"})[1] == {"counts": {"search": 1, "k_sum": 1, "hits": 1}}
    assert _post(math_search, "/nope", {})[0] == 404


# ---------------------------------------------------------------------------------------------------- model servers

MODEL_SERVERS = {
    "search_server.py": "loads Qwen3-Embedding-8B on CUDA and a FAISS index at import",
    "multi_search_server.py": "loads Qwen3-Embedding-8B and needs FAISS index directories at import",
    "gemini/search_server.py": "builds a Vertex AI client (service-account credentials, network) at import",
}


@pytest.mark.parametrize("name", sorted(MODEL_SERVERS))
def test_model_server_request_handling(name):
    pytest.skip(f"{name} {MODEL_SERVERS[name]}; its 2,000-character cut is checked statically below")


@pytest.mark.parametrize("name", sorted(MODEL_SERVERS))
def test_model_server_search_cuts_text_to_2000_chars(name):
    tree = ast.parse((RETR / name).read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "search")
    assert "[:2000]" in ast.unparse(fn)
    doc = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "get_document")
    assert "2000" not in ast.unparse(doc)             # /document returns the whole text


def test_vertex_service_account_from_env_file(monkeypatch, tmp_path, capsys):
    """vertex.py reads the service-account JSON from GCP_VERTEX_ENV only when the variable is not exported."""
    for m in ("google", "google.genai", "google.genai.types", "google.oauth2", "google.oauth2.service_account"):
        if m not in sys.modules:
            monkeypatch.setitem(sys.modules, m, types.ModuleType(m))
    monkeypatch.setattr(sys.modules["google"], "genai", sys.modules["google.genai"], raising=False)
    monkeypatch.setattr(sys.modules["google.genai"], "types", sys.modules["google.genai.types"], raising=False)
    monkeypatch.setattr(sys.modules["google.oauth2"], "service_account", sys.modules["google.oauth2.service_account"],
                        raising=False)
    var = "NNEXT_PROVIDERS_GCP_VERTEX_SERVICE_ACCOUNT_JSON"
    monkeypatch.delenv(var, raising=False)
    env = tmp_path / "vertex.env"
    env.write_text(f"OTHER=1\n{var}='{{\"project_id\": \"p-test\", \"private_key\": \"dummy\"}}'\n")
    monkeypatch.setenv("GCP_VERTEX_ENV", str(env))
    vertex = load_script(monkeypatch, RETR / "gemini" / "vertex.py", [], "vertex_under_test")
    assert vertex._sa_info() == {"project_id": "p-test", "private_key": "dummy"}
    monkeypatch.setenv(var, '{"project_id": "from-env"}')
    monkeypatch.setenv("GCP_VERTEX_ENV", str(tmp_path / "absent.env"))   # not opened when the variable is set
    assert vertex._sa_info() == {"project_id": "from-env"}
    assert capsys.readouterr().out == ""                                  # nothing printed


# ---------------------------------------------------------------------------------------------------- static checks

WRITTEN = sorted(
    [p for p in (REPO / "agents" / "shared" / "rlm_shared" / "grading").glob("*.py")]
    + [p for p in RETR.rglob("*.py")]
    + [p for p in (REPO / "scripts" / "reporting").glob("*.py")]
    + [REPO / "agents" / "shared" / "rlm_shared" / "paths.py"]
    + sorted((REPO / "tests").glob("test_shared_*.py")))


@pytest.mark.parametrize("path", WRITTEN, ids=lambda p: str(p.relative_to(REPO)))
def test_py_compile(path, tmp_path):
    py_compile.compile(str(path), cfile=str(tmp_path / "x.pyc"), doraise=True)


@pytest.mark.parametrize("path", sorted(RETR.rglob("*.sh")), ids=lambda p: str(p.relative_to(REPO)))
def test_bash_syntax(path):
    subprocess.run(["bash", "-n", str(path)], check=True)


@pytest.mark.parametrize("path", WRITTEN, ids=lambda p: str(p.relative_to(REPO)))
def test_no_machine_paths_in_code(path):
    """No string literal outside docstrings names the original machine (comments may cite it)."""
    tree = ast.parse(path.read_text())
    docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                  if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body
                  and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    bad = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
           and id(n) not in docstrings and re.search(r"/home/|/tmp/claude", n.value)]
    if path.name == Path(__file__).name:
        bad = [b for b in bad if b != r"/home/|/tmp/claude"]
    assert bad == []


def test_new_variables_are_documented():
    from rlm_shared.paths import VARIABLES
    env = (REPO / "configs" / "paths.example.env").read_text()
    for name in ("LOGS_ROOT", "ORIGINAL_REPO_ROOT", "RETRIEVAL_EVAL_ROOT", "GCP_VERTEX_ENV"):
        assert name in VARIABLES and re.search(rf"^{name}=", env, re.M)
    assert VARIABLES["LOGS_ROOT"] == "root of the original run logs (logs/ of the original repository)"
