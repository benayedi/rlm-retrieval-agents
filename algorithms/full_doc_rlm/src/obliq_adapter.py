"""Adapter: the Full-Doc RLM retrieval API in front of an OBLIQ search server.

From original/FULLDOC/src/obliq_adapter.py. Run as a server, one per OBLIQ index, for example:
    python src/obliq_adapter.py --port <port> --upstream <search server URL> --index <index name>
experiments/obliq_fd.sh expects it on :8768 (Twitter) and :8769 (Congress); Math uses its own server on :8014.

WHY. The Snippet RLM and the Full-Doc RLM are compared on OBLIQ only if BOTH hit the SAME index
with the SAME search. The upstream search server (multi_search_server.py / the Gemini-Embedding-2 server)
speaks the Snippet RLM's wire format; the Full-Doc RLM's ToolBox speaks a different one
(k/docid/tag/mode, plus /tag_counts, /tag_docids, /stats). Rather than build a second retrieval
stack -- which would make any difference between the two arms partly a difference of retriever --
this translates the Full-Doc RLM's calls onto the one server.

It adds no ranking logic of its own: it forwards the query, takes the order it is given,
and renames fields. The per-tag counters that run_one.py reads for its `search calls` and
`recall` metrics are reproduced here, because the Full-Doc RLM's own retrieval servers keep them.

NOTE (not changed): mode "snippet" returns the whole text under the key 'snippet' (no truncation), and the
arguments are parsed at import time.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8767)
ap.add_argument("--upstream", default="http://localhost:8010")
ap.add_argument("--index", default="obliq-des-twitter")
ap.add_argument("--snippet-max-tokens", type=int, default=512)
args = ap.parse_args()

UP = args.upstream.rstrip("/")
T0 = time.time()
_lock = threading.Lock()
_tag_counts: dict[str, dict] = {}
_tag_docids: dict[str, set] = {}
_n_search = 0


def _search(query: str, k: int, mode: str, tag: str | None):
    global _n_search
    r = requests.post(f"{UP}/search", json={"query": query, "top_k": int(k),
                                            "index": args.index}, timeout=300)
    r.raise_for_status()
    hits = r.json().get("results", [])
    with _lock:
        _n_search += 1
        if tag:
            c = _tag_counts.setdefault(tag, {"search": 0, "k_sum": 0, "hits": 0})
            c["search"] += 1
            c["k_sum"] += int(k)
            c["hits"] += len(hits)
            _tag_docids.setdefault(tag, set()).update(str(h["doc_id"]) for h in hits)
    out = []
    for h in hits:
        docid = str(h["doc_id"])
        text = h.get("text") or ""
        score = round(float(h.get("score", 0.0)), 6)
        if mode == "ids":
            out.append({"docid": docid, "score": score})
        elif mode == "full":
            out.append({"docid": docid, "score": score, "text": text, "n_chars": len(text),
                        "embedded_span": [0, len(text)], "snippet_span": [0, len(text)]})
        else:
            out.append({"docid": docid, "score": score, "snippet": text})
    return out


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/stats":
            return self._send(200, {"retriever": f"adapter -> {UP} [{args.index}]",
                                    "n_search": _n_search, "uptime_s": round(time.time() - T0, 1)})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n) or b"{}")
        try:
            if self.path == "/search":
                if not req.get("tag") and not req.get("untagged_ok"):
                    return self._send(400, {"error": "tag required"})
                return self._send(200, {"results": _search(
                    req["query"], req.get("k", 5), req.get("mode", "snippet"), req.get("tag"))})
            if self.path == "/tag_counts":
                return self._send(200, {"counts": _tag_counts.get(req.get("tag"))})
            if self.path == "/tag_docids":
                ids = _tag_docids.get(req.get("tag"))
                return self._send(200, {"docids": sorted(ids) if ids is not None else None})
            self._send(404, {"error": "not found"})
        except Exception as e:
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), H)
    srv.daemon_threads = True
    print(f"[adapter] READY on {args.port} -> {UP} index={args.index}", flush=True)
    srv.serve_forever()
