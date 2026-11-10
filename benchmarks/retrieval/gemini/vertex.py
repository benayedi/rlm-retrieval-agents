"""Gemini embeddings on Vertex AI with the service account in .env (NNEXT_PROVIDERS_GCP_VERTEX_SERVICE_ACCOUNT_JSON).
The key is read at runtime and never written to disk. Run with ${GEMINI_PYTHON} (original: a dedicated venv).

Refactored copy of scripts/gemini/vertex.py of the original repository; imported by search_server.py and
build_index.py next to it. CONFIGURATION REQUIREMENT: the service-account JSON comes from the environment variable
NNEXT_PROVIDERS_GCP_VERTEX_SERVICE_ACCOUNT_JSON or, when that is unset, from the line of the same name in the .env-format
file named by GCP_VERTEX_ENV (originally the .env of the original repository, a fixed absolute path). Nothing here
prints or stores it. Location: GEMINI_LOCATION, default "eu". See agents/shared/CHANGES_FROM_ORIGINAL.md.
"""
import json, os, sys

# Refactor (import wiring): <repo>/agents/shared on sys.path, computed from this file (<repo>/benchmarks/retrieval/gemini/).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "agents", "shared"))
from rlm_shared.paths import path  # noqa: E402

from google import genai
from google.genai import types
from google.oauth2 import service_account

# Refactor: ENV was the absolute path of the original repository's .env; it is now GCP_VERTEX_ENV, read only when the
# file is actually needed (below), so an exported service-account variable needs no file.
VAR = "NNEXT_PROVIDERS_GCP_VERTEX_SERVICE_ACCOUNT_JSON"


def _sa_info():
    v = os.environ.get(VAR)
    if not v:
        for line in open(path("GCP_VERTEX_ENV")):
            if line.startswith(VAR + "="):
                v = line.split("=", 1)[1].strip()
                break
    if v[:1] in "'\"" and v[-1:] == v[:1]:
        v = v[1:-1]
    return json.loads(v)


def client(location=None):
    info = _sa_info()
    creds = service_account.Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/cloud-platform"])
    # 120 s per request (2026-10-27): without a timeout one stuck call stalled a 213k-document build for good.
    return genai.Client(vertexai=True, project=info["project_id"],
                        location=location or os.environ.get("GEMINI_LOCATION", "eu"), credentials=creds,
                        http_options=types.HttpOptions(timeout=120_000))
