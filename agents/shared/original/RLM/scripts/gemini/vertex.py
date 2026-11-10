"""Gemini embeddings on Vertex AI with the service account in .env (NNEXT_PROVIDERS_GCP_VERTEX_SERVICE_ACCOUNT_JSON).
The key is read at runtime and never written to disk. Run with /home/extstu_00/venvs/gemini/bin/python."""
import json, os

from google import genai
from google.genai import types
from google.oauth2 import service_account

ENV = "/home/extstu_00/RLM/.env"
VAR = "NNEXT_PROVIDERS_GCP_VERTEX_SERVICE_ACCOUNT_JSON"


def _sa_info():
    v = os.environ.get(VAR)
    if not v:
        for line in open(ENV):
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
