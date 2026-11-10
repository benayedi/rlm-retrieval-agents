"""RAG-enabled dspy.RLM with FAISS retrieval and delegation for BrowseComp+.

Publication copy of rag/__init__.py: the package was `rag` in the original repository and is `dspy_rlm` here; the
public names are unchanged. Imported by src/run_browsecomp.py and src/run_obliq.py, with
PYTHONPATH=<repo>/agents/shared:<repo>/agents/dspy_rlm/src (see ../../experiments/).
"""

from .agent import RunMetrics, build_lm, build_rlm_agent, run_question
from .tools import FaissRetriever, RemoteRetriever, make_embedding_fn

__all__ = [
    "RemoteRetriever",
    "FaissRetriever",
    "make_embedding_fn",
    "RunMetrics",
    "build_lm",
    "build_rlm_agent",
    "run_question",
]
