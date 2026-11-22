"""pytest wiring for the search-policy study tests: puts this directory on sys.path and exposes the `cfg` fixture.
The helpers live in search_policy_testlib.py (a unique module name, so tests of other folders cannot shadow it)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from search_policy_testlib import cfg  # noqa: E402,F401
