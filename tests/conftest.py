"""Import telecomtriage.guardrails without running the package __init__.

telecomtriage/__init__.py imports agent.py, which seeds the vector store with live Gemini
embedding calls at import time. Unit tests must not need an API key or network, so register
a bare package object first; submodules then import normally.
"""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
pkg = types.ModuleType("telecomtriage")
pkg.__path__ = [str(ROOT / "telecomtriage")]
sys.modules.setdefault("telecomtriage", pkg)
