"""
Project-wide UTF-8 safeguards.

Placing ``sitecustomize.py`` at the repository root forces Python to import
this module on start-up (as long as the project is on ``sys.path``).  The code
below normalises standard I/O streams to UTF-8 and ensures any subprocesses
inherit matching environment variables.
"""

from __future__ import annotations

import os
import sys

# Guarantee child processes receive explicit UTF-8 hints even on Windows where
# the legacy ANSI code page would normally win.
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def _reconfigure_stream(stream_name: str) -> None:
    stream = getattr(sys, stream_name, None)
    if stream is None:
        return
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            # Fall back silently; replacing the stream isn't worth crashing the
            # interpreter during bootstrap.
            pass


_reconfigure_stream("stdout")
_reconfigure_stream("stderr")

# Python's text-mode stdin can still honour the console codepage on Windows;
# reconfigure it if possible so interactive prompts accept Unicode input.
_reconfigure_stream("stdin")
