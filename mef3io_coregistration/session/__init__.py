"""Session: interactive state, persistence, background loaders."""
from __future__ import annotations

from mef3io_coregistration.result import CoregResult

from .coreg_session import CoregSession
from .loader import BackgroundLoad, Cancelled, LoadProduct, LoadSpec, load_alignment_signal

__all__ = [
    "BackgroundLoad",
    "Cancelled",
    "CoregResult",
    "CoregSession",
    "LoadProduct",
    "LoadSpec",
    "load_alignment_signal",
]
