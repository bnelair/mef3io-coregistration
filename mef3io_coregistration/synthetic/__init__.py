"""Synthetic-warp test harness.

Generate reference/moving recording pairs with a known clock offset, so the whole
pipeline has ground truth to recover. Fixtures are cached on disk and regenerated only
when missing or when the harness version changes.
"""
from __future__ import annotations

from .fixtures import PROFILES, FixturePair, Signal, build_or_load
from .generate import RenderedPair, make_source, render_pair, render_pair_long
from .warp import WarpSpec

__all__ = [
    "PROFILES",
    "FixturePair",
    "RenderedPair",
    "Signal",
    "WarpSpec",
    "build_or_load",
    "make_source",
    "render_pair",
    "render_pair_long",
]
