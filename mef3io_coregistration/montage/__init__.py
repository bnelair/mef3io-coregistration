"""Montage: channel specs (unipolar/bipolar, per file) and bipolar derivations."""
from __future__ import annotations

from .bipolar import BipolarSignal, bipolar
from .spec import ChannelSpec, read_montage, spec_bounds, spec_fs, spec_gaps

__all__ = [
    "BipolarSignal",
    "ChannelSpec",
    "bipolar",
    "read_montage",
    "spec_bounds",
    "spec_fs",
    "spec_gaps",
]
