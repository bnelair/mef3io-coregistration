"""Transform: the clock map and moving-signal resampling."""
from __future__ import annotations

from .clock_map import ClockMap
from .resample import resample_onto_reference

__all__ = ["ClockMap", "resample_onto_reference"]
