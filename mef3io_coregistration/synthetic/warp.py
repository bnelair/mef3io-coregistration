"""Ground-truth clock-offset models for the synthetic-warp harness.

The offset is defined in SECONDS as a function of reference-clock time relative to
recording start (``t_rel_s``):

    offset(t) = t_moving - t_ref

The reference clock is treated as true/world time (it is "sacred" — never
transformed). The moving clock is ``t_moving = t_ref + offset(t_ref)``. Because
``|d offset/dt|`` stays well below 1, the resulting time map is strictly monotonic
(a required invariant) even when the offset itself is
non-monotonic (swings positive then negative).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DAY_S = 86_400.0


@dataclass(frozen=True)
class WarpSpec:
    """A parametric, analytic clock offset (so tests have exact ground truth).

    Components are summed:

    - ``offset0_s``      constant absolute offset — can be minutes; models a
                         second recorder whose clock is set by hand.
    - ``rate_s_per_s``   linear drift rate.
    - ``swing_amp_s``    amplitude of a smooth sinusoidal wander — models drift
                         that goes positive then negative over a day.
    - ``swing_period_s`` period of that wander.
    """

    kind: str
    offset0_s: float = 0.0
    rate_s_per_s: float = 0.0
    swing_amp_s: float = 0.0
    swing_period_s: float = DAY_S

    def offset_s(self, t_rel_s: np.ndarray | float) -> np.ndarray:
        """Offset in seconds at reference-relative time(s) ``t_rel_s``."""
        t = np.asarray(t_rel_s, dtype=float)
        o = np.full(t.shape, self.offset0_s, dtype=float)
        o = o + self.rate_s_per_s * t
        if self.swing_amp_s:
            o = o + self.swing_amp_s * np.sin(2.0 * np.pi * t / self.swing_period_s)
        return o

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "offset0_s": self.offset0_s,
            "rate_s_per_s": self.rate_s_per_s,
            "swing_amp_s": self.swing_amp_s,
            "swing_period_s": self.swing_period_s,
        }

    @classmethod
    def from_dict(cls, d: dict) -> WarpSpec:
        return cls(**d)
