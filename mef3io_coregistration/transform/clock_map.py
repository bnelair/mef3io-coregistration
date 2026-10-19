"""The clock map.

Canonical representation: reference clock → moving clock as anchor arrays ``(x_ref,
y_moving)`` (absolute uUTC), an interpolation method, and an extrapolation policy. The map
is **monotone** (``dy/dx > 0`` everywhere), so ``inverse`` is a swapped-axis interpolation.
It is the saved artifact of coregistration and subsumes both the linear case
(two anchors) and a dense drift curve (many anchors).

Timestamps are absolute uUTC (~1.7e15). To keep µs precision through interpolation, all
arithmetic is done on offsets from the first anchor and the integer origin is re-added last.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.interpolate import PchipInterpolator

from .. import __version__
from ..config import US_PER_S

InterpMethod = Literal["linear", "pchip"]
Extrapolation = Literal["hold_end_slope"]

_FORMAT = "signal_coregistration.clock_map"
_FORMAT_VERSION = 1


@dataclass(frozen=True)
class ClockMap:
    """Monotone reference→moving clock map with defined interp/extrapolation policy.

    ``x_ref`` and ``y_moving`` are strictly increasing ``int64`` uUTC anchor arrays of equal
    length (≥ 2). ``interp_method`` is ``"linear"`` (default) or ``"pchip"`` (monotone
    cubic). ``extrapolation`` beyond the anchor span holds the end slope (linear), and such
    queries are flagged by :meth:`is_extrapolated`. ``metadata`` is free-form
    provenance persisted alongside the map (the session layer fills it in).
    """

    x_ref: np.ndarray
    y_moving: np.ndarray
    interp_method: InterpMethod = "linear"
    extrapolation: Extrapolation = "hold_end_slope"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        x = np.asarray(self.x_ref, dtype=np.int64)
        y = np.asarray(self.y_moving, dtype=np.int64)
        object.__setattr__(self, "x_ref", x)
        object.__setattr__(self, "y_moving", y)
        if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
            raise ValueError("x_ref and y_moving must be 1-D arrays of equal length")
        if x.shape[0] < 2:
            raise ValueError("a clock map needs at least two anchors")
        if not np.all(np.diff(x) > 0):
            raise ValueError("x_ref must be strictly increasing")
        if not np.all(np.diff(y) > 0):
            raise ValueError("y_moving must be strictly increasing (map must be monotone)")
        if self.interp_method not in ("linear", "pchip"):
            raise ValueError(f"unknown interp_method {self.interp_method!r}")

    # -- construction helpers ----------------------------------------------
    @classmethod
    def from_reference_offset(
        cls,
        x_ref: np.ndarray,
        offset_s: np.ndarray | float,
        **kwargs: object,
    ) -> ClockMap:
        """Build a map from reference anchors and an offset (seconds): ``y = x + offset``."""
        x = np.asarray(x_ref, dtype=np.int64)
        y = x + np.rint(np.asarray(offset_s, dtype=np.float64) * US_PER_S).astype(np.int64)
        return cls(x, y, **kwargs)  # type: ignore[arg-type]

    # -- evaluation -------------------------------------------
    def end_slopes(self) -> tuple[float, float]:
        """d(moving)/d(reference) held beyond each end of the knots.

        A fitted map may store its drift rate at each end (``end_offset_slopes``, s/s,
        fitted over several windows); older maps and simple ones hold the end segment's slope.
        """
        stored = self.metadata.get("end_offset_slopes") if self.metadata else None
        if stored:
            return 1.0 + float(stored[0]), 1.0 + float(stored[1])
        xf = self.x_ref.astype(np.float64)
        yf = self.y_moving.astype(np.float64)
        return (yf[1] - yf[0]) / (xf[1] - xf[0]), (yf[-1] - yf[-2]) / (xf[-1] - xf[-2])

    def forward(self, t_ref: np.ndarray | float) -> np.ndarray | float:
        """Map reference-clock uUTC → moving-clock uUTC (float; caller rounds if needed)."""
        return _interp_extrap(t_ref, self.x_ref, self.y_moving, self.interp_method, self.end_slopes())

    def inverse(self, t_moving: np.ndarray | float) -> np.ndarray | float:
        """Map moving-clock uUTC → reference-clock uUTC (valid because the map is monotone).

        For ``linear`` the swapped-axis interpolation is the exact inverse. For ``pchip`` a
        swapped-axis cubic would not invert the forward curve, so the forward map is inverted
        numerically (monotone bisection) to keep ``inverse(forward(t)) == t``.
        """
        if self.interp_method == "linear":
            s0, s1 = self.end_slopes()
            return _interp_extrap(t_moving, self.y_moving, self.x_ref, "linear", (1.0 / s0, 1.0 / s1))
        return self._inverse_numeric(t_moving)

    def _inverse_numeric(self, t_moving: np.ndarray | float) -> np.ndarray | float:
        q = np.asarray(t_moving, dtype=np.float64)
        scalar = q.ndim == 0
        qa = np.atleast_1d(q)
        xf = self.x_ref.astype(np.float64)
        yf = self.y_moving.astype(np.float64)
        x0, x1, y0, y1 = xf[0], xf[-1], yf[0], yf[-1]
        s0, s1 = self.end_slopes()

        out = np.empty_like(qa)
        below, above = qa < y0, qa > y1
        inside = ~(below | above)
        out[below] = x0 + (qa[below] - y0) / s0  # forward extrapolates linearly, so does inverse
        out[above] = x1 + (qa[above] - y1) / s1
        if np.any(inside):
            target = qa[inside]
            lo = np.full(target.shape, x0)
            hi = np.full(target.shape, x1)
            for _ in range(60):  # 2^-60 of the domain → far below µs resolution
                mid = 0.5 * (lo + hi)
                fm = np.asarray(self.forward(np.rint(mid).astype(np.int64)), dtype=np.float64)
                go_right = fm < target
                lo = np.where(go_right, mid, lo)
                hi = np.where(go_right, hi, mid)
            out[inside] = 0.5 * (lo + hi)
        return float(out[0]) if scalar else out

    def offset_s(self, t_ref: np.ndarray | float) -> np.ndarray | float:
        """Offset ``t_moving - t_ref`` in seconds at reference time(s) ``t_ref``."""
        t = np.asarray(t_ref, dtype=np.float64)
        return (np.asarray(self.forward(t_ref), dtype=np.float64) - t) / US_PER_S

    def is_extrapolated(self, t_ref: np.ndarray | float) -> np.ndarray | bool:
        """True where ``t_ref`` falls outside the measured span."""
        lo, hi = self.measured_span
        t = np.asarray(t_ref)
        flag = (t < lo) | (t > hi)
        return bool(flag) if flag.ndim == 0 else flag

    @property
    def measured_span(self) -> tuple[int, int]:
        """The reference-clock span the map was actually measured over.

        Usually the knots' span; but a map built without drift carries knots out to the ends of
        the recording (the held offset) while only its anchors were measured — it records that
        span in ``metadata["measured_span_uutc"]``, which then decides what is extrapolated.
        """
        span = self.metadata.get("measured_span_uutc") if self.metadata else None
        if span:
            return int(span[0]), int(span[1])
        return self.domain

    @property
    def domain(self) -> tuple[int, int]:
        """The measured reference-clock span ``(x_ref[0], x_ref[-1])``."""
        return int(self.x_ref[0]), int(self.x_ref[-1])

    # -- persistence ------------------------------------------
    def to_dict(self) -> dict:
        return {
            "format": _FORMAT,
            "format_version": _FORMAT_VERSION,
            "tool_version": __version__,
            "interp_method": self.interp_method,
            "extrapolation": self.extrapolation,
            "x_ref": [int(v) for v in self.x_ref],
            "y_moving": [int(v) for v in self.y_moving],
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> Path:
        """Persist the map as JSON (integer anchors → bit-stable reload)."""
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return path

    @classmethod
    def from_dict(cls, d: dict) -> ClockMap:
        if d.get("format") != _FORMAT:
            raise ValueError(f"not a clock map file (format={d.get('format')!r})")
        return cls(
            x_ref=np.asarray(d["x_ref"], dtype=np.int64),
            y_moving=np.asarray(d["y_moving"], dtype=np.int64),
            interp_method=d.get("interp_method", "linear"),
            extrapolation=d.get("extrapolation", "hold_end_slope"),
            metadata=d.get("metadata", {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> ClockMap:
        return cls.from_dict(json.loads(Path(path).read_text()))


def _interp_extrap(
    query: np.ndarray | float,
    xp: np.ndarray,
    yp: np.ndarray,
    method: InterpMethod,
    end_slopes: tuple[float, float] | None = None,
) -> np.ndarray | float:
    """Interpolate ``yp`` at ``query`` over anchors ``xp`` with hold-end-slope extrapolation
    (``end_slopes`` dy/dx at each end; default: the end segments').

    Interior uses ``method``; outside the anchor span the end segment's slope is held so the
    result stays monotone and continuous. Arithmetic is on offsets from ``xp[0]``/``yp[0]`` to
    preserve µs precision at absolute-uUTC magnitudes.
    """
    qf = np.asarray(query, dtype=np.float64)  # never truncate to int (inverse gets fractions)
    scalar = qf.ndim == 0
    x0, y0 = int(xp[0]), int(yp[0])
    xf = (xp - x0).astype(np.float64)
    yf = (yp - y0).astype(np.float64)
    xq = np.atleast_1d(qf - x0)

    if method == "linear":
        yq = np.interp(xq, xf, yf)
    else:  # pchip (monotone cubic) interior; NaN outside, filled by extrapolation below
        yq = PchipInterpolator(xf, yf, extrapolate=False)(xq)

    if end_slopes is not None:
        s0, s1 = end_slopes
    else:
        s0 = (yf[1] - yf[0]) / (xf[1] - xf[0])
        s1 = (yf[-1] - yf[-2]) / (xf[-1] - xf[-2])
    yq = np.where(xq < xf[0], yf[0] + s0 * (xq - xf[0]), yq)
    yq = np.where(xq > xf[-1], yf[-1] + s1 * (xq - xf[-1]), yq)

    result = yq + y0
    return float(result[0]) if scalar else result
