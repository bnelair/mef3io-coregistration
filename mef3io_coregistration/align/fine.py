"""Stage 2 — fine alignment / anchor.

Normalized cross-correlation within a user ROI on the band-limited decimated signal, seeded
by the gross offset, with sub-sample refinement. Emits a single anchor ``(t_ref, t_moving)`` +
confidence (R5.2); call it at several ROIs to accumulate anchors (R5.3). Reference and moving
traces must be at the same (harmonized) target Fs — the lower channel's Fs by default.

Windows are cut by integer sample slicing (no interpolation, which would distort near-Nyquist
content and bias the peak); the sub-sample fractional alignment between the two windows' start
timestamps is carried explicitly and added to the measured lag.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from mef3io_coregistration.config import (
    DRIFT_MAX_GAP_FRACTION,
    FINE_WINDOW_DEFAULT_S,
    LOCAL_SEED_HALF_S,
    LOCAL_SEED_LAG_S,
    MIN_LOCK_CONFIDENCE,
    US_PER_S,
)
from mef3io_coregistration.result import Anchor

from .conditioning import DETECTION_FILTERS, FilterSpec, condition_pair
from .tde import normxcorr_delay, normxcorr_surface


class Trace(Protocol):
    """A uniformly sampled trace with an absolute start (e.g. ``Decimated``/``ChannelRead``).

    Read-only members, so frozen dataclasses satisfy it.
    """

    @property
    def data(self) -> np.ndarray: ...
    @property
    def start_uutc(self) -> int | None: ...
    @property
    def fs(self) -> float: ...


class GappyWindow(ValueError):
    """The requested window overlaps too much recording gap to align on (R1.3, REVIEW B3)."""


def estimate_anchor(
    ref: Trace,
    mov: Trace,
    roi_center_uutc: int,
    seed_offset_s: float,
    half_window_s: float = FINE_WINDOW_DEFAULT_S / 2,
    max_residual_s: float = 0.25,
    max_gap_fraction: float = DRIFT_MAX_GAP_FRACTION,
    filters: FilterSpec | None = None,
) -> Anchor:
    """Refine the offset at an ROI, seeded by ``seed_offset_s``, into an anchor (R5.1, R5.2).

    A window of half-width ``half_window_s`` is cut from the reference at ``roi_center_uutc``
    and from the moving trace at the seeded position ``roi_center + seed_offset``; both are
    conditioned (high-passed, robust-scaled, clipped) on the *estimation* band, normalized
    cross-correlation gives the residual lag (bounded by ``max_residual_s``), and the anchor
    offset is ``seed + residual``.

    Raises :class:`GappyWindow` if either window is mostly recording gap — better to say so
    than to correlate against interpolated zeros and return a confident number (REVIEW B3).
    """
    if abs(ref.fs - mov.fs) > 1e-6:
        raise ValueError("reference and moving traces must be harmonized to the same Fs")
    fs = ref.fs

    mov_center = int(round(roi_center_uutc + seed_offset_s * US_PER_S))
    a, a_start, b, b_start = paired_windows(
        ref, mov, roi_center_uutc, mov_center, half_window_s
    )
    n = min(a.shape[0], b.shape[0])

    ca, cb = condition_pair(a[:n], b[:n], fs, filters)
    _reject_if_gappy(ca.gap_fraction, cb.gap_fraction, max_gap_fraction)

    delay = normxcorr_delay(ca.data, cb.data, fs, max_lag_s=max_residual_s)
    # offset = (window start difference) + (measured residual lag). The start difference
    # carries the seed plus the sub-sample slice alignment; the lag carries the correction.
    offset_s = (b_start - a_start) / US_PER_S + delay.lag_s
    t_ref = int(roi_center_uutc)
    t_moving = int(round(t_ref + offset_s * US_PER_S))
    return Anchor(
        t_ref, t_moving, offset_s, delay.confidence, ncc=delay.ncc, window_s=2 * half_window_s
    )


@dataclass(frozen=True)
class CostCurve:
    """The correlation-vs-offset curve for an ROI — the cost surface Refine picks its peak from.

    ``offset_s`` and ``corr`` are equal-length arrays: the (absolute) offset each lag corresponds
    to, and the max-normalised correlation there. ``best_offset_s`` is the argmax (what Refine
    would return), ``seed_offset_s`` is where the window was cut, and the search spanned
    ``seed ± max_residual_s``. Draw the curve, mark the seed and the current on-screen offset, and
    the user can see whether the peak is real and where they are on it.
    """

    offset_s: list[float]
    corr: list[float]
    best_offset_s: float
    confidence: float
    ncc: float
    seed_offset_s: float
    window_s: float


def cost_surface(
    ref: Trace,
    mov: Trace,
    roi_center_uutc: int,
    seed_offset_s: float,
    half_window_s: float = FINE_WINDOW_DEFAULT_S / 2,
    max_residual_s: float = 0.5,
    max_gap_fraction: float = DRIFT_MAX_GAP_FRACTION,
    filters: FilterSpec | None = None,
    step: int = 1,
) -> CostCurve:
    """Correlation curve for the ROI, over ``seed ± max_residual_s`` — same windows and
    conditioning as :func:`estimate_anchor`, so its argmax is exactly the anchor Refine produces.

    ``step`` decimates the returned curve (every ``step``-th lag) to bound the payload on a wide
    search; the reported ``best_offset_s`` keeps full sub-sample resolution regardless.
    """
    if abs(ref.fs - mov.fs) > 1e-6:
        raise ValueError("reference and moving traces must be harmonized to the same Fs")
    fs = ref.fs

    mov_center = int(round(roi_center_uutc + seed_offset_s * US_PER_S))
    a, a_start, b, b_start = paired_windows(
        ref, mov, roi_center_uutc, mov_center, half_window_s
    )
    n = min(a.shape[0], b.shape[0])
    ca, cb = condition_pair(a[:n], b[:n], fs, filters)
    _reject_if_gappy(ca.gap_fraction, cb.gap_fraction, max_gap_fraction)

    surf = normxcorr_surface(ca.data, cb.data, fs, max_lag_s=max_residual_s)
    # Each lag maps to an absolute offset the same way the anchor does: window-start difference
    # (seed + sub-sample slice alignment) plus the lag itself.
    base = (b_start - a_start) / US_PER_S
    offsets = base + surf.lags_samples / fs
    stride = max(1, int(step))
    return CostCurve(
        offset_s=[float(o) for o in offsets[::stride]],
        corr=[float(c) for c in surf.values[::stride]],
        best_offset_s=base + surf.best_lag_samples / fs,
        confidence=float(surf.confidence),
        ncc=float(surf.ncc),
        seed_offset_s=float(seed_offset_s),
        window_s=2 * half_window_s,
    )


@dataclass(frozen=True)
class LocalSeed:
    """A seed refined around an ROI. ``refined`` is False when the neighbourhood gave no
    trustworthy peak (gap, flat signal, or a peak on the search boundary) and ``offset_s`` is
    then the seed as given."""

    offset_s: float
    refined: bool
    confidence: float = 0.0


def local_seed(
    ref: Trace,
    mov: Trace,
    roi_center_uutc: int,
    seed_offset_s: float,
    half_span_s: float = LOCAL_SEED_HALF_S,
    max_lag_s: float = LOCAL_SEED_LAG_S,
    max_gap_fraction: float = DRIFT_MAX_GAP_FRACTION,
) -> LocalSeed:
    """Refine a coarse seed from the ROI's neighbourhood, before Stage 2 narrows in.

    The Stage 1 lock is ONE offset for the whole recording. Over a day the clocks drift by
    seconds (GROSS_CONSENSUS_S exists to absorb exactly that), so at any given ROI the global
    offset can be further from the truth than Stage 2's narrow search reaches — and Stage 2 then
    returns the best peak *inside* its window, a confident-looking anchor that is simply wrong
    (measured: 604.51 s for a true 605.00 s, confidence 0.45, 6 h into a ±5 s day). The same
    holds for the last anchor's offset used far from that anchor.

    So: correlate ±``half_span_s`` of the neighbourhood on the *detection* band, searching
    ±``max_lag_s`` around the seed, and move the seed there only if the peak is distinctive and
    not on the search boundary. A seed placed by hand is not passed through this (the caller
    decides); a failed refinement returns the seed unchanged.
    """
    try:
        mov_center = int(round(roi_center_uutc + seed_offset_s * US_PER_S))
        a, a_start, b, b_start = paired_windows(ref, mov, roi_center_uutc, mov_center, half_span_s)
    except GappyWindow:
        return LocalSeed(seed_offset_s, False)
    n = min(a.shape[0], b.shape[0])
    if n < 4:
        return LocalSeed(seed_offset_s, False)
    ca, cb = condition_pair(a[:n], b[:n], ref.fs, DETECTION_FILTERS)
    if max(ca.gap_fraction, cb.gap_fraction) > max_gap_fraction:
        return LocalSeed(seed_offset_s, False)
    delay = normxcorr_delay(ca.data, cb.data, ref.fs, max_lag_s=max_lag_s)
    if delay.at_edge or delay.confidence < MIN_LOCK_CONFIDENCE:
        return LocalSeed(seed_offset_s, False, delay.confidence)
    return LocalSeed((b_start - a_start) / US_PER_S + delay.lag_s, True, delay.confidence)


def _reject_if_gappy(gap_a: float, gap_b: float, limit: float) -> None:
    worst = max(gap_a, gap_b)
    if worst > limit:
        raise GappyWindow(
            f"window is {worst:.0%} recording gap (limit {limit:.0%}) — pick an ROI clear of "
            "the gaps, or widen the window"
        )


def paired_windows(
    ref: Trace,
    mov: Trace,
    ref_center_uutc: int,
    mov_center_uutc: int,
    half_s: float,
    min_half_s: float = 0.5,
) -> tuple[np.ndarray, int, np.ndarray, int]:
    """Cut the SAME window from both traces, shrinking it symmetrically if either runs out.

    The offset a caller derives from these windows is ``(b_start - a_start) + lag``, and that
    arithmetic is only valid if both windows sit at the same position **relative to their own
    centres**. Slicing them independently breaks exactly that: near a recording's edge one window
    clamps and the other does not, the two centres silently drift apart, and the clamping
    asymmetry is then read back out as though it were signal.

    That is not hypothetical. With the viewport free to pan beyond the reference recording (so the
    user can inspect the extrapolated region), an ROI placed out there produced a reference window
    clamped to the file's first sample while the moving window sat where it was asked — and Stage
    2, seeded at -67.5 s and searching +/-0.5 s, returned an anchor at **-3588 s**. A bounded
    search cannot produce an unbounded error; the error came from the windows, not the estimator.

    So: shrink both windows to the half-width that BOTH can honour, and refuse if what remains is
    too small to align on.
    """
    if ref.start_uutc is None or mov.start_uutc is None:
        raise ValueError("both traces need an absolute start_uutc for alignment")

    half = min(
        half_s,
        _available_half_s(ref, ref_center_uutc),
        _available_half_s(mov, mov_center_uutc),
    )
    if half < min_half_s:
        raise GappyWindow(
            "the ROI does not fit inside both recordings at this offset — it is at or beyond the "
            "edge of one of them. Move it into the span the two recordings share."
        )

    a, a_start = sample_window(ref, ref_center_uutc, half)
    b, b_start = sample_window(mov, mov_center_uutc, half)
    return a, a_start, b, b_start


def _available_half_s(sig: Trace, center_uutc: int) -> float:
    """The largest half-window centred at ``center_uutc`` that fits inside ``sig``."""
    base = int(sig.start_uutc or 0)
    i_center = (center_uutc - base) / US_PER_S * sig.fs
    return min(i_center, sig.data.shape[0] - i_center) / sig.fs


def sample_window(sig: Trace, center_uutc: int, half_s: float) -> tuple[np.ndarray, int]:
    """Slice ``2*half_s`` seconds of ``sig`` centred on the sample nearest ``center_uutc``.

    Returns ``(data, start_uutc)`` where ``start_uutc`` is the exact uUTC of the first sliced
    sample on ``sig``'s native grid. No interpolation is performed.
    """
    if sig.start_uutc is None:
        raise ValueError("trace needs an absolute start_uutc for alignment")
    base = sig.start_uutc
    fs = sig.fs
    i_center = int(round((center_uutc - base) / US_PER_S * fs))
    n_half = int(round(half_s * fs))
    i0 = max(0, i_center - n_half)
    i1 = min(sig.data.shape[0], i_center + n_half)
    start = int(round(base + i0 * US_PER_S / fs))
    return sig.data[i0:i1], start
