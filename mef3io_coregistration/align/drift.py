"""Stage 3 — continuous drift estimation.

Seeded sliding-window cross-correlation across the full length. Every Stage 2 anchor is ground
truth for its neighbourhood: each window belongs to its nearest anchor, and each anchor's windows
are walked **outward from it**. A window's search is narrow (``max_residual_s``) around a
prediction from the recent confident estimates — last value plus local slope — so it locks onto
the *nearby* peak and does not track-jump (R6.1). A peak on the search boundary is rejected: the
true peak lies outside the search. (Walking from the start of the span with the anchor's offset
as the seed broke down whenever the anchor sat far from the start: with realistic drift the seed
was already off by more than the search range at the first window, the walk locked onto a side
peak and followed it — a constant 0.5–1 s error with nearly every window "accepted".) Each window keeps its peak height as a confidence weight; low-
confidence windows are dropped from the fit and do not advance the seed (R6.2). A weighted
smoothing spline through the surviving ``(t_i, offset_i)`` gives a continuous ``offset(t)``,
sampled to the monotone ``(x_ref, y_moving)`` clock map (R6.3, R6.4). The smooth-drift
assumption (no clock jumps within a session) is a documented assumption.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import median_filter

from mef3io_coregistration.config import (
    DRIFT_GRID_S,
    DRIFT_MAX_GAP_FRACTION,
    DRIFT_MAX_SLOPE,
    DRIFT_WINDOW_S,
    MIN_LOCK_CONFIDENCE,
    PEAK_GUARD_S,
    US_PER_S,
)
from mef3io_coregistration.transform.clock_map import ClockMap

from .conditioning import FilterSpec, condition_pair, condition_rows
from .fine import Anchor, GappyWindow, Trace, paired_windows
from .tde import normxcorr_delay, parabolic_peak, peak_quality


@dataclass(frozen=True)
class DriftEstimate:
    """Stage 3 output: the clock map plus per-window QC arrays (R6.2, R12).

    Every array is finite — a window that could not be estimated (gap, no lock) is marked
    ``rejected`` and carries its seeded offset, never ``NaN``. NaN used to reach the API and
    serialise as the bare token ``NaN``, which is not valid JSON and killed the drift panel in
    the browser with a parse error that mentioned neither gaps nor drift (REVIEW B3).
    """

    clock_map: ClockMap
    t_ref_uutc: np.ndarray  # window centres (reference clock)
    offset_s: np.ndarray  # raw per-window offset estimate
    confidence: np.ndarray  # per-window confidence weight in [0, 1]
    smoothed_offset_s: np.ndarray  # spline evaluated at the window centres
    accepted: np.ndarray  # bool mask: windows that passed min_confidence and were gap-clear
    gapped: np.ndarray  # bool mask: windows rejected because they overlap a recording gap
    measured_span_uutc: tuple[int, int]  # first..last ACCEPTED window centre — the honest span
    scatter_s: float = 1.0  # measured per-window lag scatter — kept so the map can be refit
                            # from a user-edited keep set without recomputing correlations
    anchors: tuple[Anchor, ...] = ()  # the Stage 2 anchors the walk started from
    anchor_residual_s: np.ndarray = field(default_factory=lambda: np.empty(0))  # fit − anchor


def estimate_drift(
    ref: Trace,
    mov: Trace,
    anchor: Anchor | Sequence[Anchor],
    start_uutc: int | None = None,
    stop_uutc: int | None = None,
    window_s: float = DRIFT_WINDOW_S,
    grid_s: float = DRIFT_GRID_S,
    max_residual_s: float = 0.25,
    min_confidence: float = MIN_LOCK_CONFIDENCE,
    max_gap_fraction: float = DRIFT_MAX_GAP_FRACTION,
    smoothing: float | None = None,
    filters: FilterSpec | None = None,
) -> DriftEstimate:
    """Estimate ``offset(t)`` over the anchored span and return a clock map (R6).

    ``ref``/``mov`` are harmonized traces at the same Fs. ``anchor`` is one or more Stage 2
    anchors; every window belongs to its nearest anchor and is reached by walking outward from
    it, each window seeded from the recent confident estimates. Windows that overlap a recording
    gap, whose correlation peak is not distinctive, or whose peak sits on the search boundary are
    rejected: they do not advance the seed, do not enter the spline fit, and do not become map
    anchors. ``smoothing`` is the spline's ``s`` (weighted sum-of-squares budget).
    """
    anchors = _as_anchors(anchor)
    if abs(ref.fs - mov.fs) > 1e-6:
        raise ValueError("reference and moving traces must be harmonized to the same Fs")
    if ref.start_uutc is None:
        raise ValueError("reference trace needs an absolute start_uutc")
    fs = ref.fs
    half = window_s / 2.0

    ref_start = int(ref.start_uutc)
    ref_stop = ref_start + int(round((ref.data.shape[0] - 1) * US_PER_S / fs))
    # Windows only where BOTH recordings exist: the moving extent mapped back through the anchors'
    # offsets. (Over the whole reference, the windows beyond the moving file were counted as
    # "recording gaps" — a different thing, and a misleading one.)
    if mov.start_uutc is not None:
        mov_start = int(mov.start_uutc)
        mov_stop = mov_start + int(round((mov.data.shape[0] - 1) * US_PER_S / mov.fs))
        ref_start = max(ref_start, mov_start - int(round(anchors[0].offset_s * US_PER_S)))
        ref_stop = min(ref_stop, mov_stop - int(round(anchors[-1].offset_s * US_PER_S)))
    lo = int(start_uutc if start_uutc is not None else ref_start) + int(round(half * US_PER_S))
    hi = int(stop_uutc if stop_uutc is not None else ref_stop) - int(round(half * US_PER_S))
    centers = np.arange(lo, hi + 1, int(round(grid_s * US_PER_S)), dtype=np.int64)
    if centers.shape[0] < 2:
        raise ValueError("span too short for the requested drift window/grid")

    n_win = centers.shape[0]
    offsets = np.empty(n_win, dtype=np.float64)
    confid = np.zeros(n_win, dtype=np.float64)
    gapped = np.zeros(n_win, dtype=bool)
    fallback_slope = _anchor_slope(anchors)

    t_anchor = np.array([a.t_ref_uutc for a in anchors], dtype=np.int64)
    owner = np.argmin(np.abs(centers[:, None] - t_anchor[None, :]), axis=1)
    for k, anc in enumerate(anchors):
        mine = np.flatnonzero(owner == k)
        after = mine[centers[mine] >= anc.t_ref_uutc]
        before = mine[centers[mine] < anc.t_ref_uutc][::-1]
        for path in (after, before):
            history = [(anc.t_ref_uutc, anc.offset_s)]
            pos, size = 0, _BLOCK_START
            while pos < path.size:
                block = path[pos : pos + size]
                line = _prediction_line(history, fallback_slope)
                predicted = [line(int(centers[i])) for i in block]
                tol = _admission_tol(history, fs, max_residual_s)
                results = _measure_block(
                    ref, mov, centers[block], predicted, half, fs, filters,
                    max_residual_s, max_gap_fraction,
                )
                # Keep results up to the first window the prediction did not hold for (no clear
                # peak, or a peak far from the prediction), then re-predict from there with a
                # smaller block. At block size 1 this is exactly the sequential walk.
                used = block.size
                for j, (i, r) in enumerate(zip(block, results, strict=True)):
                    off, conf, gap, edge = r
                    poor = not gap and (
                        edge or conf < min_confidence or abs(off - predicted[j]) > tol
                    )
                    if poor and size > 1:
                        used = j
                        break
                    # A rejected window still needs a finite offset: carry the prediction, so
                    # the arrays the UI plots are always plottable.
                    offsets[i] = predicted[j]
                    if gap:
                        gapped[i] = True
                        continue
                    if edge:
                        continue  # true peak outside the search: no measurement, no advance
                    offsets[i], confid[i] = off, conf
                    # Advance the seed only from confident windows that land where the walk
                    # expected: true drift moves the offset by milliseconds per step, so a
                    # confident window far from the prediction is a spurious peak (R6.2).
                    if conf >= min_confidence and abs(off - predicted[j]) <= tol:
                        history.append((int(centers[i]), float(off)))
                size = min(_BLOCK, size * 2) if used == block.size else max(1, size // 2)
                pos += used

    # Robust outlier rejection against a local median. The seeded chain can, in principle, walk
    # off: a window that locks onto the wrong peak advances the seed, and its neighbours follow
    # it. Drift is smooth by assumption (R6.5), so a window that disagrees with its neighbours
    # is wrong no matter how confident it is — and rejecting it costs nothing, because a smooth
    # curve is well determined by the windows that remain.
    outlier, scatter_s = _local_outliers(offsets, gapped, fs)

    accepted = (confid >= min_confidence) & ~gapped & ~outlier
    n_ok = int(accepted.sum())
    if n_ok < 2:
        raise ValueError(
            f"drift failed: only {n_ok} of {n_win} windows locked "
            f"({int(gapped.sum())} fell in recording gaps). The two files may share too little "
            "signal on this channel — try a different alignment channel, or check the anchor."
        )

    smoothed = _fit_offset(centers, offsets, confid, accepted, scatter_s, smoothing)
    end_slopes = _end_offset_slopes(centers, smoothed, accepted)
    smoothed = _extend(centers, smoothed, accepted, end_slopes)

    # Map anchors come from ACCEPTED windows only. Building them from every window centre —
    # including the ~65% this dataset rejects — and then reporting the full span as "measured"
    # is extrapolation dressed up as measurement (REVIEW B5).
    ok_centers = centers[accepted]
    ok_offsets = smoothed[accepted]
    measured_span = (int(ok_centers[0]), int(ok_centers[-1]))

    clock_map = ClockMap.from_reference_offset(
        ok_centers,
        ok_offsets,
        metadata={
            "stage": "drift",
            "end_offset_slopes": list(end_slopes),
            "window_s": window_s,
            "grid_s": grid_s,
            "max_residual_s": max_residual_s,
            "min_confidence": min_confidence,
            "max_gap_fraction": max_gap_fraction,
            "n_windows": n_win,
            "n_accepted": n_ok,
            "n_gapped": int(gapped.sum()),
            "measured_span_uutc": list(measured_span),
            "anchor": {"t_ref_uutc": anchors[0].t_ref_uutc, "offset_s": anchors[0].offset_s},
            "anchors": [
                {"t_ref_uutc": int(a.t_ref_uutc), "offset_s": float(a.offset_s)} for a in anchors
            ],
        },
    )
    return DriftEstimate(
        clock_map, centers, offsets, confid, smoothed, accepted, gapped, measured_span, scatter_s,
        anchors, _anchor_residuals(centers, smoothed, accepted, anchors),
    )


def refit_drift(
    est: DriftEstimate,
    keep: np.ndarray,
    smoothing: float | None = None,
) -> DriftEstimate:
    """Rebuild the clock map from a user-edited **keep** mask, reusing the per-window estimates.

    Stage 3's expensive part is the per-window correlation; accepting or discarding a window does
    not change any of it. So a manual review — the user overriding which windows are trustworthy —
    is a pure refit: the same weighted smoothing spline through a different subset, at no
    correlation cost. Gapped windows can never be kept (there is no measurement there), so they are
    masked out regardless of what the caller sends.
    """
    keep = np.asarray(keep, dtype=bool) & ~est.gapped
    if int(keep.sum()) < 2:
        raise ValueError("keep at least two (non-gap) windows to rebuild the map")

    smoothed = _fit_offset(
        est.t_ref_uutc, est.offset_s, est.confidence, keep, est.scatter_s, smoothing
    )
    end_slopes = _end_offset_slopes(est.t_ref_uutc, smoothed, keep)
    smoothed = _extend(est.t_ref_uutc, smoothed, keep, end_slopes)
    ok_centers = est.t_ref_uutc[keep]
    ok_offsets = smoothed[keep]
    measured_span = (int(ok_centers[0]), int(ok_centers[-1]))

    meta = dict(est.clock_map.metadata)
    meta.update(
        reviewed=True,
        n_accepted=int(keep.sum()),
        measured_span_uutc=list(measured_span),
        end_offset_slopes=list(end_slopes),
    )
    clock_map = ClockMap.from_reference_offset(ok_centers, ok_offsets, metadata=meta)
    return DriftEstimate(
        clock_map, est.t_ref_uutc, est.offset_s, est.confidence, smoothed,
        keep, est.gapped, measured_span, est.scatter_s,
        est.anchors, _anchor_residuals(est.t_ref_uutc, smoothed, keep, est.anchors),
    )


# Confident estimates used to predict the local drift slope while walking.
_SLOPE_HISTORY = 12
# A window advances the walk only if it lands within _ADMIT_SIGMA robust sigmas of the walk's
# own recent scatter (floor _ADMIT_FLOOR_SAMPLES), and never beyond _TRUST of the search range.
# Windows measured per batch: few until the walk has a local slope, doubling while predictions
# hold, halving (down to the sequential walk) when one does not.
_BLOCK_START = 2
_BLOCK = 32
_TRUST = 0.5
_ADMIT_SIGMA = 4.0
_ADMIT_FLOOR_SAMPLES = 2.0


def _as_anchors(anchor: Anchor | Sequence[Anchor]) -> tuple[Anchor, ...]:
    """Anchors sorted by time; a later anchor at the same time supersedes an earlier one."""
    items = (anchor,) if isinstance(anchor, Anchor) else tuple(anchor)
    if not items:
        raise ValueError("Stage 3 needs at least one Stage 2 anchor")
    latest = {a.t_ref_uutc: a for a in items}
    return tuple(sorted(latest.values(), key=lambda a: a.t_ref_uutc))


def _anchor_slope(anchors: Sequence[Anchor]) -> float:
    """Mean drift rate (s/s) between the outermost anchors; 0 with a single anchor."""
    if len(anchors) < 2:
        return 0.0
    dt = (anchors[-1].t_ref_uutc - anchors[0].t_ref_uutc) / US_PER_S
    return (anchors[-1].offset_s - anchors[0].offset_s) / dt if dt > 0 else 0.0


def _prediction_line(history: list[tuple[int, float]], fallback_slope: float):
    """:func:`_predict` for a whole block: the robust line is fitted once and evaluated at each
    centre (the median is translation-equivariant, so this is the same line)."""
    recent = history[-_SLOPE_HISTORY:]
    t_ref = recent[-1][0]
    x = np.array([h[0] - t_ref for h in recent], dtype=np.float64) / US_PER_S
    y = np.array([h[1] for h in recent])
    slope = _theil_sen_slope(x, y, fallback_slope) if len(recent) >= 3 else fallback_slope
    slope = float(np.clip(slope, -DRIFT_MAX_SLOPE, DRIFT_MAX_SLOPE))
    level = float(y[-1]) if len(recent) < 3 else float(np.median(y - slope * x))

    def at(t: int) -> float:
        return level + slope * (t - t_ref) / US_PER_S

    return at


def _predict(history: list[tuple[int, float]], t: int, fallback_slope: float) -> float:
    """Offset predicted at ``t`` from a robust line through the recent confident estimates.

    Theil–Sen: the slope is the median of pairwise slopes and the level the median of the
    points' intercepts, both over the recent history — so one spurious window moves neither.
    (Predicting from the *last* point let a single noise window that slipped through set every
    later prediction; the genuine windows that followed then looked like outliers and were
    refused, freezing the walk on the bad value.) The slope is clamped to a physical bound:
    clocks do not drift faster than a few seconds per hour.
    """
    recent = history[-_SLOPE_HISTORY:]
    x = np.array([h[0] - t for h in recent], dtype=np.float64) / US_PER_S  # relative to t
    y = np.array([h[1] for h in recent])
    slope = fallback_slope
    if len(recent) >= 3:
        slope = _theil_sen_slope(x, y, fallback_slope)
    slope = float(np.clip(slope, -DRIFT_MAX_SLOPE, DRIFT_MAX_SLOPE))
    if len(recent) < 3:
        return float(y[-1] - slope * x[-1])
    return float(np.median(y - slope * x))  # the line's value at t (x = 0)


def _theil_sen_slope(x: np.ndarray, y: np.ndarray, fallback: float) -> float:
    """Median of the pairwise slopes. Every pair counts, whatever order the points are in: the
    walk runs backward as well as forward, and a backward history is stored newest-first (only
    keeping pairs with increasing x once silently dropped every pair of a backward walk, which
    then extrapolated the fallback slope — of the wrong sign — until it lost the track)."""
    i, j = np.triu_indices(x.shape[0], k=1)
    dx = x[j] - x[i]
    ok = dx != 0
    if not ok.any():
        return fallback
    return float(np.median((y[j] - y[i])[ok] / dx[ok]))


def _measure_block(
    ref: Trace,
    mov: Trace,
    centers: np.ndarray,
    predicted: list[float],
    half: float,
    fs: float,
    filters: FilterSpec | None,
    max_residual_s: float,
    max_gap_fraction: float,
) -> list[tuple[float, float, bool, bool]]:
    """Measure a block of windows: ``[(offset_s, confidence, gapped, at_edge), ...]``.

    Per window this is exactly the sequential step — paired windows, conditioning on the
    estimation band, normalized cross-correlation within ±``max_residual_s``, parabolic peak,
    distinctiveness, edge rejection — but gap-free windows of equal length are conditioned and
    correlated together (one vectorised pass, one batched FFT). Windows with gaps take the
    per-window path unchanged.
    """
    out: list[tuple[float, float, bool, bool] | None] = [None] * len(predicted)
    batch: dict[int, list[tuple[int, np.ndarray, np.ndarray, float]]] = {}
    for j, (c, p) in enumerate(zip(centers, predicted, strict=True)):
        try:
            a, a_start, b, b_start = paired_windows(
                ref, mov, int(c), int(round(c + p * US_PER_S)), half
            )
        except GappyWindow:
            out[j] = (p, 0.0, True, False)
            continue
        n = min(a.shape[0], b.shape[0])
        if n < 2:
            out[j] = (p, 0.0, True, False)
            continue
        base = (b_start - a_start) / US_PER_S
        a, b = a[:n], b[:n]
        if np.isnan(a).any() or np.isnan(b).any():
            ca, cb = condition_pair(a, b, fs, filters)
            if max(ca.gap_fraction, cb.gap_fraction) > max_gap_fraction:
                out[j] = (p, 0.0, True, False)
                continue
            d = normxcorr_delay(ca.data, cb.data, fs, max_lag_s=max_residual_s)
            out[j] = (base + d.lag_s, d.confidence, False, d.at_edge)
            continue
        batch.setdefault(n, []).append((j, a, b, base))

    guard = int(round(PEAK_GUARD_S * fs))
    for n, items in batch.items():
        A = condition_rows(np.vstack([it[1] for it in items]), fs, filters)
        B = condition_rows(np.vstack([it[2] for it in items]), fs, filters)
        A -= A.mean(axis=1, keepdims=True)
        B -= B.mean(axis=1, keepdims=True)
        m = min(n - 1, int(round(max_residual_s * fs)))
        nfft = next_fast_len(2 * n - 1)
        C = irfft(rfft(B, nfft, axis=1) * np.conj(rfft(A, nfft, axis=1)), nfft, axis=1)
        seg = np.concatenate((C[:, nfft - m :], C[:, : m + 1]), axis=1)
        for r, (j, _a, _b, base) in enumerate(items):
            k = int(np.argmax(seg[r]))
            lag = (k - m) + parabolic_peak(seg[r], k)
            if k == 0 or k == seg.shape[1] - 1:
                out[j] = (base + lag / fs, 0.0, False, True)
                continue
            conf, _ratio = peak_quality(seg[r], k, guard)
            out[j] = (base + lag / fs, conf, False, False)
    return [o for o in out if o is not None]


def _admission_tol(history: list[tuple[int, float]], fs: float, max_residual_s: float) -> float:
    """How far from its prediction a window may land and still advance the walk (s).

    Scaled to the walk's own measured scatter — the robust spread of the recent history about
    its Theil–Sen line: a few samples on clean data, so noise windows (whose lags are spread over
    the whole search range) are shut out; up to ``_TRUST`` of the search range on noisy data.
    """
    cap = _TRUST * max_residual_s
    floor = _ADMIT_FLOOR_SAMPLES / fs
    recent = history[-_SLOPE_HISTORY:]
    if len(recent) < 4:
        return cap
    x = np.array([h[0] for h in recent], dtype=np.float64) / US_PER_S
    y = np.array([h[1] for h in recent])
    slope = _theil_sen_slope(x, y, 0.0)
    r = y - slope * x
    sigma = 1.4826 * float(np.median(np.abs(r - np.median(r))))
    return float(np.clip(_ADMIT_SIGMA * sigma, floor, cap))


def _anchor_residuals(
    centers: np.ndarray, smoothed: np.ndarray, accepted: np.ndarray, anchors: Sequence[Anchor]
) -> np.ndarray:
    """Fitted map minus each anchor's own offset (s) — an independent consistency check (R12.1).

    The anchors seed the walk but are not in the fit, so a wrong anchor shows up here instead of
    bending the map. Evaluated on the fitted curve (held flat beyond the measured span).
    """
    if not anchors or not accepted.any():
        return np.zeros(len(anchors))
    t = np.array([a.t_ref_uutc for a in anchors], dtype=np.float64)
    fit = np.interp(t, centers[accepted].astype(np.float64), smoothed[accepted])
    return fit - np.array([a.offset_s for a in anchors])


def _local_outliers(
    offsets: np.ndarray, gapped: np.ndarray, fs: float
) -> tuple[np.ndarray, float]:
    """Windows that disagree with their neighbours, plus the robust per-window scatter.

    Returns ``(outlier_mask, scatter_s)``. ``scatter_s`` is the measurement noise on a single
    window's lag estimate — the number the spline's smoothing budget must be set from.
    """
    n = offsets.shape[0]
    size = min(11, n if n % 2 else n - 1) or 1
    local_median = median_filter(offsets, size=size, mode="nearest")
    resid = offsets - local_median

    usable = ~gapped
    if usable.sum() < 3:
        return np.zeros(n, dtype=bool), 1.0 / fs

    scatter = 1.4826 * float(np.median(np.abs(resid[usable] - np.median(resid[usable]))))
    # Floor well below a sample: parabolic peak interpolation genuinely resolves ~0.1 sample at
    # high SNR, and on clean data the measured scatter is what should bind, not the floor.
    scatter = max(scatter, 0.1 / fs)

    outlier = np.abs(resid) > 4.0 * scatter
    return outlier & usable, scatter


def _end_offset_slopes(
    centers: np.ndarray, smoothed: np.ndarray, accepted: np.ndarray
) -> tuple[float, float]:
    """The drift rate (s/s) the map holds beyond its measured span, at each end.

    One slope per end, fitted by least squares over the fit's first / last few accepted windows
    (a fifth of them, 2–20) — not the last knot-to-knot segment, whose integer-µs knots quantise
    it to 1 µs per grid step and which is the noisiest estimate available. Stored in the map
    (``end_offset_slopes``) so the map, the plotted curve and the QC budget extrapolate alike.
    """
    idx = np.flatnonzero(accepted)
    if idx.size < 2:
        return 0.0, 0.0
    k = max(2, min(idx.size // 5, 20))
    t = centers[idx].astype(np.float64) / US_PER_S
    y = smoothed[idx]
    lo = float(np.polyfit(t[:k] - t[0], y[:k], 1)[0])
    hi = float(np.polyfit(t[-k:] - t[-1], y[-k:], 1)[0])
    return lo, hi


def _extend(
    centers: np.ndarray, smoothed: np.ndarray, accepted: np.ndarray, slopes: tuple[float, float]
) -> np.ndarray:
    """The fitted curve beyond the measured span, continued with the stored end slopes."""
    idx = np.flatnonzero(accepted)
    if idx.size == 0:
        return smoothed
    out = smoothed.copy()
    t = centers.astype(np.float64) / US_PER_S
    a, b = idx[0], idx[-1]
    before, after = t < t[a], t > t[b]
    out[before] = smoothed[a] + slopes[0] * (t[before] - t[a])
    out[after] = smoothed[b] + slopes[1] * (t[after] - t[b])
    return out


def _fit_offset(
    centers: np.ndarray,
    offsets: np.ndarray,
    confid: np.ndarray,
    accepted: np.ndarray,
    scatter_s: float,
    smoothing: float | None,
) -> np.ndarray:
    """Weighted smoothing spline through accepted windows, evaluated at every centre.

    The smoothing budget is set from the **measured** per-window scatter. Weighting each point
    by ``1/sigma`` makes ``UnivariateSpline``'s objective a chi-square, so the natural budget is
    ``s = n`` — one unit of misfit per point. The previous default asked the spline to pass
    within 0.3 *samples* (≈1 ms) of every window, which on real data (where a single 20 s window
    locates the lag to ~0.1 s) forces the curve to chase the noise: it swung 0.56 s on a
    recording whose true drift is 0.15 s. Drift is smooth; the fit should say so.

    Evaluation is **clamped** to the accepted windows' span. ``UnivariateSpline`` extrapolates
    cubically, which diverges fast: beyond the last accepted window a cubic will happily invent
    a multi-second offset. Outside the measured span the caller continues the curve with the
    fitted end slopes (:func:`_end_offset_slopes`, as the map does) and marks that region
    extrapolated rather than pretending it was measured (REVIEW B5).
    """
    t0 = int(centers[0])
    x_all = (centers - t0).astype(np.float64) / US_PER_S
    xa = x_all[accepted]
    ya = offsets[accepted]

    n = xa.shape[0]
    k = min(3, n - 1)
    # w = sqrt(confidence)/sigma: confident windows pull harder, and dividing by the scatter
    # puts the objective in chi-square units so `s = n` is the principled budget.
    wa = np.sqrt(np.clip(confid[accepted], 1e-3, 1.0)) / scatter_s
    s = smoothing if smoothing is not None else float(n)

    spline = UnivariateSpline(xa, ya, w=wa, k=k, s=s)
    x_eval = np.clip(x_all, xa[0], xa[-1])
    return np.asarray(spline(x_eval), dtype=np.float64)
