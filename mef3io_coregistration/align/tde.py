"""Time-delay estimation primitives.

Cross-correlation with sub-sample peak refinement, shared by the gross (Stage 1), fine
(Stage 2) and drift (Stage 3) stages. FFT-based and batchable.

Sign convention: :func:`gcc_phat` / :func:`normxcorr_delay` return the lag ``d`` (in samples,
``a``→``b``) such that ``b[n] ≈ a[n - d]`` — i.e. ``d > 0`` means ``b`` is delayed relative to
``a``. With ``a = reference`` and ``b = moving`` this is the local offset ``t_moving - t_ref``
in samples.

**Confidence is peak distinctiveness, not correlation height**. The raw normalized
cross-correlation of two windows *falls* as the window lengthens, because two different
amplifiers accumulate more independent content — even though the lag estimate simultaneously
gets much more reliable. Thresholding on raw NCC therefore rejects precisely the windows whose
estimates are best. What "did I lock on?" actually asks is whether the correlation surface has
*one* peak that stands clear of its rivals, so that is what we measure:

    peak_ratio  = best / (best runner-up outside a guard band around the best)
    confidence  = 1 - 1/peak_ratio        ∈ [0, 1), monotone in distinctiveness

That number is high exactly when the lock is unambiguous, is comparable across window lengths,
and is near zero on a flat correlation surface — where the old metric happily returned 0.34.
The raw NCC is still reported as :attr:`Delay.ncc`, for information only.

Inputs must be finite. Gap (``NaN``) samples are the caller's problem: condition the trace
(which zeroes gaps and records a validity mask) and reject over-gappy windows *before*
estimating, rather than letting a single NaN turn a lag into ``NaN`` and a JSON body into
something a front end cannot parse.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.signal import correlate

from mef3io_coregistration.config import PEAK_GUARD_S


@dataclass(frozen=True)
class Delay:
    """A time-delay estimate.

    ``confidence`` is peak distinctiveness in [0, 1] (see module docstring) and is the number
    to threshold on. ``ncc`` is the raw normalized cross-correlation at the chosen lag, kept
    for display. ``peak_ratio`` is the underlying best/runner-up ratio (>= 1).
    """

    lag_samples: float
    lag_s: float
    confidence: float
    ncc: float = 0.0
    peak_ratio: float = 1.0
    at_edge: bool = False  # bounded search peaked on its boundary: true peak lies outside


def parabolic_peak(y: np.ndarray, k: int) -> float:
    """Sub-sample offset of a peak at integer index ``k`` from a 3-point parabola fit."""
    if 0 < k < len(y) - 1:
        denom = y[k - 1] - 2.0 * y[k] + y[k + 1]
        if denom != 0.0:
            return 0.5 * (y[k - 1] - y[k + 1]) / denom
    return 0.0


def peak_quality(surface: np.ndarray, k: int, guard_samples: int) -> tuple[float, float]:
    """Distinctiveness of the peak at ``k``: ``(confidence, peak_ratio)``.

    The runner-up is the largest value at least ``guard_samples`` away from ``k`` — the guard
    band excludes the peak's own shoulders, which would otherwise always be the runner-up and
    make every peak look indistinct.

    The guard is clamped to a quarter of the surface. Stages 2 and 3 search a narrow seeded
    range (±0.25 s), which is *narrower than the default guard*: without the clamp the guard
    swallows the whole surface, no runner-up exists, and every window — including the good ones
    — reports confidence 0.
    """
    n = surface.shape[0]
    best = float(surface[k])
    if n < 3 or best <= 0.0:
        return 0.0, 1.0

    guard = max(1, min(int(guard_samples), (n - 1) // 4))
    lo, hi = max(0, k - guard), min(n, k + guard + 1)
    outside = np.concatenate((surface[:lo], surface[hi:]))
    if outside.size == 0:
        return 0.0, 1.0

    runner_up = float(outside.max())
    if runner_up <= 0.0:
        return 1.0, float("inf")
    ratio = best / runner_up
    if ratio <= 1.0:
        return 0.0, float(ratio)
    return float(1.0 - 1.0 / ratio), float(ratio)


def peak_snr(surface: np.ndarray, k: int) -> float:
    """Height of the peak at ``k`` in units of the correlation surface's own robust sigma.

    This is matched-filter detection SNR. Over a long search (24 h of iEEG is ~20 M lags) the
    *best rival* peak is always close to the true peak — iEEG is self-similar, and somewhere in
    a day of recording there is always another stretch that resembles this one. So
    :func:`peak_quality`'s best/runner-up ratio, which is the right statistic inside a short
    seeded window, is far too harsh full-length: it reports ~0 even for a correct lock.

    The peak's height against the *distribution* of the surface is the statistic that does
    discriminate: a true match stands tens of sigma above the background, while the background's
    own maximum sits around 5–7 sigma.
    """
    n = surface.shape[0]
    if n < 3:
        return 0.0
    med = float(np.median(surface))
    mad = float(np.median(np.abs(surface - med)))
    sigma = 1.4826 * mad
    std = float(np.std(surface))

    # The MAD is robust but degenerate on a spiky or piled-up distribution: if a large fraction
    # of the surface holds the same value, the MAD collapses towards zero and every point becomes
    # "thousands of sigma above background". That is not a hypothetical — clipping the NCC
    # surface at zero once made half of it exactly 0, and this function duly reported 3158σ for
    # an offset that was an hour wrong. Fall back to the standard deviation whenever the robust
    # estimate is implausibly small next to it.
    if sigma <= 0.1 * std:
        sigma = std
    if sigma <= 0.0:
        return 0.0
    return float((float(surface[k]) - med) / sigma)


def gcc_phat(
    a: np.ndarray,
    b: np.ndarray,
    fs: float,
    max_lag_s: float | None = None,
    guard_s: float = PEAK_GUARD_S,
) -> Delay:
    """Estimate the delay of ``b`` relative to ``a`` via GCC-PHAT.

    The phase transform whitens the cross-spectrum, sharpening the peak. Kept for
    completeness and for narrow, seeded searches on band-limited traces; Stages 1–3 use
    :func:`normxcorr_delay`, which does not whiten and is therefore robust to the
    within-window time dilation that drift induces.
    """
    a = _finite(a, "a")
    b = _finite(b, "b")
    a = a - a.mean()
    b = b - b.mean()

    n = a.shape[0] + b.shape[0]
    nfft = next_fast_len(n)
    # R = B·conj(A) so that b[n]=a[n-d] (b delayed by d) peaks at positive lag +d.
    R = rfft(b, nfft) * np.conj(rfft(a, nfft))
    R /= np.abs(R) + 1e-12  # phase transform (whiten)
    cc = irfft(R, nfft)

    half = nfft // 2
    max_shift = half if max_lag_s is None else min(half, int(round(max_lag_s * fs)))
    # Reorder to lags [-max_shift .. +max_shift]; negative lags live at the array tail.
    window = np.concatenate((cc[nfft - max_shift :], cc[: max_shift + 1]))
    k = int(np.argmax(window))
    lag = (k - max_shift) + parabolic_peak(window, k)
    conf, ratio = peak_quality(window, k, int(round(guard_s * fs)))

    return Delay(
        lag_samples=lag,
        lag_s=lag / fs,
        confidence=conf,
        ncc=_ncc_at_lag(a, b, int(round(lag))),
        peak_ratio=ratio,
    )


def normxcorr_delay(
    a: np.ndarray,
    b: np.ndarray,
    fs: float,
    max_lag_s: float | None = None,
    guard_s: float = PEAK_GUARD_S,
) -> Delay:
    """Estimate the delay of ``b`` relative to ``a`` via normalized cross-correlation.

    Same sign convention as :func:`gcc_phat` (``b[n] ≈ a[n - d]`` → ``d``). This is the
    estimator of choice for Stages 2 and 3 on band-limited, conditioned traces of equal
    length.
    """
    a = _finite(a, "a")
    b = _finite(b, "b")
    a = a - a.mean()
    b = b - b.mean()

    corr = correlate(b, a, mode="full", method="fft")
    mid = a.shape[0] - 1  # zero-lag index (equal-length inputs)
    if max_lag_s is None:
        seg, origin = corr, mid
    else:
        m = min(mid, int(round(max_lag_s * fs)))
        seg, origin = corr[mid - m : mid + m + 1], m

    k = int(np.argmax(seg))
    lag = (k - origin) + parabolic_peak(seg, k)
    conf, ratio = peak_quality(seg, k, int(round(guard_s * fs)))
    # A bounded search whose maximum sits on its boundary has not found a peak — the curve is
    # still rising there. Its "distinctiveness" can nonetheless be high, which is how a seed
    # that is off by more than the search range used to be accepted as a confident window.
    at_edge = max_lag_s is not None and (k == 0 or k == seg.shape[0] - 1)
    return Delay(
        lag_samples=lag,
        lag_s=lag / fs,
        confidence=0.0 if at_edge else conf,
        ncc=_ncc_at_lag(a, b, int(round(lag))),
        peak_ratio=ratio,
        at_edge=at_edge,
    )


@dataclass(frozen=True)
class Surface:
    """A correlation surface over a lag window — the evidence behind a single :class:`Delay`.

    ``lags_samples`` and ``values`` are the correlation-vs-lag curve the estimator's argmax runs
    on; ``values`` is max-normalised to [0, 1] for display. ``best_lag_samples`` is the sub-sample
    peak and ``confidence``/``peak_ratio`` are computed on the *raw* surface, so they match exactly
    what :func:`normxcorr_delay` would report on the same window.
    """

    lags_samples: np.ndarray
    values: np.ndarray
    best_lag_samples: float
    confidence: float
    peak_ratio: float
    ncc: float
    fs: float


def normxcorr_surface(
    a: np.ndarray,
    b: np.ndarray,
    fs: float,
    max_lag_s: float | None = None,
    guard_s: float = PEAK_GUARD_S,
) -> Surface:
    """The full normalized cross-correlation surface of ``b`` vs ``a`` over ±``max_lag_s``.

    Same math and sign convention as :func:`normxcorr_delay`, but returns the whole curve rather
    than only its argmax — so a front end can show *why* a lag was chosen, and whether the peak is one
    clear winner or one of several equal bumps.
    """
    a = _finite(a, "a")
    b = _finite(b, "b")
    a = a - a.mean()
    b = b - b.mean()

    corr = correlate(b, a, mode="full", method="fft")
    mid = a.shape[0] - 1
    if max_lag_s is None:
        seg, origin = corr, mid
    else:
        m = min(mid, int(round(max_lag_s * fs)))
        seg, origin = corr[mid - m : mid + m + 1], m

    k = int(np.argmax(seg))
    lag = (k - origin) + parabolic_peak(seg, k)
    conf, ratio = peak_quality(seg, k, int(round(guard_s * fs)))
    lags = np.arange(seg.shape[0], dtype=np.float64) - origin

    peak = float(seg.max()) if seg.size else 0.0
    values = seg / peak if peak > 0.0 else np.zeros_like(seg)
    return Surface(
        lags_samples=lags,
        values=values,
        best_lag_samples=lag,
        confidence=conf,
        peak_ratio=ratio,
        ncc=_ncc_at_lag(a, b, int(round(lag))),
        fs=fs,
    )


def _finite(x: np.ndarray, name: str) -> np.ndarray:
    """Reject non-finite input rather than quietly returning a NaN lag."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.size and not np.isfinite(arr).all():
        raise ValueError(
            f"time-delay estimation got non-finite samples in {name!r}; condition the trace "
            "and reject gap-overlapping windows before estimating"
        )
    return arr


def _ncc_at_lag(a: np.ndarray, b: np.ndarray, d: int) -> float:
    """Normalized cross-correlation of ``a`` and ``b`` aligned so ``b[n] ~ a[n - d]``."""
    n = min(a.shape[0], b.shape[0])
    if d >= 0:
        x, y = a[: n - d], b[d:n]
    else:
        x, y = a[-d:n], b[: n + d]
    if x.shape[0] < 2:
        return 0.0
    xn, yn = np.linalg.norm(x), np.linalg.norm(y)
    if xn == 0.0 or yn == 0.0:
        return 0.0
    return float(np.clip(np.dot(x, y) / (xn * yn), 0.0, 1.0))
