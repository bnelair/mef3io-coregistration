"""Band-limited evaluation of a uniformly sampled signal at arbitrary times.

Under ``coregister=True`` the moving sample *values* change, not just their timestamps: each
output time on the reference clock is mapped to a moving-clock time ``forward(t_ref)`` and the
moving signal is evaluated there. The mapped times are non-uniform (drift) and the output rate
may differ from the source rate, so evaluation is band-limited:

1. if the output rate is lower than the source rate, the source is first anti-alias decimated
   to the output rate (polyphase FIR, zero group delay);
2. the result is oversampled ×4 (polyphase interpolation);
3. it is evaluated at the requested times with 4-point cubic (Lagrange) interpolation, which on
   a ×4-oversampled band-limited trace is accurate to ~1e-3 of full scale.

``NaN`` gap samples are never interpolated across: they are held at the mean while filtering
and every output sample whose filter support touches a gap (or falls outside the data) is
``NaN``.
The reference recording is never passed through here unless the caller asks for a different
output rate.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

from ..config import US_PER_S
from .clock_map import ClockMap

_OVERSAMPLE = 4
# Kaiser β for the ×4 interpolation filter: ~5e-5 in-band error up to ~0.8 Nyquist (the
# scipy default β=5 leaves ~2e-3 ripple).
_OVERSAMPLE_WINDOW = ("kaiser", 8.0)
# Half-support of the ×4 interpolation (10 base samples) + the cubic's own sample.
_INTERP_SUPPORT = 11


def support_samples(fs: float, out_fs: float | None = None, max_denominator: int = 10_000) -> int:
    """Half-support, in source samples, of the filters :func:`evaluate_at` applies — how much
    source data beyond the requested span a read must fetch so no output near its ends is lost."""
    support = _INTERP_SUPPORT
    if out_fs is not None and out_fs < fs:
        up, down = _ratio(out_fs / fs, max_denominator)
        support = int(np.ceil(10 * max(up, down) / up)) + int(np.ceil(_INTERP_SUPPORT * fs / (fs * up / down)))
    return support


def evaluate_at(
    data: np.ndarray,
    start_uutc: float,
    fs: float,
    t_uutc: np.ndarray,
    out_fs: float | None = None,
    max_denominator: int = 10_000,
) -> np.ndarray:
    """Evaluate the uniformly sampled ``data`` (first sample at ``start_uutc``, rate ``fs``) at
    times ``t_uutc`` (same clock, float µs OK), band-limited to ``out_fs`` (default ``fs``).

    For µs-exact positions pass both times *relative* to a nearby origin (absolute uUTC in
    float64 only resolves ~0.25 µs, and an integer-rounded start is off by up to 0.5 µs).

    Times outside the data or next to a gap give ``NaN`` (never invented).
    """
    x = np.asarray(data, dtype=np.float64)
    t = np.asarray(t_uutc, dtype=np.float64)
    out = np.full(t.shape, np.nan)
    if x.shape[0] < 4 or t.size == 0:
        return out

    gap = np.isnan(x)
    mean = float(np.nanmean(x)) if not gap.all() else 0.0
    x = np.where(gap, mean, x)

    base, base_fs = x, float(fs)
    # FIR half-support in *source* samples (resample_poly: 10·max(up, down) taps per side at
    # the upsampled rate). Outputs this close to a gap (mean-filled) are not trustworthy.
    support = support_samples(fs, out_fs, max_denominator)
    if out_fs is not None and out_fs < fs:
        up, down = _ratio(out_fs / fs, max_denominator)
        base = resample_poly(x, up, down)
        base_fs = fs * up / down
    over = resample_poly(base, _OVERSAMPLE, 1, window=_OVERSAMPLE_WINDOW)
    over_fs = base_fs * _OVERSAMPLE
    last = (x.shape[0] - 1) * (over_fs / fs)  # index of the final source sample in ``over``

    pos = (t - start_uutc) * (over_fs / US_PER_S)  # fractional index into ``over``
    ok = (pos >= -1e-9) & (pos <= last + 1e-9)
    if np.any(ok):
        padded = np.pad(over, 2, mode="edge")  # 4-point support at the two ends
        p = np.clip(pos[ok], 0.0, last) + 2
        i1 = np.minimum(np.floor(p).astype(np.int64), padded.shape[0] - 3)
        out[ok] = _lagrange4(padded, i1, p - i1)

    decimating = out_fs is not None and out_fs < fs
    if decimating:
        # The anti-alias filter sees zero padding past the data ends: treat them like gaps.
        edge = min(support, gap.shape[0])
        gap = gap.copy()
        gap[:edge] = True
        gap[gap.shape[0] - edge :] = True
    if gap.any():
        # Any output whose filter support (in source samples) touches a gap → NaN.
        near = np.convolve(gap.astype(np.int32), np.ones(2 * support + 1, dtype=np.int32), "same") > 0
        src = (t - start_uutc) * (fs / US_PER_S)
        j = np.clip(np.rint(src).astype(np.int64), 0, gap.shape[0] - 1)
        out[near[j]] = np.nan
    return out


def resample_onto_grid(
    moving_data: np.ndarray,
    moving_start_uutc: int,
    moving_fs: float,
    clock_map: ClockMap,
    grid_uutc: np.ndarray,
    out_fs: float,
    max_denominator: int = 10_000,
) -> np.ndarray:
    """Evaluate the moving signal at reference-clock times ``grid_uutc`` via the clock map."""
    t_moving = np.asarray(clock_map.forward(np.asarray(grid_uutc)), dtype=np.float64)
    return evaluate_at(moving_data, moving_start_uutc, moving_fs, t_moving, out_fs=out_fs,
                       max_denominator=max_denominator)


def resample_onto_reference(
    moving_data: np.ndarray,
    moving_start_uutc: int,
    moving_fs: float,
    clock_map: ClockMap,
    ref_start_uutc: int,
    ref_fs: float,
    n: int,
    max_denominator: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the moving signal on the uniform reference grid ``ref_start + k/ref_fs``.

    Returns ``(values, ref_grid_uutc)`` with ``values`` float64 and ``ref_grid_uutc`` int64.
    """
    ref_grid = uniform_grid(ref_start_uutc, ref_fs, 0, n)
    values = resample_onto_grid(
        moving_data, moving_start_uutc, moving_fs, clock_map, ref_grid, out_fs=ref_fs,
        max_denominator=max_denominator,
    )
    return values, ref_grid


def uniform_grid(origin_uutc: int, fs: float, k0: int, n: int) -> np.ndarray:
    """``origin + round(k/fs)`` µs for ``k = k0 .. k0+n-1`` (integer epoch kept exact)."""
    k = k0 + np.arange(n, dtype=np.int64)
    return int(origin_uutc) + np.rint(k * (US_PER_S / fs)).astype(np.int64)


def grid_indices(origin_uutc: int, fs: float, start: int, stop: int) -> tuple[int, int]:
    """Index range ``[k0, k1)`` of grid points (as :func:`uniform_grid`) inside ``[start, stop)``."""
    k0 = first_index_at_or_after(origin_uutc, US_PER_S / fs, int(start))
    k1 = first_index_at_or_after(origin_uutc, US_PER_S / fs, int(stop))
    return k0, max(k0, k1)


def first_index_at_or_after(origin_uutc: int, period_us: float, t: int) -> int:
    """Smallest ``k`` whose grid time ``origin + rint(k·period)`` is ``>= t``.

    Uses exactly the rounding the timestamps are built with (``np.rint``, half-to-even), so a
    half-open ``[start, stop)`` selection agrees with the reported timestamps even at ties.
    """
    k = int(np.floor((t - origin_uutc) / period_us)) - 1
    while origin_uutc + int(np.rint(k * period_us)) < t:
        k += 1
    return k


def _ratio(r: float, max_denominator: int) -> tuple[int, int]:
    f = Fraction(r).limit_denominator(max_denominator)
    return f.numerator, f.denominator


def _lagrange4(y: np.ndarray, i1: np.ndarray, frac: np.ndarray) -> np.ndarray:
    """4-point cubic Lagrange interpolation between ``y[i1]`` and ``y[i1+1]`` at ``frac``."""
    ym1, y0, y1, y2 = y[i1 - 1], y[i1], y[i1 + 1], y[i1 + 2]
    d = frac
    return (
        -d * (d - 1) * (d - 2) / 6 * ym1
        + (d + 1) * (d - 1) * (d - 2) / 2 * y0
        - (d + 1) * d * (d - 2) / 2 * y1
        + (d + 1) * d * (d - 1) / 6 * y2
    )
