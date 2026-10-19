"""Derived-signal products.

Two products per channel, both pure functions of ``(data, fs)`` so the core stays
signal-agnostic:

* :func:`minmax_envelope` — ~50 Hz min/max envelope for display.
  Preserves each bin's visual extremes but is **not** time-accurate within a bin:
  the min and max are placed at the bin boundary, not their true sub-bin positions.
* :func:`decimate_bandlimited` — anti-aliased, group-delay-compensated decimation to a
  target Fs, where sample-level timing matters.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

from ..config import ENVELOPE_FS_HZ


@dataclass(frozen=True)
class Envelope:
    """A ~``fs`` Hz min/max envelope. NOT time-accurate within a bin.

    ``mins``/``maxs`` hold one value per bin. ``fs`` is the envelope (bin) rate, ``bin_size``
    the number of source samples per bin. A bin covering only gap (``NaN``) samples yields
    ``NaN`` for both traces.
    """

    mins: np.ndarray
    maxs: np.ndarray
    fs: float
    bin_size: int
    start_uutc: int | None = None


@dataclass(frozen=True)
class Decimated:
    """A band-limited decimated trace. Time-accurate (group-delay compensated)."""

    data: np.ndarray
    fs: float
    start_uutc: int | None = None


def envelope_bin_size(fs: float, envelope_fs: float = ENVELOPE_FS_HZ) -> int:
    """Source samples per envelope bin: ``round(fs / envelope_fs)``, but at least 2 — a
    one-sample bin has ``min == max`` and carries no activity (low-rate recordings get a
    coarser envelope instead of an empty one)."""
    return max(2, int(round(fs / envelope_fs)))


def minmax_envelope(
    data: np.ndarray,
    fs: float,
    envelope_fs: float = ENVELOPE_FS_HZ,
    start_uutc: int | None = None,
) -> Envelope:
    """Bin ``data`` into ~``1/envelope_fs`` s bins and take each bin's min and max.

    The bin size is ``round(fs / envelope_fs)`` source samples (≥ 2, see
    :func:`envelope_bin_size`). Trailing samples that
    do not fill a whole bin are dropped. ``NaN`` gap samples are ignored within a bin; an
    all-``NaN`` bin yields ``NaN`` (gaps are surfaced, never filled).
    """
    data = np.asarray(data, dtype=np.float64)
    bin_size = envelope_bin_size(fs, envelope_fs)
    n_bins = data.shape[0] // bin_size
    if n_bins == 0:
        empty = np.empty(0, dtype=np.float64)
        return Envelope(empty, empty.copy(), fs / bin_size, bin_size, start_uutc)

    binned = data[: n_bins * bin_size].reshape(n_bins, bin_size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN bins -> NaN, expected
        mins = np.nanmin(binned, axis=1)
        maxs = np.nanmax(binned, axis=1)
    return Envelope(mins, maxs, fs / bin_size, bin_size, start_uutc)


def decimate_bandlimited(
    data: np.ndarray,
    fs: float,
    target_fs: float,
    start_uutc: int | None = None,
    max_denominator: int = 10_000,
) -> Decimated:
    """Anti-aliased decimation of ``data`` from ``fs`` to ~``target_fs``.

    Uses polyphase resampling (``scipy.signal.resample_poly``), a linear-phase FIR whose
    group delay is compensated, so output samples stay time-aligned to the input start —
    phase is preserved (a sinusoid is reproduced within a fraction of a target-Fs sample).
    ``target_fs`` must not exceed ``fs`` (this is decimation, not interpolation).
    """
    data = np.asarray(data, dtype=np.float64)
    if target_fs > fs:
        raise ValueError(f"target_fs ({target_fs}) > fs ({fs}); this function only decimates")
    if target_fs == fs:
        return Decimated(data.copy(), fs, start_uutc)

    ratio = Fraction(target_fs / fs).limit_denominator(max_denominator)
    up, down = ratio.numerator, ratio.denominator
    out = resample_poly(data, up, down)
    return Decimated(out, fs * up / down, start_uutc)
