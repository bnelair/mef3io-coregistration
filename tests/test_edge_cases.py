"""Edge cases found in review: clipped windows at recording starts, repeated anchors at one
ROI, gap masking on the decimating resample path, and half-open bounds at rounding ties."""
from __future__ import annotations

import numpy as np

from mef3io_coregistration.align import estimate_anchor
from mef3io_coregistration.config import US_PER_S
from mef3io_coregistration.io.derived import Decimated
from mef3io_coregistration.transform.resample import evaluate_at, grid_indices, uniform_grid

_T0 = 1_700_000_000 * US_PER_S


def _shifted_pair(offset_s=3.0, mov_late_s=5.0, fs=200.0, dur=120.0):
    rng = np.random.default_rng(0)
    x = np.convolve(rng.standard_normal(int((dur + 20) * fs)), np.ones(5) / 5, "same")
    ref = Decimated(x[: int(dur * fs)], fs, _T0)
    # moving clock = reference + offset; the moving recording starts `mov_late_s` later
    k = int(mov_late_s * fs)
    mov = Decimated(x[k : k + int(dur * fs)], fs, _T0 + int((offset_s + mov_late_s) * US_PER_S))
    return ref, mov


def test_anchor_near_moving_start_is_not_biased():
    """A window clipped by the start of one recording is trimmed on both sides together."""
    ref, mov = _shifted_pair()
    for roi in (10.0, 7.0, 6.0):
        a = estimate_anchor(ref, mov, _T0 + int(roi * US_PER_S), 3.0, half_window_s=5.0)
        assert abs(a.offset_s - 3.0) < 1e-3 and a.confidence > 0.5, (roi, a)  # distinctiveness


def test_decimating_resample_masks_the_filter_support_around_gaps():
    fs = 1000.0
    t = np.arange(5000) / fs
    x = np.sin(2 * np.pi * 3 * t)
    x[2000:2500] = np.nan
    q = np.arange(0, 5000, 5) * (US_PER_S / fs)
    y = evaluate_at(x, 0.0, fs, q, out_fs=100.0)
    truth = np.sin(2 * np.pi * 3 * q / US_PER_S)
    ok = np.isfinite(y)
    assert np.max(np.abs(y[ok] - truth[ok])) < 5e-3  # nothing mean-filled leaks through
    assert np.isnan(y[(q / 1000 >= 1995) & (q / 1000 < 2505)]).all()


def test_half_open_bounds_at_rounding_ties():
    """fs = 32 kHz: grid times are 31.25 µs apart, so rint hits .5 ties (half-to-even)."""
    for start in range(0, 400):
        k0, k1 = grid_indices(0, 32000.0, start, start + 1000)
        ts = uniform_grid(0, 32000.0, k0, k1 - k0)
        assert ts[0] >= start and uniform_grid(0, 32000.0, k0 - 1, 1)[0] < start
        assert ts[-1] < start + 1000
