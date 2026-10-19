"""Tests for time-delay primitives (shared by Stages 1, 2 and 3).

Sign convention (``b[n] ≈ a[n - d]`` → ``d``), sub-sample accuracy, confidence, bounded
search, and — the property that drove the design — robustness of normalized cross-correlation
to within-window dilation where GCC-PHAT track-jumps.
"""
from __future__ import annotations

import numpy as np
import pytest

from mef3io_coregistration.align.tde import gcc_phat, normxcorr_delay, parabolic_peak

FS = 512.0


def _signal(n=4096, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    # a couple of sharp transients give a distinct correlation peak
    x[n // 3] += 8.0
    x[2 * n // 3] -= 6.0
    return x


@pytest.mark.parametrize("estimator", [gcc_phat, normxcorr_delay])
@pytest.mark.parametrize("shift", [7, -5, 0])
def test_integer_shift_sign_and_confidence(estimator, shift):
    x = _signal()
    d = estimator(x, np.roll(x, shift), FS)
    assert d.lag_samples == pytest.approx(shift, abs=0.05)
    assert d.lag_s == pytest.approx(shift / FS, abs=0.05 / FS)
    assert d.ncc > 0.99  # a pure shift is a perfect match
    assert d.confidence > 0.5  # ...and the peak stands clear of its rivals


def test_confidence_is_distinctiveness_not_match_height():
    """`confidence` must answer "did I lock on?", not "how similar are these?".

    A signal correlated against itself plus heavy independent noise still has ONE unambiguous
    peak, so it must stay confident even though the raw NCC has collapsed. The old metric was
    the raw NCC, which falls as windows lengthen and as amplifiers differ — so thresholding on
    it rejected precisely the estimates that were most reliable.
    """
    x = _signal(seed=7)
    rng = np.random.default_rng(8)
    b = np.roll(x, 5) + rng.standard_normal(x.size) * np.std(x) * 3.0

    d = normxcorr_delay(x, b, FS)
    assert d.lag_samples == pytest.approx(5, abs=0.5)
    assert d.ncc < 0.5  # raw similarity is poor...
    assert d.confidence > 0.5  # ...but the lock is unambiguous, and that is what we report


@pytest.mark.parametrize("estimator", [gcc_phat, normxcorr_delay])
def test_subsample_accuracy(estimator):
    x = _signal(seed=1)
    t = np.arange(x.size)
    b = np.interp(t - 3.25, t, x)  # fractional delay of +3.25 samples
    d = estimator(x, b, FS)
    assert d.lag_samples == pytest.approx(3.25, abs=0.2)


def test_confidence_low_for_unrelated_signals():
    a, b = _signal(seed=2), _signal(seed=3)
    assert gcc_phat(a, b, FS).confidence < 0.3
    assert normxcorr_delay(a, b, FS).confidence < 0.3


def test_bounded_search_ignores_far_peak():
    x = _signal(seed=4)
    b = np.roll(x, 40)
    # true lag 40 is outside a ±10-sample search, so it must not be returned
    d = normxcorr_delay(x, b, FS, max_lag_s=10 / FS)
    assert abs(d.lag_samples) <= 10


def test_normxcorr_handles_within_window_dilation():
    """Under a mild symmetric stretch (the within-window time-dilation that drift induces),
    normalized cross-correlation stays sub-sample and is at least as accurate as PHAT — whose
    high-frequency emphasis degrades faster. This is why the fine/drift stages use it."""
    x = _signal(n=8192, seed=5)
    t = np.arange(x.size)
    c = x.size // 2
    b = np.interp((t - c) / 1.0005 + c, t, x)  # symmetric 0.05 % stretch, best lag = 0
    core = slice(c - 3000, c + 3000)
    err_norm = abs(normxcorr_delay(x[core], b[core], FS).lag_samples)
    err_phat = abs(gcc_phat(x[core], b[core], FS).lag_samples)
    assert err_norm < 1.0
    assert err_norm <= err_phat


def test_parabolic_peak():
    assert parabolic_peak(np.array([0.0, 1.0, 0.0]), 1) == pytest.approx(0.0)
    # peak skewed towards the right neighbour -> positive sub-sample offset
    assert parabolic_peak(np.array([0.0, 1.0, 0.6]), 1) > 0.0
    assert parabolic_peak(np.array([1.0, 2.0, 3.0]), 0) == 0.0  # edge -> no refinement
