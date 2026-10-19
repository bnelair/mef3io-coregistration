"""Tests for derived signals.

min-max envelope preserves each bin's extremes; band-limited decimation is
phase-accurate (within 1 sample at the target Fs); min-max is explicitly not
time-accurate within a bin, decimation is.
"""
from __future__ import annotations

import numpy as np
import pytest

from mef3io_coregistration.io.derived import decimate_bandlimited, minmax_envelope


# --------------------------------------------------------------------------- #
# min-max envelope
# --------------------------------------------------------------------------- #
def test_envelope_preserves_bin_extremes():
    """Each envelope bin holds the true min and max of the source samples in that bin."""
    fs = 2000.0
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20_000)

    env = minmax_envelope(data, fs, envelope_fs=50.0)

    bin_size = int(round(fs / 50.0))
    assert env.bin_size == bin_size
    assert env.fs == pytest.approx(fs / bin_size)

    n_bins = data.shape[0] // bin_size
    binned = data[: n_bins * bin_size].reshape(n_bins, bin_size)
    np.testing.assert_array_equal(env.mins, binned.min(axis=1))
    np.testing.assert_array_equal(env.maxs, binned.max(axis=1))


def test_envelope_captures_sharp_spike():
    """A one-sample spike survives into the bin's max (the display/gross use case)."""
    fs = 1000.0
    data = np.zeros(10_000)
    data[3_333] = 42.0
    env = minmax_envelope(data, fs, envelope_fs=50.0)
    assert env.maxs.max() == pytest.approx(42.0)


def test_envelope_all_nan_bin_is_nan():
    """A bin covering only gap (NaN) samples yields NaN; partial-NaN bins ignore the NaN."""
    fs = 100.0
    data = np.arange(400, dtype=float)
    data[:2] = np.nan  # first bin (samples 0..1) all NaN at envelope_fs=50 -> bin_size 2
    env = minmax_envelope(data, fs, envelope_fs=50.0)
    assert np.isnan(env.mins[0]) and np.isnan(env.maxs[0])
    assert not np.isnan(env.mins[1])


# --------------------------------------------------------------------------- #
# band-limited decimation (phase-accurate)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("f0", [7.0, 37.0, 101.0])
def test_decimation_preserves_phase(f0):
    """Decimating a sinusoid preserves phase within 1 sample at the target Fs."""
    fs, target_fs = 2048.0, 512.0
    n = int(20 * fs)
    phase = 0.7
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * f0 * t + phase)

    dec = decimate_bandlimited(x, fs, target_fs)
    assert dec.fs == pytest.approx(target_fs)

    tt = np.arange(dec.data.shape[0]) / dec.fs
    ideal = np.sin(2 * np.pi * f0 * tt + phase)
    core = slice(int(0.1 * dec.data.shape[0]), int(0.9 * dec.data.shape[0]))
    # max amplitude error << 1; equivalently the timing error is < 1 target sample.
    assert np.max(np.abs(dec.data[core] - ideal[core])) < 0.05

    lag = np.argmax(
        np.correlate(dec.data[core] - dec.data[core].mean(), ideal[core] - ideal[core].mean(), "full")
    ) - (dec.data[core].shape[0] - 1)
    assert abs(lag) <= 1  # within one target-Fs sample


def test_decimation_reduces_rate():
    fs, target_fs = 2000.0, 500.0
    x = np.random.default_rng(0).standard_normal(10_000)
    dec = decimate_bandlimited(x, fs, target_fs)
    assert dec.fs == pytest.approx(target_fs)
    assert dec.data.shape[0] == pytest.approx(x.shape[0] * target_fs / fs, abs=2)


def test_decimation_rejects_upsampling():
    with pytest.raises(ValueError):
        decimate_bandlimited(np.zeros(100), 512.0, 2048.0)


def test_decimation_identity_when_equal():
    x = np.arange(10, dtype=float)
    dec = decimate_bandlimited(x, 512.0, 512.0)
    np.testing.assert_array_equal(dec.data, x)
    assert dec.fs == pytest.approx(512.0)
