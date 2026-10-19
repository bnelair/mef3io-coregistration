"""Tests for bipolar derivations.

bipolar from two channels in one file (with Fs harmonization); per-use channel
selection is independent; differing references handled (subtraction is well-defined).
"""
from __future__ import annotations

import numpy as np
import pytest

from mef3io_coregistration.config import US_PER_S
from mef3io_coregistration.io import MefChannelReader
from mef3io_coregistration.io.derived import decimate_bandlimited
from mef3io_coregistration.montage import bipolar

_EPOCH = 1_700_000_000 * US_PER_S


def _three_channel_file(write_mef, fs=512.0, n=4096):
    rng = np.random.default_rng(0)
    a = rng.standard_normal(n) * 30
    b = rng.standard_normal(n) * 30
    c = rng.standard_normal(n) * 30
    path = write_mef(
        "montage.mefd",
        {"a": [(a, _EPOCH, fs)], "b": [(b, _EPOCH, fs)], "c": [(c, _EPOCH, fs)]},
    )
    return path


def test_bipolar_equals_difference(write_mef):
    """bipolar(a, b) == channel a − channel b sample-for-sample (same Fs)."""
    path = _three_channel_file(write_mef)
    with MefChannelReader(path) as reader:
        a = reader.read("a")
        b = reader.read("b")
        bip = bipolar(a, b)

    assert bip.fs == pytest.approx(a.fs)
    assert bip.anode == "a" and bip.cathode == "b"
    np.testing.assert_allclose(bip.data, a.data - b.data)
    np.testing.assert_array_equal(bip.ts, a.ts)


def test_bipolar_selection_is_independent(write_mef):
    """choosing a different channel pair (as a later stage may) yields a different
    derivation from the same file — selection is not tied to any single global choice."""
    path = _three_channel_file(write_mef)
    with MefChannelReader(path) as reader:
        a, b, c = reader.read("a"), reader.read("b"), reader.read("c")
        ab = bipolar(a, b)
        ac = bipolar(a, c)

    np.testing.assert_allclose(ab.data, a.data - b.data)
    np.testing.assert_allclose(ac.data, a.data - c.data)
    assert not np.allclose(ab.data, ac.data)


def test_bipolar_harmonizes_differing_fs(write_mef):
    """channels at different Fs are harmonized (higher decimated to lower) before
    subtraction; the result is on the lower-Fs grid and equals the difference there."""
    n_hi, fs_hi, fs_lo = 8192, 2048.0, 512.0
    rng = np.random.default_rng(1)
    hi = rng.standard_normal(n_hi) * 20
    # Band-limit the low channel below both Nyquists so decimation is a clean reference.
    lo_full = decimate_bandlimited(rng.standard_normal(n_hi) * 20, fs_hi, fs_lo).data
    path = write_mef(
        "mixed.mefd",
        {"hi": [(hi, _EPOCH, fs_hi)], "lo": [(lo_full, _EPOCH, fs_lo)]},
    )
    with MefChannelReader(path) as reader:
        hi_rd = reader.read("hi")
        lo_rd = reader.read("lo")
        bip = bipolar(hi_rd, lo_rd)

    assert bip.fs == pytest.approx(fs_lo)
    expected_hi = decimate_bandlimited(hi_rd.data, fs_hi, fs_lo).data
    n = min(expected_hi.shape[0], lo_rd.data.shape[0], bip.data.shape[0])
    np.testing.assert_allclose(bip.data[:n], expected_hi[:n] - lo_rd.data[:n])
    assert np.all(np.abs(np.diff(bip.ts) - US_PER_S / fs_lo) <= 1)
