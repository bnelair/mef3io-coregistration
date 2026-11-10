"""Tests for the MEF3 IO wrapper.

Covers: read == the legacy mef_tools API (mef3io.compat.MefReader); per-channel Fs (incl. differing rates); gaps surfaced
as NaN and reported, never silently filled.
"""
from __future__ import annotations

import numpy as np
import pytest
from mef3io.compat import MefReader

from mef3io_coregistration.config import US_PER_S
from mef3io_coregistration.io import MefChannelReader
from mef3io_coregistration.synthetic import build_or_load

_EPOCH = 1_700_000_000 * US_PER_S


# --------------------------------------------------------------------------- #
# read matches the legacy mef_tools API (mef3io.compat)
# --------------------------------------------------------------------------- #
def test_full_read_matches_legacy_mef_tools_api(shared_data_dir):
    """A full-channel read returns byte-identical samples to mef3io.compat.MefReader (the mef_tools API), with timestamps
    reconstructed on the channel's nominal grid."""
    fx = build_or_load("short_constant", data_dir=shared_data_dir)

    with MefChannelReader(fx.reference_path) as reader:
        rd = reader.read("ch1")

    ref = MefReader(str(fx.reference_path))
    expected = np.asarray(ref.get_data("ch1"), dtype=np.float64)
    start = int(ref.get_property("start_time", "ch1"))
    fs = float(ref.get_property("fsamp", "ch1"))
    ref.close()

    np.testing.assert_array_equal(rd.data, expected)
    assert rd.fs == pytest.approx(fs)
    assert rd.start_uutc == start
    assert rd.ts[0] == start
    assert rd.ts.shape == rd.data.shape
    assert np.all(np.diff(rd.ts) > 0)
    # per-sample step is one sample period (rounded to whole µs)
    assert np.all(np.abs(np.diff(rd.ts) - US_PER_S / fs) <= 1)


def test_windowed_read_grid_aligned(shared_data_dir):
    """A grid-aligned windowed read returns the exact sample count and slice of the full
    read, with the requested start timestamp."""
    fx = build_or_load("short_constant", data_dir=shared_data_dir)

    with MefChannelReader(fx.reference_path) as reader:
        fs = reader.fs("ch1")
        start, _end = reader.time_bounds("ch1")
        full = reader.read("ch1")

        k1, k2 = 1000, 5000
        t1 = int(round(start + k1 * US_PER_S / fs))
        t2 = int(round(start + k2 * US_PER_S / fs))
        win = reader.read("ch1", t1, t2)

    assert win.n == k2 - k1
    assert win.start_uutc == t1
    np.testing.assert_array_equal(win.data, full.data[k1:k2])
    np.testing.assert_array_equal(win.ts, full.ts[k1:k2])


# --------------------------------------------------------------------------- #
# per-channel sampling rate (differing rates)
# --------------------------------------------------------------------------- #
def test_per_channel_fs_differs(write_mef):
    """Two channels in one file may carry different sampling rates."""
    n_hi, n_lo = 4096, 1024
    path = write_mef(
        "mixed.mefd",
        {
            "hi": [(np.random.default_rng(0).standard_normal(n_hi) * 10, _EPOCH, 2048.0)],
            "lo": [(np.random.default_rng(1).standard_normal(n_lo) * 10, _EPOCH, 512.0)],
        },
    )
    with MefChannelReader(path) as reader:
        assert set(reader.channels) == {"hi", "lo"}
        assert reader.fs("hi") == pytest.approx(2048.0)
        assert reader.fs("lo") == pytest.approx(512.0)
        assert reader.read("hi").n == n_hi
        assert reader.read("lo").n == n_lo


# --------------------------------------------------------------------------- #
# gaps surfaced, never silently filled
# --------------------------------------------------------------------------- #
def test_gap_is_reported_and_not_filled(write_mef):
    """A 2 s discontinuity between two 5 s segments is reported by ``gaps`` and appears as
    NaN (not interpolated/zero-filled) in the read data."""
    fs = 512.0
    n_seg = int(5 * fs)
    seg1 = np.sin(2 * np.pi * 3 * np.arange(n_seg) / fs) * 50
    seg2 = np.cos(2 * np.pi * 3 * np.arange(n_seg) / fs) * 50
    start2 = int(_EPOCH + 7 * US_PER_S)  # 5 s data + 2 s gap
    path = write_mef("gapped.mefd", {"ch1": [(seg1, _EPOCH, fs), (seg2, start2, fs)]})

    with MefChannelReader(path) as reader:
        gaps = reader.gaps("ch1")
        rd = reader.read("ch1")

    assert len(gaps) == 1
    gap = gaps[0]
    assert gap.start_uutc == int(_EPOCH + 5 * US_PER_S)
    assert gap.end_uutc == start2
    assert gap.n_missing == int(2 * fs)

    # The read spans the full extent (both segments + the gap) with NaN across the gap only.
    assert rd.n == int(12 * fs)
    assert np.isnan(rd.data).sum() == int(2 * fs)
    gap_mask = (rd.ts >= gap.start_uutc) & (rd.ts < gap.end_uutc)
    assert np.all(np.isnan(rd.data[gap_mask]))
    assert not np.any(np.isnan(rd.data[~gap_mask]))
    assert rd.gaps == gaps


def test_continuous_channel_has_no_gaps(shared_data_dir):
    """A contiguous recording reports no gaps and contains no NaN."""
    fx = build_or_load("short_constant", data_dir=shared_data_dir)
    with MefChannelReader(fx.reference_path) as reader:
        rd = reader.read("ch1")
        assert reader.gaps("ch1") == ()
    assert not np.any(np.isnan(rd.data))


# --------------------------------------------------------------------------- #
# channel extent comes from the block index, not the .tmet header times (#5)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def _one_channel(tmp_path_factory):
    import mef3io
    path = tmp_path_factory.mktemp("span") / "s.mefd"
    n = 100 * 35  # 35 s at 100 Hz: several blocks
    with mef3io.Writer(str(path), overwrite=True) as w:
        w.write("x", np.sin(np.arange(n) / 20.0), _EPOCH, 100.0, precision=3)
    return {"a": path, "b": path}, n


@pytest.mark.parametrize("offset,fn", [(24, lambda e: e - 10 * US_PER_S), (16, lambda s: s + 20 * US_PER_S)])
def test_a_wrong_header_time_does_not_shorten_the_channel(_one_channel, tmp_path, offset, fn):
    from _synth import patched_world
    world, n = _one_channel
    a, _ = patched_world(world, tmp_path, "A", "x", offset, fn)
    with MefChannelReader(a) as r:
        assert r.info("x")["start_time_header"] != r.info("x")["start_time"] or \
            r.info("x")["end_time_header"] != r.info("x")["end_time"]  # the header really is off
        t0, t1 = r.time_bounds("x")
        assert (t0, t1) == (_EPOCH, _EPOCH + 35 * US_PER_S)
        rd = r.read("x", t0, t1)
    assert rd.n == n and not np.isnan(rd.data).any()
