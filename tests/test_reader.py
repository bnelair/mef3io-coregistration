"""Tests for the runtime reader and resampler.

Covers: raw pulls byte-identical to the legacy mef_tools API (mef3io.compat) from both files; coregistered moving
resampled onto the reference grid; reference never transformed; reference-clock
query domain under coregistration. Validated end-to-end against the synthetic-warp harness.
"""
from __future__ import annotations

import numpy as np
import pytest
from mef3io.compat import MefReader

from mef3io_coregistration.config import US_PER_S
from mef3io_coregistration.reader import CoregisteredReader
from mef3io_coregistration.transform import ClockMap
from mef3io_coregistration.transform.resample import resample_onto_reference


def _ground_truth_map(fx, n_anchors=200):
    ref = fx.load_reference()
    anchors = np.linspace(ref.ts[0], ref.ts[-1], n_anchors).astype(np.int64)
    offset = fx.offset_s((anchors - ref.start_uutc) / US_PER_S)
    return ClockMap.from_reference_offset(anchors, offset)


def _reader(fx):
    # reference.mefd is the reference (role "A"); moving.mefd is moving (role "B").
    return CoregisteredReader(_ground_truth_map(fx), fx.reference_path, fx.moving_path, reference="A")


# --------------------------------------------------------------------------- #
# raw pulls from both files match the legacy mef_tools API
# --------------------------------------------------------------------------- #
def test_raw_pull_matches_legacy_api_both_files(shared_data_dir):
    from mef3io_coregistration.synthetic import build_or_load

    fx = build_or_load("short_linear", data_dir=shared_data_dir)
    with _reader(fx) as reader:
        raw_ref = reader.get_data("ch1", file="A", coregister=False)
        raw_mov = reader.get_data("ch1", file="B", coregister=False)

    for path, raw in ((fx.reference_path, raw_ref), (fx.moving_path, raw_mov)):
        mr = MefReader(str(path))
        expected = np.asarray(mr.get_data("ch1"), dtype=np.float64)
        mr.close()
        np.testing.assert_array_equal(raw.data, expected)


# --------------------------------------------------------------------------- #
# reference is never transformed
# --------------------------------------------------------------------------- #
def test_reference_never_transformed(shared_data_dir):
    """Reading the reference file returns native samples whether coregister is on or off."""
    from mef3io_coregistration.synthetic import build_or_load

    fx = build_or_load("short_linear", data_dir=shared_data_dir)
    with _reader(fx) as reader:
        native = reader.get_data("ch1", file="A", coregister=False)
        coreg = reader.get_data("ch1", file="A", coregister=True)
    np.testing.assert_array_equal(native.data, coreg.data)
    np.testing.assert_array_equal(native.ts, coreg.ts)


# --------------------------------------------------------------------------- #
# coregistered moving onto the reference grid
# --------------------------------------------------------------------------- #
def test_coregistered_moving_reconstructs_reference(shared_data_dir):
    """The moving signal read coregistered reproduces the reference signal (they share the
    same source), on the reference grid, in the reference-clock domain."""
    from mef3io_coregistration.synthetic import build_or_load

    fx = build_or_load("drift_swing", data_dir=shared_data_dir)
    ref = fx.load_reference()
    t0 = int(ref.ts[0] + 120 * US_PER_S)
    t1 = int(ref.ts[0] + 180 * US_PER_S)

    with _reader(fx) as reader:
        coreg = reader.get_data("ch1", t0, t1, file="B", coregister=True)
        ref_win = reader.get_data("ch1", t0, t1, file="A", coregister=False)

    assert coreg.fs == pytest.approx(fx.fs_ref)  # placed on the reference grid
    np.testing.assert_array_equal(coreg.ts, ref_win.ts)  # reference-clock timestamps

    core = slice(int(0.05 * coreg.n), int(0.95 * coreg.n))
    a, b = coreg.data[core], ref_win.data[core]
    ok = ~np.isnan(a)
    assert np.corrcoef(a[ok], b[ok])[0, 1] > 0.99


def test_coregistered_read_of_a_moving_only_channel(tmp_path):
    """A coregistered read must work for channels that exist ONLY in the moving file — which is
    the tool's whole purpose.

    Typically a clinical file carries many channels a second recorder does not, and much of the
    analysis lives on them. An earlier reader built its output grid by reading the requested
    channel from the *reference* file, so every such channel raised
    `TypeError: float() argument must be ... not 'NoneType'`, naming neither channel nor file.
    """
    from _synth import write_pair

    from mef3io_coregistration.synthetic.warp import WarpSpec

    spec = WarpSpec("linear", offset0_s=120.0, rate_s_per_s=5.0 / (12 * 3600))
    pair = write_pair(tmp_path, spec, channels=("eeg",), moving_only=("clinical_only",))

    ref_start = pair.start_uutc
    anchors = np.linspace(ref_start, ref_start + int(1100 * US_PER_S), 200).astype(np.int64)
    cmap = ClockMap.from_reference_offset(
        anchors, spec.offset_s((anchors - ref_start) / US_PER_S)
    )

    with CoregisteredReader(cmap, pair.reference_path, pair.moving_path, reference="A") as r:
        t0 = ref_start + 300 * US_PER_S
        t1 = ref_start + 360 * US_PER_S

        coreg = r.get_data("clinical_only", t0, t1, file="B", coregister=True)
        assert coreg.n > 0
        assert coreg.fs == pytest.approx(256.0)  # the channel's OWN rate, not the reference's
        assert coreg.ts[0] >= t0 and coreg.ts[-1] <= t1  # reference-clock domain

        # it carries the same content as the shared channel, read the same way
        shared = r.get_data("eeg", t0, t1, file="B", coregister=True)
        n = min(coreg.n, shared.n)
        a = np.interp(
            np.linspace(0, 1, n), np.linspace(0, 1, coreg.n), coreg.data
        )
        b = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, shared.n), shared.data)
        ok = ~np.isnan(a) & ~np.isnan(b)
        assert np.corrcoef(a[ok], b[ok])[0, 1] > 0.99

        # a channel in neither file fails with a message that names it
        with pytest.raises(KeyError, match="nonexistent"):
            r.get_data("nonexistent", t0, t1, file="B", coregister=True)


def test_coregistered_moving_timing_within_one_lower_fs_sample(shared_data_dir):
    """The coregistered moving signal aligns with the reference to within 1 lower-Fs sample
    (the acceptance bar for placing stimulation pulses)."""
    from mef3io_coregistration.synthetic import build_or_load

    fx = build_or_load("drift_swing", data_dir=shared_data_dir)
    ref = fx.load_reference()
    t0 = int(ref.ts[0] + 120 * US_PER_S)
    t1 = int(ref.ts[0] + 180 * US_PER_S)

    with _reader(fx) as reader:
        coreg = reader.get_data("ch1", t0, t1, file="B", coregister=True)
        ref_win = reader.get_data("ch1", t0, t1, file="A", coregister=False)

    core = slice(int(0.1 * coreg.n), int(0.9 * coreg.n))
    a = coreg.data[core]
    b = ref_win.data[core]
    ok = ~np.isnan(a)
    a = a[ok] - np.mean(a[ok])
    b = b[ok] - np.mean(b[ok])
    lag = int(np.argmax(np.correlate(a, b, "full"))) - (a.shape[0] - 1)
    # lag is in reference samples; 1 lower-Fs sample == fs_ref / fs_moving reference samples.
    assert abs(lag) <= fx.fs_ref / fx.fs_moving


def test_from_map_file_roundtrip(shared_data_dir, tmp_path):
    from mef3io_coregistration.synthetic import build_or_load

    fx = build_or_load("short_linear", data_dir=shared_data_dir)
    map_path = _ground_truth_map(fx).save(tmp_path / "map.json")
    with CoregisteredReader.from_map_file(
        map_path, fx.reference_path, fx.moving_path, reference="A"
    ) as reader:
        raw = reader.get_data("ch1", file="A", coregister=False)
    assert raw.n > 0


# --------------------------------------------------------------------------- #
# resample_onto_reference — unit behaviour
# --------------------------------------------------------------------------- #
def test_resample_identity_map_same_fs():
    """An identity map at matching Fs returns the moving samples on the reference grid.

    The reference grid is integer-µs while the moving samples sit on exact 1/fs fractions, so
    identity resampling carries only a sub-µs interpolation error (< 1e-4 here), not a shift.
    """
    fs = 512.0
    start = 1_700_000_000 * US_PER_S
    n = 2000
    data = np.sin(2 * np.pi * 7 * np.arange(n + 16) / fs)  # a few extra samples cover the grid
    cm = ClockMap.from_reference_offset(
        np.array([start, start + int((n + 15) / fs * US_PER_S)], dtype=np.int64), 0.0
    )
    values, grid = resample_onto_reference(data, start, fs, cm, start, fs, n)
    assert grid[0] == start
    assert not np.any(np.isnan(values))
    np.testing.assert_allclose(values, data[:n], atol=1e-4)


def test_resample_nan_outside_moving_span():
    """Grid samples mapping outside the provided moving data come back as NaN, not invented."""
    fs = 512.0
    start = 1_700_000_000 * US_PER_S
    data = np.ones(500)
    cm = ClockMap.from_reference_offset(
        np.array([start, start + int(500 / fs * US_PER_S)], dtype=np.int64), 0.0
    )
    # Request more samples than the moving data covers.
    values, _ = resample_onto_reference(data, start, fs, cm, start, fs, 800)
    assert not np.any(np.isnan(values[:400]))
    assert np.all(np.isnan(values[600:]))
