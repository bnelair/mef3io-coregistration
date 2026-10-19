"""Generic channels and montages + reader fixes.

Nothing may depend on the two files sharing channel names or montage: channels are resolved
per file, a unipolar channel on one file can be aligned against a bipolar pair on the other,
and a channel present only in the moving file can be read coregistered.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest
from _synth import _START, _write_mef

from mef3io_coregistration.config import US_PER_S
from mef3io_coregistration.io import MefChannelReader
from mef3io_coregistration.montage import ChannelSpec
from mef3io_coregistration.reader import CoregisteredReader
from mef3io_coregistration.session import CoregResult, CoregSession
from mef3io_coregistration.synthetic.generate import render_pair
from mef3io_coregistration.synthetic.warp import WarpSpec
from mef3io_coregistration.transform import ClockMap

_DRIFT = WarpSpec("linear", offset0_s=120.0, rate_s_per_s=5.0 / (12 * 3600))


def _named_pair(tmp_path, duration_s=900.0, fs_ref=1024.0, fs_mov=256.0, seed=5):
    """Reference file: ``shared`` (the warped signal) + ``ref_ground`` (a zero channel).
    Moving file: differently named ``mov_x`` = signal + common-mode ``cm``, ``mov_cm`` = ``cm``
    (so bipolar ``mov_x-mov_cm`` recovers the signal), and a moving-only ``clinical``."""
    rp = render_pair(_DRIFT, duration_s, fs_ref, fs_mov, _START, seed=seed)
    rng = np.random.default_rng(99)
    cm = np.cumsum(rng.standard_normal(rp.mov.shape[0])) * 5.0  # large wandering common mode
    ref_p, mov_p = tmp_path / "ref.mefd", tmp_path / "mov.mefd"
    _write_mef(ref_p, {"shared": rp.ref, "ref_ground": np.zeros_like(rp.ref)}, fs_ref, rp.ref_start_uutc)
    _write_mef(
        mov_p,
        {"mov_x": rp.mov + cm, "mov_cm": cm, "clinical": 2.0 * rp.mov},
        fs_mov,
        rp.mov_start_uutc,
    )
    return str(ref_p), str(mov_p), rp


def _gt_map(duration_s=900.0):
    x = _START + np.linspace(0, duration_s, 50) * US_PER_S
    return ClockMap.from_reference_offset(x.astype(np.int64), _DRIFT.offset_s(np.linspace(0, duration_s, 50)))


# --------------------------------------------------------------------------- #
# ChannelSpec parsing
# --------------------------------------------------------------------------- #
def test_channel_spec_parse_forms():
    chans = ["LA1", "LA2", "EEG Fp1-Ref", "Fp2-Ref"]
    assert ChannelSpec.parse("LA1", chans) == ChannelSpec("LA1")
    assert ChannelSpec.parse("LA1-LA2", chans) == ChannelSpec("LA1", "LA2")
    assert ChannelSpec.parse("EEG Fp1-Ref", chans) == ChannelSpec("EEG Fp1-Ref")  # '-' in name
    assert ChannelSpec.parse("EEG Fp1-Ref-Fp2-Ref", chans) == ChannelSpec("EEG Fp1-Ref", "Fp2-Ref")
    assert ChannelSpec.parse(("LA1", None)) == ChannelSpec("LA1")
    assert ChannelSpec.parse({"anode": "LA1", "cathode": "LA2"}) == ChannelSpec("LA1", "LA2")
    assert ChannelSpec.parse("A-B") == ChannelSpec("A", "B")  # no channel list: one '-' splits
    with pytest.raises(KeyError):
        ChannelSpec.parse("nope", chans)


# --------------------------------------------------------------------------- #
# IO: off-grid, out-of-range bounds (review fix)
# --------------------------------------------------------------------------- #
def test_read_bounds_are_half_open_on_grid_and_tile(write_mef):
    fs = 100.0
    path = write_mef("g.mefd", {"c": [(np.arange(1000.0), _START, fs)]})
    with MefChannelReader(path) as r:
        full = r.read("c")
        # off-grid bounds: samples with t in [start, stop)
        w = r.read("c", _START + 4_000, _START + 1_005_000)
        assert w.data[0] == pytest.approx(1.0) and w.data[-1] == pytest.approx(100.0)
        assert w.ts[0] == _START + 10_000 and w.n == 100
        # consecutive half-open reads tile exactly
        cuts = [_START + k for k in (0, 123_456, 555_555, 10_000_000)]
        parts = [r.read("c", a, b).data for a, b in itertools.pairwise(cuts)]
        np.testing.assert_array_equal(np.concatenate(parts), full.data)
        # before the recording: NaN padding on the grid, then data
        pre = r.read("c", _START - 1_000_000, _START + 1_000_000)
        assert pre.n == 200 and np.isnan(pre.data[:100]).all() and pre.data[100] == pytest.approx(0.0)
        assert pre.ts[0] == _START - 1_000_000
        with pytest.raises(KeyError, match="available"):
            r.fs("missing")


# --------------------------------------------------------------------------- #
# Reader: moving-only channels, differing names/montages, tiling, roles
# --------------------------------------------------------------------------- #
def test_moving_only_channel_reads_coregistered(tmp_path):
    ref_p, mov_p, _rp = _named_pair(tmp_path)
    with CoregisteredReader(_gt_map(), ref_p, mov_p, reference="A") as reader:
        t0, t1 = _START + 200 * US_PER_S, _START + 260 * US_PER_S
        ref = reader.get_data("shared", t0, t1, file="A")
        clin = reader.get_data("clinical", t0, t1, file="B", coregister=True)
        assert clin.fs == pytest.approx(256.0)  # native moving rate by default
        assert clin.ts[0] >= t0 and clin.ts[-1] < t1
        # overlay on the reference grid: clinical == 2 × the shared signal
        on_ref = reader.get_data("clinical", file="B", coregister=True, grid=ref.ts)
        core = slice(200, -200)
        np.testing.assert_allclose(on_ref.data[core], 2.0 * ref.data[core], atol=0.05 * np.std(ref.data) * 2)


def test_bipolar_on_one_file_vs_unipolar_on_other(tmp_path):
    ref_p, mov_p, rp = _named_pair(tmp_path)
    with CoregisteredReader(_gt_map(), ref_p, mov_p, reference="A") as reader:
        t0, t1 = _START + 300 * US_PER_S, _START + 330 * US_PER_S
        ref = reader.get_data("shared", t0, t1, file="A")
        bip = reader.get_data("mov_x-mov_cm", file="B", coregister=True, grid=ref.ts)
        assert bip.channel == "mov_x-mov_cm"
        core = slice(200, -200)
        assert np.corrcoef(bip.data[core], ref.data[core])[0, 1] > 0.999
        raw = reader.get_data(("mov_x", "mov_cm"), file="B", coregister=False)
        np.testing.assert_allclose(raw.data, rp.mov[: raw.n], atol=1e-3)


def test_consecutive_coregistered_reads_tile(tmp_path):
    ref_p, mov_p, _rp = _named_pair(tmp_path)
    with CoregisteredReader(_gt_map(), ref_p, mov_p, reference="A") as reader:
        cuts = [_START + int(s * US_PER_S) for s in (100.0, 131.3, 170.0)]
        whole = reader.get_data("clinical", cuts[0], cuts[-1], file="B")
        parts = [reader.get_data("clinical", a, b, file="B") for a, b in itertools.pairwise(cuts)]
        np.testing.assert_array_equal(np.concatenate([p.ts for p in parts]), whole.ts)
        np.testing.assert_allclose(np.concatenate([p.data for p in parts]), whole.data, atol=1e-9)
        assert np.all(np.diff(whole.ts) > 0)


def test_reader_from_result_takes_roles_from_result(tmp_path):
    """Reference = file B: the reader built from the saved result applies the map the right
    way round (the old bare-map constructor defaulted to A)."""
    ref_p, mov_p, _rp = _named_pair(tmp_path)
    res = CoregResult(_gt_map(), reference_file=ref_p, moving_file=mov_p, reference_role="B")
    assert res.file_a == mov_p and res.file_b == ref_p
    path = res.save(tmp_path / "res.json")
    with CoregisteredReader.from_result(path) as reader:
        assert reader.reference_file == "B" and reader.moving_file == "A"
        t0, t1 = _START + 300 * US_PER_S, _START + 330 * US_PER_S
        ref = reader.get_data("shared", t0, t1, file="B")
        mov = reader.get_data("clinical", file="A", coregister=True, grid=ref.ts)
        assert np.corrcoef(mov.data[100:-100], ref.data[100:-100])[0, 1] > 0.999


def test_v1_result_files_still_load(tmp_path):
    res = CoregResult(_gt_map(), reference_file="r", moving_file="m")
    d = res.to_dict()
    d["format_version"] = 1
    for k in ("reference_role", "file_a", "file_b", "stage_filters", "qc"):
        d.pop(k)
    old = CoregResult.from_dict(d)
    assert old.reference_role == "A" and old.file_a == "r"


# --------------------------------------------------------------------------- #
# Session: differing names and montages per file (alignment)
# --------------------------------------------------------------------------- #
def test_session_aligns_unipolar_against_bipolar_with_other_names(tmp_path):
    ref_p, mov_p, _rp = _named_pair(tmp_path, duration_s=900.0)
    sess = CoregSession(background=False, chunk_s=120.0).open(ref_p, mov_p, reference="A")
    sess.set_channels("gross", reference=("shared", None), moving=("clinical", None))
    sess.set_channels("fine", reference=("shared", None), moving=("mov_x", "mov_cm"))
    gross = sess.run_gross()
    assert gross.locked and abs(gross.offset_s - float(_DRIFT.offset_s(450.0))) <= 1.0
    anchor = sess.run_fine(_START + 450 * US_PER_S)
    assert abs(anchor.offset_s - float(_DRIFT.offset_s(450.0))) * 256.0 <= 1.0
    sess.run_fine(_START + 450 * US_PER_S)  # a re-run at the same ROI replaces the anchor
    assert len(sess.anchors) == 1
    drift = sess.run_drift()
    t_rel = (drift.t_ref_uutc - _START) / US_PER_S
    ok = drift.accepted
    err = (drift.smoothed_offset_s[ok] - _DRIFT.offset_s(t_rel[ok])) * 256.0
    assert np.sqrt(np.mean(err**2)) < 1.0
    res = sess.result()
    assert res.stage_channels["fine"] == {"reference": "shared", "moving": "mov_x-mov_cm"}
    assert res.reference_role == "A"
    rec = sess.open_recording()
    assert {"shared", "clinical", "mov_x"} <= set(rec.channels)
    rec.close()
