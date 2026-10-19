"""The unified recording: two coregistered files used as one.

Logical channels map to a signal in file A and/or B (any names, any montage) with a preferred
source; reads serve the preferred file wherever it has data and fall back to the other file
elsewhere, on the reference clock, with per-sample source and extrapolation flags.
"""
from __future__ import annotations

import numpy as np
import pytest
from _synth import _START
from mef3io.compat import MefWriter

from mef3io_coregistration.config import US_PER_S
from mef3io_coregistration.reader import CoregisteredRecording, LogicalChannel
from mef3io_coregistration.session import CoregResult
from mef3io_coregistration.synthetic.generate import render_pair
from mef3io_coregistration.synthetic.warp import WarpSpec
from mef3io_coregistration.transform import ClockMap

_DRIFT = WarpSpec("linear", offset0_s=120.0, rate_s_per_s=5.0 / (12 * 3600))
_FS_A, _FS_B = 1024.0, 256.0
_DUR = 600.0
# The moving file B covers only reference-clock [150, 450) s, with a gap at [280, 300) s.
_B_SPANS = ((150.0, 280.0), (300.0, 450.0))


def _write(path, chans, fs, segments):
    w = MefWriter(str(path), overwrite=True)
    for name, data in chans.items():
        for i0, i1, t0 in segments:
            w.write_data(np.asarray(data[i0:i1], dtype=np.float64), name, int(t0), float(fs), precision=5)
    del w


@pytest.fixture(scope="module")
def rec_files(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("unified")
    rp = render_pair(_DRIFT, _DUR, _FS_A, _FS_B, _START, seed=11)
    a_p, b_p = tmp / "A.mefd", tmp / "B.mefd"
    _write(a_p, {"LA1": rp.ref, "LA2": 0.5 * rp.ref, "Fz": rp.ref[::-1].copy()}, _FS_A, [(0, rp.ref.shape[0], rp.ref_start_uutc)])
    segs = []
    for s0, s1 in _B_SPANS:  # reference-clock seconds → moving sample indices
        m0 = s0 + float(_DRIFT.offset_s(s0)) - float(_DRIFT.offset_s(0.0))
        m1 = s1 + float(_DRIFT.offset_s(s1)) - float(_DRIFT.offset_s(0.0))
        i0, i1 = int(np.ceil(m0 * _FS_B)), int(np.floor(m1 * _FS_B))
        segs.append((i0, i1, rp.mov_start_uutc + round(i0 * US_PER_S / _FS_B)))
    cm = np.random.default_rng(1).standard_normal(rp.mov.shape[0]).cumsum()
    _write(b_p, {"EEG LA1-Ref": rp.mov + cm, "CM": cm, "STIM": 3.0 * rp.mov}, _FS_B, segs)

    x = _START + (np.linspace(0, _DUR, 60) * US_PER_S).astype(np.int64)
    cmap = ClockMap.from_reference_offset(x, _DRIFT.offset_s(np.linspace(0, _DUR, 60)))
    res = CoregResult(cmap, reference_file=str(a_p), moving_file=str(b_p), reference_role="A")
    res_path = res.save(tmp / "result.json")
    return res_path, rp


def _t(s):
    return _START + int(s * US_PER_S)


def test_auto_channels_cover_both_files(rec_files):
    res_path, _ = rec_files
    with CoregisteredRecording.open(res_path) as rec:
        # "LA1" and "EEG LA1-Ref" differ only by a prefix/suffix: linked as one logical channel
        assert set(rec.channels) == {"LA1", "LA2", "Fz", "CM", "STIM"}
        assert rec.channel("LA1").b.label == "EEG LA1-Ref"
        assert rec.channel("STIM").a is None and rec.channel("STIM").b is not None
        assert rec.default_prefer == "A"  # the reference by default


def test_prefer_a_falls_back_to_b_where_a_has_no_data(rec_files):
    """Link differing names + montages: A's unipolar LA1 ≡ B's bipolar 'EEG LA1-Ref'-'CM'.
    Preferring B: B where B has data, A in B's gap and outside B's span."""
    res_path, rp = rec_files
    with CoregisteredRecording.open(res_path) as rec:
        rec.set_channel("LA1", a="LA1", b="EEG LA1-Ref-CM", prefer="B")
        sig = rec.get_data("LA1", _t(100.0), _t(500.0))
        assert sig.fs == pytest.approx(_FS_B)  # the preferred source's grid
        t = (sig.ts - _START) / US_PER_S
        in_b = ((t > 151) & (t < 279)) | ((t > 301) & (t < 449))
        out_b = (t < 149) | ((t > 281) & (t < 299)) | (t > 451)
        assert np.all(sig.source[in_b] == 2)
        assert np.all(sig.source[out_b] == 1)
        assert not np.isnan(sig.data).any()  # A fills every hole
        # values: whichever file served a sample, it is the same physical signal
        truth = rec.get_data("LA1", _t(100.0), _t(500.0), fs=_FS_B)  # same grid
        ref = np.interp(sig.ts, _START + np.arange(rp.ref.shape[0]) * US_PER_S / _FS_A, rp.ref)
        core = (t > 105) & (t < 495)
        err = np.abs(sig.data[core] - ref[core])
        assert np.percentile(err, 99) < 0.05 * np.std(rp.ref)
        assert truth.n == sig.n


def test_prefer_reference_returns_native_samples(rec_files):
    res_path, rp = rec_files
    with CoregisteredRecording.open(res_path) as rec:
        rec.set_channel("LA1", a="LA1", b="EEG LA1-Ref-CM", prefer="A")
        sig = rec.get_data("LA1", _t(200.0), _t(210.0))
        assert np.all(sig.source == 1)
        i0 = int(round(200.0 * _FS_A))
        np.testing.assert_allclose(sig.data, rp.ref[i0 : i0 + sig.n], atol=1e-4)  # MEF precision
        assert not sig.extrapolated.any()


def test_moving_only_channel_and_nan_outside(rec_files):
    res_path, _ = rec_files
    with CoregisteredRecording.open(res_path) as rec:
        sig = rec.get_data("STIM", _t(100.0), _t(200.0))
        t = (sig.ts - _START) / US_PER_S
        assert np.all(np.isnan(sig.data[t < 149])) and np.all(sig.source[t < 149] == 0)
        assert np.all(sig.source[(t > 151)] == 2)
        lo, _hi = rec.time_bounds("STIM")
        assert abs((lo - _START) / US_PER_S - 150.0) < 0.1


def test_extrapolated_flag_outside_map(rec_files):
    """Samples from the moving file outside the measured map span are flagged."""
    res_path, _ = rec_files
    res = CoregResult.load(res_path)
    x = res.clock_map.x_ref
    short = ClockMap(x[20:40], res.clock_map.y_moving[20:40])  # map measured over ~[200, 400] s
    res.clock_map = short
    with CoregisteredRecording.open(res) as rec:
        sig = rec.get_data("STIM", _t(160.0), _t(440.0))
        t = (sig.ts - _START) / US_PER_S
        lo_s, hi_s = ((np.array(short.domain) - _START) / US_PER_S)
        assert sig.extrapolated[(t < lo_s - 1) & (sig.source == 2)].all()
        assert not sig.extrapolated[(t > lo_s + 1) & (t < hi_s - 1)].any()


def test_config_roundtrip_and_multi_channel_read(rec_files, tmp_path):
    res_path, _ = rec_files
    with CoregisteredRecording.open(res_path, default_prefer="B") as rec:
        rec.set_channel("brain", a=("LA1", "LA2"), b="EEG LA1-Ref-CM", prefer="A", fs=512.0)
        rec.remove_channel("Fz")
        cfg = rec.save_config(tmp_path / "rec.json", result_path=res_path)
        first = rec.get_data(["brain", "STIM"], _t(300.0), _t(305.0))
    with CoregisteredRecording.from_config(cfg) as again:
        assert again.channels == rec.channels
        assert again.channel("brain") == LogicalChannel.from_dict(rec.channel("brain").to_dict())
        assert again.get_property("fsamp", "brain") == 512.0
        second = again.get_data(["brain", "STIM"], _t(300.0), _t(305.0))
    for x, y in zip(first, second, strict=False):
        np.testing.assert_array_equal(x.ts, y.ts)
        np.testing.assert_array_equal(x.data, y.data)


def test_invalid_mappings_raise(rec_files):
    res_path, _ = rec_files
    with CoregisteredRecording.open(res_path) as rec:
        with pytest.raises(KeyError):
            rec.set_channel("x", a="not-a-channel")
        with pytest.raises(ValueError):
            rec.set_channel("x")
        with pytest.raises(KeyError):
            rec.get_data("nope")


def test_string_specs_resolve_against_each_file(rec_files):
    """Names containing '-' (e.g. 'EEG LA1-Ref') are channels, not bipolar pairs, when given
    as strings in a mapping (API / config path)."""
    res_path, _ = rec_files
    chans = [
        {"name": "x", "a": "LA1", "b": "EEG LA1-Ref", "prefer": "B"},
        {"name": "y", "a": None, "b": "EEG LA1-Ref-CM", "prefer": "B"},
    ]
    with CoregisteredRecording.open(res_path, channels=chans) as rec:
        assert rec.channel("x").b.label == "EEG LA1-Ref" and not rec.channel("x").b.is_bipolar
        assert rec.channel("y").b.cathode == "CM"


def test_display_buckets_follow_preference_zoomed_out_and_in(rec_files):
    """Viewer buckets (R10.3): zoomed out from cached envelopes, zoomed in from exact data —
    both honour the preferred source with fallback, and never exceed ``max_points``."""
    res_path, _ = rec_files
    with CoregisteredRecording.open(res_path) as rec:
        rec.set_channel("LA1", a="LA1", b="EEG LA1-Ref-CM", prefer="B")
        for start, stop, n in ((_t(0.0), _t(600.0), 600), (_t(270.0), _t(310.0), 400)):
            d = rec.display("LA1", start, stop, max_points=n)
            assert len(d["mins"]) == n and np.all(d["mins"][~np.isnan(d["mins"])] <= d["maxs"][~np.isnan(d["maxs"])])
            t = (d["start_uutc"] + (np.arange(n) + 0.5) * 1e6 / d["fs"] - _START) / US_PER_S
            assert np.all(d["source"][(t > 160) & (t < 270)] == 2)   # B preferred where present
            assert np.all(d["source"][(t > 285) & (t < 295)] == 1)   # A fills B's gap
            if stop - start > 100 * US_PER_S:
                assert np.all(d["source"][t < 140] == 1)             # and before B starts


# --------------------------------------------------------------------------- #
# Merge safety (clinical review): a logical channel may only fall back from one file to the
# other when the two sources agree over their overlap — sign, gain, units — or the user says so.
# --------------------------------------------------------------------------- #
_GOOD_B = ("EEG LA1-Ref", "CM")  # EEG LA1-Ref − CM is the same signal as LA1


def _gap_filled(rec, name):
    sig = rec.get_data(name, _t(270.0), _t(310.0))  # spans B's gap at [280, 300)
    t = (sig.ts - _START) / US_PER_S
    return sig, bool(np.all(sig.source[(t > 282) & (t < 298)] == 1))


def test_agreeing_sources_merge(rec_files):
    res_path, _ = rec_files
    with CoregisteredRecording.open(res_path) as rec:
        rec.set_channel("LA1", a="LA1", b=_GOOD_B, prefer="B")
        ag = rec.agreement("LA1")
        assert ag.status == "ok" and ag.corr > 0.9 and 0.5 < ag.gain_b_over_a < 2.0
        sig, filled = _gap_filled(rec, "LA1")
        assert filled and sig.merge_note == ""


def test_inverted_source_is_not_merged_until_corrected(rec_files):
    res_path, _ = rec_files
    with CoregisteredRecording.open(res_path) as rec:
        rec.set_channel("LA1", a="LA1", b=_GOOD_B, prefer="B", scale_b=-1.0)
        ag = rec.agreement("LA1")
        assert ag.status == "inverted" and ag.corr < -0.9
        sig, filled = _gap_filled(rec, "LA1")
        assert not filled and "opposite polarity" in sig.merge_note
        assert np.isnan(sig.data[sig.source == 0]).all()  # the gap stays empty, never mixed in
        # correcting the other side restores agreement (and the merge)
        rec.set_channel("LA1", a="LA1", b=_GOOD_B, prefer="B", scale_a=-1.0, scale_b=-1.0)
        assert rec.agreement("LA1").ok and _gap_filled(rec, "LA1")[1]


def test_scale_mismatch_is_not_merged_until_corrected(rec_files):
    res_path, _ = rec_files
    with CoregisteredRecording.open(res_path) as rec:
        rec.set_channel("x", a="LA1", b="STIM", prefer="B")  # STIM = 3 × the signal
        ag = rec.agreement("x")
        assert ag.status == "scale" and 2.5 < ag.gain_b_over_a < 3.5
        assert not _gap_filled(rec, "x")[1]
        rec.set_channel("x", a="LA1", b="STIM", prefer="B", scale_b=1 / 3)
        assert rec.agreement("x").ok and _gap_filled(rec, "x")[1]


def test_allow_mismatch_is_an_explicit_override(rec_files):
    res_path, _ = rec_files
    with CoregisteredRecording.open(res_path) as rec:
        rec.set_channel("x", a="LA1", b="STIM", prefer="B", allow_mismatch=True)
        assert rec.agreement("x").status == "scale"
        assert _gap_filled(rec, "x")[1]


def test_runs_and_clock_conversion(rec_files):
    res_path, _ = rec_files
    with CoregisteredRecording.open(res_path) as rec:
        rec.set_channel("LA1", a="LA1", b=_GOOD_B, prefer="B")
        sig = rec.get_data("LA1", _t(270.0), _t(310.0))
        runs = sig.runs()
        assert [r[0] for r in runs] == [2, 1, 2]  # B, A in B's gap, B
        assert runs[0][1] == sig.ts[0] and runs[-1][2] > sig.ts[-1]
        t = np.array([_t(100.0), _t(400.0)])
        t_b = rec.to_file_time(t, "B")
        assert np.all(np.abs((t_b - t) / US_PER_S - _DRIFT.offset_s(np.array([100.0, 400.0]))) < 1e-3)
        np.testing.assert_allclose(rec.to_reference_time(t_b, "B"), t, atol=2)
        np.testing.assert_array_equal(rec.to_file_time(t, "A"), t)


def test_a_file_with_another_span_is_refused(rec_files, tmp_path):
    res_path, rp = rec_files
    res = CoregResult.load(res_path)
    from mef3io_coregistration.io.mef_reader import fingerprint

    res.file_fingerprints = {"A": fingerprint(res.file_a), "B": fingerprint(res.file_b)}
    saved = res.save(tmp_path / "fp_result.json")
    with CoregisteredRecording.open(saved) as rec:  # the right files: fine
        assert rec.channels
    other = tmp_path / "other.mefd"
    from _synth import _write_mef

    _write_mef(other, {"EEG LA1-Ref": rp.mov[:10000]}, _FS_B, rp.mov_start_uutc)
    with pytest.raises(ValueError, match="not the recording"):
        CoregisteredRecording.open(saved, file_b=other)
    CoregisteredRecording.open(saved, file_b=other, verify=False).close()


def test_recording_config_is_bound_to_its_result(rec_files, tmp_path):
    res_path, _ = rec_files
    res = CoregResult.load(res_path)
    saved = res.save(tmp_path / "bound_result.json")
    with CoregisteredRecording.open(saved) as rec:
        cfg = rec.save_config(tmp_path / "rec.json", result_path=saved)
    CoregisteredRecording.from_config(cfg).close()
    res.anchors = []  # the result changes after the mapping was saved
    res.created_utc = "changed"
    res.save(saved)
    with pytest.raises(ValueError, match="changed since"):
        CoregisteredRecording.from_config(cfg)
    CoregisteredRecording.from_config(cfg, verify=False).close()
