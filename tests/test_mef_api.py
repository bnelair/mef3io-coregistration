"""MEF-style access to a coregistration (mef3io ``Reader``-like and legacy ``mef_tools`` API).

Either file, any channel, unipolar or bipolar, over a partial window, on the reference clock; the
result is only the transformation, so the recordings may move.
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import pytest
from mef3io.compat import MefReader as LegacyMefReader

from mef3io_coregistration.config import US_PER_S
from mef3io_coregistration.io.mef_reader import fingerprint
from mef3io_coregistration.reader import (
    CoregisteredMefReader,
    CoregisteredReader,
    CoregisteredSession,
)
from mef3io_coregistration.result import CoregResult
from mef3io_coregistration.transform import ClockMap

_START = 1_700_000_000_000_000
_OFFSET_S = 30.0
_DRIFT = 2e-4  # s/s: 0.72 s over the hour


def _offset(t_ref_s):
    return _OFFSET_S + _DRIFT * t_ref_s


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    """Reference A (256 Hz, a/b/c) and moving B (1024 Hz, a/b/c) recording the same three
    sources; B's clock runs ``_offset`` ahead. Saved with a ground-truth clock map."""
    from mef3io.compat import MefWriter

    tmp = tmp_path_factory.mktemp("mefapi")
    rng = np.random.default_rng(0)
    dur = 900.0
    t_hi = np.arange(0, dur + 200, 1 / 4096.0)  # a fine "true time" axis for the sources
    src = {k: np.cumsum(rng.standard_normal(t_hi.size)) * 0.05 + rng.standard_normal() for k in "abc"}
    src = {k: np.convolve(v, np.ones(64) / 64, mode="same") for k, v in src.items()}

    def write(path, fs, t0_file_s, to_true):
        n = int(dur * fs)
        t_file = t0_file_s + np.arange(n) / fs
        w = MefWriter(str(path), overwrite=True)
        for k, v in src.items():
            w.write_data(np.interp(to_true(t_file), t_hi, v) * 100, k, _START + int(t0_file_s * US_PER_S), fs,
                         precision=3)
        del w

    a_p, b_p = tmp / "ref.mefd", tmp / "mov.mefd"
    write(a_p, 256.0, 0.0, lambda t: t)
    # B's own clock reads t_ref + offset(t_ref): invert (drift is tiny) to find the true time.
    write(b_p, 1024.0, 60.0, lambda t: (t - _OFFSET_S) / (1 + _DRIFT))
    x = np.linspace(0, dur + 200, 400)
    cmap = ClockMap.from_reference_offset((_START + x * US_PER_S).astype(np.int64), _offset(x))
    res = CoregResult(cmap, reference_file=str(a_p), moving_file=str(b_p), reference_role="A",
                      file_fingerprints={"A": fingerprint(a_p), "B": fingerprint(b_p)})
    return res.save(tmp / "result.json"), a_p, b_p


def _t(s):
    return _START + int(s * US_PER_S)


def test_reference_file_reads_exactly_as_mef3io(pair):
    res, a_p, _ = pair
    with CoregisteredSession(res, file="reference") as r:
        assert r.file == "A" and r.is_reference
        x = r.read("a", _t(100), _t(110))
        ts = r.times("a", _t(100), _t(110))
    leg = LegacyMefReader(str(a_p))
    np.testing.assert_array_equal(x, leg.get_data("a", int(ts[0]), int(ts[0]) + 10 * US_PER_S))
    assert x.size == 2560


def test_moving_file_lands_on_the_reference_clock(pair):
    """The moving file, read on the reference clock, matches the reference file's signal."""
    res, _, _ = pair
    ref = CoregisteredSession(res, file="reference")
    mov = CoregisteredSession(ref.reader, file="moving")
    assert mov.file == "B" and mov.fs("a") == 1024.0
    t0, t1 = _t(300), _t(310)
    xm, tm = mov.read("a", t0, t1), mov.times("a", t0, t1)
    xr, tr = ref.read("a", t0, t1), ref.times("a", t0, t1)
    assert xm.size == 10 * 1024 and np.all(np.diff(tm) > 0)
    assert tm[0] >= t0 and tm[-1] < t1
    np.testing.assert_allclose(np.interp(tr, tm, xm), xr, atol=0.05 * np.std(xr))
    ref.close()


def test_bipolar_is_derived_natively_then_resampled(pair):
    res, _, _ = pair
    with CoregisteredSession(res, file="moving") as mov:
        t0, t1 = _t(200), _t(205)
        bip = mov.read("a-b", t0, t1)
        np.testing.assert_array_equal(bip, mov.read(("a", "b"), t0, t1))
        # resampling is linear, so (a − b) resampled equals a resampled − b resampled
        np.testing.assert_allclose(bip, mov.read("a", t0, t1) - mov.read("b", t0, t1), atol=1e-6)
        assert mov.info("a-b")["montage"] == "bipolar"
        with pytest.raises(KeyError):
            mov.read("a-zz", t0, t1)


def test_partial_reads_tile(pair):
    res, _, _ = pair
    with CoregisteredSession(res, file="moving") as mov:
        whole = mov.read_channel("c", _t(400), _t(402))
        a = mov.read_channel("c", _t(400), _t(401))
        b = mov.read_channel("c", _t(401), _t(402))
    np.testing.assert_array_equal(np.r_[a.ts, b.ts], whole.ts)
    np.testing.assert_allclose(np.r_[a.data, b.data], whole.data, rtol=0, atol=1e-9 * np.nanstd(whole.data))


def test_info_times_are_on_the_reference_clock(pair):
    res, _, _ = pair
    with CoregisteredSession(res, file="moving") as mov:
        i = mov.info("a")
        assert i["start_time_file"] == _t(60)
        # file time 60 s on B's clock is reference time t with t + offset(t) = 60
        t_ref = (60 - _OFFSET_S) / (1 + _DRIFT)
        assert abs(i["start_time"] - _t(t_ref)) < 2_000
        assert i["coregistered"] and i["clock"] == "reference"
        seg = mov.segments("a")[0]
        assert seg["start_time_file"] == _t(60) and abs(seg["start_time"] - _t(t_ref)) < 2_000


def test_legacy_api_one_to_one(pair):
    res, a_p, _ = pair
    leg = LegacyMefReader(str(a_p))
    ref = CoregisteredMefReader(res, file="reference")
    assert ref.channels == sorted(leg.channels)
    for prop in ("fsamp", "nsamp", "unit", "start_time", "ufact"):
        assert ref.get_property(prop, "a") == leg.get_property(prop, "a"), prop
    assert ref.get_property("fsamp") == [256.0, 256.0, 256.0]
    x = ref.get_data(["a", "b"], _t(10), _t(11))
    assert isinstance(x, list) and [v.shape for v in x] == [(256,), (256,)]
    np.testing.assert_array_equal(x[0], leg.get_data("a", _t(10), _t(11)))
    info = ref.get_channel_info("a")
    assert info["fsamp"] == [256.0] and info["name"] == "a"
    np.testing.assert_array_equal(ref.get_raw_data("a", _t(10), _t(11))[0], leg.get_raw_data("a", _t(10), _t(11))[0])

    mov = CoregisteredMefReader(res, file="moving")
    y = mov.get_data(np.array(["a", "b-c"]), _t(10), _t(11))
    assert [v.shape for v in y] == [(1024,), (1024,)]
    assert mov.get_property("fsamp", "b-c") == 1024.0
    with pytest.raises(NotImplementedError):
        mov.get_raw_data("a", _t(10), _t(11))
    for r in (ref, mov):
        r.close()


def test_recordings_can_move(pair, tmp_path):
    """The result is only the transformation: moved recordings are found next to the result,
    or passed explicitly; a missing one says how to fix it."""
    res, a_p, b_p = pair
    share = tmp_path / "share"
    share.mkdir()
    shutil.copytree(a_p, share / a_p.name)
    os.symlink(b_p, share / b_p.name)
    d = CoregResult.load(res)
    d.reference_file, d.moving_file = "/gone/" + a_p.name, "/gone/" + b_p.name
    moved = d.save(share / "result.json")
    with CoregisteredSession(moved, file="reference") as r:
        assert r.path == str(share / a_p.name)
        assert r.read("a", _t(10), _t(11)).size == 256  # fingerprints match the copy
    with CoregisteredSession(moved, file_a=a_p, file_b=b_p) as r:
        assert r.path == str(b_p)
    (share / b_p.name).unlink()
    with pytest.raises(FileNotFoundError, match="file_b="):
        CoregisteredReader.from_result(moved)


def test_a_different_recording_is_refused(pair, tmp_path):
    res, a_p, _ = pair
    with pytest.raises(ValueError, match="one file was given for both"):
        CoregisteredSession(res, file_b=a_p)  # file A passed as file B
    from mef3io.compat import MefWriter

    other = tmp_path / "other.mefd"  # another recording: another time span
    w = MefWriter(str(other), overwrite=True)
    w.write_data(np.zeros(2560), "a", _t(5000), 256.0, precision=3)
    del w
    with pytest.raises(ValueError, match="not the recording"):
        CoregisteredSession(res, file_b=other)


def test_record_body_times_land_on_the_reference_clock(pair, monkeypatch):
    """mef3io >= 1.2 records carry absolute body times (Seiz onsets/offsets, Epoc, Curs): they map
    like ``time``, the originals kept as ``*_file``; mef3io's no-entry sentinels stay untouched."""
    res, _, _ = pair
    seiz = {"type": "Seiz", "time": _t(70), "level": "channel", "channel": "a", "segment": None,
            "earliest_onset": _t(70), "latest_offset": _t(95), "timestamp": None, "end_timestamp": None,
            "trace_timestamp": 2**63 - 1,  # MEF's end-of-time marker
            "channels": [{"name": "a", "onset": _t(71), "offset": _t(90)},
                         {"name": "b", "onset": -1, "offset": -1}]}
    with CoregisteredSession(res, file="moving") as mov:
        rdr = mov._rd.reader(mov.file)
        seen = {}

        def fake(channel=None, **kw):
            seen.update(kw, channel=channel)
            return [dict(seiz)]

        monkeypatch.setattr(rdr, "records", fake)
        (r,) = mov.records("a", level="channel")
        assert seen == {"channel": "a", "level": "channel"}

        def ref(s):  # file second s on B's clock -> reference uUTC
            return _t((s - _OFFSET_S) / (1 + _DRIFT))

        for key, s in (("time", 70), ("earliest_onset", 70), ("latest_offset", 95)):
            assert r[key + "_file"] == _t(s) and abs(r[key] - ref(s)) < 2_000, key
        assert r["timestamp"] is None and r["trace_timestamp"] == 2**63 - 1 and "trace_timestamp_file" not in r
        a, b = r["channels"]
        assert a["onset_file"] == _t(71) and abs(a["onset"] - ref(71)) < 2_000
        assert abs(a["offset"] - ref(90)) < 2_000
        assert b == {"name": "b", "onset": -1, "offset": -1}
        assert seiz["channels"][0]["onset"] == _t(71)  # the backend's dicts are not mutated


def test_records_through_mef3io_and_legacy_parity(pair, tmp_path):
    """Real records through mef3io: ``level=`` reaches the backend, and the legacy
    ``get_annotations`` returns what ``mef3io.compat`` returns, with times on the reference clock."""
    from mef3io.compat import MefWriter

    res, _, b_p = pair
    b2 = tmp_path / b_p.name
    shutil.copytree(b_p, b2)
    w = MefWriter(str(b2))
    w.write_annotations([{"time": _t(80), "text": "session note"}])
    w.write_annotations([{"type": "EDFA", "time": _t(85), "text": "channel note", "duration": 2_000_000}], channel="a")
    del w

    def ref(s):
        return _t((s - _OFFSET_S) / (1 + _DRIFT))

    with CoregisteredSession(res, file="moving", file_b=b2) as mov:
        (sess,) = mov.records()
        assert sess["text"] == "session note" and sess["time_file"] == _t(80)
        assert abs(sess["time"] - ref(80)) < 2_000
        (ch,) = mov.records("a", level="channel")
        assert ch["text"] == "channel note" and abs(ch["time"] - ref(85)) < 2_000
        assert ch["duration"] == 2_000_000  # durations stay on the file's clock
        with pytest.raises(ValueError):
            mov.records("a", level="session")  # mef3io's own scope check: the kwarg got there
    leg = LegacyMefReader(str(b2))
    mr = CoregisteredMefReader(res, file="moving", file_b=b2)
    for channel in (None, "a"):
        want, got = leg.get_annotations(channel), mr.get_annotations(channel)
        assert [g["time_file"] for g in got] == [x["time"] for x in want]
        assert [g["text"] for g in got] == [x["text"] for x in want]
    # mef3io >= 1.2 adds segment-level records to records(channel); the legacy call, like
    # mef3io.compat, asks for the channel level only (mef3io cannot write segment records to test).
    calls = []
    real = mr._s.records
    mr._s.records = lambda channel=None, **kw: calls.append((channel, kw)) or real(channel, **kw)
    mr.get_annotations("a")
    mr.get_annotations()
    assert calls == [("a", {"level": "channel"}), (None, {})]
    mr.close()
