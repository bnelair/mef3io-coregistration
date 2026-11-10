"""Export of a coregistered session: a brand-new .mefd from a result and the two recordings.

Synthetic data only. Reference clock = true time; the moving recorder's clock runs ``_OFFSET_S``
ahead and drifts by ``_DRIFT``, so every moving sample's true value is known.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import mef3io
import numpy as np
import pytest
from _synth import patched_world

from mef3io_coregistration.config import US_PER_S
from mef3io_coregistration.export import (
    ExportCancelled,
    export_coregistered,
)
from mef3io_coregistration.io.mef_reader import fingerprint
from mef3io_coregistration.result import CoregResult
from mef3io_coregistration.transform import ClockMap

_START = 1_700_000_000_000_000
_OFFSET_S = 12.0
_DRIFT = 1e-4  # s/s
_DUR_S = 90.0


def _sig(t_true_s: np.ndarray, k: int = 0) -> np.ndarray:
    return (50 * np.sin(2 * np.pi * (1.3 + 0.1 * k) * t_true_s)
            + 20 * np.sin(2 * np.pi * 4.7 * t_true_s + 1.0 + k))


def _true_of_moving(t_file_s):
    return (np.asarray(t_file_s) - _OFFSET_S) / (1 + _DRIFT)


def _write(path, chans: dict, password: str = "", units="uV"):
    pw2 = password + "-2" if password else ""
    """chans: name -> (data, start_uutc, fs, precision)."""
    with mef3io.Writer(str(path), overwrite=True, units=units, password1=password, password2=pw2) as w:
        for name, (data, start, fs, prec) in chans.items():
            w.write(name, data, int(start), float(fs), precision=prec)


def _t(s):
    return _START + int(round(s * US_PER_S))


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("export")
    fs_r, fs_m = 256.0, 200.0
    # reference: true time 0 .. 90 s
    tr = np.arange(int(_DUR_S * fs_r)) / fs_r
    ref = {"x": (_sig(tr, 0), _START, fs_r, 3), "r_only": (_sig(tr, 1), _START, fs_r, 3)}
    # moving: starts at true time -10 s, on its own clock; m2 has a 5 s gap
    t0_file = -10.0 + _OFFSET_S + _DRIFT * -10.0
    tf = t0_file + np.arange(int(100 * fs_m)) / fs_m
    true = _true_of_moving(tf)
    m2 = _sig(true, 3)
    gap = (tf > 30.0 + _OFFSET_S) & (tf < 35.0 + _OFFSET_S)
    m2[gap] = np.nan
    tf3 = t0_file + np.arange(int(100 * 100.0)) / 100.0
    mov = {
        "x": (_sig(true, 0), _t(t0_file), fs_m, 3),
        "m2": (m2, _t(t0_file), fs_m, 3),
        "m3": (_sig(_true_of_moving(tf3), 2), _t(t0_file), 100.0, 3),
    }
    a_p, b_p = tmp / "recording_a.mefd", tmp / "recording_b.mefd"
    _write(a_p, ref)
    _write(b_p, mov)
    xs = np.linspace(-30, 110, 60)
    cmap = ClockMap.from_reference_offset(
        (_START + xs * US_PER_S).astype(np.int64), _OFFSET_S + _DRIFT * xs)
    res = CoregResult(cmap, reference_file=str(a_p), moving_file=str(b_p), reference_role="A",
                      file_fingerprints={"A": fingerprint(a_p), "B": fingerprint(b_p)})
    res_p = res.save(tmp / "pair.coreg.json")
    return dict(tmp=tmp, a=a_p, b=b_p, res=res_p, cmap=cmap, gap=(30.0, 35.0))


def _read_all(path, password=""):
    out = {}
    with mef3io.Reader(str(path), password=password) as r:
        for c in sorted(r.channels):
            info = r.info(c)
            out[c] = (np.asarray(r.read(c)), int(info["start_time"]), float(info["sampling_frequency"]),
                      int(info["number_of_samples"]))
    return out


def _export(world, name, **kw):
    out = world["tmp"] / name
    rep = export_coregistered(world["res"], out, world["a"], world["b"],
                              moving_prefix=kw.pop("moving_prefix", "B_"), **kw)
    return out, rep


def test_reference_untouched_and_moving_resampled_to_truth(world):
    out, rep = _export(world, "o1.mefd")
    got = _read_all(out)
    assert set(got) == {"x", "r_only", "B_x", "B_m2", "B_m3"}
    src = _read_all(world["a"])
    for c in ("x", "r_only"):  # reference: same samples, same start, same rate
        np.testing.assert_array_equal(got[c][0], src[c][0])
        assert got[c][1:3] == src[c][1:3]
    assert not rep.inverted and rep.reference_role == "A"
    for c, k, fs in (("B_x", 0, 200.0), ("B_m3", 2, 100.0)):
        x, start, rate, _ = got[c]
        assert rate == fs
        ts = start + np.rint(np.arange(x.size) * US_PER_S / fs)
        ok = ~np.isnan(x)
        assert ok.mean() > 0.99
        np.testing.assert_allclose(x[ok], _sig((ts[ok] - _START) / US_PER_S, k), atol=0.15)
    # the moving channel begins at the reference time of its first sample (true time -10 s)
    assert 0 <= got["B_x"][1] - _t(-10.0) < 100_000  # the filter-support margin at the start


def test_gaps_stay_gaps(world):
    out, _ = _export(world, "o2.mefd")
    x, start, fs, _ = _read_all(out)["B_m2"]
    t = (start + np.rint(np.arange(x.size) * US_PER_S / fs) - _START) / US_PER_S
    g0, g1 = world["gap"]
    inside = (t > g0 + 0.2) & (t < g1 - 0.2)
    assert np.isnan(x[inside]).all()
    far = (t < g0 - 0.5) | (t > g1 + 0.5)
    assert not np.isnan(x[far]).any()
    ok = ~np.isnan(x)
    np.testing.assert_allclose(x[ok], _sig(t[ok], 3), atol=0.15)


def test_direction_swap_uses_the_inverse(world):
    out, rep = _export(world, "o3.mefd", reference="B", moving_prefix="A_", reference_prefix="")
    assert rep.inverted and rep.reference_role == "B"
    got = _read_all(out)
    assert "m2" in got and "A_x" in got and "A_r_only" in got
    # B is the reference now: its channels are native
    srcb = _read_all(world["b"])
    np.testing.assert_array_equal(got["m2"][0], srcb["m2"][0])
    # A is resampled onto B's clock: value at B-clock time t is sig(true(t))
    x, start, fs, _ = got["A_r_only"]
    ts = start + np.rint(np.arange(x.size) * US_PER_S / fs)
    ok = ~np.isnan(x)
    assert ok.mean() > 0.99
    truth = _sig(_true_of_moving((ts[ok] - _START) / US_PER_S), 1)
    np.testing.assert_allclose(x[ok], truth, atol=0.15)
    # the A grid starts at A's first sample on B's clock
    assert 0 <= start - _t(_OFFSET_S) < 100_000


def test_result_saved_with_b_as_reference_gives_the_same_default_direction(world, tmp_path):
    cm = world["cmap"]
    res = CoregResult(ClockMap(cm.y_moving, cm.x_ref), reference_file=str(world["b"]),
                      moving_file=str(world["a"]), reference_role="B",
                      file_fingerprints={"A": fingerprint(world["a"]), "B": fingerprint(world["b"])})
    res_p = res.save(tmp_path / "b_ref.coreg.json.gz")
    out = tmp_path / "o.mefd"
    rep = export_coregistered(res_p, out, world["a"], world["b"], moving_prefix="A_")
    assert rep.reference_role == "B" and not rep.inverted
    a = _read_all(out)
    swapped, _ = _export(world, "o3b.mefd", reference="B", moving_prefix="A_")
    c = _read_all(swapped)
    assert a.keys() == c.keys()
    for k in a:
        np.testing.assert_array_equal(a[k][0], c[k][0])  # the same inverse either way


def test_pchip_map_inverted(world, tmp_path):
    cm = world["cmap"]
    pm = ClockMap(cm.x_ref, cm.y_moving, interp_method="pchip")
    res = CoregResult(pm, str(world["a"]), str(world["b"]), reference_role="A",
                      file_fingerprints={"A": fingerprint(world["a"]), "B": fingerprint(world["b"])})
    out = tmp_path / "p.mefd"
    export_coregistered(res, out, world["a"], world["b"], reference="B", moving_prefix="A_",
                        reference_channels=["m3"], moving_channels=["r_only"])
    x, start, fs, _ = _read_all(out)["A_r_only"]
    ts = start + np.rint(np.arange(x.size) * US_PER_S / fs)
    ok = ~np.isnan(x)
    np.testing.assert_allclose(x[ok], _sig(_true_of_moving((ts[ok] - _START) / US_PER_S), 1), atol=0.15)


def test_channel_subset(world):
    out, rep = _export(world, "o4.mefd", reference_channels=["r_only"], moving_channels=["m3"])
    assert set(_read_all(out)) == {"r_only", "B_m3"}
    assert [c.source for c in rep.channels] == ["reference", "moving"]
    out2, _ = _export(world, "o4b.mefd", reference_channels=[], moving_channels=["m2"], moving_prefix="")
    assert set(_read_all(out2)) == {"m2"}


def test_unknown_channel_and_empty_selection_refused(world):
    with pytest.raises(KeyError):
        _export(world, "bad.mefd", moving_channels=["nope"])
    with pytest.raises(ValueError, match="no channels"):
        _export(world, "bad.mefd", reference_channels=[], moving_channels=[])
    assert not (world["tmp"] / "bad.mefd").exists()


def test_name_collision_is_refused_unless_prefixed(world):
    out = world["tmp"] / "col.mefd"
    with pytest.raises(ValueError, match=r"written twice.*'x'"):
        export_coregistered(world["res"], out, world["a"], world["b"])
    assert not out.exists() and not (world["tmp"] / "col.mefd.partial").exists()
    rep = export_coregistered(world["res"], out, world["a"], world["b"], moving_prefix="B_")
    assert {c.name for c in rep.channels} >= {"x", "B_x"}


def _tree_bytes(root: Path, skip_headers=True):
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            b = p.read_bytes()
            out[str(p.relative_to(root))] = b[1024:] if skip_headers else b
    return out


def test_n_threads_does_not_change_the_output(world):
    for d in ("n1", "n4"):
        (world["tmp"] / d).mkdir(exist_ok=True)
    kw = dict(moving_prefix="B_")  # same session name in both folders: mef3io names files after it
    o1, o4 = world["tmp"] / "n1" / "o.mefd", world["tmp"] / "n4" / "o.mefd"
    export_coregistered(world["res"], o1, world["a"], world["b"], n_threads=1, **kw)
    export_coregistered(world["res"], o4, world["a"], world["b"], n_threads=4, **kw)
    a, b = _tree_bytes(o1), _tree_bytes(o4)
    assert a.keys() == b.keys()
    for k in a:
        assert a[k] == b[k], k
    ra, rb = _read_all(o1), _read_all(o4)
    for c in ra:
        np.testing.assert_array_equal(ra[c][0], rb[c][0])


def test_chunk_size_does_not_change_the_values_or_times(world):
    big, _ = _export(world, "c60.mefd", chunk_seconds=60.0)
    small, _ = _export(world, "c7.mefd", chunk_seconds=7.3)
    tiny, _ = _export(world, "c1.mefd", chunk_seconds=1.0)
    rb = _read_all(big)
    for other in (small, tiny):
        ro = _read_all(other)
        assert ro.keys() == rb.keys()
        for c in rb:
            np.testing.assert_array_equal(ro[c][0], rb[c][0])
            assert ro[c][1:] == rb[c][1:]


@pytest.mark.parametrize("fs", [333.33, 1000 / 3, 125.0])
def test_odd_rates_survive_chunking(tmp_path, fs):
    n = int(40 * fs)
    t = np.arange(n) / fs
    _write(tmp_path / "a.mefd", {"c": (_sig(t), _START, fs, 3)})
    _write(tmp_path / "b.mefd", {"d": (_sig(t, 1), _START, fs, 3)})
    xs = np.array([-5.0, 60.0])
    cm = ClockMap.from_reference_offset((_START + xs * US_PER_S).astype(np.int64), np.array([0.5, 0.5]))
    res = CoregResult(cm, str(tmp_path / "a.mefd"), str(tmp_path / "b.mefd"), reference_role="A",
                      file_fingerprints={"A": fingerprint(tmp_path / "a.mefd"),
                                         "B": fingerprint(tmp_path / "b.mefd")})
    outs = []
    for i, cs in enumerate((30.0, 2.0, 0.7)):
        o = tmp_path / f"o{i}.mefd"
        export_coregistered(res, o, tmp_path / "a.mefd", tmp_path / "b.mefd", chunk_seconds=cs)
        outs.append(_read_all(o))
    for o in outs[1:]:
        for c in outs[0]:
            np.testing.assert_array_equal(o[c][0], outs[0][c][0])
            assert o[c][1:] == outs[0][c][1:]


def test_existing_output_is_refused_and_untouched(world):
    out = world["tmp"] / "exists.mefd"
    out.mkdir()
    (out / "keep.txt").write_text("mine")
    with pytest.raises(FileExistsError):
        export_coregistered(world["res"], out, world["a"], world["b"], moving_prefix="B_")
    assert (out / "keep.txt").read_text() == "mine"
    part = world["tmp"] / "other.mefd.partial"
    part.mkdir()
    with pytest.raises(FileExistsError):
        export_coregistered(world["res"], world["tmp"] / "other.mefd", world["a"], world["b"], moving_prefix="B_")
    assert part.is_dir() and not (world["tmp"] / "other.mefd").exists()


def test_inputs_are_not_modified(world):
    before = {p: _tree_bytes(world[p], skip_headers=False) for p in ("a", "b")}
    _export(world, "ro.mefd")
    assert before == {p: _tree_bytes(world[p], skip_headers=False) for p in ("a", "b")}


def test_cancel_removes_only_its_own_files(world):
    tmp = world["tmp"]
    sibling = tmp / "sibling.txt"
    sibling.write_text("mine")
    ev = threading.Event()
    calls = []

    def progress(done, total):
        calls.append((done, total))
        if len(calls) == 2:
            ev.set()

    before = sorted(p.name for p in tmp.iterdir())
    with pytest.raises(ExportCancelled):
        export_coregistered(world["res"], tmp / "cancelled.mefd", world["a"], world["b"],
                            moving_prefix="B_", chunk_seconds=2.0, progress=progress, cancel=ev)
    assert sorted(p.name for p in tmp.iterdir()) == before
    assert sibling.read_text() == "mine"
    # callable cancel, cancelled before anything is written
    with pytest.raises(ExportCancelled):
        export_coregistered(world["res"], tmp / "cancelled.mefd", world["a"], world["b"],
                            moving_prefix="B_", cancel=lambda: True)
    assert sorted(p.name for p in tmp.iterdir()) == before


def test_progress_reaches_the_total(world):
    calls = []
    _export(world, "prog.mefd", chunk_seconds=10.0, progress=lambda d, t: calls.append((d, t)))
    done = [c[0] for c in calls]
    assert done == sorted(done) and done[-1] == calls[-1][1] > 0


def test_failure_in_progress_callback_cleans_up(world):
    before = sorted(p.name for p in world["tmp"].iterdir())

    def boom(d, t):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _export(world, "fail.mefd", progress=boom)
    assert sorted(p.name for p in world["tmp"].iterdir()) == before


def _narrow_result(world, tmp_path, lo_s, hi_s):
    xs = np.linspace(lo_s, hi_s, 20)
    cm = ClockMap.from_reference_offset((_START + xs * US_PER_S).astype(np.int64), _OFFSET_S + _DRIFT * xs)
    res = CoregResult(cm, str(world["a"]), str(world["b"]), reference_role="A",
                      file_fingerprints={"A": fingerprint(world["a"]), "B": fingerprint(world["b"])})
    return res


def test_out_of_span_refuses_by_default_and_can_trim_or_extrapolate(world, tmp_path):
    res = _narrow_result(world, tmp_path, 10.0, 70.0)  # moving reaches -10 .. 90 s
    kw = dict(moving_prefix="B_", moving_channels=["m3"], reference_channels=[])
    with pytest.raises(ValueError, match="measured over"):
        export_coregistered(res, tmp_path / "r.mefd", world["a"], world["b"], **kw)
    assert not (tmp_path / "r.mefd").exists() and not (tmp_path / "r.mefd.partial").exists()

    rep = export_coregistered(res, tmp_path / "t.mefd", world["a"], world["b"], out_of_span="trim", **kw)
    x, start, _fs, _ = _read_all(tmp_path / "t.mefd")["B_m3"]
    assert start >= _t(10.0 - 1.0) - 20_000 and start + x.size * 1e4 <= _t(70.0 + 1.0) + 20_000
    assert rep.channels[0].trimmed_us[0] > 5 * US_PER_S and rep.notes

    rep = export_coregistered(res, tmp_path / "e.mefd", world["a"], world["b"], out_of_span="extrapolate", **kw)
    x, start, _fs, _ = _read_all(tmp_path / "e.mefd")["B_m3"]
    assert 0 <= start - _t(-10.0) < 200_000 and rep.channels[0].extrapolated_us[0] > 5 * US_PER_S
    ts = (start + np.rint(np.arange(x.size) * 1e4) - _START) / US_PER_S
    np.testing.assert_allclose(x, _sig(ts, 2), atol=0.15)  # a linear drift extrapolates exactly


def test_provenance_note_has_no_paths(world):
    out, rep = _export(world, "prov.mefd")
    with mef3io.Reader(str(out)) as r:
        recs = r.records()
    notes = [x for x in recs if "export_coregistered" in x.get("text", "")]
    assert len(notes) == 1
    doc = json.loads(notes[0]["text"])
    assert doc["result_sha256"] == hashlib.sha256(Path(world["res"]).read_bytes()).hexdigest() == rep.result_sha256
    assert doc["mef3io_version"] == mef3io.__version__ and doc["reference_recording"] == "recording_a.mefd"
    assert doc["direction"].startswith("as saved") and doc["moving_recording"] == "recording_b.mefd"
    blob = b"".join(p.read_bytes() for p in out.rglob("*") if p.is_file())
    assert str(world["tmp"]).encode() not in blob


def test_units_conversion_factor_and_units_label_follow_the_source(world, tmp_path):
    out, rep = _export(world, "u.mefd")
    with mef3io.Reader(str(out)) as r:
        assert r.info("B_m3")["units_conversion_factor"] == pytest.approx(1e-3)
        assert r.info("B_m3")["units_description"] == "uV"
    assert rep.units == "uV"


def test_non_decimal_factor_is_kept_verbatim(tmp_path):
    fs = 100.0
    n = int(60 * fs)
    counts = (np.sin(np.arange(n) / 7.0) * 3000).astype(np.int32)
    with mef3io.Writer(str(tmp_path / "a.mefd"), overwrite=True, units="uV") as w:
        w.write_int32("c", counts, 0.195, _START, fs)
    with mef3io.Writer(str(tmp_path / "b.mefd"), overwrite=True, units="uV") as w:
        w.write_int32("d", counts, 0.195, _START, fs)
    cm = ClockMap.from_reference_offset(np.array([_START, _START + 60 * US_PER_S]), np.array([0.0, 0.0]))
    res = CoregResult(cm, str(tmp_path / "a.mefd"), str(tmp_path / "b.mefd"),
                      file_fingerprints={"A": fingerprint(tmp_path / "a.mefd"), "B": fingerprint(tmp_path / "b.mefd")})
    export_coregistered(res, tmp_path / "o.mefd", tmp_path / "a.mefd", tmp_path / "b.mefd", moving_prefix="B_")
    with mef3io.Reader(str(tmp_path / "o.mefd")) as r:
        assert r.info("c")["units_conversion_factor"] == 0.195
        assert r.info("B_d")["units_conversion_factor"] == 0.195
        raw = r.read_raw("c")["samples"]
        np.testing.assert_array_equal(raw, counts)  # bit-exact reference counts
        m = np.asarray(r.read("B_d")) / 0.195
        i0 = int(round((r.info("B_d")["start_time"] - _START) * fs / US_PER_S))
        assert 0 < i0 < 20  # the margin at the start
        np.testing.assert_allclose(m, counts[i0 : i0 + m.size], atol=1.0)  # identity map: ~ no-op


def test_mixed_units_need_an_explicit_label(world, tmp_path):
    _write(tmp_path / "m.mefd", {"v": (_sig(np.arange(5000) / 100.0), _START, 100.0, 3)}, units="mV")
    cm = world["cmap"]
    res = CoregResult(cm, str(world["a"]), str(tmp_path / "m.mefd"), reference_role="A",
                      file_fingerprints={"A": fingerprint(world["a"]), "B": fingerprint(tmp_path / "m.mefd")})
    with pytest.raises(ValueError, match="units"):
        export_coregistered(res, tmp_path / "o.mefd", world["a"], tmp_path / "m.mefd", verify_inputs=False)
    rep = export_coregistered(res, tmp_path / "o.mefd", world["a"], tmp_path / "m.mefd",
                              verify_inputs=False, units="uV", out_of_span="extrapolate")
    assert rep.units == "uV"


def test_encrypted_input_and_encrypted_output(world, tmp_path):
    tr = np.arange(int(30 * 100.0)) / 100.0
    _write(tmp_path / "ea.mefd", {"c": (_sig(tr), _START, 100.0, 3)}, password="pw-a")
    _write(tmp_path / "eb.mefd", {"d": (_sig(tr, 1), _START, 100.0, 3)}, password="pw-b")
    cm = ClockMap.from_reference_offset(np.array([_START, _START + 30 * US_PER_S]), np.array([0.0, 0.0]))
    res = CoregResult(cm, "a", "b", reference_role="A", file_fingerprints={})
    kw = dict(password_a="pw-a-2", password_b="pw-b-2", verify_inputs=False, moving_prefix="B_")
    export_coregistered(res, tmp_path / "plain.mefd", tmp_path / "ea.mefd", tmp_path / "eb.mefd", **kw)
    assert set(_read_all(tmp_path / "plain.mefd")) == {"c", "B_d"}  # readable without a password
    export_coregistered(res, tmp_path / "enc.mefd", tmp_path / "ea.mefd", tmp_path / "eb.mefd",
                        output_passwords=("out-pw", "out-pw2"), **kw)
    assert set(_read_all(tmp_path / "enc.mefd", password="out-pw2")) == {"c", "B_d"}
    with pytest.raises(Exception):
        _read_all(tmp_path / "enc.mefd", password="wrong")


def test_wrong_file_for_the_result_is_refused(world, tmp_path):
    with pytest.raises(ValueError, match=r"swapped|not the recording"):
        export_coregistered(world["res"], tmp_path / "s.mefd", world["b"], world["a"], moving_prefix="B_")


def test_argument_checks(world, tmp_path):
    for kw in (dict(chunk_seconds=0), dict(out_of_span="x"), dict(reference="C"), dict(n_threads=-1)):
        with pytest.raises(ValueError):
            export_coregistered(world["res"], tmp_path / "o.mefd", world["a"], world["b"], **kw)
    with pytest.raises(ValueError, match=r"\.mefd"):
        export_coregistered(world["res"], tmp_path / "o.tar", world["a"], world["b"])
    with pytest.raises(FileNotFoundError):
        export_coregistered(world["res"], tmp_path / "nodir" / "o.mefd", world["a"], world["b"])


# -- review round 1 ----------------------------------------------------------------------------
def test_segments_with_different_factors_use_the_finest(tmp_path):
    """mef3io scales each sample by its own segment's factor (info gives None then): the export
    must not crash and must keep every reference value exactly (0.01 and 0.1 -> 0.01)."""
    fs = 100.0
    x = np.round(np.sin(np.arange(3000) / 7.0) * 30, 2)
    with mef3io.Writer(str(tmp_path / "a.mefd"), units="uV") as w:
        w.write("c", x[:1500], _START, fs, precision=2)
        w.write("c", x[1500:], _START + 15_500_000, fs, precision=1, new_segment=True)
    _write(tmp_path / "b.mefd", {"d": (x, _START, fs, 2)})
    with mef3io.Reader(str(tmp_path / "a.mefd")) as r:
        assert r.info("c")["units_conversion_factor"] is None  # the case under test
    cm = ClockMap.from_reference_offset(np.array([_START, _START + 40 * US_PER_S]), np.array([0.0, 0.0]))
    res = CoregResult(cm, str(tmp_path / "a.mefd"), str(tmp_path / "b.mefd"), file_fingerprints={})
    rep = export_coregistered(res, tmp_path / "o.mefd", tmp_path / "a.mefd", tmp_path / "b.mefd",
                              moving_prefix="B_", verify_inputs=False)
    ch = {c.name: c for c in rep.channels}
    assert ch["c"].units_conversion_factor == pytest.approx(0.01) and ch["c"].precision == 2
    assert any("different conversion factors" in n for n in rep.notes)
    src, got = _read_all(tmp_path / "a.mefd"), _read_all(tmp_path / "o.mefd")
    # both segments' values, as the same decimals (292 x 0.1 vs 2920 x 0.01 differ by one ulp)
    assert np.array_equal(np.isnan(got["c"][0]), np.isnan(src["c"][0]))
    np.testing.assert_allclose(got["c"][0], src["c"][0], rtol=1e-15, atol=0)
    with mef3io.Reader(str(tmp_path / "o.mefd")) as r:
        raw = r.read_raw("c")
    ok = np.asarray(raw["valid"]) != 0
    sv = src["c"][0][~np.isnan(src["c"][0])]  # the source's values: 0.01 and 0.1 steps
    np.testing.assert_array_equal(np.asarray(raw["samples"])[ok], np.rint(sv * 100).astype(np.int64))
    assert got["c"][1] == src["c"][1]


def test_a_negative_factor_keeps_the_values():
    from mef3io_coregistration.export import _counts, _factor_precision

    for f, want in ((-0.1, (0.1, 1)), (-0.195, (0.195, None)), (0.0, (1.0, 0)), (1e-3, (1e-3, 3))):
        got = _factor_precision(f)
        assert got[1] == want[1] and got[0] == pytest.approx(want[0])
    counts = np.array([-5, 0, 7, 123456], dtype=np.int64)
    for f in (-0.1, -0.195):
        values = counts * f  # what mef3io reads from a negative-factor segment
        factor, prec = _factor_precision(f)
        back = _counts(values, factor, prec) * (factor if prec is None else 10.0 ** -prec)
        np.testing.assert_allclose(back, values, rtol=1e-12, atol=0)  # not rounded to whole units


def test_units_override_warns_and_records_the_source_units(world, tmp_path):
    _write(tmp_path / "m.mefd", {"v": (_sig(np.arange(5000) / 100.0), _START, 100.0, 3)}, units="mV")
    res = CoregResult(world["cmap"], str(world["a"]), str(tmp_path / "m.mefd"), reference_role="A",
                      file_fingerprints={})
    with pytest.warns(UserWarning, match=r"NOT converted.*'v' \(mV\)"):
        rep = export_coregistered(res, tmp_path / "o.mefd", world["a"], tmp_path / "m.mefd",
                                  verify_inputs=False, units="uV", out_of_span="extrapolate",
                                  reference_channels=["x"])
    assert rep.source_units == {"v": "mV"} and any("NOT converted" in n for n in rep.notes)
    with mef3io.Reader(str(tmp_path / "o.mefd")) as r:
        doc = json.loads(next(x["text"] for x in r.records() if "export_coregistered" in x["text"]))
    assert doc["source_units"] == {"v": "mV"} and doc["units"] == "uV"


def test_each_input_is_opened_once(world, tmp_path, monkeypatch):
    opened = []
    real = mef3io.Reader

    class Counting(real):
        def __init__(self, path, *a, **k):
            opened.append(Path(path).name)
            super().__init__(path, *a, **k)

    monkeypatch.setattr(mef3io, "Reader", Counting)
    export_coregistered(world["res"], tmp_path / "once.mefd", world["a"], world["b"], moving_prefix="B_")
    assert sorted(opened) == ["once.mefd", "recording_a.mefd", "recording_b.mefd"]  # + read-back


def test_channel_batches_do_not_change_the_output(world, monkeypatch):
    import mef3io_coregistration.export as ex

    ref, _ = _export(world, "batch_ref.mefd")
    monkeypatch.setattr(ex, "_MAX_CHUNK_SAMPLES", 1)  # one channel per batch
    one, _ = _export(world, "batch_one.mefd")
    a, b = _read_all(ref), _read_all(one)
    assert a.keys() == b.keys()
    for c in a:
        np.testing.assert_array_equal(a[c][0], b[c][0])
        assert a[c][1:] == b[c][1:]


def test_a_partial_folder_appearing_meanwhile_is_not_removed(world, tmp_path, monkeypatch):
    import mef3io_coregistration.export as ex

    out = tmp_path / "race.mefd"
    real = ex._load_result

    def racing(result):  # another process takes the staging name after the checks
        (tmp_path / "race.mefd.partial").mkdir()
        (tmp_path / "race.mefd.partial" / "theirs.txt").write_text("theirs")
        return real(result)

    monkeypatch.setattr(ex, "_load_result", racing)
    with pytest.raises(FileExistsError):
        export_coregistered(world["res"], out, world["a"], world["b"], moving_prefix="B_")
    assert (tmp_path / "race.mefd.partial" / "theirs.txt").read_text() == "theirs"


# -- channel span comes from the block index, never from the .tmet header times -----------------
@pytest.mark.parametrize("which,channel", [("A", "x"), ("B", "m3")])
def test_a_header_end_at_the_last_block_start_loses_no_samples(world, tmp_path, which, channel):
    base, _ = _export(world, f"span_base_{which}.mefd")
    block_us = 10 * US_PER_S
    a, b = patched_world(world, tmp_path, which, channel, 24, lambda e: e - block_us)
    with mef3io.Reader(str({"A": a, "B": b}[which])) as r:  # the header really is short
        assert int(r.info(channel)["end_time"]) < int(r.block_envelope(channel)["end_uutc"][-1]) - 5 * US_PER_S
    out = tmp_path / "span_out.mefd"
    export_coregistered(world["res"], out, a, b, moving_prefix="B_", verify_inputs=False)
    want, got = _read_all(base), _read_all(out)
    assert want.keys() == got.keys()
    for c in want:
        np.testing.assert_array_equal(got[c][0], want[c][0])
        assert got[c][1:] == want[c][1:]


@pytest.mark.parametrize("which,channel", [("A", "x"), ("B", "m3")])
def test_a_late_header_start_loses_no_samples(world, tmp_path, which, channel):
    base, _ = _export(world, f"late_base_{which}.mefd")
    a, b = patched_world(world, tmp_path, which, channel, 16, lambda s: s + 20 * US_PER_S)
    out = tmp_path / "late_out.mefd"
    export_coregistered(world["res"], out, a, b, moving_prefix="B_", verify_inputs=False)
    want, got = _read_all(base), _read_all(out)
    for c in want:
        np.testing.assert_array_equal(got[c][0], want[c][0])
        assert got[c][1:] == want[c][1:]

