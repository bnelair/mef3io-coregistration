"""Opening a coregistration result with the reader alone: the time-span check (the only check),
swapped files, and the two files read as one recording.

Self-contained (numpy + mef3io + the reader), so it also runs in the public reader repository.
"""
from __future__ import annotations

import numpy as np
import pytest
from mef3io.compat import MefWriter

from mef3io_coregistration.config import US_PER_S
from mef3io_coregistration.io.mef_reader import fingerprint
from mef3io_coregistration.reader import CoregisteredRecording, CoregisteredSession
from mef3io_coregistration.result import CoregResult
from mef3io_coregistration.transform import ClockMap

_START = 1_700_000_000_000_000
_OFFSET_S = 12.0
_DRIFT = 1e-4  # s/s


def _t(s: float) -> int:
    return _START + int(s * US_PER_S)


def _signal(t_true_s: np.ndarray) -> np.ndarray:
    return 50 * np.sin(2 * np.pi * 1.3 * t_true_s) + 20 * np.sin(2 * np.pi * 4.7 * t_true_s + 1.0)


def _write(path, fs: float, t0_file_s: float, dur_s: float, to_true) -> None:
    t_file = t0_file_s + np.arange(int(dur_s * fs)) / fs
    w = MefWriter(str(path), overwrite=True)
    w.write_data(_signal(to_true(t_file)), "x", _t(t0_file_s), fs, precision=3)
    del w


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    """Reference A covers reference time 0–300 s; moving B (its clock ``offset`` ahead) covers
    reference time −60–240 s, so each file has a stretch the other lacks."""
    tmp = tmp_path_factory.mktemp("openmerge")
    a_p, b_p, other_p = tmp / "a.mefd", tmp / "b.mefd", tmp / "other.mefd"
    _write(a_p, 256.0, 0.0, 300.0, lambda t: t)
    b0 = -60.0 + _OFFSET_S + _DRIFT * -60.0  # B's own clock at reference time −60 s
    _write(b_p, 200.0, b0, 300.0, lambda t: (t - _OFFSET_S) / (1 + _DRIFT))
    _write(other_p, 256.0, 1000.0, 300.0, lambda t: t)  # another recording: another span
    x = np.linspace(-120, 360, 100)
    cmap = ClockMap.from_reference_offset((_START + x * US_PER_S).astype(np.int64), _OFFSET_S + _DRIFT * x)
    res = CoregResult(cmap, reference_file=str(a_p), moving_file=str(b_p), reference_role="A",
                      file_fingerprints={"A": fingerprint(a_p), "B": fingerprint(b_p)})
    return res.save(tmp / "pair.coreg.json"), a_p, b_p, other_p


def test_result_round_trips_byte_stable(pair, tmp_path):
    res_p = pair[0]
    again = CoregResult.load(res_p).save(tmp_path / "again.json")
    assert again.read_bytes() == res_p.read_bytes()


def test_moving_file_matches_the_true_signal_on_the_reference_clock(pair):
    with CoregisteredSession(pair[0], file="moving") as mov:
        t0, t1 = _t(100), _t(110)
        x, ts = mov.read("x", t0, t1), mov.times("x", t0, t1)
    want = _signal((ts - _START) / US_PER_S)
    np.testing.assert_allclose(x, want, atol=0.5)  # 70-unit signal; stored at precision=3


def test_a_recording_with_another_span_is_refused(pair):
    res_p, a_p, _, other_p = pair
    with pytest.raises(ValueError, match="spans"):
        CoregisteredSession(res_p, file_a=a_p, file_b=other_p)
    with CoregisteredSession(res_p, file="moving", file_a=a_p, file_b=other_p, verify=False) as s:
        assert s.path == str(other_p)


def test_swapped_files_are_refused(pair):
    res_p, a_p, b_p, _ = pair
    with pytest.raises(ValueError, match="swapped"):
        CoregisteredSession(res_p, file_a=b_p, file_b=a_p)


def test_both_files_read_as_one_recording(pair):
    rec = CoregisteredRecording.open(pair[0])
    rec.set_channel("x", a="x", b="x", prefer="A")
    sig = rec.get_data("x", _t(-30), _t(30))  # B only before 0 s, A (preferred) after
    t_s = (sig.ts - _START) / US_PER_S
    assert np.all(sig.source[t_s < -0.1] == 2) and np.all(sig.source[t_s > 0.1] == 1)
    ok = sig.source > 0
    np.testing.assert_allclose(sig.data[ok], _signal(t_s[ok]), atol=0.5)

    t_ref = np.array([_t(50), _t(150)])
    t_b = rec.to_file_time(t_ref, "B")
    np.testing.assert_allclose(t_b - t_ref, (_OFFSET_S + _DRIFT * np.array([50, 150])) * US_PER_S, atol=2)
    np.testing.assert_array_equal(rec.to_reference_time(t_b, "B"), t_ref)
    np.testing.assert_array_equal(rec.to_file_time(t_ref, "A"), t_ref)
