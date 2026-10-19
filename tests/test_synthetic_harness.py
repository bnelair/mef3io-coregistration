"""Tests for the synthetic-warp harness (Milestone M1).

The harness is the foundation every later milestone is tested against, so these tests
validate two distinct things:

1. **Caching contract** — fixtures are generated only when absent from the designated
   folder.
2. **Harness correctness** — the fixtures actually encode the clock offset we claim.
   This is checked two ways:
     * *with* ground truth: mapping the moving recording onto the reference timeline via
       the known clock map reproduces the reference signal (the exact relationship the
       alignment stages must later recover);
     * *without* ground truth: a plain cross-correlation of the two recordings recovers
       the local offset, proving alignment is even feasible on these fixtures.
"""
from __future__ import annotations

import numpy as np
import pytest

from mef3io_coregistration.config import US_PER_S
from mef3io_coregistration.synthetic import PROFILES, build_or_load

ALL_PROFILES = sorted(PROFILES)


# --------------------------------------------------------------------------- #
# 1. Caching contract
# --------------------------------------------------------------------------- #
def test_profiles_registered():
    assert {"short_constant", "short_linear", "drift_swing"} <= set(PROFILES)


def test_build_creates_then_reuses(tmp_path):
    """First call generates on disk; a second call reuses the cache (does not regenerate).

    Uses an isolated ``tmp_path`` so the assertion does not depend on whether the shared
    fixture folder was already populated by a previous run.
    """
    first = build_or_load("short_constant", data_dir=tmp_path)
    assert first.generated is True
    assert first.reference_path.exists()
    assert first.moving_path.exists()

    second = build_or_load("short_constant", data_dir=tmp_path)
    assert second.generated is False  # reused, not regenerated


def test_force_regenerates(tmp_path):
    build_or_load("short_constant", data_dir=tmp_path)
    forced = build_or_load("short_constant", data_dir=tmp_path, force=True)
    assert forced.generated is True


def test_unknown_profile_raises():
    with pytest.raises(KeyError):
        build_or_load("does_not_exist")


# --------------------------------------------------------------------------- #
# 2. Round-trip / structural correctness
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.parametrize("name", ALL_PROFILES)
def test_roundtrip_mef3(name, shared_data_dir):
    """Both recordings read back with the requested Fs, strictly increasing timestamps,
    and a moving-file start that reflects the clock offset at t=0."""
    fx = build_or_load(name, data_dir=shared_data_dir)
    ref = fx.load_reference()
    mov = fx.load_moving()

    assert ref.fs == pytest.approx(fx.fs_ref)
    assert mov.fs == pytest.approx(fx.fs_moving)
    assert ref.data.size > 0 and mov.data.size > 0
    assert np.all(np.diff(ref.ts) > 0)
    assert np.all(np.diff(mov.ts) > 0)

    # The moving file's absolute start is later by the offset at world time 0.
    expected_mov_start = ref.start_uutc + float(fx.offset_s(0.0)) * US_PER_S
    assert mov.start_uutc == pytest.approx(expected_mov_start, abs=US_PER_S)  # within 1 s


@pytest.mark.slow
def test_ground_truth_map_is_monotonic(shared_data_dir):
    """The clock map must be strictly monotonic even though the offset swings both ways
    (invariant R6.4) — otherwise it would not be invertible."""
    fx = build_or_load("drift_swing", data_dir=shared_data_dir)
    ref = fx.load_reference()
    moving_clock = fx.forward_uutc(ref.ts)
    assert np.all(np.diff(moving_clock) > 0)


# --------------------------------------------------------------------------- #
# 3. Harness encodes the offset (the reasoned correctness tests)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_map_reconstructs_reference(shared_data_dir):
    """Using ONLY the ground-truth clock map, the moving recording mapped onto the
    reference timeline reproduces the reference signal.

    Rationale: this is precisely the relationship the alignment stages must recover. The
    source is band-limited below both Nyquist rates, so with the *correct* map the
    reconstruction is near-exact (>0.99 correlation). Edges are excluded because interp
    clamps outside the moving span.
    """
    fx = build_or_load("drift_swing", data_dir=shared_data_dir)
    ref = fx.load_reference()
    mov = fx.load_moving()

    moving_clock = fx.forward_uutc(ref.ts)                 # reference time -> moving clock
    reconstructed = np.interp(moving_clock, mov.ts, mov.data)

    core = slice(int(0.05 * ref.data.size), int(0.95 * ref.data.size))
    corr = np.corrcoef(reconstructed[core], ref.data[core])[0, 1]
    assert corr > 0.99


@pytest.mark.slow
@pytest.mark.parametrize("name", ALL_PROFILES)
def test_offset_detectable_without_ground_truth(name, shared_data_dir):
    """Independent of ground truth: a blind cross-correlation of the two recordings
    recovers the local clock offset. Proves the fixtures contain real shared structure,
    i.e. that alignment is feasible (the premise of the whole tool).

    A short reference template is matched against the *entire* moving recording, so the
    search is agnostic to the offset's magnitude (which can be minutes — the case Stage 1
    must absorb). The offset is checked a quarter into the recording: for ``drift_swing``
    (whose swing period equals its duration) that point is an offset extremum where the
    local drift *rate* is zero, so the rigid template is not defeated by the deliberately
    exaggerated local time-dilation. Recovering the offset *through* that dilation is
    Stage 3's job, not this feasibility check's.
    """
    fx = build_or_load(name, data_dir=shared_data_dir)
    ref = fx.load_reference()
    mov = fx.load_moving()

    fs = min(fx.fs_ref, fx.fs_moving)
    t_rel = fx.duration_s * 0.25
    gt_offset = float(fx.offset_s(t_rel))

    measured = _template_offset_seconds(ref, mov, t_rel, fs, tmpl_s=4.0)
    assert measured == pytest.approx(gt_offset, abs=3.0 / fs)  # within a few samples


# --------------------------------------------------------------------------- #
# helpers (deliberately independent of the package's own alignment code)
# --------------------------------------------------------------------------- #
def _uniform_grid(sig, fs: float) -> np.ndarray:
    """Resample ``sig`` onto a uniform grid at ``fs`` spanning its full extent."""
    n = int((sig.ts[-1] - sig.ts[0]) / US_PER_S * fs)
    grid = sig.ts[0] + np.arange(n) / fs * US_PER_S
    return np.interp(grid, sig.ts, sig.data)


def _parabolic_shift(y: np.ndarray, k: int) -> float:
    """Sub-sample offset of a peak at integer index ``k`` via a 3-point parabola fit."""
    if 0 < k < len(y) - 1:
        denom = y[k - 1] - 2.0 * y[k] + y[k + 1]
        if denom != 0.0:
            return 0.5 * (y[k - 1] - y[k + 1]) / denom
    return 0.0


def _template_offset_seconds(ref, mov, t_rel, fs, tmpl_s) -> float:
    """Estimate offset(t_rel) = t_moving - t_ref by matching a short reference template
    against the whole moving recording (normalized cross-correlation, sub-sample refined).

    A ``tmpl_s`` window of the reference centered at world time ``t_rel`` is slid across the
    entire moving recording; the lag maximizing the normalized correlation locates the same
    feature on the moving clock, and the difference of the two absolute timestamps is the
    offset. Sliding over the full recording — rather than a fixed same-timestamp window —
    is what makes this robust to arbitrarily large offsets.
    """
    n = int(round(tmpl_s * fs))
    tmpl_start = ref.start_uutc + (t_rel - tmpl_s / 2.0) * US_PER_S
    grid = tmpl_start + np.arange(n) / fs * US_PER_S
    tmpl = np.interp(grid, ref.ts, ref.data)
    tmpl = tmpl - tmpl.mean()

    mv = _uniform_grid(mov, fs)
    mv = mv - mv.mean()
    corr = np.correlate(mv, tmpl, mode="valid")
    energy = np.sqrt(np.convolve(mv * mv, np.ones(n), mode="valid"))
    ncc = corr / (energy * np.linalg.norm(tmpl) + 1e-12)

    k = int(np.argmax(ncc))
    matched_start = mov.ts[0] + (k + _parabolic_shift(ncc, k)) / fs * US_PER_S
    return (matched_start - tmpl_start) / US_PER_S
