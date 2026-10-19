"""Tests for Stage 3 — continuous drift estimation.

Covers: seeded sliding-window search; per-window confidence weights, low-confidence
windows dropped; weighted smoothing spline → offset(t); non-monotonic offset yields
a monotone map. Accept: recovered offset(t) RMS < 1 lower-Fs sample over the anchored span.
"""
from __future__ import annotations

import numpy as np
import pytest
from _synth import build_pair, corrupt_segment

from mef3io_coregistration.align import estimate_anchor, estimate_drift
from mef3io_coregistration.align.drift import refit_drift
from mef3io_coregistration.config import US_PER_S
from mef3io_coregistration.synthetic.warp import WarpSpec

# Physical drift rates (|d offset/dt| ~ 1e-4). A swing amplitude of 0.07 s over a 1200 s period
# reproduces the ±5 s / 24 h *rate* in a fast-to-render span, and is non-monotonic.
_LINEAR = WarpSpec("linear", offset0_s=120.0, rate_s_per_s=5.0 / (12 * 3600))
_SWING = WarpSpec("swing", offset0_s=120.0, swing_amp_s=0.07, swing_period_s=1200.0)


def _run(pair, spec, roi_s=300.0, **kw):
    anchor = estimate_anchor(
        pair.ref_dec, pair.mov_dec, pair.roi_uutc(roi_s), pair.gt_offset_s(roi_s), half_window_s=20.0
    )
    drift = estimate_drift(pair.ref_dec, pair.mov_dec, anchor, window_s=20.0, grid_s=30.0, **kw)
    t_rel = (drift.t_ref_uutc - pair.start_uutc) / US_PER_S
    gt = np.asarray(spec.offset_s(t_rel))
    rms = float(np.sqrt(np.mean((drift.smoothed_offset_s - gt) ** 2)) * pair.fs_target)
    return drift, gt, rms


def test_recovers_linear_drift_under_one_sample():
    pair = build_pair(_LINEAR)
    drift, _gt, rms = _run(pair, _LINEAR)
    assert rms < 1.0
    assert np.all(np.diff(drift.clock_map.y_moving) > 0)  # monotone map


def test_recovers_swing_with_monotone_map():
    """A non-monotonic offset (swings up then down) still yields a strictly monotone map
, recovered within 1 sample RMS."""
    pair = build_pair(_SWING)
    drift, gt, rms = _run(pair, _SWING)
    d = np.diff(gt)
    assert d.max() > 0 and d.min() < 0  # offset really is non-monotonic
    assert rms < 1.0
    assert np.all(np.diff(drift.clock_map.y_moving) > 0)


def test_refit_from_kept_windows_reuses_estimates():
    """Manual review is a pure refit: same per-window offsets/confidences, a different keep set.
    Discarding the outer windows must shrink the measured span without touching the correlations."""
    pair = build_pair(_LINEAR)
    drift, _gt, _rms = _run(pair, _LINEAR)

    keep = drift.accepted.copy()
    kept_idx = np.flatnonzero(keep)
    # discard the first and last accepted windows
    keep[kept_idx[0]] = False
    keep[kept_idx[-1]] = False

    refit = refit_drift(drift, keep)
    assert int(refit.accepted.sum()) == int(drift.accepted.sum()) - 2
    # per-window measurements are untouched — only the fit and span change
    assert np.array_equal(refit.offset_s, drift.offset_s)
    assert np.array_equal(refit.confidence, drift.confidence)
    assert refit.measured_span_uutc[0] > drift.measured_span_uutc[0]
    assert refit.measured_span_uutc[1] < drift.measured_span_uutc[1]
    assert refit.clock_map.metadata["reviewed"] is True
    assert np.all(np.diff(refit.clock_map.y_moving) > 0)  # still monotone


def test_refit_never_keeps_a_gap_window_and_needs_two():
    pair = build_pair(_LINEAR)
    pair.mov_dec = corrupt_segment(pair.mov_dec, 500.0, 650.0)
    drift, _gt, _rms = _run(pair, _LINEAR)

    # asking to keep everything must still exclude gap windows
    refit = refit_drift(drift, np.ones_like(drift.accepted, dtype=bool))
    assert not np.any(refit.accepted & drift.gapped)

    with pytest.raises(ValueError, match="two"):
        refit_drift(drift, np.zeros_like(drift.accepted, dtype=bool))


def test_low_confidence_windows_are_dropped_and_span_still_recovered():
    """A dead/low-SNR span loses shared structure; those windows fall below min_confidence,
    are excluded from the fit, and the rest of the span is still recovered."""
    pair = build_pair(_LINEAR)
    pair.mov_dec = corrupt_segment(pair.mov_dec, 500.0, 650.0)  # noise over [500, 650] s
    drift, gt, _rms = _run(pair, _LINEAR)

    t_rel = (drift.t_ref_uutc - pair.start_uutc) / US_PER_S
    corrupted = (t_rel >= 500.0) & (t_rel < 650.0)
    assert np.mean(drift.confidence[corrupted]) < np.mean(drift.confidence[~corrupted])
    assert not np.all(drift.accepted[corrupted])  # at least some corrupted windows dropped
    # away from the dead span the offset is still recovered to sub-sample accuracy
    clean = ~corrupted
    err = (drift.smoothed_offset_s[clean] - gt[clean]) * pair.fs_target
    assert np.sqrt(np.mean(err**2)) < 1.0


def test_qc_arrays_are_exposed():
    pair = build_pair(_LINEAR, duration_s=600.0)
    drift, _gt, _rms = _run(pair, _LINEAR)
    n = drift.t_ref_uutc.shape[0]
    assert drift.offset_s.shape == (n,)
    assert drift.confidence.shape == (n,)
    assert drift.smoothed_offset_s.shape == (n,)
    assert drift.accepted.shape == (n,)
    assert drift.clock_map.metadata["stage"] == "drift"


# --------------------------------------------------------------------------- #
# Walk outward from the anchors. Seeding from the start of the span with the anchor's offset
# broke whenever the anchor sat far from the start: with realistic drift the first window's seed
# was already off by more than the ±0.25 s search, the walk locked onto a side peak and tracked
# it — a constant 0.5 s error with 229/240 windows "accepted".
# --------------------------------------------------------------------------- #
def _anchors(pair, rois, half=60.0):
    return [
        estimate_anchor(pair.ref_dec, pair.mov_dec, pair.roi_uutc(r), pair.gt_offset_s(r), half_window_s=half)
        for r in rois
    ]


def _rms_samples(pair, spec, drift):
    t_rel = (drift.t_ref_uutc - pair.start_uutc) / US_PER_S
    ok = drift.accepted
    err = (drift.smoothed_offset_s[ok] - np.asarray(spec.offset_s(t_rel[ok]))) * pair.fs_target
    return float(np.sqrt(np.mean(err**2)))


def test_anchor_far_from_start_realistic_rate():
    pair = build_pair(_LINEAR, duration_s=7200.0, fs_ref=128.0, fs_mov=64.0)
    drift = estimate_drift(pair.ref_dec, pair.mov_dec, _anchors(pair, [6600.0]))
    assert drift.accepted.mean() > 0.95
    assert _rms_samples(pair, _LINEAR, drift) < 0.1
    assert np.all(np.abs(drift.anchor_residual_s * pair.fs_target) < 0.5)


def test_all_anchors_are_used_and_residuals_reported():
    pair = build_pair(_LINEAR, duration_s=3600.0, fs_ref=128.0, fs_mov=64.0)
    drift = estimate_drift(pair.ref_dec, pair.mov_dec, _anchors(pair, [3000.0, 500.0]))
    assert [a.t_ref_uutc for a in drift.anchors] == [pair.roi_uutc(500.0), pair.roi_uutc(3000.0)]
    assert drift.anchor_residual_s.shape == (2,)
    assert len(drift.clock_map.metadata["anchors"]) == 2
    assert _rms_samples(pair, _LINEAR, drift) < 0.1


def test_later_anchor_at_same_time_supersedes():
    from mef3io_coregistration.align import Anchor

    pair = build_pair(_LINEAR, duration_s=900.0)
    good = _anchors(pair, [400.0], half=20.0)[0]
    bad = Anchor(good.t_ref_uutc, good.t_ref_uutc + 125 * US_PER_S, 125.0, 0.1)
    drift = estimate_drift(pair.ref_dec, pair.mov_dec, [bad, good])
    assert drift.anchors == (good,)


def test_bad_anchor_shows_in_residual_not_in_map():
    """Anchors seed the walk but are not fitted: a wrong anchor is reported, the map holds."""
    from mef3io_coregistration.align import Anchor

    pair = build_pair(_LINEAR, duration_s=900.0)
    good = _anchors(pair, [300.0], half=20.0)[0]
    t = pair.roi_uutc(700.0)
    off = pair.gt_offset_s(700.0) + 0.05  # 12.8 samples wrong, inside the search range
    drift = estimate_drift(pair.ref_dec, pair.mov_dec, [good, Anchor(t, t + int(off * US_PER_S), off, 0.9)])
    assert abs(drift.anchor_residual_s[1] * pair.fs_target) > 5.0
    assert _rms_samples(pair, _LINEAR, drift) < 1.0


def test_dead_span_then_distant_region_recovers():
    """A noise span whose windows can look 'distinctive' must not teach the walk a wrong
    slope that it then carries on for the rest of the recording."""
    pair = build_pair(_LINEAR, duration_s=3600.0, fs_ref=128.0, fs_mov=64.0)
    pair.mov_dec = corrupt_segment(pair.mov_dec, 900.0, 1500.0)
    drift = estimate_drift(pair.ref_dec, pair.mov_dec, _anchors(pair, [300.0]))
    t_rel = (drift.t_ref_uutc - pair.start_uutc) / US_PER_S
    late = drift.accepted & (t_rel > 1600.0)
    err = (drift.offset_s[late] - np.asarray(_LINEAR.offset_s(t_rel[late]))) * pair.fs_target
    assert late.sum() > 50 and np.max(np.abs(err)) < 1.0


def test_search_edge_peak_is_not_a_measurement():
    """A slow signal whose true lag (0.8 s) lies outside a ±0.5 s search: the correlation is
    still rising at the boundary, which is not a peak — confidence 0, flagged."""
    from mef3io_coregistration.align.tde import normxcorr_delay

    fs = 100.0
    t = np.arange(6000) / fs
    x = np.sin(2 * np.pi * 0.1 * t) + 0.3 * np.sin(2 * np.pi * 0.037 * t)
    d = normxcorr_delay(x[:4000], x[80:4080], fs, max_lag_s=0.5)
    assert d.at_edge and d.confidence == 0.0


def test_backward_walk_learns_its_own_slope():
    """Anchors at the peak and trough of a swing: the slope between them has the opposite sign
    to the drift before the first anchor, which the backward walk must learn from its own
    windows (a backward history is stored newest-first; a slope estimator that only used pairs
    in increasing order never updated there and walked off with the anchors' slope)."""
    spec = WarpSpec("swing", offset0_s=120.0, swing_amp_s=0.5, swing_period_s=7200.0)
    pair = build_pair(spec, duration_s=7200.0, fs_ref=128.0, fs_mov=64.0)
    drift = estimate_drift(pair.ref_dec, pair.mov_dec, _anchors(pair, [1800.0, 5400.0]))
    t_rel = (drift.t_ref_uutc - pair.start_uutc) / US_PER_S
    early = drift.accepted & (t_rel < 1700.0)
    err = (drift.offset_s[early] - np.asarray(spec.offset_s(t_rel[early]))) * pair.fs_target
    assert early.sum() > 40 and np.max(np.abs(err)) < 1.0
    assert _rms_samples(pair, spec, drift) < 0.2


def test_windows_only_where_both_recordings_exist():
    """A moving recording covering part of the reference: no windows outside it — they used to
    be counted as 'recording gaps', a different (and misleading) thing."""
    from mef3io_coregistration.io.derived import Decimated

    pair = build_pair(_LINEAR, duration_s=1200.0)
    lo, hi = int(300 * pair.mov_dec.fs), int(900 * pair.mov_dec.fs)
    start = pair.mov_dec.start_uutc + int(round(lo * US_PER_S / pair.mov_dec.fs))
    pair.mov_dec = Decimated(pair.mov_dec.data[lo:hi], pair.mov_dec.fs, start)
    drift = estimate_drift(pair.ref_dec, pair.mov_dec, _anchors(pair, [600.0], half=60.0))
    t_rel = (drift.t_ref_uutc - pair.start_uutc) / US_PER_S
    assert t_rel.min() > 290.0 and t_rel.max() < 910.0
    assert not drift.gapped.any()
