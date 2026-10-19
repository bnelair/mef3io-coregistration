"""Tests for Stage 2 — fine alignment / anchor.

Covers: ROI GCC seeded by the gross offset with sub-sample refinement; emit an anchor
(t_ref, t_moving) + confidence; repeatable at multiple ROIs. Accept: recovered anchor
lag within 1 sample of the lower Fs of offset_gt at the ROI centre.
"""
from __future__ import annotations

import numpy as np
import pytest
from _synth import build_pair

from mef3io_coregistration.align import estimate_anchor, estimate_gross_offset
from mef3io_coregistration.align.fine import GappyWindow
from mef3io_coregistration.config import US_PER_S
from mef3io_coregistration.io.derived import Decimated
from mef3io_coregistration.synthetic.warp import WarpSpec

_DRIFT = WarpSpec("linear", offset0_s=120.0, rate_s_per_s=5.0 / (12 * 3600))  # ±5 s / 12 h


def test_anchor_within_one_lower_fs_sample():
    pair = build_pair(_DRIFT)
    seed = estimate_gross_offset(pair.ref_dec, pair.mov_dec).offset_s
    for roi_s in (200.0, 600.0, 1000.0):
        anchor = estimate_anchor(
            pair.ref_dec, pair.mov_dec, pair.roi_uutc(roi_s), seed, half_window_s=20.0
        )
        gt = pair.gt_offset_s(roi_s)
        assert abs(anchor.offset_s - gt) * pair.fs_target <= 1.0
        # the emitted anchor is self-consistent: t_moving = t_ref + offset
        assert anchor.t_moving_uutc == pytest.approx(
            anchor.t_ref_uutc + anchor.offset_s * US_PER_S, abs=US_PER_S / pair.fs_target
        )
        assert anchor.t_ref_uutc == pair.roi_uutc(roi_s)
        assert anchor.confidence > 0.5


def test_multiple_rois_accumulate_independent_anchors():
    pair = build_pair(_DRIFT)
    seed = estimate_gross_offset(pair.ref_dec, pair.mov_dec).offset_s
    anchors = [
        estimate_anchor(pair.ref_dec, pair.mov_dec, pair.roi_uutc(t), seed, half_window_s=20.0)
        for t in (300.0, 900.0)
    ]
    assert anchors[0].t_ref_uutc != anchors[1].t_ref_uutc
    for a, t in zip(anchors, (300.0, 900.0), strict=False):
        assert abs(a.offset_s - pair.gt_offset_s(t)) * pair.fs_target <= 1.0


def test_rejects_a_window_that_is_mostly_gap():
    """A window over a recording gap must be refused, not correlated against interpolated
    zeros. Previously one NaN sample made the lag NaN, which a JSON encoder writes
    as the bare token `NaN` — not valid JSON — and a front end reading it died on a parse
    error that mentioned neither gaps nor drift."""
    pair = build_pair(_DRIFT)
    gappy = pair.mov_dec.data.copy()
    gappy[:] = np.nan  # the whole moving trace is a gap
    mov = Decimated(gappy, pair.mov_dec.fs, pair.mov_dec.start_uutc)

    with pytest.raises(GappyWindow, match="gap"):
        estimate_anchor(pair.ref_dec, mov, pair.roi_uutc(600.0), 120.0, half_window_s=60.0)


def test_requires_same_fs():
    pair = build_pair(_DRIFT, duration_s=300.0)
    mismatched = Decimated(pair.mov_dec.data, pair.mov_dec.fs * 2, pair.mov_dec.start_uutc)
    with pytest.raises(ValueError, match="harmonized"):
        estimate_anchor(pair.ref_dec, mismatched, pair.roi_uutc(150.0), 120.0, half_window_s=20.0)


def test_seed_is_what_the_caller_passes_not_a_stale_offset():
    """Stage 2 refines from the seed it is GIVEN, and its search cannot reach past its own
    seed's error.

    This is the property an earlier front end violated: its refine request sent no seed, so the
    engine fell back to the stored Stage 1 offset. A user who dragged the traces into alignment and hit
    Refine had that alignment silently ignored — the search started from the old automatic value
    and, being bounded, could not walk back to the truth. It returned a confidently wrong anchor
    seconds away. The fix is in the client (it now seeds from the offset on screen at the ROI);
    this test pins the estimator's half of the contract.
    """
    pair = build_pair(_DRIFT)
    roi = pair.roi_uutc(600.0)
    truth = pair.gt_offset_s(600.0)

    good = estimate_anchor(pair.ref_dec, pair.mov_dec, roi, truth + 0.1, max_residual_s=0.5)
    assert abs(good.offset_s - truth) * pair.fs_target <= 1.0

    # a seed further away than the search range cannot be recovered from — the anchor inherits
    # the error rather than correcting it, which is exactly why the seed must be the user's
    stale = estimate_anchor(pair.ref_dec, pair.mov_dec, roi, truth + 3.0, max_residual_s=0.5)
    assert abs(stale.offset_s - truth) > 1.0


def test_filters_travel_with_the_call():
    """Filters are chosen PER CALCULATION, not per session.

    A notch on the stim rate may be what makes one ROI alignable and pointless on the next, so
    the spec travels with the request. This pins the plumbing — that a filter set actually
    reaches the estimator and changes what it correlates. It deliberately does NOT assert that
    every band gives a good answer: a badly chosen band gives a bad anchor, which is exactly why
    the choice belongs to the person looking at the signal.
    """
    from mef3io_coregistration.align.conditioning import FilterSpec

    pair = build_pair(_DRIFT)
    roi = pair.roi_uutc(600.0)
    truth = pair.gt_offset_s(600.0)
    seed = truth + 0.1

    # sensible bands recover the truth
    for f in (
        FilterSpec(highpass_hz=0.5, lowpass_hz=20.0),
        FilterSpec(highpass_hz=1.0, lowpass_hz=20.0, notch_hz=60.0),
        FilterSpec(highpass_hz=0.5, lowpass_hz=20.0, clip_mad=3.0),
    ):
        a = estimate_anchor(
            pair.ref_dec, pair.mov_dec, roi, seed, max_residual_s=0.5, filters=f
        )
        assert abs(a.offset_s - truth) * pair.fs_target <= 2.0, f

    # and the filters really are being applied: a crushingly narrow band cannot resolve the lag
    # that the default band resolves to within a sample
    narrow = estimate_anchor(
        pair.ref_dec, pair.mov_dec, roi, seed, max_residual_s=0.5,
        filters=FilterSpec(highpass_hz=0.5, lowpass_hz=1.0),
    )
    default = estimate_anchor(pair.ref_dec, pair.mov_dec, roi, seed, max_residual_s=0.5)
    assert abs(narrow.offset_s - default.offset_s) > 1.0 / pair.fs_target


def test_roi_outside_a_recording_is_refused_not_silently_clamped():
    """An ROI that does not fit inside BOTH recordings is refused (a real failure: an anchor 3588 s off).

    The two windows must sit at the same position relative to their own centres, because the
    caller reads the offset back out as `(b_start - a_start) + lag`. Slicing them independently
    let one clamp at a recording's edge while the other did not — and the clamping asymmetry came
    back as a lag. Seeded at -67.5 s with a +/-0.5 s search, Stage 2 returned an anchor at
    -3588 s: a bounded search cannot produce an unbounded error, so the error was in the windows.
    """
    pair = build_pair(_DRIFT)
    ref_start = pair.start_uutc

    # an ROI a long way before the reference recording even begins
    with pytest.raises(GappyWindow, match="does not fit inside both"):
        estimate_anchor(
            pair.ref_dec, pair.mov_dec,
            ref_start - int(3600 * US_PER_S), 120.0, half_window_s=60.0,
        )

    # ...and one comfortably inside still works, shrinking rather than failing near the edge
    a = estimate_anchor(
        pair.ref_dec, pair.mov_dec, pair.roi_uutc(600.0),
        pair.gt_offset_s(600.0), half_window_s=60.0,
    )
    assert abs(a.offset_s - pair.gt_offset_s(600.0)) * pair.fs_target <= 1.0


def test_local_seed_recovers_a_seed_that_is_seconds_off():
    """The Stage 1 lock is one offset for the whole recording; over a day it can be seconds off
    at a given ROI. local_seed refines it from the neighbourhood before Stage 2 narrows in."""
    from mef3io_coregistration.align.fine import local_seed

    pair = build_pair(_DRIFT, duration_s=1800.0)
    roi = pair.roi_uutc(900.0)
    truth = pair.gt_offset_s(900.0)
    ls = local_seed(pair.ref_dec, pair.mov_dec, roi, truth + 4.0)  # 4 s off, far outside ±0.5
    assert ls.refined and abs(ls.offset_s - truth) * pair.fs_target < 2.0
    anchor = estimate_anchor(pair.ref_dec, pair.mov_dec, roi, ls.offset_s, half_window_s=150.0)
    assert abs(anchor.offset_s - truth) * pair.fs_target <= 1.0


def test_local_seed_keeps_the_seed_without_a_trustworthy_peak():
    from mef3io_coregistration.align.fine import local_seed

    pair = build_pair(_DRIFT, duration_s=1800.0)
    noise = Decimated(
        np.random.default_rng(0).standard_normal(pair.mov_dec.data.size),
        pair.mov_dec.fs,
        pair.mov_dec.start_uutc,
    )
    ls = local_seed(pair.ref_dec, noise, pair.roi_uutc(900.0), 123.0)
    assert ls.offset_s == 123.0 or not ls.refined
