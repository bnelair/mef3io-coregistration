"""Cost surface (Stage-2 correlation curve) and the Hilbert-envelope alignment feature.

The cost curve must be the *same evidence* the anchor is picked from — its argmax is the anchor
Refine returns. The envelope feature must recover the offset on real-shaped signal and, unlike the
waveform, be blind to a polarity flip (a bipolar pair swapped on one file).
"""
from __future__ import annotations

import numpy as np
import pytest
from _synth import build_pair

from mef3io_coregistration.align.conditioning import FilterSpec
from mef3io_coregistration.align.fine import cost_surface, estimate_anchor
from mef3io_coregistration.align.tde import normxcorr_delay, normxcorr_surface
from mef3io_coregistration.io.derived import Decimated

_DRIFT = None  # constant offset is enough here; drift is exercised elsewhere


def _pair():
    from mef3io_coregistration.synthetic.warp import WarpSpec

    return build_pair(WarpSpec("constant", offset0_s=120.0))


# --------------------------------------------------------------------------- cost surface --- #
def test_cost_curve_argmax_is_the_anchor():
    """The curve is the evidence: its peak must equal the anchor Refine would emit."""
    pair = _pair()
    roi = pair.roi_uutc(600.0)
    seed = pair.gt_offset_s(600.0)
    # Same search width, so the curve's peak_quality matches the anchor's exactly.
    anchor = estimate_anchor(
        pair.ref_dec, pair.mov_dec, roi, seed, half_window_s=20.0, max_residual_s=0.25
    )
    curve = cost_surface(
        pair.ref_dec, pair.mov_dec, roi, seed, half_window_s=20.0, max_residual_s=0.25
    )

    assert curve.best_offset_s == pytest.approx(anchor.offset_s, abs=1.0 / pair.fs_target)
    assert curve.confidence == pytest.approx(anchor.confidence, abs=1e-9)


def test_cost_curve_arrays_align_and_span_the_search():
    pair = _pair()
    roi = pair.roi_uutc(600.0)
    seed = pair.gt_offset_s(600.0)
    curve = cost_surface(
        pair.ref_dec, pair.mov_dec, roi, seed, half_window_s=20.0, max_residual_s=0.25
    )
    assert len(curve.offset_s) == len(curve.corr)
    assert len(curve.offset_s) > 3
    # corr is max-normalised for display; anti-correlation lobes may dip negative (down to -1)
    assert max(curve.corr) == pytest.approx(1.0, abs=1e-6)
    assert min(curve.corr) >= -1.0001
    # the search spans seed ± max_residual, so the seed sits inside the offset axis
    assert min(curve.offset_s) <= seed <= max(curve.offset_s)
    # the peak is inside the searched band
    assert min(curve.offset_s) <= curve.best_offset_s <= max(curve.offset_s)


def test_cost_curve_step_decimates_without_moving_the_peak():
    pair = _pair()
    roi = pair.roi_uutc(600.0)
    seed = pair.gt_offset_s(600.0)
    full = cost_surface(pair.ref_dec, pair.mov_dec, roi, seed, half_window_s=20.0)
    coarse = cost_surface(pair.ref_dec, pair.mov_dec, roi, seed, half_window_s=20.0, step=4)
    assert len(coarse.offset_s) < len(full.offset_s)
    # sub-sample peak is reported at full resolution regardless of the display step
    assert coarse.best_offset_s == pytest.approx(full.best_offset_s, abs=1e-6)


def test_normxcorr_surface_matches_normxcorr_delay():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(2048)
    b = np.roll(a, 7) + 0.05 * rng.standard_normal(2048)
    fs = 256.0
    d = normxcorr_delay(a, b, fs, max_lag_s=0.25)
    s = normxcorr_surface(a, b, fs, max_lag_s=0.25)
    assert s.best_lag_samples == pytest.approx(d.lag_samples, abs=1e-9)
    assert s.confidence == pytest.approx(d.confidence, abs=1e-9)
    assert s.values.shape == s.lags_samples.shape


# --------------------------------------------------------------------------- envelope --- #
def test_envelope_feature_recovers_the_offset():
    """Aligning on the amplitude envelope must still land within a sample of the truth."""
    pair = _pair()
    roi = pair.roi_uutc(600.0)
    seed = pair.gt_offset_s(600.0)
    env = FilterSpec(feature="envelope")
    anchor = estimate_anchor(
        pair.ref_dec, pair.mov_dec, roi, seed, half_window_s=20.0, filters=env
    )
    assert abs(anchor.offset_s - pair.gt_offset_s(600.0)) * pair.fs_target <= 1.5


def test_envelope_is_polarity_blind_where_waveform_is_not():
    """Flip the moving trace's sign (a swapped bipolar pair). The waveform correlation is wrecked
    by the inversion; the envelope is not, because it discards carrier phase."""
    pair = _pair()
    roi = pair.roi_uutc(600.0)
    seed = pair.gt_offset_s(600.0)
    flipped = Decimated(-pair.mov_dec.data, pair.mov_dec.fs, pair.mov_dec.start_uutc)

    wave = estimate_anchor(pair.ref_dec, flipped, roi, seed, half_window_s=20.0)
    env = estimate_anchor(
        pair.ref_dec, flipped, roi, seed, half_window_s=20.0, filters=FilterSpec(feature="envelope")
    )
    gt = pair.gt_offset_s(600.0)
    # envelope survives the flip to within a couple of samples
    assert abs(env.offset_s - gt) * pair.fs_target <= 2.0
    # waveform does not (anti-correlated peak lands elsewhere or is far less confident)
    assert abs(wave.offset_s - gt) * pair.fs_target > abs(env.offset_s - gt) * pair.fs_target
