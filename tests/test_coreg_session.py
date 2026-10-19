"""Tests for the CoregSession orchestration.

End-to-end recover-known-drift through the session, product reuse vs. distinct loads (R11.5),
in-memory products held until close (R11.1/R11.4), and the emitted result driving the reader.
Uses ``background=False`` for deterministic in-process loading (the multiprocessing paths are
covered in test_loader.py).
"""
from __future__ import annotations

import numpy as np
import pytest
from _synth import write_pair

from mef3io_coregistration.align.gross import GrossOffset
from mef3io_coregistration.config import US_PER_S
from mef3io_coregistration.session import CoregResult, CoregSession
from mef3io_coregistration.session.coreg_session import NotLocked
from mef3io_coregistration.synthetic.warp import WarpSpec

_DRIFT = WarpSpec("linear", offset0_s=120.0, rate_s_per_s=5.0 / (12 * 3600))  # ±5 s / 12 h


def _session(tmp_path, channels=("eeg",)):
    pair = write_pair(tmp_path, _DRIFT, channels=channels)
    sess = CoregSession(background=False, chunk_s=120.0).open(
        pair.reference_path, pair.moving_path, reference="A"
    )
    return sess, pair


def test_end_to_end_recovers_drift_and_reader_reconstructs(tmp_path):
    sess, pair = _session(tmp_path)
    sess.set_channels("gross", "eeg")
    sess.set_channels("fine", "eeg")

    gross = sess.run_gross()
    assert abs(gross.offset_s - pair.gt_offset_s(600.0)) <= 1.0

    anchor = sess.run_fine(pair.roi_uutc(600.0))
    assert abs(anchor.offset_s - pair.gt_offset_s(600.0)) * 256.0 <= 1.0

    drift = sess.run_drift()
    t_rel = (drift.t_ref_uutc - pair.start_uutc) / US_PER_S
    gt = np.asarray(_DRIFT.offset_s(t_rel))
    assert np.sqrt(np.mean((drift.smoothed_offset_s - gt) ** 2)) * 256.0 < 1.0
    assert np.all(np.diff(drift.clock_map.y_moving) > 0)

    # the emitted result drives the runtime reader to reconstruct the reference from the moving
    result = sess.result()
    # provenance: the version of the engine that aligned it
    import mef3io_coregistration

    assert result.tool_version == mef3io_coregistration.__version__
    reader = sess.open_reader()
    t0, t1 = pair.roi_uutc(500.0), pair.roi_uutc(560.0)
    coreg = reader.get_data("eeg", t0, t1, file="B", coregister=True)
    ref = reader.get_data("eeg", t0, t1, file="A", coregister=False)
    core = slice(int(0.05 * coreg.n), int(0.95 * coreg.n))
    ok = ~np.isnan(coreg.data[core])
    assert np.corrcoef(coreg.data[core][ok], ref.data[core][ok])[0, 1] > 0.99
    reader.close()

    assert result.clock_map is sess.clock_map
    assert result.stage_channels["gross"] == {"reference": "eeg", "moving": "eeg"}
    assert len(result.anchors) == 1


def test_matching_channels_reuse_one_load(tmp_path):
    """Fine channels equal to gross channels reuse the loaded products (R11.5): only the two
    files' products are held, not four."""
    sess, _pair = _session(tmp_path)
    sess.set_channels("gross", "eeg")
    sess.set_channels("fine", "eeg")
    sess.load("gross")
    sess.load("fine")
    assert len(sess._products) == 2  # reference + moving, reused across the two stages


def test_distinct_channels_load_separately(tmp_path):
    """Different fine channels load their own products (R11.5): four products in memory."""
    sess, _pair = _session(tmp_path, channels=("eeg1", "eeg2"))
    sess.set_channels("gross", "eeg1")
    sess.set_channels("fine", "eeg2")
    sess.load("gross")
    sess.load("fine")
    assert len(sess._products) == 4


def test_close_releases_products(tmp_path):
    sess, _pair = _session(tmp_path)
    sess.set_channels("gross", "eeg")
    sess.load("gross")
    assert len(sess._products) > 0
    sess.close()
    assert len(sess._products) == 0


def test_result_roundtrips_from_session(tmp_path):
    sess, pair = _session(tmp_path)
    sess.set_channels("gross", "eeg")
    sess.set_channels("fine", "eeg")
    sess.run_gross()
    sess.run_fine(pair.roi_uutc(600.0))
    sess.run_drift()

    path = sess.result().save(tmp_path / "res.json")
    reloaded = CoregResult.load(path)
    q = np.linspace(*sess.clock_map.domain, 200).astype(np.int64)
    np.testing.assert_array_equal(
        np.asarray(sess.clock_map.forward(q)), np.asarray(reloaded.clock_map.forward(q))
    )
    assert reloaded.drift is not None and len(reloaded.drift["confidence"]) > 0


def test_errors_when_stage_prerequisites_missing(tmp_path):
    sess, pair = _session(tmp_path)
    sess.set_channels("fine", "eeg")
    with pytest.raises(RuntimeError, match="Stage 1"):
        sess.run_fine(pair.roi_uutc(600.0))  # no gross seed
    with pytest.raises(RuntimeError, match="clock map"):
        sess.result()  # no clock map yet


def test_refuses_to_build_on_a_failed_lock(tmp_path):
    """Stage 2 must not run on an unlocked Stage 1 (REVIEW B1). The old pipeline reported
    confidence 0.000 and happily refined a three-hour error into an anchor."""
    sess, pair = _session(tmp_path)
    sess.set_channels("gross", "eeg")
    sess.set_channels("fine", "eeg")

    sess.gross_offset = GrossOffset(
        offset_s=9999.0, confidence=0.0, locked=False, spread_s=0.0,
        n_agree=0, n_templates=4, reason="NO LOCK — templates disagree",
    )
    with pytest.raises(NotLocked, match="NO LOCK"):
        sess.run_fine(pair.roi_uutc(600.0))

    # ...but a human who has looked at the overlay may place the offset by hand (R4.3/R9.1),
    # and that IS a lock — the escape hatch the tool exists to provide.
    sess.manual_offset(pair.gt_offset_s(600.0))
    anchor = sess.run_fine(pair.roi_uutc(600.0))
    assert abs(anchor.offset_s - pair.gt_offset_s(600.0)) * 256.0 <= 1.0


def test_an_unlocked_estimate_can_be_accepted_and_is_recorded(tmp_path):
    """Never a dead end: stimulation can leave too few clean templates to corroborate a right
    offset. The user may continue with the estimate; that decision is recorded and reported."""
    sess, pair = _session(tmp_path)
    sess.set_channels("gross", "eeg")
    sess.set_channels("fine", "eeg")
    truth = pair.gt_offset_s(600.0)
    sess.gross_offset = GrossOffset(
        offset_s=truth + 0.3, confidence=0.5, locked=False, spread_s=0.0,
        n_agree=1, n_templates=4, reason="NO LOCK — templates disagree",
    )
    g = sess.accept_gross()
    assert g.locked and g.accepted_by == "user" and g.offset_s == truth + 0.3
    anchor = sess.run_fine(pair.roi_uutc(600.0))  # Stage 2 corrects the 0.3 s
    assert abs(anchor.offset_s - truth) * 256.0 <= 1.0
    sess.build_map_without_drift()
    assert sess.result().qc["stage1_accepted_by"] == "user"


def test_per_file_channels_and_polarity(tmp_path):
    """Channels are selected INDEPENDENTLY per file, and swapping a bipolar pair on one file is
    exactly a polarity inversion (R3.2/R3.3).

    This is what makes the tool usable when the two systems reference their channels in opposite
    senses — the same physiology arrives inverted — and when they simply do not name or carry the
    same electrodes. `C1-C2` on one file against `C2-C1` on the other IS the x-1, so no special
    handling is needed anywhere in the estimators; `invert` exists only for single-channel
    selections, where there is no pair to swap.
    """
    pair = write_pair(tmp_path, _DRIFT, channels=("a", "b"))

    with CoregSession(background=False) as sess:
        sess.open(pair.reference_path, pair.moving_path, reference="A")
        # different derivations per file: a-b on the reference, b-a on the moving
        sess.set_channels("gross", reference=("a", "b"), moving=("b", "a"))
        swapped = sess.load("gross")["moving"].decimated.data

        # ...and the same thing expressed with the invert flag instead
        sess.set_channels("gross", reference=("a", "b"), moving=("a", "b"), invert_moving=True)
        inverted = sess.load("gross")["moving"].decimated.data

    n = min(swapped.size, inverted.size)
    np.testing.assert_allclose(swapped[:n], inverted[:n], atol=1e-9)


def test_manual_offset_uses_the_offset_the_user_placed(tmp_path):
    """A hand-placed offset is ACCEPTED OR REFUSED — never silently moved.

    Verification decides whether the shared signal supports the placement. It does not get to
    replace it. An earlier version adopted the templates' own estimate whenever the two were
    within the tolerance window, which meant a user who deliberately hand-aligned watched the
    tool discard their placement and put the automatic answer back the moment they hit Accept.
    """
    sess, pair = _session(tmp_path)
    sess.set_channels("gross", "eeg")
    sess.set_channels("fine", "eeg")

    truth = pair.gt_offset_s(600.0)
    placed = truth + 0.40  # a deliberate hand placement, near the automatic answer but not on it

    g = sess.manual_offset(placed)
    assert g.locked
    assert g.offset_s == pytest.approx(placed, abs=1e-9), "the user's placement was overwritten"

    # the templates' own estimate is reported, for the user to adopt if they choose
    assert sess.offset_verification is not None
    assert abs(sess.offset_verification.refined_offset_s - truth) < 0.5

    # ...and the stage that follows is seeded from the USER's offset
    anchor = sess.run_fine(pair.roi_uutc(600.0))
    assert abs(anchor.offset_s - truth) * 256.0 <= 1.0


def test_map_without_drift_from_a_constant_offset(tmp_path):
    """Stage 3 can be bypassed and still yield a real, usable, verified clock map (R6, R7).

    On a short recording — or a short overlap — the clocks have no room to drift, and running a
    sliding-window estimator anyway does not measure drift: it fits the estimator's own scatter
    and presents the wiggle as a clock. The bypass builds the map from what is actually known and
    records the assumption it made.
    """
    sess, pair = _session(tmp_path)
    sess.set_channels("gross", "eeg")
    sess.set_channels("fine", "eeg")
    sess.run_gross()

    cmap = sess.build_map_without_drift()
    assert sess.drift_estimate is None  # no windows were invented
    assert cmap.metadata["stage"] == "no-drift"
    assert "constant offset" in cmap.metadata["method"]

    # the map is monotone, spans the recording, and holds the offset it was given
    assert np.all(np.diff(cmap.y_moving) > 0)
    t = pair.roi_uutc(600.0)
    assert abs(cmap.offset_s(t) - pair.gt_offset_s(600.0)) <= 1.0

    # and it drives the runtime reader like any other map
    result = sess.result()
    assert result.drift is None
    with sess.open_reader() as reader:
        got = reader.get_data("eeg", pair.roi_uutc(500.0), pair.roi_uutc(520.0),
                              file="B", coregister=True)
    assert got.n > 0 and np.isfinite(got.data).any()


def test_map_without_drift_uses_the_anchors_when_there_are_several(tmp_path):
    """Two or more anchors give a straight line THROUGH them — which absorbs a constant
    difference in clock RATE without ever estimating drift."""
    sess, pair = _session(tmp_path)
    sess.set_channels("gross", "eeg")
    sess.set_channels("fine", "eeg")
    sess.run_gross()
    for t in (300.0, 900.0):
        sess.run_fine(pair.roi_uutc(t))

    cmap = sess.build_map_without_drift()
    assert "anchors" in cmap.metadata["method"]
    assert cmap.metadata["n_anchors"] == 2

    # the line passes through the anchors, so the drifting truth is tracked between them
    for t in (300.0, 600.0, 900.0):
        assert abs(cmap.offset_s(pair.roi_uutc(t)) - pair.gt_offset_s(t)) * 256.0 < 2.0
