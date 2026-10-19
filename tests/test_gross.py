"""Tests for Stage 1 — gross alignment.

Stage 1 correlates the **decimated waveform**, not the min-max envelope: the envelope
summarises each file's raw signal at that file's own bandwidth, so it is not a shared physical
quantity across two amplifiers and cannot align them (see align/gross.py, REVIEW B1).

R4.1 full-length correlation; R4.2 absorbs offsets >= 10 minutes; recovered offset within
±1 s. Plus the property the old design lacked: Stage 1 must *know when it has not locked*.
"""
from __future__ import annotations

import numpy as np
import pytest
from _synth import build_pair

from mef3io_coregistration.align import estimate_gross_offset
from mef3io_coregistration.align.gross import verify_offset
from mef3io_coregistration.config import GROSS_MIN_SNR
from mef3io_coregistration.io.derived import Decimated
from mef3io_coregistration.synthetic.warp import WarpSpec


@pytest.mark.parametrize("offset0", [120.0, 600.0])  # 2 min and 10 min (R4.2)
def test_recovers_constant_offset_within_one_second(offset0):
    pair = build_pair(WarpSpec("constant", offset0_s=offset0), duration_s=900.0)
    g = estimate_gross_offset(pair.ref_dec, pair.mov_dec)
    assert abs(g.offset_s - offset0) <= 1.0
    assert g.locked
    assert g.n_agree >= 2


def test_recovers_offset_with_small_drift():
    spec = WarpSpec("linear", offset0_s=600.0, rate_s_per_s=5.0 / (12 * 3600))  # 10 min + drift
    pair = build_pair(spec, duration_s=900.0)
    g = estimate_gross_offset(pair.ref_dec, pair.mov_dec)
    assert abs(g.offset_s - pair.gt_offset_s(pair.duration_s / 2)) <= 1.0
    assert g.locked


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_reports_no_lock_on_unrelated_signals(seed):
    """The property that matters most (REVIEW B1): when the two recordings share no signal,
    Stage 1 must say so rather than return the argmax of a flat correlation surface. The old
    implementation reported confidence 0.000, drew it in a stat box, marked the stage done, and
    let Stage 2 refine the garbage into an anchor."""
    pair = build_pair(WarpSpec("constant", offset0_s=120.0), duration_s=900.0)
    rng = np.random.default_rng(seed)
    noise = Decimated(
        rng.standard_normal(pair.mov_dec.data.size) * 50.0,
        pair.mov_dec.fs,
        pair.mov_dec.start_uutc,
    )
    g = estimate_gross_offset(pair.ref_dec, noise)
    assert not g.locked
    assert "NO LOCK" in g.reason


def test_peak_height_cannot_separate_true_locks_from_false_ones():
    """Peak height must never become the lock criterion, and this test is what stops it.

    Two unrelated recordings still produce a correlation peak that *looks* real by any absolute
    measure. Measured across true and deliberately false pairs (on a real clinical recording pair), the
    detection SNR of a true lock spans 3.4–22 sigma and of a false one 2.7–4.4 sigma: the
    distributions OVERLAP, so no threshold separates them — a false pair here out-scores a
    genuine synthetic lock. Only template agreement separates the cases cleanly (true 4/4,
    false 0/4), because independent templates cut from different hours cannot be fooled into the
    *same* wrong answer.
    """
    # Unrelated recordings can still share a steady rhythm (a persistent oscillation, a periodic
    # artifact), and that is what makes a false peak tall. The synthetic source's own rhythms
    # wander (narrowband noise), so the shared steady component is added explicitly here.
    true_pair = build_pair(WarpSpec("constant", offset0_s=120.0), duration_s=1200.0, seed=12)
    false_a = build_pair(WarpSpec("constant", offset0_s=120.0), duration_s=1200.0, seed=12)
    false_b = build_pair(WarpSpec("constant", offset0_s=120.0), duration_s=1200.0, seed=62)
    true_ref = _with_rhythm(true_pair.ref_dec, 0.0, phase=0.3)
    true_mov = _with_rhythm(true_pair.mov_dec, 120.0, phase=0.3)  # same rhythm, world-aligned
    false_ref = _with_rhythm(false_a.ref_dec, 0.0, phase=0.3)
    false_mov = _with_rhythm(false_b.mov_dec, 120.0, phase=2.1)  # unrelated phase

    good = estimate_gross_offset(true_ref, true_mov)
    bad = estimate_gross_offset(false_ref, false_mov)

    # The false pair's peak clears the absolute height bar a peak-height criterion would use —
    # it "looks locked" (and in real data true and false heights overlap outright)...
    assert bad.snr > 2 * GROSS_MIN_SNR
    # ...but independent templates refuse to corroborate it, and that is what decides.
    assert good.n_agree >= 2 and good.locked
    assert bad.n_agree < 2 and not bad.locked


def _with_rhythm(dec: Decimated, offset_s: float, phase: float, hz: float = 2.0) -> Decimated:
    """``dec`` plus a steady ``hz`` rhythm in world time (moving clock = world + offset)."""
    t = np.arange(dec.data.size) / dec.fs - offset_s
    amp = 0.8 * float(np.std(dec.data))
    return Decimated(dec.data + amp * np.sin(2 * np.pi * hz * t + phase), dec.fs, dec.start_uutc)


def test_verification_rejects_a_globally_wrong_offset():
    """The check that stands between a wrong seed and a confident clock map (REVIEW-2 N1).

    Stage 2/3 confidence is computed on a +/-0.25 s surface around wherever it was pointed, so it
    cannot tell a right seed from one that is an hour out — measured, it scores the wrong seed
    HIGHER. Verification asks the question those statistics cannot: do independent templates put
    their best match at this offset at all?
    """
    pair = build_pair(WarpSpec("linear", offset0_s=120.0, rate_s_per_s=5.0 / (12 * 3600)))
    truth = pair.gt_offset_s(600.0)

    ok = verify_offset(pair.ref_dec, pair.mov_dec, truth)
    assert ok.verified and ok.n_support >= 2

    for wrong in (0.0, truth + 500.0, truth - 300.0):
        bad = verify_offset(pair.ref_dec, pair.mov_dec, wrong)
        assert not bad.verified, f"a {wrong - truth:+.0f}s error was accepted"
        assert "NOT SUPPORTED" in bad.reason


def test_verification_snaps_a_sloppy_drag_onto_the_signal():
    """Verification tolerates a few seconds of slop, because the true offset genuinely drifts —
    but that slack would be a trap if the sloppy value were then handed to a Stage 2 that only
    searches +/-0.5 s. So it reports where the signal actually is, and the caller adopts it."""
    pair = build_pair(WarpSpec("linear", offset0_s=120.0, rate_s_per_s=5.0 / (12 * 3600)))
    truth = pair.gt_offset_s(600.0)

    v = verify_offset(pair.ref_dec, pair.mov_dec, truth + 3.0)
    assert v.verified
    assert abs(v.refined_offset_s - truth) < 0.5  # snapped back onto the shared signal
    assert abs(v.offset_s - (truth + 3.0)) < 1e-6  # and remembers what it was asked


def test_requires_absolute_start():
    pair = build_pair(WarpSpec("constant", offset0_s=120.0), duration_s=300.0)
    no_start = Decimated(np.zeros(1000), 256.0)  # start_uutc defaults to None
    with pytest.raises(ValueError, match="start_uutc"):
        estimate_gross_offset(no_start, pair.mov_dec)


def _stimulate(dec, blocks_s, kind):
    """Stimulation blocks (seconds from the trace start): ``"rail"`` pins the amplifier at its
    rail with a recovery kick every 0.7 s; ``"blank"`` zeroes the samples (a blanking amplifier)."""
    x = np.array(dec.data, dtype=np.float64)
    fs, sd = dec.fs, float(np.nanstd(x))
    for b0, b1 in blocks_s:
        i0, i1 = int(b0 * fs), min(int(b1 * fs), x.size)
        if kind == "blank":
            x[i0:i1] = 0.0
            continue
        per, kick = int(0.7 * fs), int(0.25 * fs)
        for j in range(i0, i1, per):
            x[j : min(j + per, i1)] = 20 * sd
            k = min(kick, i1 - j)
            x[j : j + k] = 20 * sd * (1 - 2 * np.exp(-np.arange(k) / fs / 0.05))
    return Decimated(x, fs, dec.start_uutc)


def test_locks_through_saturation_on_one_side_and_blanking_on_the_other():
    """Stimulation for 8 of every 12 minutes: one amplifier saturates, the other blanks. The
    stimulated stretches must not decide the match — they are masked as no-signal, and templates
    are correlated over their clean samples only — so every template still agrees."""
    offset0 = 300.0
    pair = build_pair(WarpSpec("constant", offset0_s=offset0), duration_s=3600.0)
    ref_blocks = [(k * 720 + 60, k * 720 + 540) for k in range(6)]
    ref = _stimulate(pair.ref_dec, ref_blocks, "rail")
    # The same moments on the moving clock (its start differs from the reference's).
    shift = offset0 - (pair.mov_dec.start_uutc - pair.ref_dec.start_uutc) / 1e6
    mov = _stimulate(pair.mov_dec, [(a + shift, b + shift) for a, b in ref_blocks], "blank")
    g = estimate_gross_offset(ref, mov)
    assert g.locked
    assert abs(g.offset_s - offset0) <= 1.0
    assert g.n_agree == g.n_templates >= 3


def test_mask_flat_marks_rail_and_blanking_but_not_physiology():
    from mef3io_coregistration.align.gross import mask_flat

    fs = 256.0
    x = np.random.default_rng(0).standard_normal(int(60 * fs))
    x[int(10 * fs) : int(20 * fs)] = 5.0   # rail
    x[int(40 * fs) : int(41 * fs)] = 0.0   # blanking
    x[int(50 * fs) : int(50 * fs) + 5] = 1.0  # 20 ms of repeats: not a flat stretch
    y = mask_flat(x, fs)
    assert np.isnan(y[int(15 * fs)]) and np.isnan(y[int(40.5 * fs)])
    assert np.isnan(y[int(9.9 * fs)])  # the padding either side
    assert np.isfinite(y[int(50 * fs)]) and np.isfinite(y[int(30 * fs)])
    assert np.isfinite(x).all()  # the input is not modified
