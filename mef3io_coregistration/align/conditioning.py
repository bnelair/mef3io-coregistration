"""Signal conditioning for the alignment path.

Every stage correlates *conditioned* traces, never raw ones. The two amplifiers do not agree
on scale, offset, or bandwidth: on the reference dataset the cadence bipolar LD1-LD3 has mean
-11,438 uV and excursions to -226,729 uV, while the same derivation on natus has mean -38 uV
and sigma 2,411 uV — a ~250x amplitude difference plus a large slow component on one amplifier
only. Correlating those directly lets a DC ramp on one system dominate the physiology on the
other.

Conditioning is four steps, in order:

1. **Gap interpolation.** ``NaN`` gap samples are linearly bridged *before* filtering so the
   IIR does not ring off a step edge, then re-zeroed afterwards. The ``valid`` mask records
   where the signal was real; correlation windows that overlap too much gap are rejected
   upstream rather than silently correlated against zeros (REVIEW B3).
2. **High-pass** (zero-phase, so no group delay is introduced — timing is the whole point).
   Removes the DC/slow drift that differs per amplifier.
3. **Robust scaling** by the MAD, not the standard deviation: a few ±200 mV stim excursions
   would set sigma and crush the physiology to nothing.
4. **Clipping** at ``clip_mad`` robust sigmas, which bounds saturation and stim artifact
   without discarding the window.

The output is dimensionless (units of robust sigma), so traces from the two systems are
directly comparable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import hilbert, sosfiltfilt

from mef3io_coregistration.config import CLIP_MAD, FINE_LOWPASS_HZ, HIGHPASS_HZ, LOWPASS_HZ
from mef3io_coregistration.transform.filters import (  # noqa: F401  (shared with the reader)
    _design,
    _filter,
    _sosfiltfilt,
)

_MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class Conditioned:
    """A conditioned trace: dimensionless, zero-mean, gap-zeroed, with a validity mask."""

    data: np.ndarray
    valid: np.ndarray  # False where the source was a gap (NaN)
    fs: float
    start_uutc: int | None = None

    @property
    def gap_fraction(self) -> float:
        if self.valid.size == 0:
            return 1.0
        return float(1.0 - self.valid.mean())


@dataclass(frozen=True)
class FilterSpec:
    """The filters applied before correlation — chosen by the user, per calculation.

    Not a session-wide mode. The right band genuinely differs from one ROI to the next: a stretch
    dominated by stim artifact may need a tight low-pass and a notch to be alignable at all, while
    the same recording an hour later aligns best wide open. So a filter set travels *with* each
    request — the view, the anchor, the drift fit — and the viewer sends the same one it is
    drawing, so what you align on is what you were looking at.

    ``notch_hz`` is a band-stop (mains and its harmonics, or a stimulation rate). Zero-phase
    throughout: a causal filter's group delay would shift the trace and bias every lag estimate.

    ``feature`` chooses what the correlation actually runs on:

    * ``"waveform"`` — the conditioned band-limited signal (the default).
    * ``"envelope"`` — its Hilbert amplitude envelope. Two amplifiers with different analog/digital
      filter chains give a broadband transient a *different shape* (their group delays and ringing
      differ), which biases a waveform cross-correlation of that transient. The amplitude envelope
      largely discards carrier phase, so it is far less sensitive to that mismatch — and, as a
      bonus, blind to polarity (a bipolar pair swapped C1-C3 vs C3-C1). It trades a little timing
      sharpness for robustness. This is the envelope of the **common-band** signal, computed on
      both traces identically; it is NOT the per-file min-max display envelope, which is not an
      alignment feature.
    """

    highpass_hz: float | None = HIGHPASS_HZ
    lowpass_hz: float | None = FINE_LOWPASS_HZ
    notch_hz: float | None = None
    notch_q: float = 30.0
    clip_mad: float = CLIP_MAD
    feature: str = "waveform"

    @property
    def key(self) -> tuple:
        return (self.highpass_hz, self.lowpass_hz, self.notch_hz, self.notch_q,
                self.clip_mad, self.feature)


DETECTION_FILTERS = FilterSpec(highpass_hz=HIGHPASS_HZ, lowpass_hz=LOWPASS_HZ)
ESTIMATION_FILTERS = FilterSpec(highpass_hz=HIGHPASS_HZ, lowpass_hz=FINE_LOWPASS_HZ)


def condition(
    data: np.ndarray,
    fs: float,
    start_uutc: int | None = None,
    highpass_hz: float | None = HIGHPASS_HZ,
    lowpass_hz: float | None = LOWPASS_HZ,
    clip_mad: float = CLIP_MAD,
    notch_hz: float | None = None,
    notch_q: float = 30.0,
    feature: str = "waveform",
) -> Conditioned:
    """Condition a trace for correlation: bridge gaps, band-pass, robust-scale, clip.

    ``highpass_hz`` / ``lowpass_hz`` of 0 or ``None`` disable that end of the band. The
    returned data has gap samples set to exactly 0 (the conditioned signal's mean), so they
    contribute nothing to a correlation numerator; ``valid`` says where that happened.

    With ``feature="envelope"`` the band-limited signal is replaced by its Hilbert amplitude
    envelope (centred) before scaling, so the correlation runs on amplitude structure rather than
    phase — robust to the per-amplifier transient-shape mismatch. See :class:`FilterSpec`.
    """
    x = np.asarray(data, dtype=np.float64)
    valid = np.isfinite(x)

    if not valid.any():
        return Conditioned(np.zeros_like(x), valid, fs, start_uutc)

    x = _bridge_gaps(x, valid)
    x = _filter(x, fs, highpass_hz, lowpass_hz, notch_hz, notch_q)
    if feature == "envelope":
        x = _envelope(x)

    scale = _robust_scale(x[valid])
    x = np.clip(x / scale, -clip_mad, clip_mad)
    x[~valid] = 0.0
    return Conditioned(x, valid, fs, start_uutc)


def _envelope(x: np.ndarray) -> np.ndarray:
    """Centred Hilbert amplitude envelope of a (already band-limited) trace.

    The magnitude of the analytic signal is the instantaneous amplitude, which is non-negative
    and phase-blind. We subtract its median so the result is zero-centred like the waveform path
    (the clip that follows is symmetric) and so a slowly-varying background amplitude does not
    dominate the correlation over the discrete events that carry the timing.
    """
    if x.shape[0] < 4:
        return x - x.mean()
    env = np.abs(hilbert(x))
    return env - np.median(env)


def _bandpass(
    x: np.ndarray, fs: float, highpass_hz: float | None, lowpass_hz: float | None
) -> np.ndarray:
    """Back-compat shim for callers that only want a band-pass."""
    return _filter(x, fs, highpass_hz, lowpass_hz)


def condition_pair(
    a: np.ndarray, b: np.ndarray, fs: float, filters: FilterSpec | None = None
) -> tuple[Conditioned, Conditioned]:
    """Condition a reference/moving window pair with ``filters`` (default: the estimation band).

    BOTH traces get the SAME filters. That is the whole point: the two systems disagree about
    scale, offset and bandwidth, and correlation is only meaningful once they have been put on
    common terms. Filtering one and not the other would introduce a phase difference that is not
    in the data.
    """
    f = filters or ESTIMATION_FILTERS
    return (
        condition(a, fs, highpass_hz=f.highpass_hz, lowpass_hz=f.lowpass_hz,
                  clip_mad=f.clip_mad, notch_hz=f.notch_hz, notch_q=f.notch_q, feature=f.feature),
        condition(b, fs, highpass_hz=f.highpass_hz, lowpass_hz=f.lowpass_hz,
                  clip_mad=f.clip_mad, notch_hz=f.notch_hz, notch_q=f.notch_q, feature=f.feature),
    )


def condition_rows(x: np.ndarray, fs: float, filters: FilterSpec | None = None) -> np.ndarray:
    """:func:`condition` applied to every row of ``x`` (equal-length, gap-free windows) at once.

    The same steps, vectorised across rows — filter cascade, optional envelope, per-row robust
    scale, clip — so Stage 3 can condition a block of windows in one pass instead of paying
    Python overhead window by window. Rows with gaps must go through :func:`condition`.
    """
    f = filters or ESTIMATION_FILTERS
    x = np.asarray(x, dtype=np.float64)
    stages = _design(float(fs), f.highpass_hz, f.lowpass_hz, f.notch_hz, float(f.notch_q))
    out = x.copy()
    if not stages:
        out = out - out.mean(axis=1, keepdims=True)
    for sos in stages:
        if out.shape[1] <= 3 * (sos.shape[0] * 2):
            out = out - out.mean(axis=1, keepdims=True)
            break
        out = np.asarray(sosfiltfilt(sos, out, axis=1), dtype=np.float64)
    if f.feature == "envelope":
        if out.shape[1] < 4:
            out = out - out.mean(axis=1, keepdims=True)
        else:
            env = np.abs(hilbert(out, axis=1))
            out = env - np.median(env, axis=1, keepdims=True)
    med = np.median(out, axis=1, keepdims=True)
    scale = _MAD_TO_SIGMA * np.median(np.abs(out - med), axis=1, keepdims=True)
    flat = scale <= 0.0
    if flat.any():
        scale = np.where(flat, np.std(out, axis=1, keepdims=True), scale)
    scale = np.where(scale > 0.0, scale, 1.0)
    return np.clip(out / scale, -f.clip_mad, f.clip_mad)


def _bridge_gaps(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Linearly interpolate across NaN runs so the filter sees no step edges."""
    if valid.all():
        return x.copy()
    idx = np.arange(x.shape[0], dtype=np.float64)
    out = x.copy()
    out[~valid] = np.interp(idx[~valid], idx[valid], x[valid])
    return out


def _robust_scale(x: np.ndarray) -> float:
    """Median absolute deviation as a sigma estimate; robust to stim/saturation outliers."""
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    scale = _MAD_TO_SIGMA * mad
    if scale <= 0.0:  # flat or near-flat channel
        scale = float(np.std(x))
    return scale if scale > 0.0 else 1.0
