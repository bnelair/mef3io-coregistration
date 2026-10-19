"""Zero-phase filters shared by the reader and the alignment pipeline.

Part of the reader package: the unified recording compares its two sources band-passed with
these, and the alignment pipeline uses the same ones — so what the reader calls "agree" is what
alignment saw.
Zero-phase throughout: a causal filter's group delay would shift a trace and bias every lag.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, sosfiltfilt, tf2sos


@lru_cache(maxsize=64)
def _design(
    fs: float,
    hp: float | None,
    lp: float | None,
    notch: float | None,
    q: float,
) -> tuple[np.ndarray, ...]:
    """Design the filter cascade. Cached, so dragging a filter slider does not redesign it every
    frame — the design is the expensive part; applying it to a screenful is not."""
    nyq = fs / 2.0
    stages: list[np.ndarray] = []

    if hp and lp and 0 < hp < lp < nyq:
        stages.append(butter(2, (hp, lp), btype="bandpass", fs=fs, output="sos"))
    else:
        if hp and 0 < hp < nyq:
            stages.append(butter(2, hp, btype="highpass", fs=fs, output="sos"))
        if lp and 0 < lp < nyq:
            stages.append(butter(2, lp, btype="lowpass", fs=fs, output="sos"))

    # Band-stop: mains and its harmonics, or a stimulation rate. Every harmonic below Nyquist is
    # notched — a stim comb is not a single line, and leaving the harmonics in leaves the artifact.
    if notch and 0 < notch < nyq:
        f = notch
        while f < nyq * 0.98:
            b, a = iirnotch(f, q, fs=fs)
            stages.append(tf2sos(b, a))
            f += notch

    return tuple(stages)


def _filter(
    x: np.ndarray,
    fs: float,
    highpass_hz: float | None,
    lowpass_hz: float | None,
    notch_hz: float | None = None,
    notch_q: float = 30.0,
) -> np.ndarray:
    """Zero-phase filter cascade. Zero-phase matters: a causal filter's group delay would shift
    the trace and bias every lag estimate the tool produces."""
    stages = _design(float(fs), highpass_hz, lowpass_hz, notch_hz, float(notch_q))
    if not stages:
        return x - x.mean()

    out = x
    for sos in stages:
        # filtfilt needs padlen samples of runway; a window shorter than that is left alone
        # rather than filtered into nonsense.
        padlen = 3 * (sos.shape[0] * 2)
        if out.shape[0] <= padlen:
            return out - out.mean()
        out = _sosfiltfilt(sos, out)
    return out


_ZI: dict[bytes, np.ndarray] = {}


def _sosfiltfilt(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``scipy.signal.sosfiltfilt(sos, x)`` (odd padding, default pad length) with the filter's
    initial-state solution cached.

    Numerically the same operation; the difference is speed when many short windows go through
    identical filters: SciPy re-solves the initial-state equations (``sosfilt_zi``) on every call.
    """
    n_sections = sos.shape[0]
    ntaps = 2 * n_sections + 1 - min(int((sos[:, 2] == 0).sum()), int((sos[:, 5] == 0).sum()))
    edge = 3 * ntaps
    if x.ndim != 1 or x.shape[0] <= edge:
        return np.asarray(sosfiltfilt(sos, x), dtype=np.float64)
    key = sos.tobytes()
    zi = _ZI.get(key)
    if zi is None:
        zi = _ZI[key] = sosfilt_zi(sos)
    ext = np.concatenate((2.0 * x[0] - x[edge:0:-1], x, 2.0 * x[-1] - x[-2 : -(edge + 2) : -1]))
    y, _ = sosfilt(sos, ext, zi=zi * ext[0])
    y, _ = sosfilt(sos, y[::-1], zi=zi * y[-1])
    return np.asarray(y[::-1][edge:-edge], dtype=np.float64)
