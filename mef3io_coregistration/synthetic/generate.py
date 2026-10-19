"""Render a reference/moving recording pair from a source signal + a WarpSpec.

Generative model (see ``warp.py`` for the offset convention):

* A single band-limited "world" signal ``s(tau)`` is created at a high master rate.
* The **reference** recording samples ``s`` on a uniform grid in world time
  (reference clock == world time; it is never transformed).
* The **moving** recording samples uniformly in its OWN clock ``m = tau + offset(tau)``.
  We build its uniform clock grid, invert ``m(tau)`` to recover the world time of each
  moving sample, and read ``s`` there.

The source is band-limited below the lower Nyquist so both sampling rates capture the
same content — this makes the harness self-consistency test (map moving back onto the
reference and compare) essentially exact, isolating the geometry under test from
aliasing artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mef3io_coregistration.config import US_PER_S

from .warp import WarpSpec

_PAD_S = 2.0  # small margin so edge interpolation never clamps within [0, duration]


def make_source(
    duration_s: float,
    fs: float,
    seed: int,
    cutoff_hz: float,
    pad_s: float = _PAD_S,
) -> tuple[np.ndarray, np.ndarray]:
    """A broadband-ish "physiological" signal: 1/f noise + oscillations + transients.

    Returns ``(tau_s, s)`` with world time in seconds (starting at ``-pad_s``). The
    signal is low-passed at ``cutoff_hz`` so it is fully captured at both sampling
    rates. Transients give sharp features for cross-correlation to lock onto.
    """
    rng = np.random.default_rng(seed)
    n = int(round((duration_s + 2 * pad_s) * fs))
    tau = -pad_s + np.arange(n) / fs

    # 1/f-shaped broadband background.
    x = rng.standard_normal(n)
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    f_safe = f.copy()
    f_safe[0] = f_safe[1] if n > 1 else 1.0
    spec = spec / np.sqrt(f_safe)
    x = np.fft.irfft(spec, n)

    # A few oscillatory bands (delta/alpha/gamma-ish) as *narrowband noise*: amplitude and
    # phase wander like real rhythms, so the signal is not periodic (a constant sinusoid gives
    # cross-correlation side peaks every period and invites false locks). RMS matches a sine
    # of amplitude ``amp``.
    for fc, amp in ((2.0, 0.6), (10.0, 0.35), (37.0, 0.2)):
        band = np.fft.rfft(rng.standard_normal(n)) * np.exp(-0.5 * ((f - fc) / (0.15 * fc)) ** 2)
        osc = np.fft.irfft(band, n)
        x = x + osc * (amp / np.sqrt(2.0) / (np.std(osc) + 1e-12))

    # Sparse transients (spike-like) for sharp correlation features.
    n_events = max(4, int(duration_s / 20))
    idx = rng.integers(0, n, size=n_events)
    x[idx] = x[idx] + rng.uniform(5.0, 9.0, size=n_events) * rng.choice([-1.0, 1.0], size=n_events)

    # Band-limit: both fs values fully capture content below cutoff (clean self-test).
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    spec[f > cutoff_hz] = 0.0
    x = np.fft.irfft(spec, n)

    x = x / np.std(x) * 50.0  # ~µV scale
    return tau, x.astype(np.float64)


@dataclass
class RenderedPair:
    ref: np.ndarray
    mov: np.ndarray
    fs_ref: float
    fs_moving: float
    ref_start_uutc: int
    mov_start_uutc: int


def render_pair(
    spec: WarpSpec,
    duration_s: float,
    fs_ref: float,
    fs_moving: float,
    start_uutc: int,
    seed: int,
) -> RenderedPair:
    """Render reference and moving sample arrays plus their absolute uUTC starts."""
    cutoff_hz = 0.4 * min(fs_ref, fs_moving)
    fs_master = max(fs_ref, fs_moving) * 4.0
    tau, s = make_source(duration_s, fs_master, seed, cutoff_hz)

    # Reference: uniform in world time over [0, duration].
    n_ref = int(round(duration_s * fs_ref))
    t_ref_rel = np.arange(n_ref) / fs_ref
    ref = np.interp(t_ref_rel, tau, s)

    # Moving: uniform in moving clock; invert m(tau) to find each sample's world time.
    m_of_tau = tau + spec.offset_s(tau)          # world -> moving clock (seconds), monotonic
    m0 = float(spec.offset_s(0.0))               # moving clock at world 0
    m_end = float(duration_s + spec.offset_s(duration_s))
    n_mov = int(round((m_end - m0) * fs_moving))
    m_rel = m0 + np.arange(n_mov) / fs_moving
    tau_j = np.interp(m_rel, m_of_tau, tau)       # moving clock -> world time
    mov = np.interp(tau_j, tau, s)

    return RenderedPair(
        ref=ref,
        mov=mov,
        fs_ref=fs_ref,
        fs_moving=fs_moving,
        ref_start_uutc=int(start_uutc),
        mov_start_uutc=int(round(start_uutc + m0 * US_PER_S)),
    )


def render_pair_long(
    spec: WarpSpec,
    duration_s: float,
    fs_ref: float,
    fs_moving: float,
    start_uutc: int,
    seed: int,
    chunk_s: float = 3600.0,
) -> RenderedPair:
    """Like :func:`render_pair` but memory-light for day-long recordings.

    The source is synthesized once at the higher of the two rates (band-limited below
    ``0.4 × min(fs)``) instead of a ×4 master rate; the moving recording is then evaluated at
    its warped world times in chunks with the band-limited resampler. The clock inversion
    ``m(τ) → τ`` uses a coarse τ grid (the offset is smooth).
    """
    from mef3io_coregistration.transform.resample import evaluate_at

    fs_src = max(fs_ref, fs_moving)
    cutoff_hz = 0.4 * min(fs_ref, fs_moving)
    tau, s = make_source(duration_s, fs_src, seed, cutoff_hz)
    src_start_us = tau[0] * US_PER_S  # world time of the first source sample (µs, relative)

    n_ref = int(round(duration_s * fs_ref))
    t_ref_us = np.arange(n_ref) * (US_PER_S / fs_ref)
    ref = _chunked_eval(s, src_start_us, fs_src, t_ref_us, fs_ref, chunk_s, evaluate_at)

    tau_c = np.arange(tau[0], tau[-1], 0.5)
    m_of_tau = tau_c + spec.offset_s(tau_c)
    m0 = float(spec.offset_s(0.0))
    m_end = float(duration_s + spec.offset_s(duration_s))
    n_mov = int(round((m_end - m0) * fs_moving))
    m_rel = m0 + np.arange(n_mov) / fs_moving
    tau_j_us = np.interp(m_rel, m_of_tau, tau_c) * US_PER_S
    mov = _chunked_eval(s, src_start_us, fs_src, tau_j_us, fs_moving, chunk_s, evaluate_at)

    return RenderedPair(
        ref=ref,
        mov=mov,
        fs_ref=fs_ref,
        fs_moving=fs_moving,
        ref_start_uutc=int(start_uutc),
        mov_start_uutc=int(round(start_uutc + m0 * US_PER_S)),
    )


def _chunked_eval(s, s_start_us, fs_src, t_us, out_fs, chunk_s, evaluate_at) -> np.ndarray:
    """Evaluate source ``s`` at times ``t_us`` in chunks (bounded memory)."""
    out = np.empty(t_us.shape[0])
    step = int(chunk_s * out_fs)
    pad = 512
    for i0 in range(0, t_us.shape[0], step):
        i1 = min(i0 + step, t_us.shape[0])
        t = t_us[i0:i1]
        j0 = max(0, int(np.floor((t[0] - s_start_us) * fs_src / US_PER_S)) - pad)
        j1 = min(s.shape[0], int(np.ceil((t[-1] - s_start_us) * fs_src / US_PER_S)) + pad)
        seg_start = s_start_us + j0 * US_PER_S / fs_src
        out[i0:i1] = evaluate_at(s[j0:j1], seg_start, fs_src, t, out_fs=out_fs)
    return out
