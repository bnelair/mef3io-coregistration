"""In-memory rendered pairs for the alignment-stage tests (Milestone M4).

Not a test module (leading underscore → not collected). ``build_pair`` renders a
reference/moving pair with a known warp and returns the derived products the stages consume:
the ~50 Hz min-max envelopes (Stage 1) and the decimated traces at a common target Fs
(Stages 2/3). Rendering is in-memory (no MEF3), so it is fast enough to call per test.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from mef3io.compat import MefWriter

from mef3io_coregistration.config import US_PER_S
from mef3io_coregistration.io.derived import (
    Decimated,
    Envelope,
    decimate_bandlimited,
    minmax_envelope,
)
from mef3io_coregistration.synthetic.generate import render_pair
from mef3io_coregistration.synthetic.warp import WarpSpec

_START = 1_700_000_000 * US_PER_S


@dataclass
class Pair:
    spec: WarpSpec
    start_uutc: int
    duration_s: float
    fs_target: float
    ref_env: Envelope
    mov_env: Envelope
    ref_dec: Decimated
    mov_dec: Decimated

    def gt_offset_s(self, t_rel_s: float) -> float:
        return float(self.spec.offset_s(t_rel_s))

    def roi_uutc(self, t_rel_s: float) -> int:
        return int(self.start_uutc + t_rel_s * US_PER_S)


def build_pair(
    spec: WarpSpec,
    duration_s: float = 1200.0,
    fs_ref: float = 1024.0,
    fs_mov: float = 256.0,
    seed: int = 5,
) -> Pair:
    rp = render_pair(spec, duration_s, fs_ref, fs_mov, _START, seed=seed)
    ref_env = minmax_envelope(rp.ref, fs_ref, start_uutc=rp.ref_start_uutc)
    mov_env = minmax_envelope(rp.mov, fs_mov, start_uutc=rp.mov_start_uutc)
    ref_dec = decimate_bandlimited(rp.ref, fs_ref, fs_mov, start_uutc=rp.ref_start_uutc)
    mov_dec = Decimated(rp.mov, fs_mov, rp.mov_start_uutc)
    return Pair(spec, _START, duration_s, fs_mov, ref_env, mov_env, ref_dec, mov_dec)


@dataclass
class WrittenPair:
    reference_path: str
    moving_path: str
    spec: WarpSpec
    start_uutc: int
    duration_s: float

    def gt_offset_s(self, t_rel_s: float) -> float:
        return float(self.spec.offset_s(t_rel_s))

    def roi_uutc(self, t_rel_s: float) -> int:
        return int(self.start_uutc + t_rel_s * US_PER_S)


def write_pair(
    dir_path,
    spec: WarpSpec,
    channels=("eeg",),
    duration_s: float = 1200.0,
    fs_ref: float = 1024.0,
    fs_mov: float = 256.0,
    seed: int = 5,
    moving_only=(),
) -> WrittenPair:
    """Render a warped pair and write it to two MEF3 sessions (``channels`` all carry the
    warped signal, for exercising channel selection/reuse).

    ``moving_only`` channels are written to the moving file **and not the reference** — the
    real case, where the clinical system records 209 channels and the cadence system a subset
    of 19. A coregistered read of those channels is what the tool exists for (REVIEW B4).
    """
    rp = render_pair(spec, duration_s, fs_ref, fs_mov, _START, seed=seed)
    ref_p = Path(dir_path) / "reference.mefd"
    mov_p = Path(dir_path) / "moving.mefd"
    _write_mef(ref_p, {c: rp.ref for c in channels}, fs_ref, rp.ref_start_uutc)
    _write_mef(
        mov_p,
        {c: rp.mov for c in (*channels, *moving_only)},
        fs_mov,
        rp.mov_start_uutc,
    )
    return WrittenPair(str(ref_p), str(mov_p), spec, _START, duration_s)


def _write_mef(path, chans, fs, start):
    writer = MefWriter(str(path), overwrite=True)
    for name, data in chans.items():
        writer.write_data(np.asarray(data, dtype=np.float64), name, int(start), float(fs), precision=5)
    del writer


def corrupt_segment(dec: Decimated, t0_rel_s: float, t1_rel_s: float, seed: int = 0) -> Decimated:
    """Return a copy of ``dec`` with samples in ``[t0, t1)`` replaced by noise (a dead/low-SNR
    span, so the windows there lose shared structure)."""
    data = dec.data.copy()
    i0 = int(round(t0_rel_s * dec.fs))
    i1 = int(round(t1_rel_s * dec.fs))
    rng = np.random.default_rng(seed)
    data[i0:i1] = rng.standard_normal(i1 - i0) * float(np.std(dec.data))
    return Decimated(data, dec.fs, dec.start_uutc)
