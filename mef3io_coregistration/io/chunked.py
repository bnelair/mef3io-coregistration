"""Full-length chunked reading of one alignment signal.

Reads a file's channel (or bipolar pair) across the whole recording in ~5-minute chunks and
produces two in-memory products: a band-limited decimated full-length trace and a ~50 Hz min-max
envelope. The raw full-rate signal is never held — only one chunk at a time.

Part of the reader package (the unified recording's display envelopes come from here); the
alignment pipeline runs the same function in worker processes.

Decimation is streamed with overlap-save: each chunk is read with a margin on both sides so the
polyphase FIR's edge transient falls outside the retained core, matching a full-length decimation
in the interior.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from typing import TYPE_CHECKING, Protocol

import numpy as np
from scipy.signal import resample_poly

from ..config import ENVELOPE_FS_HZ, US_PER_S
from ..montage.spec import ChannelSpec, read_montage
from .derived import Decimated, Envelope, envelope_bin_size
from .mef_reader import MefChannelReader

if TYPE_CHECKING:
    from ..montage.spec import ChannelSpec


class CancelToken(Protocol):
    def is_set(self) -> bool: ...


class Cancelled(Exception):
    """Raised inside a load when its cancel token is set."""


@dataclass(frozen=True)
class LoadSpec:
    """What to load: a file's alignment channel(s) and the target products' parameters.

    ``cathode`` set → a bipolar derivation ``anode - cathode``; otherwise the single
    ``anode`` channel. ``target_fs`` is the decimated trace's rate (defaults to the source
    Fs, i.e. no decimation). The identity ``key`` drives reuse across stages.

    ``invert`` multiplies the derivation by −1. Two systems can record the same physiology with
    opposite polarity — a bipolar pair wired the other way round, or an amplifier that inverts;
    flipping the sign makes the two comparable.
    """

    file_path: str
    anode: str
    cathode: str | None = None
    target_fs: float | None = None
    envelope_fs: float = ENVELOPE_FS_HZ
    chunk_s: float = 300.0
    invert: bool = False
    decimate: bool = True  # False: envelope only (display) — no full-length trace is kept

    @classmethod
    def for_channel(cls, file_path: str, channel: ChannelSpec, **kwargs) -> LoadSpec:
        return cls(file_path, channel.anode, channel.cathode, **kwargs)

    @property
    def key(self) -> tuple:
        return (
            self.file_path,
            self.anode,
            self.cathode,
            self.target_fs,
            self.envelope_fs,
            self.invert,
            self.decimate,
        )

    @property
    def label(self) -> str:
        base = f"{self.anode}-{self.cathode}" if self.cathode else self.anode
        return f"−({base})" if self.invert else base


@dataclass
class LoadProduct:
    """In-memory products for one file's alignment signal."""

    decimated: Decimated
    envelope: Envelope
    spec: LoadSpec


def load_alignment_signal(
    spec: LoadSpec,
    progress: ProgressHook | None = None,
    cancel: CancelToken | None = None,
) -> LoadProduct:
    """Chunked full-length load → decimated + envelope products.

    ``progress(fraction, message)`` is called after each chunk; ``cancel`` is polled before
    each chunk and raises :class:`Cancelled` if set.
    """
    reader = MefChannelReader(spec.file_path)
    try:
        # A bipolar pair is derived at the lower of its two rates, on the anode's grid (see
        # montage.bipolar) — channels of different rates or grid phases are fine.
        fs = reader.fs(spec.anode)
        ch_start, ch_end = reader.time_bounds(spec.anode)
        if spec.cathode is not None:
            fs = min(fs, reader.fs(spec.cathode))
            c_start, c_end = reader.time_bounds(spec.cathode)
            ch_end = min(ch_end, c_end)
            if c_start > ch_start:  # start on the anode's grid, at or after the cathode's start
                period = US_PER_S / fs
                ch_start += int(round(np.ceil((c_start - ch_start) / period) * period))
        n_total = int(round((ch_end - ch_start) * fs / US_PER_S))

        target_fs = spec.target_fs or fs
        ratio = Fraction(target_fs / fs).limit_denominator(10_000)
        up, down = ratio.numerator, ratio.denominator
        bin_size = envelope_bin_size(fs, spec.envelope_fs)

        # Core chunk aligned to both the envelope bin and the decimation factor; overlap is a
        # whole number of decimation steps so chunk output indices divide cleanly.
        align = _lcm(bin_size, down)
        chunk = max(align, (int(round(spec.chunk_s * fs)) // align) * align)
        overlap = 32 * down if spec.decimate else 0  # envelope bins need no FIR context

        dec_parts: list[np.ndarray] = []
        min_parts: list[np.ndarray] = []
        max_parts: list[np.ndarray] = []
        for c0 in range(0, n_total, chunk):
            if cancel is not None and cancel.is_set():
                raise Cancelled(spec.label)
            c1 = min(c0 + chunk, n_total)
            sig, core_lo = _read_core(reader, spec, fs, ch_start, c0, c1, overlap, n_total)

            core = sig[core_lo : core_lo + (c1 - c0)]
            _accumulate_envelope(core, bin_size, min_parts, max_parts)
            if spec.decimate:
                dec_parts.append(_decimate_core(sig, core_lo, c1 - c0, up, down))

            if progress is not None:
                progress(c1 / n_total, f"loaded {c1 // fs:.0f}/{n_total // fs:.0f}s of {spec.label}")

        decimated = Decimated(
            np.concatenate(dec_parts) if dec_parts else np.empty(0),
            fs * up / down,
            int(ch_start),
        )
        envelope = Envelope(
            np.concatenate(min_parts) if min_parts else np.empty(0),
            np.concatenate(max_parts) if max_parts else np.empty(0),
            fs / bin_size,
            bin_size,
            int(ch_start),
        )
        return LoadProduct(decimated=decimated, envelope=envelope, spec=spec)
    finally:
        reader.close()


def _read_core(reader, spec, fs, ch_start, c0, c1, overlap, n_total):
    """Read ``[c0, c1)`` with an ``overlap`` margin each side; return (signal, core offset)."""
    lo = max(0, c0 - overlap)
    hi = min(n_total, c1 + overlap)
    # The grid times of samples lo and hi exactly as the reader stamps them (integer epoch + a
    # rounded offset). Adding the float offset to the ~1.7e15 µs epoch first (0.25 µs float
    # spacing, then banker's rounding) put t_lo 1 µs past sample lo at some rates (2048 Hz), and
    # the read snapped to the NEXT sample: whole chunks shifted by one.
    t_lo = int(ch_start) + int(np.rint(lo * US_PER_S / fs))
    t_hi = int(ch_start) + int(np.rint(hi * US_PER_S / fs))
    if spec.cathode is None:
        r = reader.read(spec.anode, t_lo, t_hi)
    else:
        # The same derivation the reader uses: rates harmonised, both channels taken at the same
        # instants.
        r = read_montage(reader, ChannelSpec(spec.anode, spec.cathode), t_lo, t_hi)
    # Place the read on this loader's grid by its own timestamps (NaN where it has no sample),
    # never by position — a read that starts a sample late must not shift the chunk.
    sig = np.full(hi - lo, np.nan)
    if r.n:
        k = int(round((int(r.ts[0]) - t_lo) * fs / US_PER_S))
        src = r.data[max(0, -k):]
        dst0 = max(0, k)
        m = min(src.shape[0], sig.shape[0] - dst0)
        if m > 0:
            sig[dst0 : dst0 + m] = src[:m]
    if spec.invert:
        sig = -sig
    return sig, c0 - lo


def _accumulate_envelope(core, bin_size, min_parts, max_parts):
    n_bins = core.shape[0] // bin_size
    if n_bins == 0:
        return
    binned = core[: n_bins * bin_size].reshape(n_bins, bin_size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN (gap) bins -> NaN
        min_parts.append(np.nanmin(binned, axis=1))
        max_parts.append(np.nanmax(binned, axis=1))


def _decimate_core(sig, core_lo, core_len, up, down):
    if (up, down) == (1, 1):
        return sig[core_lo : core_lo + core_len].copy()
    out = resample_poly(sig, up, down)
    lo = (core_lo * up) // down
    return out[lo : lo + (core_len * up) // down]


def _lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)


class ProgressHook(Protocol):
    def __call__(self, fraction: float, message: str) -> None: ...


# --------------------------------------------------------------------------- #
# multiprocessing wrapper
# --------------------------------------------------------------------------- #
