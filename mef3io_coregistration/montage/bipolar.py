"""Bipolar derivations.

Build a bipolar alignment signal ``anode - cathode`` from two channels within one file, so
recordings with differing hardware references become comparable. The alignment
channels are chosen by the caller and may be selected independently per stage; this
module is only the derivation primitive.

If the two channels differ in sampling rate, the higher-Fs channel is band-limited and
decimated to the lower rate before subtraction (Fs harmonization). The two channels
are then subtracted at the same instants over their overlapping span: index-for-index when their
grids coincide, otherwise with the cathode evaluated at the anode's sample times.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import US_PER_S
from ..io.derived import decimate_bandlimited
from ..io.mef_reader import ChannelRead
from ..transform.resample import evaluate_at


@dataclass(frozen=True)
class BipolarSignal:
    """A bipolar derivation ``anode - cathode`` over the channels' overlapping span."""

    data: np.ndarray
    ts: np.ndarray
    fs: float
    start_uutc: int
    anode: str
    cathode: str


def bipolar(anode: ChannelRead, cathode: ChannelRead) -> BipolarSignal:
    """Return ``anode - cathode`` after Fs harmonization and uUTC alignment.

    Both reads should come from the same file. When their sampling rates differ, the
    higher-Fs channel is decimated to the lower rate first; the result is sampled on the
    (harmonized) lower-Fs grid over the two channels' overlapping time span.
    """
    fs = min(anode.fs, cathode.fs)
    a_data, a_start = _harmonize(anode, fs)
    c_data, c_start = _harmonize(cathode, fs)

    period = US_PER_S / fs
    start = max(a_start, c_start)
    ia = int(round((start - a_start) / period))
    ic = int(round((start - c_start) / period))
    n = min(a_data.shape[0] - ia, c_data.shape[0] - ic)
    if n <= 0:
        raise ValueError(
            f"channels {anode.channel!r} and {cathode.channel!r} do not overlap in time"
        )

    # Subtract values measured at the SAME instants. When the two grids coincide (the usual case
    # within one recording) that is index-for-index; when they are offset by a fraction of a
    # sample, the cathode is evaluated at the anode's sample times (band-limited interpolation) —
    # rounding to the nearest index would subtract values up to half a sample apart.
    phase = ((a_start - c_start) / period) % 1.0
    if min(phase, 1.0 - phase) * period <= 1.0:  # within 1 µs: the same grid
        cath = c_data[ic : ic + n]
    else:
        t_a = (a_start - c_start) + (ia + np.arange(n, dtype=np.float64)) * period  # rel. to cathode
        cath = evaluate_at(c_data, 0.0, fs, t_a)
    data = a_data[ia : ia + n] - cath
    # Offset term stays in float; the integer epoch is added after rounding (µs precision).
    t0 = a_start + int(round(ia * period))
    ts = int(t0) + np.rint(np.arange(n, dtype=np.int64) * period).astype(np.int64)
    return BipolarSignal(
        data=data, ts=ts, fs=fs, start_uutc=int(t0), anode=anode.channel, cathode=cathode.channel
    )


def _harmonize(read: ChannelRead, target_fs: float) -> tuple[np.ndarray, int]:
    """Return ``(data, start_uutc)`` decimated to ``target_fs`` if needed."""
    if read.fs == target_fs:
        return read.data, read.start_uutc
    dec = decimate_bandlimited(read.data, read.fs, target_fs, start_uutc=read.start_uutc)
    return dec.data, read.start_uutc
