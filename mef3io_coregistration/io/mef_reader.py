"""Thin wrapper over ``mef3io.Reader``.

Reads channels from a MEF3 session (a ``.mefd`` folder, or a ``.mefd.tar`` read in place) over
absolute uUTC intervals, reconstructs per-sample uUTC timestamps on the channel's nominal grid,
reports per-channel sampling rate, and surfaces recording gaps/discontinuities. Windowed reads
fetch only the blocks they need, so reads stay cheap on sessions of any size. Gaps come back as
``NaN`` on the uniform sample grid; we never fill or interpolate a gap, and additionally expose
the discontinuities as :class:`Gap` records.

Timestamps are computed here, not taken from the backend: a window that does not start on a
sample is snapped to the channel grid first, so every sample's uUTC is exact and consecutive
half-open reads tile without overlap or loss.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import mef3io
import numpy as np

from ..config import US_PER_S
from ..transform.resample import first_index_at_or_after


@dataclass(frozen=True)
class Gap:
    """A recording discontinuity: no samples exist in ``[start_uutc, end_uutc)``.

    ``start_uutc`` is the nominal uUTC one sample after the last sample before the gap;
    ``end_uutc`` is the uUTC of the first sample after the gap; ``n_missing`` is the
    nominal number of absent samples on the channel grid.
    """

    start_uutc: int
    end_uutc: int
    n_missing: int
    # How far the recording resumes from the channel's nominal grid (µs, signed): samples after
    # the gap are stamped on the grid, so their timestamps are off by this much (see ``read``).
    grid_offset_us: float = 0.0


@dataclass(frozen=True)
class ChannelRead:
    """Samples for one channel over a uUTC interval, with reconstructed timestamps.

    ``data`` is ``float64`` in physical units (scaled by the units-conversion factor) and contains ``NaN``
    wherever the recording has a gap — gaps are never filled. ``ts`` is the per-sample uUTC
    on the channel's nominal grid (``int64``), aligned to ``data`` index-for-index (gap
    samples included). ``gaps`` lists the discontinuities intersecting this read.
    """

    channel: str
    data: np.ndarray
    ts: np.ndarray
    fs: float
    start_uutc: int
    gaps: tuple[Gap, ...]

    @property
    def n(self) -> int:
        return int(self.data.shape[0])

    @property
    def stop_uutc(self) -> int:
        """Nominal uUTC one sample past the last returned sample (half-open end)."""
        return self.start_uutc + int(round(self.n * US_PER_S / self.fs))


class MefChannelReader:
    """Per-session MEF3 reader exposing uUTC reads, per-channel Fs, and gaps.

    Wraps a single ``mef3io.Reader``. Usable as a context manager; call :meth:`close` (or exit
    the ``with`` block) to release the underlying session. ``password`` opens encrypted sessions.
    """

    def __init__(self, path: str | Path, password: str | None = None) -> None:
        self.path = str(path)
        with warnings.catch_warnings():  # unset optional header fields are not our concern
            warnings.simplefilter("ignore", mef3io.SessionDeclarationWarning)
            self._reader = mef3io.Reader(self.path, password=password or "")
        self._channels = list(self._reader.channels)
        if not self._channels:  # mef3io opens any *.mefd folder; one without channels is not data
            self._reader.close()
            raise ValueError(f"{self.path} is not a MEF3 recording: it has no channels")
        self._info: dict[str, dict] = {}
        self._gaps: dict[str, tuple[Gap, ...]] = {}

    # -- context management -------------------------------------------------
    def __enter__(self) -> MefChannelReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._reader.close()

    # -- metadata ----------------------------------------------------
    @property
    def channels(self) -> list[str]:
        return list(self._channels)

    def has_channel(self, channel: str) -> bool:
        return channel in self._channels

    def info(self, channel: str) -> dict:
        """The backend's channel metadata (``mef3io.Reader.info``), cached."""
        self._require(channel)
        if channel not in self._info:
            self._info[channel] = dict(self._reader.info(channel))
        return self._info[channel]

    def segments(self, channel: str) -> list[dict]:
        """Per-segment map of ``channel`` (``mef3io.Reader.segments``), from metadata only."""
        self._require(channel)
        return list(self._reader.segments(channel))

    def read_raw_counts(self, channel: str, t0: int | None = None, t1: int | None = None) -> np.ndarray:
        """The stored int32 counts of ``[t0, t1)`` as the backend returns them (legacy API)."""
        self._require(channel)
        return np.asarray(self._reader.read_raw(channel, t0, t1)["samples"])

    def records(self, channel: str | None = None, **kwargs) -> list[dict]:
        """Annotations (``mef3io.Reader.records``): session-level, or one channel's (channel- and
        segment-level). ``kwargs`` pass through (``level=``, ``segment=``)."""
        return list(self._reader.records(channel, **kwargs))

    def _require(self, channel: str) -> None:
        if not self.has_channel(channel):
            raise KeyError(
                f"channel {channel!r} not in {self.path}; available: {sorted(self.channels)}"
            )

    def fs(self, channel: str) -> float:
        """Sampling rate (Hz) of ``channel``. Differs freely across channels/files."""
        return float(self.info(channel)["sampling_frequency"])

    def n_samples(self, channel: str) -> int:
        """Number of *recorded* samples (excludes gap-filler samples)."""
        return int(self.info(channel)["number_of_samples"])

    def time_bounds(self, channel: str) -> tuple[int, int]:
        """``(start_uutc, end_uutc)`` for ``channel``; ``end`` is the exclusive grid end."""
        i = self.info(channel)
        return int(i["start_time"]), int(i["end_time"])

    def unit(self, channel: str) -> str:
        u = self.info(channel).get("units_description") or ""
        return u.decode(errors="replace") if isinstance(u, bytes) else str(u)

    # -- gaps / discontinuities --------------------------------------
    def gaps(self, channel: str) -> tuple[Gap, ...]:
        """Recording discontinuities, derived from the MEF3 block table of contents (cached).

        A block flagged as a discontinuity whose start uUTC lies at least one sample past the
        nominal end of the previous block marks a gap.
        """
        self._require(channel)
        if channel in self._gaps:
            return self._gaps[channel]
        toc = self._reader.toc(channel)
        fs = self.fs(channel)
        ch0 = self.time_bounds(channel)[0]
        out: list[Gap] = []
        for prev, blk in pairwise(toc):
            # Add the small block-duration offset to the integer start (avoids losing µs
            # precision when the absolute uUTC epoch dominates a float64 sum).
            prev_end = int(prev["start_uutc"]) + int(round(int(prev["number_of_samples"]) * US_PER_S / fs))
            block_start = int(blk["start_uutc"])
            if blk["discontinuity"] and block_start > prev_end:
                n_missing = int(round((block_start - prev_end) * fs / US_PER_S))
                if n_missing > 0:  # a sub-sample step in block timing is jitter, not a gap
                    k = (block_start - int(ch0)) * fs / US_PER_S
                    off = (k - round(k)) * US_PER_S / fs
                    out.append(Gap(start_uutc=prev_end, end_uutc=block_start, n_missing=n_missing,
                                   grid_offset_us=float(off)))
        self._gaps[channel] = tuple(out)
        worst = max((abs(g.grid_offset_us) for g in out), default=0.0)
        if worst > 0.01 * US_PER_S / fs:
            warnings.warn(
                f"{self.path} {channel}: after a gap the recording resumes up to {worst:.1f} µs "
                f"({worst * fs / US_PER_S:.0%} of a sample) off the channel's sampling grid; "
                "samples after such gaps are timestamped on the grid (see Gap.grid_offset_us)",
                stacklevel=2,
            )
        return self._gaps[channel]

    # -- data --------------------------------------------------
    def read(self, channel: str, start: int | None = None, stop: int | None = None) -> ChannelRead:
        """Read ``channel``: the nominal-grid samples with uUTC in ``[start, stop)``.

        ``None`` bounds default to the channel extent. Timestamps are reconstructed on the
        channel's nominal grid anchored at the recording start. Bounds need not be grid-aligned:
        they are snapped to the grid here and the backend is asked for exactly those samples
        (it rounds a mid-sample start to the nearest sample and reports the requested time, which
        would put timestamps up to half a sample off), so consecutive half-open reads tile
        without overlap or loss. Only the blocks covering the window are read. Samples outside
        the recording or inside a gap are ``NaN`` and gaps are listed in ``gaps``.

        Timestamps follow the channel's nominal grid across gaps: where a recording resumes a
        fraction of a sample off that grid, the samples after the gap are stamped on it — off by
        ``Gap.grid_offset_us`` (at most half a sample; a warning is issued above 1 % of one).
        On the recordings this was checked on it is < 1 µs.
        """
        fs = self.fs(channel)
        ch_start, ch_end = self.time_bounds(channel)
        period = US_PER_S / fs

        if start is None and stop is None:
            data = self._read_window(channel, None, None)
            first_idx = 0
        else:
            lo = ch_start if start is None else int(start)
            hi = ch_end if stop is None else int(stop)
            # First/last grid index whose (µs-rounded) timestamp lies in [lo, hi).
            first_idx = first_index_at_or_after(ch_start, period, lo)
            end_idx = first_index_at_or_after(ch_start, period, hi)
            n_req = max(0, end_idx - first_idx)
            if n_req == 0:
                data = np.empty(0, dtype=np.float64)
            else:
                t_lo = ch_start + int(round(first_idx * period))
                t_hi = ch_start + int(round(end_idx * period))
                data = _fit_length(self._read_window(channel, t_lo, t_hi), n_req)

        n = data.shape[0]
        start_uutc = ch_start + int(round(first_idx * period))
        # Reconstruct on the nominal grid: keep the (small) sample-offset term in float and
        # add the integer epoch, so µs precision is not lost to the ~1e15 absolute uUTC.
        idx = first_idx + np.arange(n, dtype=np.int64)
        ts = ch_start + np.rint(idx * period).astype(np.int64)

        stop_uutc = start_uutc + int(round(n * period))
        gaps = tuple(
            g for g in self.gaps(channel) if g.start_uutc < stop_uutc and g.end_uutc > start_uutc
        )
        return ChannelRead(
            channel=channel, data=data, ts=ts, fs=fs, start_uutc=start_uutc, gaps=gaps
        )


    def _read_window(self, channel: str, t0: int | None, t1: int | None) -> np.ndarray:
        """Physical-unit samples of ``[t0, t1)`` with ``NaN`` wherever the backend has none."""
        r = self._reader.read_raw(channel, t0, t1)
        data = np.asarray(r["samples"], dtype=np.float64) * float(r["units_conversion_factor"])
        data[np.asarray(r["valid"]) == 0] = np.nan
        return data


def _fit_length(data: np.ndarray, n: int) -> np.ndarray:
    """Trim or NaN-pad (at the tail) to exactly ``n`` samples."""
    if data.shape[0] >= n:
        return data[:n]
    return np.concatenate((data, np.full(n - data.shape[0], np.nan)))


#: How far a file's start or end may sit from the one stored in a result and still count as the
#: recording the map was computed on (a re-export or a downsampled copy shifts its end by a sample).
SPAN_TOLERANCE_US = 1_000_000


def time_span(path: str | Path, password: str | None = None) -> tuple[int | None, int | None]:
    """``(start_uutc, end_uutc)`` of a MEF3 recording: the earliest channel start and the latest
    channel end, from metadata only. ``(None, None)`` when no channel can be described."""
    starts, ends = [], []
    with MefChannelReader(path, password=password) as r:
        for ch in r.channels:
            try:
                c0, c1 = r.time_bounds(ch)
            except Exception:  # a channel the reader cannot describe has no span
                continue
            starts.append(c0)
            ends.append(c1)
    return (min(starts), max(ends)) if starts else (None, None)


#: Samples hashed per window, and the windows (as fractions of each channel's span) whose stored
#: values go into the content hash: enough to tell two recordings with identical metadata apart,
#: few enough that fingerprinting a multi-gigabyte session stays a sub-second job.
_CONTENT_SAMPLES = 4096
_CONTENT_AT = (0.0, 0.5, 1.0)


def fingerprint(path: str | Path) -> dict:
    """Identity of a MEF3 recording by what it holds, not where it is, stored in a coregistration
    result as provenance. Opening a result checks files only by ``start_uutc``/``end_uutc`` (see
    :func:`time_span`); the hashes are a record of exactly which recording the map was made on.

    * ``sha256`` — every channel's name, rate, start, end and sample count.
    * ``content_sha256`` — the stored sample values of every channel at its start, middle and end
      (``_CONTENT_SAMPLES`` each): two recordings with identical metadata but different signals
      differ here.
    """
    import hashlib
    import json

    meta = hashlib.sha256()
    content = hashlib.sha256()
    with MefChannelReader(path) as r:
        rows = []
        for ch in sorted(r.channels):
            try:
                c0, c1 = r.time_bounds(ch)
                fs = r.fs(ch)
                rows.append([ch, fs, c0, c1, r.n_samples(ch)])
            except Exception:  # a channel the reader cannot describe still counts by name
                rows.append([ch, None, None, None, None])
                continue
            span = max(0, c1 - c0)
            win = int(round(_CONTENT_SAMPLES * US_PER_S / fs))
            content.update(ch.encode())
            for frac in _CONTENT_AT:
                t = c0 + int(frac * max(0, span - win))
                counts = r.read_raw_counts(ch, t, min(c1, t + win))
                content.update(np.ascontiguousarray(counts, dtype="<i4").tobytes())
    meta.update(json.dumps(rows, sort_keys=True).encode())
    starts = [int(row[2]) for row in rows if row[2] is not None]  # type: ignore[call-overload]
    ends = [int(row[3]) for row in rows if row[3] is not None]  # type: ignore[call-overload]
    return {
        "sha256": meta.hexdigest(),
        "content_sha256": content.hexdigest(),
        "n_channels": len(rows),
        "start_uutc": min(starts) if starts else None,
        "end_uutc": max(ends) if ends else None,
    }
