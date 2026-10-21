"""Two coregistered recordings used as **one** recording.

    rec = CoregisteredRecording.open("result.json")                     # files + roles from result
    rec.set_channel("LA1", a="LA1", b="EEG LA1-Ref", prefer="B")        # link differing names
    rec.set_channel("stim", b="STIM1-STIM2")                            # B-only, bipolar
    sig = rec.get_data("LA1", start, stop)                              # reference-clock uUTC
    sig.data, sig.ts, sig.source, sig.extrapolated

Every **logical channel** names what it is in file A and/or file B — independently, as a
channel name, an ``"anode-cathode"`` bipolar expression or a
:class:`~mef3io_coregistration.montage.spec.ChannelSpec` — plus a **preferred source**. A read
serves the preferred file wherever it has data and falls back to the other file wherever the
preferred one has none (outside its recording, or in a gap); ``source`` records, per sample,
which file it came from. All times are on the **reference clock**; samples drawn from the
moving file are resampled through the clock map and flagged ``extrapolated`` where the map is
not measured.

Without an explicit channel list the recording is built automatically: every channel name
becomes a logical channel, linked across files when both files carry that name; the default
preference is the reference file. Nothing requires the files to share names or montage.

The output grid of a logical channel is the preferred source's own sampling grid expressed on
the reference clock (or ``fs=`` if given). Reading the reference file at its native rate
returns its native samples untransformed.
"""
from __future__ import annotations

import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..config import (
    MERGE_BAND_HZ,
    MERGE_CHECK_CHUNK_S,
    MERGE_CHECK_CHUNKS,
    MERGE_GAIN_RANGE,
    MERGE_MIN_CORR,
)
from ..io.chunked import LoadSpec, load_alignment_signal
from ..io.derived import Envelope
from ..montage.spec import ChannelSpec, read_montage
from ..transform.filters import _filter
from ..transform.resample import grid_indices, uniform_grid
from .coregistered_reader import CoregisteredReader, FileRole

if TYPE_CHECKING:
    from ..result import CoregResult

_FORMAT = "signal_coregistration.recording"
_FORMAT_VERSION = 1

SOURCE_NONE, SOURCE_A, SOURCE_B = 0, 1, 2
_SOURCE_CODE: dict[str, int] = {"A": SOURCE_A, "B": SOURCE_B}


@dataclass(frozen=True)
class LogicalChannel:
    """One channel of the unified recording: its signal in file A and/or B + preference."""

    name: str
    a: ChannelSpec | None = None
    b: ChannelSpec | None = None
    prefer: FileRole = "A"
    fs: float | None = None  # output rate; default: the preferred source's rate
    # Per-source linear correction, value' = scale · value + offset. No default correction: the
    # user sets it (scale −1 inverts a source recorded with the opposite polarity).
    scale_a: float = 1.0
    offset_a: float = 0.0
    scale_b: float = 1.0
    offset_b: float = 0.0
    # Serve the other file even when the overlap check says the two sources disagree.
    allow_mismatch: bool = False

    def __post_init__(self) -> None:
        if self.a is None and self.b is None:
            raise ValueError(f"logical channel {self.name!r} needs a source in file A or B")
        if self.prefer not in ("A", "B"):
            raise ValueError('prefer must be "A" or "B"')

    def source(self, file: FileRole) -> ChannelSpec | None:
        return self.a if file == "A" else self.b

    def correction(self, file: FileRole) -> tuple[float, float]:
        return (self.scale_a, self.offset_a) if file == "A" else (self.scale_b, self.offset_b)

    @property
    def primary(self) -> FileRole:
        """The preferred file if it carries this channel, else the other one."""
        if self.source(self.prefer) is not None:
            return self.prefer
        return "B" if self.prefer == "A" else "A"

    @property
    def fallback(self) -> FileRole | None:
        other: FileRole = "B" if self.primary == "A" else "A"
        return other if self.source(other) is not None else None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "a": None if self.a is None else self.a.to_dict(),
            "b": None if self.b is None else self.b.to_dict(),
            "prefer": self.prefer,
            "fs": self.fs,
            "scale_a": self.scale_a,
            "offset_a": self.offset_a,
            "scale_b": self.scale_b,
            "offset_b": self.offset_b,
            "allow_mismatch": self.allow_mismatch,
        }

    @classmethod
    def from_dict(cls, d: dict) -> LogicalChannel:
        return cls(
            name=d["name"],
            a=None if d.get("a") is None else ChannelSpec.parse(d["a"]),
            b=None if d.get("b") is None else ChannelSpec.parse(d["b"]),
            prefer=d.get("prefer", "A"),
            fs=d.get("fs"),
            **_corrections(d),
        )


def _corrections(d: dict) -> dict:
    keys = ("scale_a", "offset_a", "scale_b", "offset_b")
    out: dict = {k: float(d[k]) for k in keys if d.get(k) is not None}
    if d.get("allow_mismatch") is not None:
        out["allow_mismatch"] = bool(d["allow_mismatch"])
    return out


@dataclass(frozen=True)
class SourceAgreement:
    """Do file A's and file B's signals for one logical channel agree where both exist?

    Compared over their overlap after each source's correction (scale, offset), band-passed alike
    so that DC and bandwidth differences between the amplifiers do not decide it. ``status``:
    ``ok`` · ``inverted`` (clearly negative correlation) · ``poor`` (no clear correlation) ·
    ``scale`` (robust gain ratio outside the accepted range) · ``units`` (different units) ·
    ``no_overlap`` · ``single_source``. Only ``ok`` lets the channel fall back from one file to
    the other (unless the logical channel sets ``allow_mismatch``). The baseline (DC) difference is
    reported (``offset_b_minus_a``, ``baseline_step``) but deliberately not part of the gate.
    """

    status: str
    corr: float = float("nan")
    gain_b_over_a: float = float("nan")  # robust (MAD) amplitude ratio, after corrections
    offset_b_minus_a: float = float("nan")  # median difference of the corrected raw values
    unit_a: str = ""
    unit_b: str = ""
    overlap_s: float = 0.0
    message: str = ""
    # The corrected baselines differ by more than the signal's spread: merging is allowed (the gate
    # is sign, gain and units — a wandering DC-coupled baseline fits no constant offset) but each
    # switch between files shows a step. Filter each run on its own (UnifiedRead.runs()).
    baseline_step: bool = False

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "single_source")


@dataclass(frozen=True)
class UnifiedRead:
    """Samples of a logical channel on the reference clock.

    ``source[i]`` is ``1`` (file A), ``2`` (file B) or ``0`` (no data: ``NaN``).
    ``extrapolated[i]`` is True where the sample came from the moving file outside the measured
    span of the clock map.
    """

    channel: str
    data: np.ndarray
    ts: np.ndarray
    fs: float
    start_uutc: int
    source: np.ndarray
    extrapolated: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    merge_note: str = ""  # why the other file was NOT used to fill this read, if it was not

    @property
    def n(self) -> int:
        return int(self.data.shape[0])

    def runs(self) -> list[tuple[int, int, int]]:
        """Contiguous stretches ``(source, start_uutc, stop_uutc)`` — filter, detect or measure
        each stretch on its own: a statistic spanning a switch between files mixes two
        amplifiers (bandwidth, hardware filters, stim-artifact handling, reference)."""
        if self.n == 0:
            return []
        edges = np.flatnonzero(np.diff(self.source)) + 1
        starts = np.concatenate(([0], edges))
        stops = np.concatenate((edges, [self.n]))
        step = int(round(1e6 / self.fs))
        return [
            (int(self.source[i0]), int(self.ts[i0]), int(self.ts[i1 - 1]) + step)
            for i0, i1 in zip(starts, stops, strict=True)
        ]


class CoregisteredRecording:
    """Two MEF3 recordings, coregistered by a clock map, served as one recording.

    **A logical channel that switches between files is a display and navigation aid.** The two
    systems differ in bandwidth, hardware filtering, stim-artifact handling and reference, so any
    statistic spanning a switch (spike rate, band power, a detector's output) is confounded by
    it: analyse within one source (:meth:`UnifiedRead.runs`), or per source. Falling back from one
    file to the other is refused unless the two agree over their overlap
    (:meth:`agreement`) or the channel explicitly sets ``allow_mismatch``.
    """

    def __init__(
        self,
        reader: CoregisteredReader,
        channels: Iterable[LogicalChannel | dict] | None = None,
        default_prefer: FileRole | None = None,
        envelope_cache: dict | None = None,
    ) -> None:
        """``envelope_cache`` may be shared between recordings on the same two files: display
        envelopes depend only on file and channel (not on the mapping or the clock map), and
        each costs a full read of that channel."""
        self.reader = reader
        self.default_prefer: FileRole = default_prefer or reader.reference_file
        if self.default_prefer not in ("A", "B"):
            raise ValueError('default_prefer must be "A" or "B"')
        self._channels: dict[str, LogicalChannel] = {}
        self._envelopes: dict[tuple, Envelope] = envelope_cache if envelope_cache is not None else {}
        self._env_ref: dict[tuple, tuple[Envelope, np.ndarray]] = {}  # + reference-clock times
        self._agreement: dict[LogicalChannel, SourceAgreement] = {}
        # MEF readers are not thread-safe and the API serves requests concurrently.
        self._lock = threading.RLock()
        if channels is None:
            self._auto_channels()
        else:
            for ch in channels:
                if isinstance(ch, LogicalChannel):
                    self._add(ch)
                else:  # strings resolve against each file's channel list (names may hold '-')
                    self.set_channel(
                        ch["name"], ch.get("a"), ch.get("b"), ch.get("prefer"), ch.get("fs"),
                        **_corrections(ch),
                    )

    # -- construction ---------------------------------------------------------
    @classmethod
    def open(
        cls,
        result: CoregResult | str | Path,
        file_a: str | Path | None = None,
        file_b: str | Path | None = None,
        channels: Iterable[LogicalChannel | dict] | None = None,
        default_prefer: FileRole | None = None,
        verify: bool = True,
        password_a: str | None = None,
        password_b: str | None = None,
    ) -> CoregisteredRecording:
        """Open the two files of a coregistration result (roles from the result; files checked
        against the time spans stored in the result unless ``verify=False``). ``file_a``/``file_b``
        override the stored paths — for recordings that have been moved or shared."""
        reader = CoregisteredReader.from_result(
            result, file_a, file_b, verify=verify, password_a=password_a, password_b=password_b
        )
        return cls(reader, channels, default_prefer)

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        result: CoregResult | str | Path | None = None,
        file_a: str | Path | None = None,
        file_b: str | Path | None = None,
        verify: bool = True,
        password_a: str | None = None,
        password_b: str | None = None,
    ) -> CoregisteredRecording:
        """Reopen a recording saved with :meth:`save_config` (result path stored inside).

        ``result``/``file_a``/``file_b`` override the stored paths; a stored result path that no
        longer exists is looked for next to the config. The config holds the result file's hash:
        a result overwritten since is refused (as are MEF3 files whose start or end differs from
        the result's by more than 1 s), unless ``verify=False``. The config and the result may be
        plain or gzip-compressed (``.json.gz``), and a result named in one form is also found in
        the other: the hash is of the JSON content, so a compressed copy matches."""
        from ..io.jsonfile import other_form, read_json

        d = read_json(config_path)
        if d.get("format") != _FORMAT:
            raise ValueError(f"not a recording config (format={d.get('format')!r})")
        res = result if result is not None else d["result"]
        if res is None:
            raise ValueError("this config names no result file; pass result=<the clock-map result>")
        if isinstance(res, (str, Path)):
            here = Path(config_path).parent
            candidates = [Path(res)] if Path(res).is_absolute() else [here / res, Path(res)]
            candidates += [here / Path(res).name]
            candidates += [other_form(c) for c in candidates]  # x.json ↔ x.json.gz
            found = next((c for c in candidates if c.exists()), None)
            if found is None:
                raise FileNotFoundError(
                    f"the result {res} is not there any more (nor compressed, nor next to the "
                    "config); pass result=<where it is now>"
                )
            res = found
        expected = d.get("result_sha256")
        if expected and isinstance(res, (str, Path)) and verify:
            actual = _sha256(res)
            if actual != expected:
                raise ValueError(
                    f"{res} is not the clock-map result this recording was saved with (it has "
                    "changed since). Re-save the mapping, or pass verify=False to use it anyway."
                )
        return cls.open(res, file_a, file_b, d["channels"], d.get("default_prefer"), verify=verify,
                        password_a=password_a, password_b=password_b)

    def save_config(self, path: str | Path, result_path: str | Path | None = None) -> Path:
        """Persist the logical-channel mapping (and the result it applies to) as JSON;
        gzip-compressed when the name ends in ``.gz``."""
        from ..io.jsonfile import write_json

        doc = self.config_dict(
            None if result_path is None else str(result_path),
            None if result_path is None else _sha256(result_path),
        )
        return write_json(path, doc)

    def config_dict(self, result: str | None, result_sha256: str | None) -> dict:
        """The mapping as :meth:`save_config` writes it. ``result`` may be a bare file name: the
        config then expects the result next to it (a downloaded pair kept in one folder)."""
        return {
            "format": _FORMAT,
            "format_version": _FORMAT_VERSION,
            "result": result,
            "result_sha256": result_sha256,
            "default_prefer": self.default_prefer,
            "channels": [lc.to_dict() for lc in self._channels.values()],
        }

    def close(self) -> None:
        # Wait for an in-flight read (a first full-length envelope, an agreement check): closing
        # under it would fail that request with "this Reader is closed".
        with self._lock:
            self.reader.close()

    def __enter__(self) -> CoregisteredRecording:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- channel mapping --------------------------------------------------------
    def _auto_channels(self) -> None:
        """One logical channel per name; a name in both files links them. Names that differ only
        by a common prefix/suffix ("EEG LA1-Ref" ~ "LA1") are linked when that is unambiguous on
        both sides. A link is a proposal: the overlap check (:meth:`agreement`) decides whether
        the two sources may actually be merged."""
        a_names = self.reader.channels("A")
        b_names = self.reader.channels("B")
        b_of: dict[str, str] = {n: n for n in a_names if n in set(b_names)}
        a_norm = _unique_by(_normalise_name, [n for n in a_names if n not in b_of])
        b_norm = _unique_by(_normalise_name, [n for n in b_names if n not in b_of.values()])
        for key, a_name in a_norm.items():
            if key in b_norm:
                b_of[a_name] = b_norm[key]
        for name in a_names:
            b = b_of.get(name)
            self._add(LogicalChannel(
                name,
                a=ChannelSpec(name),
                b=None if b is None else ChannelSpec(b),
                # a channel only one file has can only come from that file
                prefer=self.default_prefer if b is not None else "A",
            ))
        linked_b = set(b_of.values())
        for name in b_names:
            if name not in linked_b and name not in self._channels:
                self._add(LogicalChannel(name, b=ChannelSpec(name), prefer="B"))

    def _add(self, lc: LogicalChannel) -> None:
        files: tuple[FileRole, FileRole] = ("A", "B")
        for file in files:
            spec = lc.source(file)
            if spec is not None:
                self.reader.spec(spec, file)  # validates the channels exist in that file
        self._channels[lc.name] = lc

    def set_channel(
        self,
        name: str,
        a: ChannelSpec | str | tuple | None = None,
        b: ChannelSpec | str | tuple | None = None,
        prefer: FileRole | None = None,
        fs: float | None = None,
        **corrections: float | bool,
    ) -> LogicalChannel:
        """Add or replace logical channel ``name`` (its signal in A and/or B, preference).

        ``corrections``: ``scale_a``/``offset_a``/``scale_b``/``offset_b`` (value' = scale·value +
        offset; scale −1 inverts) and ``allow_mismatch`` (fall back between files even when
        :meth:`agreement` says they disagree)."""
        lc = LogicalChannel(
            name,
            a=None if a is None else self.reader.spec(a, "A"),
            b=None if b is None else self.reader.spec(b, "B"),
            prefer=prefer or self.default_prefer,
            fs=fs,
            **_corrections(dict(corrections)),
        )
        self._add(lc)
        return lc

    def set_preference(self, name: str, prefer: FileRole) -> LogicalChannel:
        from dataclasses import replace

        lc = self.channel(name)
        new = replace(lc, prefer=prefer)
        self._channels[name] = new
        return new

    def remove_channel(self, name: str) -> None:
        self.channel(name)
        del self._channels[name]

    @property
    def channels(self) -> list[str]:
        return list(self._channels)

    def channel(self, name: str) -> LogicalChannel:
        if name not in self._channels:
            raise KeyError(f"no logical channel {name!r}; channels: {self.channels}")
        return self._channels[name]

    def logical_channels(self) -> list[LogicalChannel]:
        return list(self._channels.values())

    # -- metadata (mef-tools-like) --------------------------------------------
    def fs(self, name: str) -> float:
        lc = self.channel(name)
        if lc.fs is not None:
            return float(lc.fs)
        return self.reader.fs(lc.source(lc.primary), lc.primary)  # type: ignore[arg-type]

    def time_bounds(self, name: str) -> tuple[int, int]:
        """``(start, end)`` on the reference clock: the union of the channel's sources."""
        lc = self.channel(name)
        spans = [
            self.reader.time_bounds(spec, file, coregister=True)  # type: ignore[arg-type]
            for file, spec in (("A", lc.a), ("B", lc.b))
            if spec is not None
        ]
        return min(s[0] for s in spans), max(s[1] for s in spans)

    def get_property(self, prop: str, channel: str):
        """mef-tools-style metadata: ``fsamp``, ``start_time``, ``end_time``, ``unit``."""
        if prop == "fsamp":
            return self.fs(channel)
        if prop == "start_time":
            return self.time_bounds(channel)[0]
        if prop == "end_time":
            return self.time_bounds(channel)[1]
        if prop == "unit":
            lc = self.channel(channel)
            spec = lc.source(lc.primary)
            return self.reader.reader(lc.primary).unit(spec.anode)  # type: ignore[union-attr]
        raise KeyError(f"unknown property {prop!r}")

    def units(self, name: str) -> dict[str, str]:
        """Each source's unit (``{"A": ..., "B": ...}``, sources the channel has)."""
        lc = self.channel(name)
        out = {}
        for file in ("A", "B"):
            spec = lc.source(file)
            if spec is not None:
                out[file] = self.reader.reader(file).unit(spec.anode)
        return out

    # -- clock conversion ---------------------------------------------------------
    def to_file_time(self, t_ref_uutc, file: FileRole):
        """Reference-clock time(s) → the given file's own clock (e.g. to mark an event in the
        clinical file for review in its native viewer). Identity for the reference file."""
        t = np.asarray(t_ref_uutc, dtype=np.float64)
        if file == self.reader.reference_file:
            return np.rint(t).astype(np.int64)
        return np.rint(np.asarray(self.reader.clock_map.forward(t))).astype(np.int64)

    def to_reference_time(self, t_file_uutc, file: FileRole):
        """The given file's own clock → reference-clock time(s)."""
        t = np.asarray(t_file_uutc, dtype=np.float64)
        if file == self.reader.reference_file:
            return np.rint(t).astype(np.int64)
        return np.rint(np.asarray(self.reader.clock_map.inverse(t))).astype(np.int64)

    # -- may the two sources be merged? -----------------------------------------------
    def agreement(self, name: str) -> SourceAgreement:
        """Compare file A's and file B's signal for ``name`` over their overlap (cached).

        Up to ``MERGE_CHECK_CHUNKS`` chunks of ``MERGE_CHECK_CHUNK_S`` spread over the overlap are
        read on a common reference-clock grid, corrected (scale, offset) and band-passed alike;
        the correlation, the robust gain ratio and the median offset decide the status (see
        :class:`SourceAgreement`)."""
        with self._lock:
            lc = self.channel(name)
            if lc not in self._agreement:
                self._agreement[lc] = self._compute_agreement(lc)
            return self._agreement[lc]

    def _compute_agreement(self, lc: LogicalChannel) -> SourceAgreement:
        if lc.a is None or lc.b is None:
            return SourceAgreement("single_source", message="one source only")
        units = self.units(lc.name)
        ua, ub = units.get("A", ""), units.get("B", "")
        rdr = self.reader
        (a0, a1) = rdr.time_bounds(lc.a, "A", coregister=True)
        (b0, b1) = rdr.time_bounds(lc.b, "B", coregister=True)
        lo, hi = max(a0, b0), min(a1, b1)
        if hi - lo < MERGE_CHECK_CHUNK_S * 1e6:
            return SourceAgreement("no_overlap", unit_a=ua, unit_b=ub,
                                   message="the two sources barely overlap — nothing to compare")
        fs = min(rdr.fs(lc.a, "A"), rdr.fs(lc.b, "B"))
        n_chunk = int(MERGE_CHECK_CHUNK_S * fs)
        span = hi - lo - MERGE_CHECK_CHUNK_S * 1e6
        centers = lo + (np.arange(MERGE_CHECK_CHUNKS) + 0.5) / MERGE_CHECK_CHUNKS * span
        xa_all, xb_all, ra_all, rb_all = [], [], [], []
        for c in centers:
            ts = uniform_grid(int(c), fs, 0, n_chunk)
            va = self._corrected(lc, "A", rdr.get_data(lc.a, file="A", coregister=True, grid=ts).data)
            vb = self._corrected(lc, "B", rdr.get_data(lc.b, file="B", coregister=True, grid=ts).data)
            ok = np.isfinite(va) & np.isfinite(vb)
            if ok.sum() < n_chunk // 2:
                continue
            idx = np.arange(va.shape[0])
            fa = np.interp(idx, idx[ok], va[ok])
            fb = np.interp(idx, idx[ok], vb[ok])
            ba = _band(fa, fs)
            bb = _band(fb, fs)
            xa_all.append(ba[ok])
            xb_all.append(bb[ok])
            ra_all.append(va[ok])
            rb_all.append(vb[ok])
        if not xa_all:
            return SourceAgreement("no_overlap", unit_a=ua, unit_b=ub,
                                   message="no stretch where both sources have data")
        xa, xb = np.concatenate(xa_all), np.concatenate(xb_all)
        sa = float(np.median(np.abs(xa - np.median(xa))))
        sb = float(np.median(np.abs(xb - np.median(xb))))
        corr = float(np.corrcoef(xa, xb)[0, 1]) if sa > 0 and sb > 0 else 0.0
        gain = sb / sa if sa > 0 else float("inf")
        offset = float(np.median(np.concatenate(rb_all)) - np.median(np.concatenate(ra_all)))
        overlap_s = (hi - lo) / 1e6
        lo_g, hi_g = MERGE_GAIN_RANGE
        common: dict[str, Any] = dict(corr=corr, gain_b_over_a=gain, offset_b_minus_a=offset,
                                      unit_a=ua, unit_b=ub, overlap_s=overlap_s)
        if ua and ub and ua != ub:
            return SourceAgreement("units", message=f"different units ({ua} vs {ub})", **common)
        if corr <= -MERGE_MIN_CORR:
            return SourceAgreement(
                "inverted", message=f"opposite polarity (r = {corr:.2f}) — set a scale of −1 on one "
                "source, or check the pairing", **common)
        if corr < MERGE_MIN_CORR:
            return SourceAgreement(
                "poor", message=f"the two sources barely correlate (r = {corr:.2f}) — are they the "
                "same electrode and derivation?", **common)
        if not lo_g <= gain <= hi_g:
            return SourceAgreement(
                "scale", message=f"amplitudes differ {gain:.3g}× (B/A) — set a scale to match them",
                **common)
        # Baseline is reported, not gated: a DC-coupled amplifier's baseline wanders by more than
        # its signal (a constant offset fits nowhere), so gating on it would refuse sources that
        # agree. A large difference still means a step at every switch — say so.
        spread = 1.4826 * float(np.median(np.abs(np.concatenate(ra_all) - np.median(np.concatenate(ra_all)))))
        note = ""
        if spread > 0 and abs(offset) > spread:
            note = (f"; baseline differs by {offset:.3g} ({abs(offset) / spread:.1f}× the signal's "
                    "spread) — expect a step at each switch: filter each run on its own")
        return SourceAgreement(
            "ok", message=f"agree over the overlap (r = {corr:.2f}, gain {gain:.2f}×){note}",
            baseline_step=bool(note), **common)

    def _corrected(self, lc: LogicalChannel, file: FileRole, values: np.ndarray) -> np.ndarray:
        scale, offset = lc.correction(file)
        if scale == 1.0 and offset == 0.0:
            return np.asarray(values, dtype=np.float64)
        return np.asarray(values, dtype=np.float64) * scale + offset

    def _merge_allowed(self, lc: LogicalChannel) -> tuple[bool, str]:
        if lc.fallback is None or lc.allow_mismatch:
            return True, ""
        ag = self.agreement(lc.name)
        if ag.ok:
            return True, ""
        return False, f"not filled from file {lc.fallback}: {ag.message}"

    # -- data -------------------------------------------------------------------
    def get_data(
        self,
        channel: str | list[str],
        start: int | None = None,
        stop: int | None = None,
        fs: float | None = None,
    ) -> UnifiedRead | list[UnifiedRead]:
        """Read logical channel(s) over ``[start, stop)`` reference-clock uUTC.

        The preferred source is used wherever it has data; the other file fills the rest.
        ``fs`` overrides the output rate (default: the channel's configured or native rate).
        """
        with self._lock:
            if isinstance(channel, (list, tuple)):
                return [self._read(c, start, stop, fs) for c in channel]
            return self._read(channel, start, stop, fs)

    def _read(self, name: str, start: int | None, stop: int | None, fs: float | None) -> UnifiedRead:
        lc = self.channel(name)
        rdr = self.reader
        primary = lc.primary
        pspec: ChannelSpec = lc.source(primary)  # type: ignore[assignment]
        native_fs = rdr.fs(pspec, primary)
        out_fs = float(fs if fs is not None else (lc.fs if lc.fs is not None else native_fs))

        lo, hi = self.time_bounds(name)
        start = lo if start is None else int(start)
        stop = hi if stop is None else int(stop)
        origin = rdr.grid_origin(pspec, primary)
        k0, k1 = grid_indices(origin, out_fs, start, stop)
        ts = uniform_grid(origin, out_fs, k0, k1 - k0)
        if ts.size == 0:
            empty = np.empty(0)
            return UnifiedRead(name, empty, ts, out_fs, start, empty.astype(np.uint8), empty.astype(bool))

        data = None
        if primary == rdr.reference_file and out_fs == native_fs:
            # Native reference samples, untransformed — when they lie on ``ts``. A bipolar
            # pair whose channels start a fraction of a sample apart is derived on the anode's
            # grid, which is not ``ts``: then it is evaluated on ``ts`` like any other read.
            native = read_montage(rdr.reader(primary), pspec, int(ts[0]), int(ts[-1]) + 1)
            n = min(native.n, ts.size)
            if n and np.array_equal(native.ts[:n], ts[:n]):
                data = native.data[: ts.size] if native.n >= ts.size else np.pad(
                    native.data, (0, ts.size - native.n), constant_values=np.nan
                )
        if data is None:
            data = rdr.get_data(pspec, file=primary, coregister=True, grid=ts).data
        data = self._corrected(lc, primary, np.array(data, dtype=np.float64))
        source = np.where(np.isnan(data), SOURCE_NONE, _SOURCE_CODE[primary]).astype(np.uint8)

        fallback = lc.fallback
        missing = np.flatnonzero(np.isnan(data))
        merge_ok, note = self._merge_allowed(lc) if (fallback and missing.size) else (True, "")
        if fallback is not None and missing.size and merge_ok:
            i0, i1 = int(missing[0]), int(missing[-1]) + 1
            fspec: ChannelSpec = lc.source(fallback)  # type: ignore[assignment]
            fill = self._corrected(
                lc, fallback, rdr.get_data(fspec, file=fallback, coregister=True, grid=ts[i0:i1]).data
            )
            seg = data[i0:i1]
            use = np.isnan(seg) & ~np.isnan(fill)
            seg[use] = fill[use]
            source[i0:i1][use] = _SOURCE_CODE[fallback]

        from_moving = source == _SOURCE_CODE[rdr.moving_file]
        extrapolated = from_moving & np.asarray(rdr.clock_map.is_extrapolated(ts), dtype=bool)
        return UnifiedRead(name, data, ts, out_fs, int(ts[0]), source, extrapolated, note)

    # -- display (min/max buckets) -------------------------------------
    def display(
        self, name: str, start: int | None = None, stop: int | None = None, max_points: int = 2000
    ) -> dict:
        """Min/max buckets of a logical channel for a viewer, with per-bucket source.

        ``[start, stop)`` (reference clock) is split into ``max_points`` equal buckets. Zoomed
        out, buckets are filled from each source's ~50 Hz min-max envelope (computed once by the
        chunked loader and cached — the display product, never full-rate data), mapped onto the
        reference clock; the preferred source wins per bucket, the other fills the rest.
        Zoomed in (buckets narrower than a few envelope bins), the exact :meth:`get_data`
        samples are bucketed instead. Returns ``start_uutc``, ``fs`` (buckets per second),
        ``n_source`` (samples the span represents), ``kind`` (``"minmax"`` buckets, or
        ``"line"`` with the samples themselves once every sample gets a bucket), ``dt_us``,
        ``time_accurate``, ``mins``/``maxs`` (and ``values`` for lines), ``source`` (0/1/2/3 =
        none/A/B/mixed) and ``extrapolated``.
        """
        with self._lock:
            return self._display(name, start, stop, max_points)

    def _display(self, name: str, start: int | None, stop: int | None, max_points: int) -> dict:
        lc = self.channel(name)
        lo, hi = self.time_bounds(name)
        start = lo if start is None else int(start)
        stop = hi if stop is None else int(stop)
        nb = max(1, int(max_points))
        width = max(1.0, (stop - start) / nb)  # µs per bucket

        env_bin_us = 1e6 / 50.0
        time_accurate = width < 4 * env_bin_us
        if time_accurate:
            sig = self._read(name, start, stop, None)
            if sig.n <= nb:  # every sample gets a pixel: send the samples themselves
                extrap = (sig.source == _SOURCE_CODE[self.reader.moving_file]) & np.asarray(
                    self.reader.clock_map.is_extrapolated(sig.ts), dtype=bool
                )
                return {
                    "kind": "line",
                    "time_accurate": True,
                    "start_uutc": int(sig.ts[0]) if sig.n else int(start),
                    "fs": float(sig.fs),
                    "dt_us": 1e6 / float(sig.fs),
                    "n_source": int(sig.n),
                    "values": sig.data,
                    "mins": sig.data,
                    "maxs": sig.data,
                    "source": sig.source,
                    "extrapolated": extrap,
                    "merge_note": sig.merge_note,
                }
            idx = np.floor((sig.ts - start) / width).astype(np.int64)
            mins, maxs, src, _ = _bucket(idx, sig.data, sig.data, sig.source, nb)
            note = sig.merge_note
        else:
            mins = np.full(nb, np.nan)
            maxs = np.full(nb, np.nan)
            src = np.zeros(nb, dtype=np.uint8)
            merge_ok, note = self._merge_allowed(lc) if lc.fallback else (True, "")
            for file in (lc.primary, lc.fallback if merge_ok else None):
                if file is None:
                    continue
                env, t_ref = self._envelope_on_ref(file, lc.source(file))  # type: ignore[arg-type]
                # only the bins inside the viewport (t_ref is monotone: the map is)
                i0, i1 = np.searchsorted(t_ref, [start, stop])
                sl = slice(int(i0), int(i1))
                idx = np.floor((t_ref[sl] - start) / width).astype(np.int64)
                code = np.full(idx.shape, _SOURCE_CODE[file], dtype=np.uint8)
                scale, offset = lc.correction(file)
                emin, emax = env.mins[sl] * scale + offset, env.maxs[sl] * scale + offset
                if scale < 0:  # an inverted source swaps its extremes
                    emin, emax = emax, emin
                fm, fx, fs_, has = _bucket(idx, emin, emax, code, nb)
                fill = ~np.isfinite(mins) & has
                mins[fill], maxs[fill], src[fill] = fm[fill], fx[fill], fs_[fill]

        centers = start + (np.arange(nb) + 0.5) * width
        from_moving = src == _SOURCE_CODE[self.reader.moving_file]
        extrap = from_moving & np.asarray(self.reader.clock_map.is_extrapolated(centers), dtype=bool)
        return {
            "kind": "minmax",
            "time_accurate": time_accurate,  # False: from ~50 Hz envelopes (display only)
            "start_uutc": int(start),
            "fs": 1e6 / width,
            "dt_us": float(width),
            "n_source": int(round((stop - start) / 1e6 * self.fs(name))),  # samples represented
            "mins": mins,
            "maxs": maxs,
            "source": src,
            "extrapolated": extrap,
            "merge_note": note,
        }

    def _envelope_on_ref(self, file: FileRole, spec: ChannelSpec) -> tuple[Envelope, np.ndarray]:
        """A source's display envelope plus its bin times on the reference clock (cached)."""
        key = (file, spec)
        if key not in self._env_ref:
            env = self._envelope(file, spec)
            t_native = int(env.start_uutc or 0) + np.arange(env.mins.shape[0]) * (1e6 / env.fs)
            t_ref = (
                np.asarray(self.reader.clock_map.inverse(t_native), dtype=np.float64)
                if file == self.reader.moving_file
                else t_native
            )
            self._env_ref[key] = (env, t_ref)
        return self._env_ref[key]

    def _envelope(self, file: FileRole, spec: ChannelSpec) -> Envelope:
        key = (self.reader.paths[file], spec)  # by path: safe to share across recordings
        if key not in self._envelopes:
            load = LoadSpec.for_channel(self.reader.paths[file], spec, decimate=False, chunk_s=600.0)
            self._envelopes[key] = load_alignment_signal(load).envelope
        return self._envelopes[key]


def _sha256(path: str | Path) -> str:
    """The hash a mapping stores for its result: of the JSON content, so the plain file and its
    gzip-compressed copy hash alike."""
    from ..io.jsonfile import content_sha256

    return content_sha256(path)


_NAME_PREFIX = re.compile(r"^(?:s?eeg|ecog|ieeg|pol)[\s_:-]+", re.IGNORECASE)
_NAME_SUFFIX = re.compile(r"[\s_:-]*(?:ref|reference|avg|ave|cz)$", re.IGNORECASE)


def _normalise_name(name: str) -> str:
    """A channel name without the prefixes/suffixes systems add ("EEG LA1-Ref" → "la1")."""
    n = _NAME_PREFIX.sub("", name.strip())
    n = _NAME_SUFFIX.sub("", n)
    return re.sub(r"[\s_]+", "", n).casefold()


def _unique_by(key, names: list[str]) -> dict[str, str]:
    """``{key(name): name}`` for keys that exactly one name produces (ambiguity → no link)."""
    seen: dict[str, list[str]] = {}
    for n in names:
        k = key(n)
        if k:
            seen.setdefault(k, []).append(n)
    return {k: v[0] for k, v in seen.items() if len(v) == 1}


def _band(x: np.ndarray, fs: float) -> np.ndarray:
    """The estimation band both sources are compared in (same zero-phase filters as alignment)."""
    return _filter(x, fs, *MERGE_BAND_HZ)


def _bucket(
    idx: np.ndarray, lo: np.ndarray, hi: np.ndarray, code: np.ndarray, nb: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-bucket min of ``lo``, max of ``hi`` and source code (3 if mixed), ignoring NaN."""
    ok = (idx >= 0) & (idx < nb) & ~np.isnan(lo) & ~np.isnan(hi)
    idx, lo, hi, code = idx[ok], lo[ok], hi[ok], code[ok]
    mins = np.full(nb, np.inf)
    maxs = np.full(nb, -np.inf)
    np.minimum.at(mins, idx, lo)
    np.maximum.at(maxs, idx, hi)
    has = np.isfinite(mins)
    mins[~has] = np.nan
    maxs[~has] = np.nan
    cmin = np.full(nb, 255, dtype=np.uint8)
    cmax = np.zeros(nb, dtype=np.uint8)
    np.minimum.at(cmin, idx, code)
    np.maximum.at(cmax, idx, code)
    src = np.where(has, np.where(cmin == cmax, cmax, 3), 0).astype(np.uint8)
    return mins, maxs, src, has
