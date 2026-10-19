"""MEF-style access to either file of a coregistration, with every time on the reference clock.

Two views of ONE file, so existing MEF code needs only a different constructor:

* :class:`CoregisteredSession` mirrors ``mef3io.Reader`` — ``channels``, ``info``, ``read``,
  ``records``, ``segments``, ``close``.
* :class:`CoregisteredMefReader` mirrors the legacy ``mef_tools.io.MefReader`` (as re-exported by
  ``mef3io.compat``) one to one — ``channels``, ``properties``, ``get_data``, ``get_property``,
  ``get_channel_info``, ``get_annotations``, ``close``.

::

    from mef3io_coregistration.reader import CoregisteredSession, CoregisteredMefReader

    with CoregisteredSession("result.json", file="moving") as r:   # or "A" / "B" / "reference"
        x = r.read("LTP1", t0, t1)            # any channel of that file, reference-clock uUTC
        y = r.read("LTP1-LTP3", t0, t1)       # bipolar: subtracted natively, then resampled
        ts = r.times("LTP1", t0, t1)          # the uUTC of each returned sample

    mr = CoregisteredMefReader("result.json", file="moving")
    x = mr.get_data("LTP1", t0, t1)           # ndarray;  a list of channels → list of arrays
    fs = mr.get_property("fsamp", "LTP1")

Times in and out are **reference-clock uUTC** for both files. The reference file is read as is
(its own samples, exactly as ``mef3io`` returns them). The moving file is read over only the
blocks the window needs (plus a few samples for the interpolation filter), and resampled through
the clock map onto a uniform grid on the reference clock at its own sampling rate (``fs=`` to
choose another), anchored at the channel's start — so consecutive half-open reads tile exactly.
The drift is a property of the two recording systems, not of a channel: one map serves every
channel of the file.

A channel is a name, an ``"anode-cathode"`` expression (split where both sides are channels) or
an ``(anode, cathode)`` tuple. A bipolar channel is always derived on the file's own samples
first and only then resampled — never the other way round.

Opening: ``file_a=``/``file_b=`` override the paths stored in the result (the result is only the
transformation — the recordings can be moved and shared); without them a moved recording is
looked for next to the result file. ``password_a``/``password_b`` open encrypted sessions.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..io.mef_reader import ChannelRead
from ..montage.spec import ChannelSpec, spec_bounds, spec_fs
from .coregistered_reader import CoregisteredReader, FileRole

if TYPE_CHECKING:
    from ..result import CoregResult

_ROLES = {"a": "A", "b": "B"}
# Absolute uUTC body times of mef3io records (besides the header ``time``); Seiz channels' own
# ``onset``/``offset`` are handled per channel.
_RECORD_BODY_TIMES = ("earliest_onset", "latest_offset", "timestamp", "end_timestamp", "trace_timestamp")
_END_OF_TIME = 2**63 - 1  # MEF's "no end" marker (INT64_MAX): not a time, never mapped


class CoregisteredSession:
    """One file of a coregistration, read like a ``mef3io.Reader`` on the reference clock."""

    def __init__(
        self,
        result: CoregResult | str | Path | CoregisteredReader,
        file: str = "moving",
        *,
        file_a: str | Path | None = None,
        file_b: str | Path | None = None,
        password_a: str | None = None,
        password_b: str | None = None,
        verify: bool = True,
    ) -> None:
        if isinstance(result, CoregisteredReader):
            self._rd, self._own = result, False
        else:
            self._rd = CoregisteredReader.from_result(
                result, file_a, file_b, verify=verify, password_a=password_a, password_b=password_b
            )
            self._own = True
        self.file: FileRole = self._role(file)
        self.is_reference = self.file == self._rd.reference_file
        self.path = self._rd.paths[self.file]

    def _role(self, file: str) -> FileRole:
        f = str(file).strip().lower()
        if f == "reference":
            return self._rd.reference_file
        if f == "moving":
            return self._rd.moving_file
        if f in _ROLES:
            return _ROLES[f]  # type: ignore[return-value]
        raise ValueError(f'file must be "A", "B", "reference" or "moving", got {file!r}')

    # -- context management ---------------------------------------------------------
    def __enter__(self) -> CoregisteredSession:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the files (only if this object opened them)."""
        if self._own:
            self._rd.close()

    # -- metadata ---------------------------------------------------------------------
    @property
    def reader(self) -> CoregisteredReader:
        """The underlying two-file reader (e.g. for :meth:`CoregisteredReader.get_data` with
        ``grid=`` — the moving file on the reference file's exact sample times)."""
        return self._rd

    @property
    def channels(self) -> list[str]:
        """Channel names in this file, sorted (bipolar expressions are accepted by every read)."""
        return sorted(self._rd.channels(self.file))

    def spec(self, channel: ChannelSpec | str | tuple) -> ChannelSpec:
        return self._rd.spec(channel, self.file)

    def fs(self, channel: ChannelSpec | str | tuple) -> float:
        """Native sampling rate of ``channel`` (a bipolar pair: the rate it is derived at)."""
        return self._rd.fs(channel, self.file)

    def time_bounds(self, channel: ChannelSpec | str | tuple) -> tuple[int, int]:
        """``(start, end)`` of ``channel`` in reference-clock uUTC."""
        return self._rd.time_bounds(channel, self.file, coregister=True)

    def info(self, channel: ChannelSpec | str | tuple) -> dict[str, Any]:
        """``mef3io.Reader.info`` for ``channel``, with ``start_time``/``end_time`` on the
        reference clock (``start_time_file``/``end_time_file``: the file's own clock).

        A bipolar channel reports the anode's metadata with its own name, rate and extent."""
        spec = self.spec(channel)
        rdr = self._rd.reader(self.file)
        d = dict(rdr.info(spec.anode))
        f0, f1 = spec_bounds(rdr, spec)
        r0, r1 = self.time_bounds(spec)
        d.update(
            name=spec.label,
            sampling_frequency=spec_fs(rdr, spec),
            start_time=r0,
            end_time=r1,
            start_time_file=f0,
            end_time_file=f1,
            clock="reference",
            coregistered=not self.is_reference,
            montage="bipolar" if spec.cathode else "unipolar",
        )
        return d

    # -- data ---------------------------------------------------------------------------
    def read_channel(
        self,
        channel: ChannelSpec | str | tuple,
        t0: int | None = None,
        t1: int | None = None,
        fs: float | None = None,
    ) -> ChannelRead:
        """Samples of ``[t0, t1)`` (reference-clock uUTC) with their timestamps and the gaps
        (reference clock) — the full record behind :meth:`read`."""
        spec = self.spec(channel)
        if self.is_reference and fs is None:
            return self._rd.get_data(spec, t0, t1, file=self.file, coregister=True)
        # The moving file at its own rate by default — never silently adopt another grid.
        rate = float(fs) if fs is not None else self.fs(spec)
        return self._rd.get_data(spec, t0, t1, file=self.file, coregister=True, fs=rate)

    def read(
        self,
        channel: ChannelSpec | str | tuple,
        t0: int | None = None,
        t1: int | None = None,
        fs: float | None = None,
    ) -> np.ndarray:
        """float64 samples of ``[t0, t1)`` (reference-clock uUTC) on a uniform grid; ``NaN`` in
        gaps and outside the recording; physical units. ``None`` bounds span the channel."""
        return self.read_channel(channel, t0, t1, fs).data

    def times(
        self,
        channel: ChannelSpec | str | tuple,
        t0: int | None = None,
        t1: int | None = None,
        fs: float | None = None,
    ) -> np.ndarray:
        """Reference-clock uUTC (int64) of the samples :meth:`read` returns for the same call."""
        return self.read_channel(channel, t0, t1, fs).ts

    def records(self, channel: str | None = None, **kwargs) -> list[dict]:
        """Annotations (``mef3io.Reader.records``) with every absolute time on the reference clock
        and the original kept as ``<key>_file``: ``time`` and the body times (Seiz
        ``earliest_onset``/``latest_offset`` and each channel's ``onset``/``offset``, Epoc
        ``timestamp``/``end_timestamp``, Curs ``trace_timestamp``). Values <= 0 and INT64_MAX are
        MEF's no-entry markers and stay as they are. Durations (``duration``) stay in the file's
        clock µs: on the moving file, end - start on the reference clock differs from ``duration``
        by the drift over the record. ``kwargs`` pass through (``level=``, ``segment=``)."""
        out = []
        for rec in self._rd.reader(self.file).records(channel, **kwargs):
            r = dict(rec)
            if r.get("time") is not None:
                r["time_file"] = r["time"]
                r["time"] = self._to_ref(int(r["time"]))
            self._body_times_to_ref(r, _RECORD_BODY_TIMES)
            if isinstance(r.get("channels"), list):
                r["channels"] = [
                    self._body_times_to_ref(dict(c), ("onset", "offset")) if isinstance(c, dict) else c
                    for c in r["channels"]
                ]
            out.append(r)
        return out

    def _body_times_to_ref(self, d: dict, keys) -> dict:
        for k in keys:
            v = d.get(k)
            if v is not None and 0 < int(v) < _END_OF_TIME:
                d[k + "_file"] = v
                d[k] = self._to_ref(int(v))
        return d

    def segments(self, channel: str) -> list[dict]:
        """``mef3io.Reader.segments`` with ``start_time``/``end_time`` on the reference clock."""
        rdr = self._rd.reader(self.file)
        rdr.info(channel)  # raises a readable KeyError for an unknown channel
        out = []
        for seg in rdr.segments(channel):
            s = dict(seg)
            for k in ("start_time", "end_time"):
                s[k + "_file"] = s[k]
                s[k] = self._to_ref(int(s[k]))
            out.append(s)
        return out

    def _to_ref(self, t_file: int) -> int:
        if self.is_reference:
            return int(t_file)
        return int(np.rint(np.asarray(self._rd.clock_map.inverse(float(t_file)))))


class CoregisteredMefReader:
    """One file of a coregistration through the legacy ``mef_tools.io.MefReader`` API, 1:1.

    ``MefReader(session_path, password2=None)`` becomes
    ``CoregisteredMefReader(result, file="moving", password2=None)``; every other call is
    unchanged, with times on the reference clock. ``get_raw_data`` (stored integer counts) exists
    only for the reference file: resampled moving-file data has no stored counts.
    """

    #: The per-channel properties ``get_property`` serves (as ``mef_tools`` names them).
    PROPERTIES = ("name", "fsamp", "nsamp", "ufact", "unit", "start_time", "end_time",
                  "channel_description")

    def __init__(
        self,
        result: CoregResult | str | Path | CoregisteredReader,
        file: str = "moving",
        password2: str | None = None,
        *,
        file_a: str | Path | None = None,
        file_b: str | Path | None = None,
        verify: bool = True,
    ) -> None:
        # password2 is the legacy name; it opens the file this reader serves.
        role = file.strip().lower()
        pw: dict[str, str | None] = {"password_a": None, "password_b": None}
        if password2 is not None:
            if isinstance(result, CoregisteredReader):
                raise ValueError("pass passwords when opening the CoregisteredReader")
            from ..result import CoregResult

            res = result if isinstance(result, CoregResult) else CoregResult.load(result)
            target = (res.reference_role if role == "reference" else
                      ("A" if res.reference_role == "B" else "B") if role == "moving" else
                      _ROLES.get(role, role))
            pw[f"password_{str(target).lower()}"] = password2
        self._s = CoregisteredSession(result, file, file_a=file_a, file_b=file_b, verify=verify, **pw)

    @property
    def session(self) -> CoregisteredSession:
        return self._s

    @property
    def channels(self) -> list[str]:
        return self._s.channels

    @property
    def properties(self) -> list[str]:
        return list(self.PROPERTIES)

    def close(self) -> None:
        self._s.close()

    def get_data(self, channels, t_stamp1=None, t_stamp2=None):
        """Physical-unit data of ``channels`` over ``[t_stamp1, t_stamp2)`` (reference-clock
        uUTC): an array for one channel, a list of arrays for a list. Channels may be bipolar
        (``"LTP1-LTP3"``)."""
        t1 = None if t_stamp1 is None else int(t_stamp1)
        t2 = None if t_stamp2 is None else int(t_stamp2)
        if isinstance(channels, (list, np.ndarray)):
            return [self._s.read(_name(c), t1, t2) for c in channels]
        return self._s.read(_name(channels), t1, t2)

    def get_raw_data(self, channels, t_stamp1=None, t_stamp2=None):
        """Stored integer counts — the reference file only (see the class docstring)."""
        if not self._s.is_reference:
            raise NotImplementedError(
                "raw counts exist only for the reference file; the moving file is resampled onto "
                "the reference clock — use get_data"
            )
        rdr = self._s.reader.reader(self._s.file)
        names = channels if isinstance(channels, (list, np.ndarray)) else [channels]
        t1 = None if t_stamp1 is None else int(t_stamp1)
        t2 = None if t_stamp2 is None else int(t_stamp2)
        return [rdr.read_raw_counts(_name(c), t1, t2) for c in names]

    def get_property(self, property_name: str, channel=None):
        """``fsamp``, ``nsamp``, ``ufact``, ``unit``, ``start_time``/``end_time`` (reference
        clock), ``name``, ``channel_description`` — one channel, or a list over all channels."""
        if channel is None:
            return [self.get_property(property_name, c) for c in self.channels]
        return self._props(channel).get(property_name)

    def get_channel_info(self, channel=None):
        """The ``mef_tools`` channel-info dict (values wrapped in one-element lists, as there)."""
        if channel is None:
            return [self.get_channel_info(c) for c in self.channels]
        p = self._props(channel)
        info: dict[str, Any] = {k: [v] for k, v in p.items() if k not in ("name", "channel_description")}
        info["name"] = p["name"]
        info["channel"] = p["name"]
        return info

    def get_annotations(self, channel=None):
        """Annotations as the legacy reader returns them: session-level, or a channel's
        channel-level records only (``mef3io.compat.MefReader.get_annotations``) — with every
        absolute time on the reference clock, as :meth:`CoregisteredSession.records`."""
        if channel is None:
            return self._s.records()
        return self._s.records(channel, level="channel")

    def _props(self, channel) -> dict[str, Any]:
        i = self._s.info(_name(channel))
        fs = float(i["sampling_frequency"])
        unit = i.get("units_description") or ""
        if self._s.is_reference and i["montage"] == "unipolar":
            nsamp = int(i["number_of_samples"])  # stored samples, exactly as mef_tools reports
        else:  # samples of the uniform grid this reader returns over the channel's span
            nsamp = int(round((i["end_time"] - i["start_time"]) * fs / 1e6))
        return {
            "name": i["name"],
            "fsamp": fs,
            "nsamp": nsamp,
            "ufact": float(i.get("units_conversion_factor", 1.0)),
            "unit": unit.encode() if isinstance(unit, str) else unit,
            "start_time": int(i["start_time"]),
            "end_time": int(i["end_time"]),
            "channel_description": i.get("channel_description", ""),
        }


def _name(channel) -> Any:
    """numpy string scalars and bytes (what ``mef_tools`` users often pass) → str."""
    if isinstance(channel, bytes):
        return channel.decode()
    if isinstance(channel, np.str_):
        return str(channel)
    return channel
