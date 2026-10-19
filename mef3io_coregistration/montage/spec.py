"""Channel specifications: unipolar or bipolar, chosen independently per file.

A :class:`ChannelSpec` names what to read from *one* file: a single channel (unipolar) or an
``anode - cathode`` derivation (bipolar). Nothing assumes the two files share channel names or
montage — e.g. a bipolar pair on one file may be aligned against a unipolar channel on the
other, and a scalp montage against an iEEG one. :func:`read_montage` evaluates a spec against a
:class:`~mef3io_coregistration.io.mef_reader.MefChannelReader`.

Specs can be written as strings: ``"LA1"`` (unipolar) or ``"LA1-LA2"`` (bipolar). Because channel
names may themselves contain ``-``, :meth:`ChannelSpec.parse` resolves a string against the
file's channel list when one is given.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from ..io.mef_reader import ChannelRead, Gap, MefChannelReader
from .bipolar import bipolar


@dataclass(frozen=True)
class ChannelSpec:
    """One file's signal: ``anode`` alone (unipolar) or ``anode - cathode`` (bipolar)."""

    anode: str
    cathode: str | None = None

    @property
    def is_bipolar(self) -> bool:
        return self.cathode is not None

    @property
    def channels(self) -> tuple[str, ...]:
        return (self.anode,) if self.cathode is None else (self.anode, self.cathode)

    @property
    def label(self) -> str:
        return self.anode if self.cathode is None else f"{self.anode}-{self.cathode}"

    def __str__(self) -> str:
        return self.label

    def to_dict(self) -> dict:
        return {"anode": self.anode, "cathode": self.cathode}

    @classmethod
    def parse(
        cls, spec: ChannelSpec | str | tuple | list | dict, channels: Iterable[str] | None = None
    ) -> ChannelSpec:
        """Coerce ``spec`` into a :class:`ChannelSpec`.

        Accepts a spec, a ``(anode, cathode)`` tuple (cathode may be ``None``), a dict with
        ``anode``/``cathode``, or a string. A string that is itself a channel name is unipolar;
        otherwise it is split at the one ``-`` whose two sides are both channels. Without a
        channel list a string is split only if it contains exactly one ``-``.
        """
        if isinstance(spec, ChannelSpec):
            return spec
        if isinstance(spec, dict):
            return cls(str(spec["anode"]), spec.get("cathode") or None)
        if isinstance(spec, (tuple, list)):
            if len(spec) == 1:
                return cls(str(spec[0]))
            if len(spec) == 2:
                return cls(str(spec[0]), spec[1] or None)
            raise ValueError(f"channel spec tuple must be (anode[, cathode]), got {spec!r}")
        if not isinstance(spec, str) or not spec:
            raise ValueError(f"invalid channel spec {spec!r}")

        if channels is None:
            parts = spec.split("-")
            if len(parts) == 2 and parts[0] and parts[1]:
                return cls(parts[0].strip(), parts[1].strip())
            return cls(spec)

        names = set(channels)
        if spec in names:
            return cls(spec)
        splits = [
            (spec[:i].strip(), spec[i + 1 :].strip())
            for i, ch in enumerate(spec)
            if ch == "-" and spec[:i].strip() in names and spec[i + 1 :].strip() in names
        ]
        if len(splits) == 1:
            return cls(*splits[0])
        if len(splits) > 1:
            raise ValueError(f"ambiguous bipolar spec {spec!r}; pass (anode, cathode) explicitly")
        raise KeyError(
            f"{spec!r} is neither a channel nor an 'anode-cathode' pair of channels; "
            f"available: {sorted(names)}"
        )


def spec_fs(reader: MefChannelReader, spec: ChannelSpec) -> float:
    """Sampling rate of ``spec`` (the lower of its channels' rates; bipolar harmonizes down)."""
    return min(reader.fs(ch) for ch in spec.channels)


def spec_bounds(reader: MefChannelReader, spec: ChannelSpec) -> tuple[int, int]:
    """``(start, end)`` uUTC where every channel of ``spec`` is recorded (intersection)."""
    bounds = [reader.time_bounds(ch) for ch in spec.channels]
    return max(b[0] for b in bounds), min(b[1] for b in bounds)


def spec_gaps(reader: MefChannelReader, spec: ChannelSpec) -> tuple[Gap, ...]:
    """Union of the gaps of every channel in ``spec``, sorted by start."""
    gaps = {g for ch in spec.channels for g in reader.gaps(ch)}
    return tuple(sorted(gaps, key=lambda g: g.start_uutc))


def read_montage(
    reader: MefChannelReader,
    spec: ChannelSpec,
    start: int | None = None,
    stop: int | None = None,
) -> ChannelRead:
    """Read ``spec`` over ``[start, stop)`` uUTC from one file.

    Unipolar reads pass straight through. A bipolar read subtracts the cathode from the anode
    sample-for-sample when both share a grid; otherwise (differing Fs or grid phase) the two are
    harmonized to the lower Fs first (:func:`~mef3io_coregistration.montage.bipolar.bipolar`).
    Gaps in either channel propagate as ``NaN`` and are listed in the result.
    """
    if spec.cathode is None:
        return reader.read(spec.anode, start, stop)

    a = reader.read(spec.anode, start, stop)
    c = reader.read(spec.cathode, start, stop)
    gaps = tuple(sorted(set(a.gaps) | set(c.gaps), key=lambda g: g.start_uutc))
    if a.fs == c.fs and a.n == c.n and np.array_equal(a.ts, c.ts):
        return ChannelRead(spec.label, a.data - c.data, a.ts, a.fs, a.start_uutc, gaps)
    if a.n == 0 or c.n == 0:
        empty = np.empty(0, dtype=np.float64)
        return ChannelRead(spec.label, empty, empty.astype(np.int64), min(a.fs, c.fs), a.start_uutc, ())
    b = bipolar(a, c)
    return ChannelRead(spec.label, b.data, b.ts, b.fs, b.start_uutc, gaps)
