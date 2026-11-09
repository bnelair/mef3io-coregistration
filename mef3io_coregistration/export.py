"""Write a brand-new MEF 3.0 session from a coregistration result and the two recordings.

``export_coregistered`` puts channels of both recordings on ONE clock, the reference clock, and
writes them to a new ``.mefd``:

* **reference channels** keep their own samples and their own times (no value is changed);
* **moving channels** are resampled with the band-limited evaluation of
  :mod:`~mef3io_coregistration.transform.resample` onto a uniform grid on the reference clock, at
  each channel's own sampling rate. The grid starts at the reference-clock time of the channel's
  first sample kept (the first ``support_samples`` samples are dropped, see below), so the output
  times are fixed, regular and comparable with the reference ones.

Both are read as mef3io reads them: on each channel's nominal grid from its start (a segment that
resumes a fraction of a sample off that grid is placed on it, as everywhere in this package).

Gaps stay gaps: a source gap is ``NaN`` after the resampling (plus the filter support around it,
as everywhere in this package) and ``NaN`` is stored as a discontinuity, never interpolated.

The caller says which recording is the reference, whatever the result's own direction was: the
opposite choice uses the inverse of the clock map.

Beyond the span the clock map was measured over (``ClockMap.measured_span``, with the engine's
1 s span tolerance) the map can only extrapolate. By default that is REFUSED, naming the channels
and how far they reach out; ``out_of_span="trim"`` writes only the measured part and
``"extrapolate"`` follows the map's held end slope (the reader's behaviour).

Resources and safety:

* ``n_threads`` is mef3io's one thread rule: the same count goes to the mef3io ``Reader`` (open,
  file access, decoding), the ``Writer`` (encoding) and this module's own resampling pool
  (``0`` = automatic, ``1`` = no extra threads). The output does not depend on it.
* Memory is bounded: the recordings are processed in chunks of ``chunk_seconds`` with the resampler's
  support margin on each side, never loaded whole. The decoded values and times do not depend on the
  chunk size (the RED block layout in the file does, as blocks are cut per write).
* The session is written inside ``<name>.partial/`` next to the output, read back and checked
  (channel set, rates, sample counts, start times, the first and last samples), and only then
  renamed. An existing output (or partial folder) is refused and never touched; on failure or
  cancel only what this run created is removed. The inputs are only read.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import threading
import warnings
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from . import __version__
from .config import US_PER_S
from .io.jsonfile import content_sha256, dumps, file_name
from .io.mef_reader import SPAN_TOLERANCE_US
from .reader.coregistered_reader import _check_spans, _locate
from .result import CoregResult
from .transform.clock_map import ClockMap
from .transform.resample import evaluate_at, first_index_at_or_after, support_samples, uniform_grid

OutOfSpan = Literal["refuse", "trim", "extrapolate"]

#: Source samples fetched beyond a chunk's mapped span on each side, at least (the resampler's own
#: ``2 * support_samples + 16`` is used when larger).
_MIN_PAD_SAMPLES = 256
#: Samples at each end of a channel compared exactly when the output is read back.
_VERIFY_EDGE = 512
#: Samples (all channels of a grid group together) processed per chunk at most: bounds memory
#: whatever the channel count (float64: 128 MB per copy; reading and resampling hold a few).
_MAX_CHUNK_SAMPLES = 1 << 24
#: Longest provenance text written as a record; a longer channel list is summarised by counts.
_MAX_NOTE_CHARS = 4000


class ExportCancelled(RuntimeError):
    """The export was cancelled; everything this run created has been removed."""


class ExportVerificationError(RuntimeError):
    """The written session did not read back as written; the partial folder was removed."""


@dataclass
class ExportedChannel:
    """One channel of the new session."""

    name: str  #: name in the new session
    source: Literal["reference", "moving"]
    source_name: str  #: name in the source recording
    fs: float
    start_uutc: int  #: reference-clock time of the first stored sample
    n_samples: int  #: samples stored (gap samples are not stored)
    n_grid_samples: int  #: samples on the output grid, gaps included
    units_conversion_factor: float
    precision: int | None  #: decimal digits of the float path; ``None`` = counts stored verbatim
    trimmed_us: tuple[int, int] = (0, 0)  #: time cut off at the start / end (``out_of_span="trim"``)
    extrapolated_us: tuple[int, int] = (0, 0)  #: output time beyond the measured span (start / end)


@dataclass
class ExportReport:
    """What :func:`export_coregistered` wrote."""

    path: str
    channels: list[ExportedChannel]
    result_sha256: str
    reference_role: str  #: "A" or "B": the recording that kept its clock
    result_reference_role: str  #: the result's own reference
    inverted: bool  #: the choice was opposite to the result's, so the inverse map was used
    units: str
    mef3io_version: str
    notes: list[str] = field(default_factory=list)
    #: output channel -> its source's units label, for every channel whose label differs from
    #: ``units`` (only possible with ``units=``; the values were not converted)
    source_units: dict[str, str] = field(default_factory=dict)


class _RoleMap:
    """The reference → moving map for the chosen direction (``inverted``: the result's inverse)."""

    def __init__(self, cmap: ClockMap, inverted: bool) -> None:
        self._m, self.inverted = cmap, inverted
        lo, hi = cmap.measured_span
        if inverted:
            lo, hi = (int(round(float(cmap.forward(float(v))))) for v in (lo, hi))  # old reference -> old moving
        self.measured_span = (lo, hi)

    def forward(self, t):
        """Chosen-reference clock → chosen-moving clock (float µs)."""
        return self._m.inverse(t) if self.inverted else self._m.forward(t)

    def inverse(self, t):
        """Chosen-moving clock → chosen-reference clock (float µs)."""
        return self._m.forward(t) if self.inverted else self._m.inverse(t)


@dataclass
class _Plan:
    """One output channel, fully decided before anything is written."""

    out_name: str
    source: str  # "reference" | "moving"
    src_name: str
    fs: float
    src_start: int  # source grid origin (its own clock)
    src_n: int  # source grid length (gaps included)
    factor: float
    precision: int | None
    out_origin: int  # reference-clock time of output grid index 0
    k0: int  # first / end output grid index written
    k1: int
    trimmed: tuple[int, int] = (0, 0)
    extrapolated: tuple[int, int] = (0, 0)
    # filled while writing
    first_valid: int | None = None  # output grid index
    last_valid: int | None = None
    n_stored: int = 0
    head: np.ndarray | None = None  # expected stored values, first / last _VERIFY_EDGE samples
    tail: np.ndarray | None = None

    @property
    def period(self) -> float:
        return US_PER_S / self.fs

    def time_of(self, k: int) -> int:
        return int(self.out_origin) + int(np.rint(k * self.period))


def export_coregistered(
    result: CoregResult | str | Path,
    out: str | Path,
    file_a: str | Path | None = None,
    file_b: str | Path | None = None,
    *,
    reference: Literal["A", "B"] | None = None,
    reference_channels: Sequence[str] | None = None,
    moving_channels: Sequence[str] | None = None,
    reference_prefix: str = "",
    moving_prefix: str = "",
    n_threads: int = 0,
    chunk_seconds: float = 60.0,
    out_of_span: OutOfSpan = "refuse",
    units: str | None = None,
    password_a: str | None = None,
    password_b: str | None = None,
    output_passwords: tuple[str, str] | None = None,
    progress: Callable[[int, int], object] | None = None,
    cancel: threading.Event | Callable[[], bool] | None = None,
    verify_inputs: bool = True,
) -> ExportReport:
    """Write a new MEF 3.0 session (``out``, a ``.mefd`` folder that must not exist) holding chosen
    channels of both recordings on the reference clock.

    Parameters
    ----------
    result : CoregResult or path
        The saved result (``.coreg.json`` / ``.json.gz``) or a loaded one.
    out : path
        The new session; must end in ``.mefd`` and must not exist. Its folder must.
    file_a, file_b : path, optional
        The two recordings (A and B as the result names them); omitted, the stored paths are used
        (or looked up next to the result file). Opened read-only. Checked against the result's time
        spans unless ``verify_inputs=False``.
    reference : "A" or "B", optional
        Which recording keeps its clock; the other moves. Default: the result's own reference. The
        opposite choice uses the inverse of the clock map.
    reference_channels, moving_channels : sequence of str, optional
        Channels of the reference / moving recording to include (names in that recording).
        ``None`` = all of them, ``[]`` = none.
    reference_prefix, moving_prefix : str
        Put before the names in the new session. Two recordings often use the same channel names;
        the default (no prefix) REFUSES a name that would be written twice, naming it.
    n_threads : int
        Threads for everything this call does (mef3io's rule: ``0`` = automatic, ``1`` = no extra
        threads); passed to the mef3io Reader and Writer and used for the resampling. The output
        does not depend on it.
    chunk_seconds : float
        Reference-clock seconds processed at a time (memory use). The stored values and times do
        not depend on it.
    out_of_span : "refuse", "trim" or "extrapolate"
        What to do with moving samples whose reference time lies beyond the measured span of the
        clock map (more than the engine's 1 s span tolerance): refuse the export (default), write
        only the measured part, or extrapolate with the map's held end slope.
    units : str, optional
        Units label of the new session. Default: the units of the selected channels when they all
        agree; if they differ the call is refused until ``units`` is given (a MEF session carries one
        label). It is only a label: values are never converted, so a channel recorded in other
        units would carry the wrong label; that is warned about, noted in the report
        (``source_units``) and kept in the provenance record.
    password_a, password_b : str, optional
        To read encrypted recordings.
    output_passwords : (str, str), optional
        Encrypt the new session with these level-1 and level-2 passwords (mef3io's pair; they must
        differ). Default: not encrypted, whatever the inputs were.
    progress : callable, optional
        ``progress(done, total)`` in output samples (gap samples included) over all channels.
    cancel : threading.Event or callable, optional
        Set / returning True to stop between chunks; raises :class:`ExportCancelled`.
    verify_inputs : bool
        Check the two files against the result's time spans (as the readers do).

    Returns
    -------
    ExportReport

    Notes
    -----
    Each channel is stored with the sampling rate and units conversion factor of its source (one
    whose segments store different factors: the finest of them; a negative one: its magnitude). A
    factor that is a power of ten stores the data as decimal counts of it (mef3io's float path,
    precision = digits); any other factor (an amplifier's step) is kept verbatim and the values are
    rounded to whole counts of it. Subject metadata is not copied. A session-level record holds
    the provenance: source names, the result's SHA-256, direction, versions (no paths).
    """
    import mef3io

    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")
    if out_of_span not in ("refuse", "trim", "extrapolate"):
        raise ValueError('out_of_span must be "refuse", "trim" or "extrapolate"')
    if reference not in (None, "A", "B"):
        raise ValueError('reference must be "A" or "B"')
    if int(n_threads) < 0:
        raise ValueError("n_threads must be 0 (automatic) or positive")
    out = Path(out)
    if out.suffix != ".mefd":
        raise ValueError(f"the output must be a .mefd folder name, got {out.name!r}")
    partial = out.with_name(out.name + ".partial")  # staging folder; the session is inside it
    if out.exists():
        raise FileExistsError(f"{out} already exists; an export never overwrites. Choose another name.")
    if partial.exists():
        raise FileExistsError(f"{partial} exists (a leftover of an earlier export?); remove it yourself or choose another name.")
    if not out.parent.is_dir():
        raise FileNotFoundError(f"the folder for the output does not exist: {out.parent}")

    res, res_sha, near = _load_result(result)
    paths = {
        "A": str(file_a) if file_a is not None else _locate(res.file_a, near, "A"),
        "B": str(file_b) if file_b is not None else _locate(res.file_b, near, "B"),
    }
    ref_role = reference or res.reference_role
    mov_role = "B" if ref_role == "A" else "A"
    inverted = ref_role != res.reference_role
    rmap = _RoleMap(res.clock_map, inverted)
    passwords = {"A": password_a, "B": password_b}
    n_threads = int(n_threads)

    readers: dict[str, object] = {}
    try:
        for role in ("A", "B"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", mef3io.SessionDeclarationWarning)
                readers[role] = mef3io.Reader(paths[role], password=passwords[role] or "", n_threads=n_threads)
        if verify_inputs:  # from the readers just opened: each file is opened once
            _check_spans(res, paths, {"A": password_a, "B": password_b},
                         spans={role: _time_span(readers[role]) for role in ("A", "B")})
        ref_r, mov_r = readers[ref_role], readers[mov_role]
        plans, notes, label, source_units = _plan(
            ref_r, mov_r, rmap, reference_channels, moving_channels, reference_prefix, moving_prefix,
            out_of_span, units,
        )
        if not plans:
            raise ValueError("no channels selected")
        names_by_role = {
            "reference": file_name(paths[ref_role]), "moving": file_name(paths[mov_role]),
        }
        report = ExportReport(
            path=str(out), channels=[], result_sha256=res_sha, reference_role=ref_role,
            result_reference_role=res.reference_role, inverted=inverted, units=label,
            mef3io_version=mef3io.__version__, notes=notes, source_units=source_units,
        )
        _write_session(
            plans, {"reference": ref_r, "moving": mov_r}, rmap, partial, out, label, names_by_role,
            report, n_threads, chunk_seconds, output_passwords, progress, cancel,
        )
    finally:
        for r in readers.values():
            try:
                r.close()
            except Exception:
                pass
    return report


# -- planning ---------------------------------------------------------------------------------
def _time_span(reader) -> tuple[int | None, int | None]:
    """:func:`~mef3io_coregistration.io.mef_reader.time_span` of an open mef3io Reader."""
    starts, ends = [], []
    for ch in reader.channels:
        try:
            info = reader.info(ch)
            c0, c1 = int(info["start_time"]), int(info["end_time"])
        except Exception:  # a channel that cannot be described has no span (as time_span)
            continue
        starts.append(c0)
        ends.append(c1)
    return (min(starts), max(ends)) if starts else (None, None)


def _load_result(result) -> tuple[CoregResult, str, Path | None]:
    if isinstance(result, CoregResult):
        return result, content_sha256(dumps(result.to_dict())), None
    path = Path(result)
    return CoregResult.load(path), content_sha256(path), path.parent


def _select(reader, wanted: Sequence[str] | None, which: str) -> list[str]:
    have = list(reader.channels)
    if wanted is None:
        return have
    missing = [c for c in wanted if c not in have]
    if missing:
        raise KeyError(f"{which} channels not in that recording: {missing}; available: {sorted(have)}")
    if len(set(wanted)) != len(list(wanted)):
        raise ValueError(f"{which} channels listed more than once: {list(wanted)}")
    return list(wanted)


def _factor_precision(factor: float) -> tuple[float, int | None]:
    """The factor to store and the decimal precision when it is (within rounding) 10**-p.

    A negative factor (inverted polarity) is stored as its magnitude: the physical values are
    what is written, so ``-c`` counts of ``|f|`` read back exactly as ``c`` counts of ``f``."""
    f = abs(float(factor))
    if not (math.isfinite(f) and f > 0):
        f = 1.0  # "not set": mef3io reads such data as the raw counts
    p = int(round(-math.log10(f)))
    if 0 <= p <= 15 and abs(f - 10.0 ** -p) <= 1e-9 * f:
        return 10.0 ** -p, p
    return f, None


def _source_factor(reader, source: str, c: str, info: dict, notes: list[str]):
    """The factor (and decimal precision) ``c`` is stored with in the new session.

    mef3io scales every sample by its OWN segment's factor, and ``info`` gives ``None`` when the
    segments store different ones. The output channel has one factor: the finest of the
    segments' (all powers of ten -> exact; otherwise a coarser segment's values are rounded to
    whole counts of the finest step, which the report notes)."""
    f = info.get("units_conversion_factor")
    if f is not None:
        f = float(f)
        if f == 0.0:
            notes.append(f"{source} channel {c!r} stores no conversion factor (MEF's 'no entry'): "
                         "its raw counts are written as values with factor 1")
        return _factor_precision(f)
    segs = [float(s["units_conversion_factor"]) for s in reader.segments(c)]
    eff = [abs(v) if (math.isfinite(v) and v != 0.0) else 1.0 for v in segs]
    factor, precision = _factor_precision(min(eff))
    exact = all(abs(v / factor - round(v / factor)) <= 1e-9 * max(1.0, v / factor) for v in eff)
    notes.append(
        f"{source} channel {c!r}: its segments store different conversion factors "
        f"{sorted(set(segs))}; written with the finest, {factor:g}"
        + ("" if exact else " (values of the other segments rounded to whole counts of it)")
    )
    return factor, precision


def _plan(ref_r, mov_r, rmap, ref_sel, mov_sel, ref_prefix, mov_prefix, out_of_span, units):
    notes: list[str] = []
    ref_ch = _select(ref_r, ref_sel, "reference")
    mov_ch = _select(mov_r, mov_sel, "moving")
    names: dict[str, tuple[str, str]] = {}
    clashes = []
    for source, chs, prefix in (("reference", ref_ch, ref_prefix), ("moving", mov_ch, mov_prefix)):
        for c in chs:
            n = f"{prefix}{c}"
            if n in names:
                clashes.append(n)
            names[n] = (source, c)
    if clashes:
        raise ValueError(
            f"channel names would be written twice: {sorted(set(clashes))}. The two recordings use "
            "the same names; give reference_prefix= and/or moving_prefix= (for example "
            "moving_prefix='B_'), or select different channels."
        )

    tol = SPAN_TOLERANCE_US
    lo, hi = rmap.measured_span
    plans: list[_Plan] = []
    outside: list[str] = []
    unit_of: dict[str, str] = {}
    for name, (source, c) in names.items():
        reader = ref_r if source == "reference" else mov_r
        info = dict(reader.info(c))
        fs = float(info["sampling_frequency"])
        start, end = int(info["start_time"]), int(info["end_time"])
        period = US_PER_S / fs
        n = int(round((end - start) / period))
        if n < 1:
            raise ValueError(f"{source} channel {c!r} holds no samples")
        factor, precision = _source_factor(reader, source, c, info, notes)
        u = info.get("units_description") or ""
        unit_of[name] = u.decode(errors="replace") if isinstance(u, bytes) else str(u)
        if source == "reference":
            plan = _Plan(name, source, c, fs, start, n, factor, precision, start, 0, n)
        else:
            plan = _plan_moving(name, c, fs, start, n, factor, precision, rmap, lo, hi, tol,
                                out_of_span, outside)
        plans.append(plan)
    if outside and out_of_span == "refuse":
        raise ValueError(
            "moving channels reach beyond the span the clock map was measured over "
            f"({_fmt_span(lo, hi)}, tolerance {tol / US_PER_S:g} s), where it can only extrapolate: "
            + "; ".join(outside)
            + '. Pass out_of_span="trim" to write only the measured part, or "extrapolate" to '
            "follow the map\'s held end slope."
        )
    for p in plans:
        if p.source == "moving" and (p.trimmed != (0, 0)):
            notes.append(f"{p.out_name}: trimmed {p.trimmed[0] / US_PER_S:.3f} s at the start and "
                         f"{p.trimmed[1] / US_PER_S:.3f} s at the end (outside the measured span)")
        if p.source == "moving" and (p.extrapolated != (0, 0)):
            notes.append(f"{p.out_name}: {p.extrapolated[0] / US_PER_S:.3f} s at the start and "
                         f"{p.extrapolated[1] / US_PER_S:.3f} s at the end extrapolated beyond the measured span")
    if units is not None:
        label = str(units)
        # units= is a LABEL, never a conversion: a channel whose own label differs keeps its
        # values, so say so loudly (warning + report note + provenance record), never silently.
        relabelled = {n: u for n, u in unit_of.items() if u and u != label}
        if relabelled:
            text = ", ".join(f"{n!r} ({u})" for n, u in sorted(relabelled.items()))
            msg = (f"units={label!r} only sets the label; the values are NOT converted. Channels "
                   f"recorded in other units are now labelled {label!r}: {text}")
            warnings.warn(msg, UserWarning, stacklevel=3)
            notes.append(msg)
    else:
        labels = sorted(set(unit_of.values()) - {""}) or [""]
        if len(labels) > 1:
            raise ValueError(
                f"the selected channels carry different units {labels}; a MEF session has one units "
                "label. Pass units= to choose the label, or select channels of one unit."
            )
        label = labels[0]
    source_units = {n: u for n, u in unit_of.items() if u != label}
    return plans, notes, label, source_units


def _plan_moving(name, c, fs, start, n, factor, precision, rmap, lo, hi, tol, out_of_span, outside):
    period = US_PER_S / fs
    # The resampler's filters see zero padding past the first / last source sample, so the outputs
    # within its support of either end are distorted: the grid starts and ends that far inside.
    edge = support_samples(fs, fs) * period
    if n <= 2 * support_samples(fs, fs) + 4:
        raise ValueError(f"moving channel {c!r} is too short to resample ({n} samples)")
    first_t = start + int(math.ceil(edge))
    last_t = start + int(np.rint((n - 1) * period)) - int(math.ceil(edge))
    o = int(math.ceil(float(rmap.inverse(float(first_t)))))
    while float(rmap.forward(o)) < first_t:  # the µs-rounded origin must not fall before the data
        o += 1
    last = int(math.floor(float(rmap.inverse(float(last_t)))))
    while float(rmap.forward(last)) > last_t:
        last -= 1
    k1 = first_index_at_or_after(o, period, last + 1)
    k0 = 0
    trimmed = (0, 0)
    extrap = (0, 0)
    before, after = max(0, lo - o), max(0, last - hi)
    if before > tol or after > tol:
        if out_of_span == "refuse":
            outside.append(f"{c!r} reaches {before / US_PER_S:.1f} s before and {after / US_PER_S:.1f} s after it")
        elif out_of_span == "trim":
            k0 = max(0, first_index_at_or_after(o, period, lo - tol))
            k1 = min(k1, first_index_at_or_after(o, period, hi + tol + 1))
            if k1 <= k0:
                raise ValueError(f"moving channel {c!r} lies entirely outside the measured span")
            trimmed = (int(o + np.rint(k0 * period)) - o, last - int(o + np.rint((k1 - 1) * period)))
        else:
            extrap = (before, after)
    return _Plan(name, "moving", c, fs, start, n, factor, precision, o, k0, k1, trimmed, extrap)


def _fmt_span(lo: int, hi: int) -> str:
    from datetime import UTC, datetime

    def f(t):
        return datetime.fromtimestamp(int(t) / 1e6, tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    return f"{f(lo)} to {f(hi)}"


# -- writing ----------------------------------------------------------------------------------
def _write_session(plans, readers, rmap, partial, out, label, names, report, n_threads, chunk_seconds,
                   output_passwords, progress, cancel):
    import mef3io

    def cancelled() -> bool:
        if cancel is None:
            return False
        return bool(cancel.is_set() if hasattr(cancel, "is_set") else cancel())

    total = sum((p.k1 - p.k0) for p in plans)
    done = 0
    pool = ThreadPoolExecutor(max_workers=n_threads or (os.cpu_count() or 1)) if n_threads != 1 else None
    pw, pw2 = output_passwords or ("", "")
    writer = None
    session = partial / out.name  # named as the final session: mef3io names its files after it
    # Outside the try: if the folder appeared since the check (another export racing for the
    # name), this raises FileExistsError and the cleanup below never touches what is not ours.
    partial.mkdir()
    try:
        writer = mef3io.Writer(
            str(session), n_threads=n_threads, units=label, flush="close",
            password1=pw, password2=pw2,
            metadata=mef3io.Metadata(acquisition=mef3io.Acquisition(session_description=out.stem)),
        )
        # Channels that share a grid are read and written together (read / write_many).
        groups: dict[tuple, list[_Plan]] = {}
        for p in plans:
            groups.setdefault((p.source, p.fs, p.src_start, p.out_origin, p.k0, p.k1, p.precision,
                               p.factor if p.precision is None else 0.0), []).append(p)
        for group in _batches(groups.values(), chunk_seconds):
            first = group[0]
            step = max(1, int(round(chunk_seconds * first.fs)))
            for c0 in range(first.k0, first.k1, step):
                if cancelled():
                    raise ExportCancelled("the export was cancelled")
                c1 = min(first.k1, c0 + step)
                if first.source == "reference":
                    data = _read_reference(readers["reference"], group, c0, c1)
                else:
                    data = _resample_chunk(readers["moving"], group, rmap, c0, c1, pool)
                _store(writer, group, data, c0)
                done += (c1 - c0) * len(group)
                if progress is not None:
                    progress(done, total)
        if all(p.first_valid is None for p in plans):
            raise ValueError("nothing to write: every selected channel is a gap over the exported span")
        note = _provenance(report, plans, names)
        t_first = min(p.time_of(p.first_valid) for p in plans if p.first_valid is not None)
        writer.write_annotations([{"time": int(t_first), "type": "Note", "text": note}])
        writer.close()
        writer = None
        _verify(session, plans, n_threads, pw2)
    except BaseException:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        _remove(partial)
        raise
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    if out.exists():  # someone created it while we wrote; never replace it
        _remove(partial)
        raise FileExistsError(f"{out} appeared while the export ran; nothing was replaced")
    os.rename(session, out)
    os.rmdir(partial)
    for p in plans:
        report.channels.append(ExportedChannel(
            p.out_name, p.source, p.src_name, p.fs,
            p.time_of(p.first_valid) if p.first_valid is not None else p.time_of(p.k0), p.n_stored,
            p.k1 - p.k0, p.factor, p.precision, p.trimmed, p.extrapolated,
        ))


def _batches(groups, chunk_seconds: float):
    """Split each grid group so one chunk holds at most ``_MAX_CHUNK_SAMPLES`` samples (a group
    of hundreds of high-rate channels would otherwise hold gigabytes per chunk)."""
    for group in groups:
        per = max(1, int(round(chunk_seconds * group[0].fs)))
        size = max(1, _MAX_CHUNK_SAMPLES // per)
        for i in range(0, len(group), size):
            yield group[i : i + size]


def _remove(partial: Path) -> None:
    """Remove the folder this run created (it did not exist before the run)."""
    shutil.rmtree(partial, ignore_errors=True)


def _read_window(reader, names: list[str], t0: int, t1: int, n: int) -> np.ndarray:
    """``(len(names), n)`` physical values of ``[t0, t1)``, NaN where the recording has none."""
    arr = reader.read(names if len(names) > 1 else names[0], t0, t1)
    arr = np.asarray(arr, dtype=np.float64).reshape(len(names), -1)
    if arr.shape[1] == n:
        return arr
    if arr.shape[1] > n:
        return arr[:, :n]
    return np.concatenate((arr, np.full((len(names), n - arr.shape[1]), np.nan)), axis=1)


def _read_reference(reader, group: list[_Plan], c0: int, c1: int) -> np.ndarray:
    p = group[0]
    t0, t1 = p.time_of(c0), p.time_of(c1)
    return _read_window(reader, [q.src_name for q in group], t0, t1, c1 - c0)


def _resample_chunk(reader, group: list[_Plan], rmap, c0: int, c1: int, pool) -> np.ndarray:
    p = group[0]
    ts = uniform_grid(p.out_origin, p.fs, c0, c1 - c0)
    t_src = np.asarray(rmap.forward(ts), dtype=np.float64)
    period = p.period
    s = p.src_start
    rel = t_src - s
    pad = max(_MIN_PAD_SAMPLES, 2 * support_samples(p.fs, p.fs) + 16)
    i0 = max(0, int(math.floor(rel.min() / period)) - pad)
    i1 = min(p.src_n, int(math.ceil(rel.max() / period)) + pad + 2)
    out = np.full((len(group), c1 - c0), np.nan)
    if i1 - i0 < 4:
        return out
    src = _read_window(reader, [q.src_name for q in group], s + int(np.rint(i0 * period)),
                       s + int(np.rint(i1 * period)), i1 - i0)
    origin = i0 * period  # the window's exact start, relative to the channel origin

    def one(i: int) -> None:
        out[i] = evaluate_at(src[i], origin, p.fs, rel, out_fs=p.fs)

    if pool is None:
        for i in range(len(group)):
            one(i)
    else:
        list(pool.map(one, range(len(group))))
    return out


def _counts(x: np.ndarray, factor: float, precision: int | None) -> np.ndarray:
    """Values → the int32 counts that will be stored (NaN → 0)."""
    scale = 10.0 ** precision if precision is not None else 1.0 / factor
    c = np.rint(np.where(np.isnan(x), 0.0, x) * scale)
    return np.clip(c, -2147483647, 2147483647)


def _store(writer, group: list[_Plan], data: np.ndarray, c0: int) -> None:
    p0 = group[0]
    t0 = p0.time_of(c0)
    n = data.shape[1]
    # Track the first / last valid sample and the end values for the read-back check.
    for i, q in enumerate(group):
        valid = ~np.isnan(data[i])
        scale = q.factor if q.precision is None else 10.0 ** -q.precision
        stored = np.where(valid, _counts(data[i], q.factor, q.precision) * scale, np.nan)
        if q.head is None:
            q.head = np.full(_VERIFY_EDGE, np.nan)
        lo = max(c0, q.k0)
        hi = min(c0 + n, q.k0 + _VERIFY_EDGE)
        if hi > lo:
            q.head[lo - q.k0 : hi - q.k0] = stored[lo - c0 : hi - c0]
        q.tail = np.concatenate((q.tail if q.tail is not None else np.empty(0), stored))[-_VERIFY_EDGE:]
        if valid.any():
            idx = np.flatnonzero(valid)
            if q.first_valid is None:
                q.first_valid = c0 + int(idx[0])
            q.last_valid = c0 + int(idx[-1])
            q.n_stored += int(idx.size)
    if p0.precision is not None:
        writer.write_many([q.out_name for q in group], data, t0, p0.fs, precision=p0.precision)
    else:
        for i, q in enumerate(group):
            valid = ~np.isnan(data[i])
            if not valid.any():
                continue
            writer.write_int32(q.out_name, _counts(data[i], q.factor, None).astype(np.int32), q.factor,
                               t0, q.fs, valid=valid)


def _provenance(report: ExportReport, plans: list[_Plan], names: dict[str, str]) -> str:
    doc = {
        "operation": "export_coregistered",
        "tool": "mef3io-coregistration",
        "tool_version": __version__,
        "mef3io_version": report.mef3io_version,
        "result_sha256": report.result_sha256,
        "reference_role": report.reference_role,
        "result_reference_role": report.result_reference_role,
        "direction": "inverse of the result's map" if report.inverted else "as saved in the result",
        "reference_recording": names["reference"],
        "moving_recording": names["moving"],
        "reference_channels": [p.src_name for p in plans if p.source == "reference"],
        "moving_channels": [p.src_name for p in plans if p.source == "moving"],
        "output_channels": [p.out_name for p in plans],
    }
    if report.source_units:  # labels replaced by units= (values not converted)
        doc["units"] = report.units
        doc["source_units"] = dict(sorted(report.source_units.items()))
    text = json.dumps(doc, sort_keys=True, separators=(",", ":"))
    if len(text) > _MAX_NOTE_CHARS:
        for key in ("reference_channels", "moving_channels", "output_channels"):
            doc[key] = len(doc[key])
        if "source_units" in doc:  # summarised as channels per original label
            units_count: dict[str, int] = {}
            for u in doc["source_units"].values():
                units_count[u] = units_count.get(u, 0) + 1
            doc["source_units"] = units_count
        text = json.dumps(doc, sort_keys=True, separators=(",", ":"))
    return text


# -- read-back ----------------------------------------------------------------------------------
def _verify(session: Path, plans: list[_Plan], n_threads: int, password: str) -> None:
    import mef3io

    def bad(msg: str):
        raise ExportVerificationError(f"the written session does not read back as written: {msg}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = mef3io.Reader(str(session), password=password, n_threads=n_threads)
    try:
        want = {p.out_name for p in plans if p.n_stored > 0}
        if set(r.channels) != want:
            bad(f"channels {sorted(r.channels)} instead of {sorted(want)}")
        for p in plans:
            if p.n_stored == 0:
                continue
            info = r.info(p.out_name)
            t_first = p.time_of(p.first_valid)
            if float(info["sampling_frequency"]) != p.fs:
                bad(f"{p.out_name}: sampling rate {info['sampling_frequency']} instead of {p.fs}")
            if int(info["number_of_samples"]) != p.n_stored:
                bad(f"{p.out_name}: {info['number_of_samples']} samples instead of {p.n_stored}")
            if abs(int(info["start_time"]) - t_first) > 1:
                bad(f"{p.out_name}: starts at {info['start_time']} instead of {t_first}")
            t_end = p.time_of(p.last_valid + 1)
            if abs(int(info["end_time"]) - t_end) > 2:
                bad(f"{p.out_name}: ends at {info['end_time']} instead of {t_end}")
            m = min(_VERIFY_EDGE, p.k1 - p.k0)
            for expect, g0 in ((p.head[:m], p.k0), (p.tail[-m:], p.k1 - m)):
                got = np.asarray(r.read(p.out_name, p.time_of(g0), p.time_of(g0 + m)), dtype=np.float64)
                if got.shape != expect.shape or not np.allclose(got, expect, rtol=1e-9, atol=0.0, equal_nan=True):
                    bad(f"{p.out_name}: the samples at an end differ from what was written")
    finally:
        r.close()
