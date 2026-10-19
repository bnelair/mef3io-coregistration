"""Runtime reader.

    reader = CoregisteredReader.from_result("result.json")          # roles from the result
    reader.get_data("LA1-LA2", start, stop, file="B", coregister=True)

Three access modes:

* **default coregistered** — read the *moving* file with ``coregister=True``: its signal is
  band-limited resampled onto a grid on the **reference clock**; query times are in the
  reference clock domain. If the reference file carries the same channel, that
  channel's own grid is used (sample-for-sample overlay); otherwise — e.g. a clinical channel
  the second recorder does not record — the grid is the moving signal's own rate, anchored at its
  mapped first sample so consecutive reads tile exactly. ``fs=`` or an explicit reference-clock
  ``grid=`` override either.
* **force raw — reference** — ``coregister=False`` (or reading the reference file with
  ``coregister=True``): native reference samples, untransformed.
* **force raw — moving** — ``coregister=False`` on the moving file: native moving samples on
  native timestamps, identical to mef3io.

Channels are named per file and never need to match across files: ``channel`` is a
channel name, an ``"anode-cathode"`` bipolar expression, or a
:class:`~mef3io_coregistration.montage.spec.ChannelSpec`, resolved against the file being read.
The reference recording is never transformed unless a different output ``fs`` is requested.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from ..config import US_PER_S
from ..io.mef_reader import ChannelRead, Gap, MefChannelReader
from ..montage.spec import ChannelSpec, read_montage, spec_bounds, spec_fs, spec_gaps
from ..transform.clock_map import ClockMap
from ..transform.resample import evaluate_at, grid_indices, support_samples, uniform_grid

if TYPE_CHECKING:
    from ..result import CoregResult

FileRole = Literal["A", "B"]

# Padding (in moving samples) around a mapped window so the polyphase resampler has context.
_RESAMPLE_PAD_SAMPLES = 256


class CoregisteredReader:
    """Serve channel data raw or coregistered across two MEF3 files via a clock map."""

    def __init__(
        self,
        clock_map: ClockMap,
        file_a: str | Path,
        file_b: str | Path,
        reference: FileRole = "A",
        password_a: str | None = None,
        password_b: str | None = None,
    ) -> None:
        if reference not in ("A", "B"):
            raise ValueError('reference must be "A" or "B"')
        self.clock_map = clock_map
        self.reference_file: FileRole = reference
        self.moving_file: FileRole = "B" if reference == "A" else "A"
        self.paths: dict[FileRole, str] = {"A": str(file_a), "B": str(file_b)}
        self._readers: dict[FileRole, MefChannelReader] = {
            "A": MefChannelReader(file_a, password=password_a),
            "B": MefChannelReader(file_b, password=password_b),
        }

    @classmethod
    def from_result(
        cls,
        result: CoregResult | str | Path,
        file_a: str | Path | None = None,
        file_b: str | Path | None = None,
        verify: bool = True,
        password_a: str | None = None,
        password_b: str | None = None,
    ) -> CoregisteredReader:
        """Open the files a coregistration result was computed on, with its A/B roles.

        The result is a transformation, not a copy of the data: ``file_a``/``file_b`` override the
        stored paths, for data that has been moved or shared. Without them a stored path that no
        longer exists is looked for next to the result file (the same folder name, or its
        ``.mefd.tar`` archive). The reference role always comes from the result, so the map can
        never be applied inverted. Each file is checked only by its time span: its start and end
        must match the ones stored in the result within ``SPAN_TOLERANCE_US`` (1 s). Paths,
        channels and rates do not matter, so a moved copy, a downsampled copy or one with other
        channels passes, and a different recording does not; a file closer to the other role's
        span is refused as a swapped A/B (``verify=False`` to override). ``password_a``/
        ``password_b`` are used for the check too.
        """
        from ..result import CoregResult

        res = result if isinstance(result, CoregResult) else CoregResult.load(result)
        near = None if isinstance(result, CoregResult) else Path(result).parent
        paths = {
            "A": str(file_a) if file_a is not None else _locate(res.file_a, near, "A"),
            "B": str(file_b) if file_b is not None else _locate(res.file_b, near, "B"),
        }
        if verify:
            _check_spans(res, paths, {"A": password_a, "B": password_b})
        return cls(
            res.clock_map,
            paths["A"],
            paths["B"],
            reference=res.reference_role,  # type: ignore[arg-type]
            password_a=password_a,
            password_b=password_b,
        )

    @classmethod
    def from_map_file(
        cls,
        map_path: str | Path,
        file_a: str | Path,
        file_b: str | Path,
        reference: FileRole = "A",
    ) -> CoregisteredReader:
        return cls(ClockMap.load(map_path), file_a, file_b, reference=reference)

    # -- context management -------------------------------------------------
    def __enter__(self) -> CoregisteredReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        for reader in self._readers.values():
            reader.close()

    # -- metadata -------------------------------------------------------------
    def reader(self, file: FileRole) -> MefChannelReader:
        _check_role(file)
        return self._readers[file]

    def channels(self, file: FileRole) -> list[str]:
        return self.reader(file).channels

    def spec(self, channel: ChannelSpec | str | tuple, file: FileRole) -> ChannelSpec:
        """Resolve ``channel`` (name, ``"a-b"`` expression or spec) against ``file``."""
        spec = ChannelSpec.parse(channel, self.channels(file))
        for ch in spec.channels:
            if not self.reader(file).has_channel(ch):
                raise KeyError(
                    f"channel {ch!r} not in file {file} ({self.paths[file]}); "
                    f"available: {sorted(self.channels(file))}"
                )
        return spec

    def fs(self, channel: ChannelSpec | str | tuple, file: FileRole) -> float:
        return spec_fs(self.reader(file), self.spec(channel, file))

    def time_bounds(
        self, channel: ChannelSpec | str | tuple, file: FileRole, coregister: bool = True
    ) -> tuple[int, int]:
        """``(start, end)`` of ``channel`` in ``file``; in the reference clock when
        ``coregister`` (the moving extent is mapped through the clock map)."""
        lo, hi = spec_bounds(self.reader(file), self.spec(channel, file))
        if coregister and file == self.moving_file:
            return self._to_ref(lo), self._to_ref(hi)
        return lo, hi

    def _to_ref(self, t_moving: int) -> int:
        return int(round(float(self.clock_map.inverse(float(t_moving)))))

    def grid_origin(self, channel: ChannelSpec | str | tuple, file: FileRole) -> int:
        """Reference-clock time of the channel's first sample (the origin of its coregistered
        grid, so consecutive coregistered reads tile exactly)."""
        lo, _hi = spec_bounds(self.reader(file), self.spec(channel, file))
        return self._to_ref(lo) if file == self.moving_file else lo

    # -- access -------------------------------------------------------------
    def get_data(
        self,
        channel: ChannelSpec | str | tuple,
        start: int | None = None,
        stop: int | None = None,
        file: FileRole = "A",
        coregister: bool = True,
        fs: float | None = None,
        grid: np.ndarray | None = None,
    ) -> ChannelRead:
        """Read ``channel`` from ``file`` over ``[start, stop)`` uUTC.

        With ``coregister=False`` the native samples/timestamps of ``file`` are returned. With
        ``coregister=True`` times are reference-clock uUTC: the reference file returns native
        samples, the moving file is resampled onto a reference-clock grid (rate ``fs``, default
        the moving signal's own). ``grid`` gives explicit reference-clock output timestamps
        (overrides ``start``/``stop``/``fs``; the output rate for band-limiting is then
        inferred from the grid spacing).
        """
        _check_role(file)
        spec = self.spec(channel, file)
        rdr = self._readers[file]
        transform = coregister and file == self.moving_file
        if not transform and grid is None and (fs is None or fs == spec_fs(rdr, spec)):
            return read_montage(rdr, spec, start, stop)
        if transform and grid is None and fs is None and self._in_reference(spec):
            # A channel the reference file also carries (same name/montage): adopt that channel's
            # own grid, so the coregistered moving read lines up sample-for-sample with the
            # reference read and the two can be overlaid directly.
            grid = read_montage(self._readers[self.reference_file], spec, start, stop).ts
            if grid.size == 0:
                empty = np.empty(0, dtype=np.float64)
                return ChannelRead(spec.label, empty, grid, spec_fs(rdr, spec), int(start or 0), ())
        return self._read_on_grid(spec, file, start, stop, fs, grid, transform)

    def _in_reference(self, spec: ChannelSpec) -> bool:
        ref = self._readers[self.reference_file]
        return all(ref.has_channel(ch) for ch in spec.channels)

    def _read_on_grid(
        self,
        spec: ChannelSpec,
        file: FileRole,
        start: int | None,
        stop: int | None,
        fs: float | None,
        grid: np.ndarray | None,
        through_map: bool,
    ) -> ChannelRead:
        """Evaluate ``spec`` from ``file`` on a grid (reference clock if ``through_map``)."""
        rdr = self._readers[file]
        src_fs = spec_fs(rdr, spec)
        if grid is None:
            out_fs = float(fs) if fs is not None else src_fs
            origin = self.grid_origin(spec, file) if through_map else spec_bounds(rdr, spec)[0]
            lo, hi = self.time_bounds(spec, file, coregister=through_map)
            k0, k1 = grid_indices(origin, out_fs, lo if start is None else start, hi if stop is None else stop)
            ts = uniform_grid(origin, out_fs, k0, k1 - k0)
        else:
            ts = np.asarray(grid, dtype=np.int64)
            out_fs = _grid_fs(ts, fs if fs is not None else src_fs)

        if ts.size == 0:
            empty = np.empty(0, dtype=np.float64)
            return ChannelRead(spec.label, empty, ts, out_fs, int(start or 0), ())

        t_src = (
            np.asarray(self.clock_map.forward(ts), dtype=np.float64)
            if through_map
            else ts.astype(np.float64)
        )
        # Enough source context for the filters evaluate_at applies at this rate ratio (a heavy
        # decimation needs far more than a plain interpolation), or the window's ends are lost.
        # evaluate_at treats the fetched data's ends like gaps and blanks every output whose filter
        # reaches one — within 2 × the half-support — so fetch that much beyond each end.
        pad_samples = max(_RESAMPLE_PAD_SAMPLES, 2 * support_samples(src_fs, out_fs) + 16)
        pad = int(np.ceil(pad_samples * US_PER_S / src_fs))
        src = read_montage(rdr, spec, int(np.floor(t_src.min())) - pad, int(np.ceil(t_src.max())) + pad)
        # Work relative to the channel's grid origin with the *exact* (unrounded) segment start,
        # so every read interpolates at identical positions and consecutive reads tile exactly.
        origin = rdr.time_bounds(spec.anode)[0]
        values = evaluate_at(
            src.data, _exact_offset(src.start_uutc - origin, rdr.fs(spec.anode)), src.fs,
            t_src - origin, out_fs=out_fs,
        )

        gaps = spec_gaps(rdr, spec)
        if through_map:
            gaps = tuple(
                Gap(self._to_ref(g.start_uutc), self._to_ref(g.end_uutc), g.n_missing) for g in gaps
            )
        gaps = tuple(g for g in gaps if g.start_uutc <= ts[-1] and g.end_uutc > ts[0])
        return ChannelRead(spec.label, values, ts, out_fs, int(ts[0]), gaps)


def _exact_offset(offset_us: int, anode_fs: float) -> float:
    """The exact (sub-µs) offset of a read's first sample from the channel origin.

    Every sample a read returns lies on the anode's own sampling grid (a bipolar pair of different
    rates too: it is derived on the anode's grid), but its timestamp is rounded to whole µs. Snap
    back onto that grid — not onto the read's possibly lower, harmonised rate, which put mixed-rate
    bipolar reads off by up to a high-rate sample — so every read interpolates at identical
    positions and consecutive reads tile exactly.
    """
    period = US_PER_S / anode_fs
    k = round(offset_us / period)
    exact = k * period
    return exact if abs(exact - offset_us) <= 0.5 + 1e-9 else float(offset_us)


def _locate(stored: str, near: Path | None, role: str) -> str:
    """``stored`` if it exists; else the same recording next to the result file; else an error
    that says how to point the result at wherever the recording is now."""
    p = Path(stored)
    if p.exists():
        return str(p)
    names = [p.name, p.name + ".tar"] if p.suffix == ".mefd" else [p.name]
    if p.name.endswith(".mefd.tar"):
        names.append(p.name[: -len(".tar")])
    for base in [near] if near is not None else []:
        for name in names:
            if (base / name).exists():
                return str(base / name)
    raise FileNotFoundError(
        f"file {role} of this coregistration ({stored}) is not there any more"
        + (f", nor next to the result ({near})" if near is not None else "")
        + f". Pass file_{role.lower()}=<where it is now> — the result is only the transformation."
    )


def _check_role(file: str) -> None:
    if file not in ("A", "B"):
        raise ValueError('file must be "A" or "B"')


def _grid_fs(ts: np.ndarray, default: float) -> float:
    """Rate implied by a grid's mean spacing (``default`` for a single point)."""
    if ts.size < 2 or ts[-1] == ts[0]:
        return float(default)
    return US_PER_S * (ts.size - 1) / float(ts[-1] - ts[0])


def _stored_span(res, role: str) -> tuple[int, int] | None:
    fp = (res.file_fingerprints or {}).get(role) or {}
    t0, t1 = fp.get("start_uutc"), fp.get("end_uutc")
    return None if t0 is None or t1 is None else (int(t0), int(t1))


def _span_distance(have: tuple[int, int], want: tuple[int, int]) -> int:
    return max(abs(have[0] - want[0]), abs(have[1] - want[1]))


def _check_spans(res, paths: dict[str, str], passwords: dict[str, str | None]) -> None:
    """Each file must start and end within ``SPAN_TOLERANCE_US`` of the span the result stores for
    its role — the only check (paths, channels, rates do not matter). A file that matches the OTHER
    role's span more closely is refused as a swapped A/B: the map would otherwise be applied to the
    wrong file. Two recordings with identical spans cannot be told apart this way."""
    import warnings

    from ..io.mef_reader import SPAN_TOLERANCE_US, time_span

    stored = {role: _stored_span(res, role) for role in ("A", "B")}
    for role, path in paths.items():
        want = stored[role]
        if want is None:
            warnings.warn(
                f"the result stores no time span for file {role}: {path} is used unchecked.",
                UserWarning, stacklevel=4,
            )
            continue
        h0, h1 = time_span(path, password=passwords.get(role))
        have = None if h0 is None or h1 is None else (h0, h1)
        own = None if have is None else _span_distance(have, want)
        other = "B" if role == "A" else "A"
        other_span = stored[other]
        # Swapped first: a file that IS the other role's recording would otherwise only get the
        # generic message below.
        if have is not None and own is not None and other_span is not None:
            to_other = _span_distance(have, other_span)
            if to_other <= SPAN_TOLERANCE_US and to_other < own:
                raise ValueError(
                    f"file {role} ({path}) has file {other}'s time span ({_span_text(h0, h1)}), not "
                    f"file {role}'s: A and B look swapped (or one file was given for both). Assign "
                    "each file to its role, or pass verify=False to apply the map anyway."
                )
        if own is None or own > SPAN_TOLERANCE_US:
            raise ValueError(
                f"file {role} ({path}) is not the recording this clock map was computed on: "
                f"it spans {_span_text(h0, h1)}, the result expects {_span_text(*want)}. "
                "Assign the right file, or pass verify=False to apply the map anyway."
            )


def _span_text(t0, t1) -> str:
    from datetime import UTC, datetime

    if t0 is None or t1 is None:
        return "no readable time span"

    def fmt(t):
        return datetime.fromtimestamp(int(t) / 1e6, tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    return f"{fmt(t0)} – {fmt(t1)}"
