"""Review a saved coregistration with plots — from the whole recording down to single samples.

    sigcoreg-review result.json
    sigcoreg-review result.json --channel LTP1-LTP3 --out review/ --file-a /new/a.mefd

Writes PNG figures and an ``index.html`` that shows them with the numbers behind them:

1. overview       — both recordings on the reference clock, the overlap, the measured span
2. clock map      — offset(t): drift windows, fitted map, anchors, confidence
3. multi-scale    — the two signals overlaid at 2 min, 10 s, 2 s and 250 ms, at three times
4. before/after   — a constant offset against the fitted map, early and late in the recording
5. residual lag   — an independent re-measurement of the lag left after coregistration
6. stim pulses    — sharp artifacts present in both systems, as sub-millisecond sync markers
7. unified        — the two files as one channel, across the edge of the moving recording

Needs the ``report`` extra (matplotlib): ``pip install "mef3io-coregistration[report]"``. Reads the MEF3 files named in the result (checked
against their stored start/end times); ``--file-a``/``--file-b`` point it at recordings that have moved.
"""
from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

from mef3io_coregistration.montage.spec import read_montage
from mef3io_coregistration.reader import CoregisteredReader, CoregisteredRecording, UnifiedRead
from mef3io_coregistration.reader.coregistered_reader import FileRole
from mef3io_coregistration.result import CoregResult

from ..align.tde import normxcorr_delay

US = 1_000_000
REF_C, MOV_C, MAP_C, BAD_C = "#1b9e8f", "#d9822b", "#3355aa", "#b0b0b0"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 130, "font.size": 9, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "legend.frameon": False,
})


# --------------------------------------------------------------------------- helpers ---
@dataclass
class Setup:
    res: CoregResult
    rd: CoregisteredReader
    ref: FileRole  # file role of the reference, "A" | "B"
    mov: FileRole
    ref_name: str
    mov_name: str
    ch_ref: tuple
    ch_mov: tuple
    sign_mov: float  # the polarity the alignment used for the moving channel
    overlap: tuple[int, int]  # reference clock
    measured: tuple[int, int]


def _parse_label(label: str, channels: list[str]) -> tuple[tuple, float]:
    """A result's stage-channel label ("LTP1-LTP3", "-(LTP1-LTP3)", "LA1") → (spec, sign)."""
    sign = 1.0
    m = re.fullmatch(r"-\((.*)\)", label.strip())
    if m:
        label, sign = m.group(1), -1.0
    if label in channels:
        return (label, None), sign
    for i in range(1, len(label)):  # names may contain "-": split where both sides exist
        if label[i] == "-" and label[:i] in channels and label[i + 1 :] in channels:
            return (label[:i], label[i + 1 :]), sign
    raise ValueError(f"cannot resolve channel {label!r} in {channels[:12]}…")


def _spec_label(spec: tuple) -> str:
    return spec[0] if not spec[1] else f"{spec[0]} − {spec[1]}"


def _dts(t_uutc) -> list[datetime]:
    """uUTC array → datetimes, for a matplotlib time axis."""
    return [datetime.fromtimestamp(t / US, UTC) for t in np.asarray(t_uutc, dtype=float).ravel()]


def _dt(t_uutc) -> datetime | list[datetime]:
    if np.ndim(t_uutc):
        return [datetime.fromtimestamp(t / US, UTC) for t in np.asarray(t_uutc, dtype=float)]
    return datetime.fromtimestamp(float(t_uutc) / US, UTC)


def _clock(t_uutc: float) -> str:
    return datetime.fromtimestamp(float(t_uutc) / US, UTC).strftime("%H:%M:%S.%f")[:-3]


def _band(x: np.ndarray, fs: float, lo: float | None, hi: float | None) -> np.ndarray:
    """Zero-phase band-pass; NaN gaps bridged for filtering and restored."""
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    if not ok.any():
        return x
    y = np.interp(np.arange(x.size), np.flatnonzero(ok), x[ok]) if not ok.all() else x.copy()
    ny = fs / 2
    lo = lo if lo and lo < ny else None
    hi = hi if hi and hi < ny * 0.95 else None
    if lo and hi:
        sos = butter(4, [lo / ny, hi / ny], "bandpass", output="sos")
    elif lo:
        sos = butter(4, lo / ny, "highpass", output="sos")
    elif hi:
        sos = butter(4, hi / ny, "lowpass", output="sos")
    else:
        return np.where(ok, y, np.nan)
    if y.size > 3 * 13:
        y = sosfiltfilt(sos, y)
    return np.where(ok, y, np.nan)


def _robust(x: np.ndarray) -> np.ndarray:
    """Scale to robust sigma units (MAD) — the two amplifiers differ by orders of magnitude."""
    d = x[np.isfinite(x)]
    if d.size == 0:
        return x
    mad = np.median(np.abs(d - np.median(d))) * 1.4826
    return (x - np.median(d)) / (mad if mad > 0 else 1.0)


def _read(s: Setup, role: FileRole, t0: int, t1: int, fs: float | None = None, band=None):
    """(ts_ref_clock, data) of the review channel in file ``role``, on the reference clock."""
    spec = s.ch_ref if role == s.ref else s.ch_mov
    r = s.rd.get_data(spec, int(t0), int(t1), file=role, coregister=True, fs=fs)
    x = np.asarray(r.data, dtype=np.float64) * (s.sign_mov if role == s.mov else 1.0)
    if band:
        x = _band(x, r.fs, *band)
    return np.asarray(r.ts), x, r.fs


def _read_constant(s: Setup, t0: int, t1: int, offset_s: float, band=None):
    """The moving channel shifted by ONE constant offset — what you get without drift tracking."""
    r = read_montage(s.rd.reader(s.mov), s.rd.spec(s.ch_mov, s.mov),
                     int(t0 + offset_s * US), int(t1 + offset_s * US))
    x = np.asarray(r.data, dtype=np.float64) * s.sign_mov
    if band:
        x = _band(x, r.fs, *band)
    return np.asarray(r.ts) - int(offset_s * US), x, r.fs


def _envelope(s: Setup, role: FileRole, t0: int, t1: int, n_bins: int, chunk_s: float = 300.0):
    """Min/max per bin over [t0, t1) on the reference clock, read in chunks (native samples;
    the moving file's bin edges are mapped through the clock map)."""
    edges = np.linspace(t0, t1, n_bins + 1)
    lo = np.full(n_bins, np.nan)
    hi = np.full(n_bins, np.nan)
    rdr = s.rd.reader(role)
    spec = s.rd.spec(s.ch_ref if role == s.ref else s.ch_mov, role)
    to_file = (lambda t: t) if role == s.ref else (lambda t: np.asarray(s.rd.clock_map.forward(t)))
    to_ref = (lambda t: t) if role == s.ref else (lambda t: np.asarray(s.rd.clock_map.inverse(t)))
    f0, f1 = (int(v) for v in to_file(np.array([t0, t1], dtype=float)))
    b0, b1 = rdr.time_bounds(spec.anode)
    f0, f1 = max(f0, b0), min(f1, b1)
    t = f0
    step = int(chunk_s * US)
    while t < f1:
        r = read_montage(rdr, spec, t, min(t + step, f1))
        t += step
        if r.data.size == 0:
            continue
        x = np.asarray(r.data, dtype=np.float64)
        if role == s.mov:
            x = x * s.sign_mov
        idx = np.searchsorted(edges, to_ref(np.asarray(r.ts, dtype=float))) - 1
        ok = np.isfinite(x) & (idx >= 0) & (idx < n_bins)
        if not ok.any():
            continue
        np.fmin.at(lo, idx[ok], x[ok])
        np.fmax.at(hi, idx[ok], x[ok])
    return edges, lo, hi


def _scale_pair(*xs: np.ndarray) -> list[np.ndarray]:
    return [_robust(x) for x in xs]


def _save(fig, out: Path, name: str) -> str:
    p = out / f"{name}.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p.name


def _time_axis(ax, span_s: float) -> None:
    if span_s > 1800:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    elif span_s > 20:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%M:%S.%f"))
    for lab in ax.get_xticklabels():
        lab.set_rotation(0)


def _rel_s(ts, t_center) -> np.ndarray:
    return (np.asarray(ts, dtype=float) - t_center) / US


# --------------------------------------------------------------------------- figures ---
def fig_overview(s: Setup, out: Path) -> tuple[str, str]:
    ref_t0, ref_t1 = s.rd.time_bounds(s.ch_ref, s.ref)
    fig, axes = plt.subplots(3, 1, figsize=(13, 7.2), gridspec_kw={"height_ratios": [1, 1.6, 0.8]})
    # (a) the whole reference recording, with the moving recording's span
    e, lo, hi = _envelope(s, s.ref, ref_t0, ref_t1, 1500, chunk_s=1800)
    sc = np.nanpercentile(np.abs(np.r_[lo, hi]), 99)
    axes[0].fill_between(_dts((e[:-1] + e[1:]) / 2), lo / sc, hi / sc, color=REF_C, lw=0)
    axes[0].axvspan(*_dts(np.array(s.overlap)), color=MOV_C, alpha=0.18,
                    label=f"{s.mov_name} recording (moving)")
    axes[0].set_title(f"(a) {s.ref_name} — the whole reference recording · {_spec_label(s.ch_ref)}")
    axes[0].set_ylim(-1.3, 1.3)
    axes[0].legend(loc="upper right")
    _time_axis(axes[0], (ref_t1 - ref_t0) / US)

    # (b) the overlap: both, each on its own robust scale
    t0, t1 = s.overlap
    pad = int(0.02 * (t1 - t0))
    for k, (role, name, c) in enumerate(((s.ref, s.ref_name, REF_C), (s.mov, s.mov_name, MOV_C))):
        e, lo, hi = _envelope(s, role, t0 - pad, t1 + pad, 1600)
        sc = np.nanpercentile(np.abs(np.r_[lo, hi]), 99) or 1.0
        y0 = 1.25 - 2.5 * k
        axes[1].fill_between(_dts((e[:-1] + e[1:]) / 2), y0 + lo / sc, y0 + hi / sc, color=c, lw=0,
                             label=f"{name} ({'reference' if role == s.ref else 'moving, through the map'})")
    m0, m1 = s.measured
    axes[1].axvspan(*_dts(np.array([m0, m1])), color=MAP_C, alpha=0.06, label="map measured span")
    for a in s.res.anchors:
        axes[1].axvline(_dt(a.t_ref_uutc), color=MAP_C, lw=0.8, ls=":")
    axes[1].set_yticks([])
    axes[1].set_ylim(-3.4, 2.6)
    axes[1].set_title("(b) the overlap on the reference clock — min–max envelope, each trace on "
                      "its own scale (the systems differ ~10–100× in gain) · dotted: anchors")
    axes[1].legend(loc="lower center", ncol=3)
    _time_axis(axes[1], (t1 - t0) / US)

    # (c) the fitted offset over the same span
    q = np.linspace(t0 - pad, t1 + pad, 800)
    off = (np.asarray(s.rd.clock_map.forward(q)) - q) / US
    axes[2].plot(_dts(q), off, color=MAP_C)
    axes[2].set_ylabel("offset (s)")
    axes[2].set_title("(c) offset t_moving − t_reference from the clock map")
    axes[2].sharex(axes[1])
    _time_axis(axes[2], (t1 - t0) / US)
    fig.tight_layout()
    return _save(fig, out, "1_overview"), (
        "The reference recording end to end, the moving recording's span inside it, and the overlap "
        "with both signals drawn on the reference clock. Large deflections are stimulation artifact.")


def fig_clock_map(s: Setup, out: Path) -> tuple[str, str]:
    d = s.res.drift or {}
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    q = np.linspace(s.overlap[0], s.overlap[1], 1000)
    off_map = (np.asarray(s.rd.clock_map.forward(q)) - q) / US
    ref_level = float(np.median(off_map))
    ax = axes[0]
    if d:
        t = np.asarray(d["t_ref_uutc"], dtype=float)
        off = np.asarray(d["offset_s"], dtype=float)
        acc = np.asarray(d["accepted"], dtype=bool)
        gap = np.asarray(d.get("gapped", np.zeros_like(acc)), dtype=bool)
        conf = np.asarray(d["confidence"], dtype=float)
        rej = ~acc & ~gap
        ax.scatter(_dts(t[rej]), (off[rej] - ref_level) * 1e3, s=10, marker="x", color=BAD_C,
                   label=f"window rejected ({rej.sum()})", zorder=2)
        sc = ax.scatter(_dts(t[acc]), (off[acc] - ref_level) * 1e3, s=12, c=conf[acc], cmap="viridis",
                        vmin=0, vmax=1, label=f"window accepted ({acc.sum()})", zorder=3)
        fig.colorbar(sc, ax=ax, pad=0.01, label="confidence")
        axes[1].bar(_dts(t), conf, width=(d.get("grid_s", 30) * 0.8) / 86400,
                    color=np.where(acc, MAP_C, BAD_C))
        axes[1].set_ylabel("window confidence")
        axes[1].set_ylim(0, 1)
    ax.plot(_dts(q), (off_map - ref_level) * 1e3, color="k", lw=1.4, label="fitted clock map", zorder=4)
    for a in s.res.anchors:
        ax.plot(_dt(a.t_ref_uutc), (a.offset_s - ref_level) * 1e3, marker="*", ms=12, color="#c0392b",
                ls="", zorder=5)
    ax.plot([], [], marker="*", ms=10, color="#c0392b", ls="", label="Fine anchors")
    ax.set_ylabel(f"offset − {ref_level:.3f} s  (ms)")
    lim = np.nanmax(np.abs((off_map - ref_level) * 1e3)) * 1.35 + 10
    ax.set_ylim(-lim, lim)
    ax.set_title(f"Clock map: {s.mov_name} − {s.ref_name} offset over time "
                 f"(range {np.ptp(off_map) * 1e3:.0f} ms over {(s.overlap[1] - s.overlap[0]) / US / 3600:.2f} h)")
    ax.legend(loc="upper right", ncol=4)
    _time_axis(axes[1], (s.overlap[1] - s.overlap[0]) / US)
    fig.tight_layout()
    return _save(fig, out, "2_clock_map"), (
        "Each dot is one drift window's measured offset; the black line is the map actually used. "
        "The map should follow the accepted windows and pass through the anchors (stars).")


def _pick_times(s: Setup, n: int = 3) -> list[int]:
    """Well-measured times early, mid and late in the measured span (confident accepted windows)."""
    d = s.res.drift or {}
    m0, m1 = s.measured
    targets = [m0 + (m1 - m0) * f for f in np.linspace(0.12, 0.88, n)]
    if not d:
        return [int(t) for t in targets]
    t = np.asarray(d["t_ref_uutc"], dtype=float)
    good = np.asarray(d["accepted"], dtype=bool) & (np.asarray(d["confidence"]) >= 0.6)
    cand = t[good] if good.any() else t
    return [int(cand[np.argmin(np.abs(cand - x))]) for x in targets]


def fig_multiscale(s: Setup, out: Path) -> tuple[str, str]:
    times = _pick_times(s)
    spans = [(120.0, (0.5, 40.0), "2 min · 0.5–40 Hz"), (10.0, (0.5, 40.0), "10 s · 0.5–40 Hz"),
             (2.0, (0.5, 60.0), "2 s · 0.5–60 Hz"), (0.25, (None, 100.0), "250 ms · < 100 Hz")]
    fs_mov = s.rd.fs(s.ch_mov, s.mov)
    fig, axes = plt.subplots(len(times), len(spans), figsize=(16, 3.1 * len(times)), squeeze=False)
    for i, tc in enumerate(times):
        for j, (span, band, label) in enumerate(spans):
            ax = axes[i][j]
            a, b = tc - int(span / 2 * US), tc + int(span / 2 * US)
            margin = int(max(2.0, span) * US)  # filter edge room
            tr, xr, fsr = _read(s, s.ref, a - margin, b + margin, band=band)
            tm, xm, fsm = _read(s, s.mov, a - margin, b + margin,
                                fs=fs_mov if span <= 2 else None, band=band)
            kr, km = (tr >= a) & (tr < b), (tm >= a) & (tm < b)
            yr, ym = _scale_pair(xr[kr], xm[km])
            # At 2 min the stimulation fills the band: stack the traces, or one hides the other.
            dy = 14.0 if span > 60 else 0.0
            ax.plot(_rel_s(tm[km], tc), np.clip(ym, -6 if dy else -12, 6 if dy else 12) - dy / 2,
                    color=MOV_C, lw=0.7, label=f"{s.mov_name} ({fsm:.0f} Hz, mapped)")
            yr = np.clip(yr, -6, 6) + dy / 2 if dy else yr
            style = dict(marker="o", ms=2.6, lw=0.8) if span <= 0.25 else dict(lw=0.9)
            ax.plot(_rel_s(tr[kr], tc), np.clip(yr, -12, 12), color=REF_C, **style,
                    label=f"{s.ref_name} ({fsr:.0f} Hz)")
            ax.set_xlim(-span / 2, span / 2)
            if i == 0:
                ax.set_title(label)
            if j == 0:
                ax.set_ylabel(f"{_clock(tc)}\nrobust σ")
            if i == len(times) - 1:
                ax.set_xlabel("s from centre")
            if i == 0 and j == len(spans) - 1:
                ax.legend(loc="lower right", fontsize=7)
    fig.suptitle(f"The two signals overlaid on the reference clock — {_spec_label(s.ch_ref)} "
                 f"({s.ref_name}) vs {'−' if s.sign_mov < 0 else ''}{_spec_label(s.ch_mov)} "
                 f"({s.mov_name}); each on its own robust scale, clipped at ±12σ", y=1.0)
    fig.tight_layout()
    return _save(fig, out, "3_multiscale"), (
        "Rows: three well-measured times early / mid / late. Columns zoom in from 2 minutes to "
        "250 ms; at 250 ms the reference samples are drawn as dots against the moving signal at "
        "its native rate. Aligned signals share their slow waves and their artifact onsets.")


def fig_before_after(s: Setup, out: Path, residual: dict) -> tuple[str, str]:
    q = np.linspace(s.measured[0], s.measured[1], 500)
    off_map = (np.asarray(s.rd.clock_map.forward(q)) - q) / US
    const = float(s.res.anchors[0].offset_s) if s.res.anchors else float(off_map[0])
    # where does a single constant offset hurt most?
    worst = int(q[np.argmax(np.abs(off_map - const))])
    best = int(q[np.argmin(np.abs(off_map - const))])
    fig, axes = plt.subplots(2, 2, figsize=(14, 6), sharey=True)
    for i, (tc, what) in enumerate(((best, "where the constant offset is right"),
                                    (worst, "where the clocks have drifted furthest from it"))):
        a, b = tc - 2 * US, tc + 2 * US
        band = (0.5, 40.0)
        tr, xr, _ = _read(s, s.ref, a - 3 * US, b + 3 * US, band=band)
        for j, (label, getter) in enumerate((
            (f"one constant offset ({const:.3f} s)",
             lambda a=a, b=b, band=band: _read_constant(s, a - 3 * US, b + 3 * US, const, band)),
            ("the fitted clock map",
             lambda a=a, b=b, band=band: _read(s, s.mov, a - 3 * US, b + 3 * US, band=band)),
        )):
            ax = axes[i][j]
            tm, xm, _ = getter()
            kr, km = (tr >= a) & (tr < b), (tm >= a) & (tm < b)
            yr, ym = _scale_pair(xr[kr], xm[km])
            ax.plot(_rel_s(tr[kr], tc), np.clip(yr, -10, 10), color=REF_C, lw=0.9, label=s.ref_name)
            ax.plot(_rel_s(tm[km], tc), np.clip(ym, -10, 10), color=MOV_C, lw=0.8, label=s.mov_name)
            drift_ms = (np.interp(tc, q, off_map) - const) * 1e3
            ax.set_title(f"{_clock(tc)} · {label}" + (f"\n(map − constant = {drift_ms:+.0f} ms here)" if j == 1 else f"\n{what}"))
            ax.set_xlim(-2, 2)
            if i == 1:
                ax.set_xlabel("s from centre")
    axes[0][0].legend(loc="lower left")
    fig.suptitle("Why drift tracking matters: the same 4 s, aligned with one constant offset vs the map")
    fig.tight_layout()
    return _save(fig, out, "4_before_after"), (
        f"A single offset (the first anchor's) against the fitted map. The map moves by "
        f"{np.ptp(off_map) * 1e3:.0f} ms over the recording; where it has moved furthest the constant "
        "alignment visibly slips, the map does not.")


def measure_residual(s: Setup, n: int = 60, win_s: float = 20.0) -> dict:
    """Re-measure the lag between the reference and the coregistered moving signal in ``n``
    windows across the measured span — independent of the windows the map was fitted from."""
    m0, m1 = s.measured
    centers = np.linspace(m0 + win_s * US, m1 - win_s * US, n).astype(np.int64)
    q = np.linspace(m0, m1, 500)
    off_map = (np.asarray(s.rd.clock_map.forward(q)) - q) / US
    const = float(s.res.anchors[0].offset_s) if s.res.anchors else float(off_map[0])
    rows = []
    band = (0.5, 20.0)
    for tc in centers:
        a, b = int(tc - win_s / 2 * US), int(tc + win_s / 2 * US)
        _, xr, fs = _read(s, s.ref, a, b)
        _, xm, _ = _read(s, s.mov, a, b)  # on the reference grid
        n_ = min(xr.size, xm.size)
        if n_ < fs * win_s * 0.8 or np.isnan(xr[:n_]).mean() > 0.2 or np.isnan(xm[:n_]).mean() > 0.2:
            continue
        ar = np.nan_to_num(_band(xr[:n_], fs, *band))
        am = np.nan_to_num(_band(xm[:n_], fs, *band))
        ar, am = np.clip(_robust(ar), -8, 8), np.clip(_robust(am), -8, 8)
        d = normxcorr_delay(ar, am, fs, max_lag_s=0.3)
        rows.append((int(tc), d.lag_s * 1e3, d.confidence, d.ncc,
                     (float(np.interp(tc, q, off_map)) - const) * 1e3))
    arr = np.array(rows) if rows else np.zeros((0, 5))
    good = arr[:, 2] >= 0.3 if arr.size else np.zeros(0, bool)
    lag = arr[good, 1] if arr.size else np.zeros(0)
    return {
        "rows": arr, "good": good, "const": const,
        "median_ms": float(np.median(lag)) if lag.size else float("nan"),
        "p95_abs_ms": float(np.percentile(np.abs(lag), 95)) if lag.size else float("nan"),
        "n": int(arr.shape[0]), "n_good": int(good.sum()),
    }


def fig_residual(s: Setup, out: Path, r: dict) -> tuple[str, str]:
    arr, good = r["rows"], r["good"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [3, 1]})
    ax = axes[0]
    if arr.size:
        t = _dts(arr[:, 0])
        ax.plot(t, arr[:, 4], color=BAD_C, lw=1.2, label="lag with one constant offset (= map − constant)")
        ax.scatter([x for x, g in zip(t, good, strict=True) if g], arr[good, 1], s=16, color=MAP_C,
                   label="lag left after the map", zorder=3)
        ax.scatter([x for x, g in zip(t, good, strict=True) if not g], arr[~good, 1], s=16,
                   facecolors="none", edgecolors=BAD_C, label="(low confidence window)", zorder=3)
        ax.axhline(0, color="k", lw=0.6)
        _time_axis(ax, (arr[-1, 0] - arr[0, 0]) / US)
        axes[1].hist(arr[good, 1], bins=np.arange(-30, 31, 2), color=MAP_C)
    ax.set_ylabel("lag of moving vs reference (ms)")
    ax.set_title(f"Independent check: residual lag in {r['n']} windows of 20 s (0.5–20 Hz, ±300 ms search)")
    ax.legend(loc="upper left")
    axes[1].set_title(f"residual: median {r['median_ms']:+.1f} ms, |95%| {r['p95_abs_ms']:.1f} ms")
    axes[1].set_xlabel("ms")
    fig.tight_layout()
    return _save(fig, out, "5_residual_lag"), (
        "The lag between the reference and the coregistered moving signal, re-measured in windows "
        "spread evenly over the recording (not the windows the map was fitted on). After the map it "
        "should sit at 0 within the resolution of the reference (1 sample = "
        f"{1e3 / s.rd.fs(s.ch_ref, s.ref):.1f} ms); the grey line is what a constant offset would leave.")


def measure_pulses(s: Setup, n_windows: int = 40, win_s: float = 6.0) -> dict:
    """Sharp stimulation artifacts are recorded by both systems: their onsets are sync markers
    far sharper than physiology. Find pulses in the moving file at its native rate, then the same
    pulse in the reference within ±30 ms, and report the timing difference."""
    m0, m1 = s.measured
    fs_mov = s.rd.fs(s.ch_mov, s.mov)
    rows = []
    for tc in np.linspace(m0 + 10 * US, m1 - 10 * US, n_windows).astype(np.int64):
        a, b = int(tc - win_s / 2 * US), int(tc + win_s / 2 * US)
        tm, xm, fsm = _read(s, s.mov, a, b, fs=fs_mov)
        tr, xr, _ = _read(s, s.ref, a - US, b + US)
        if xm.size < 10 or xr.size < 10:
            continue
        dm = np.abs(np.diff(np.nan_to_num(xm)))
        thr = np.median(dm) + 40 * (np.median(np.abs(dm - np.median(dm))) + 1e-12)
        on = np.flatnonzero((dm[1:] > thr) & (dm[:-1] <= thr)) + 1
        # one onset per pulse: at least 100 ms apart
        keep: list[int] = []
        last = -np.inf
        for onset in on.tolist():
            if (onset - last) / fsm > 0.1:
                keep.append(onset)
                last = float(onset)
        er = np.abs(np.diff(np.nan_to_num(xr)))
        for k in keep[:6]:
            t_pulse = tm[k]
            w = (tr[1:] >= t_pulse - 30_000) & (tr[1:] <= t_pulse + 30_000)
            if w.sum() < 3:
                continue
            idx = np.flatnonzero(w)
            j = idx[np.argmax(er[idx])]
            if er[j] < np.median(er) + 20 * (np.median(np.abs(er - np.median(er))) + 1e-12):
                continue
            # the reference's steepest step lies between samples j and j+1: take its midpoint
            t_ref_step = (tr[j] + tr[j + 1]) / 2
            rows.append((int(t_pulse), (t_ref_step - t_pulse) / 1e3))
    arr = np.array(rows) if rows else np.zeros((0, 2))
    return {
        "rows": arr, "n": int(arr.shape[0]),
        "median_ms": float(np.median(arr[:, 1])) if arr.size else float("nan"),
        "iqr_ms": float(np.subtract(*np.percentile(arr[:, 1], [75, 25]))) if arr.size else float("nan"),
    }


def fig_pulses(s: Setup, out: Path, p: dict) -> tuple[str, str]:
    arr = p["rows"]
    fs_mov = s.rd.fs(s.ch_mov, s.mov)
    fig = plt.figure(figsize=(15, 7.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
    picks = arr[np.linspace(0, len(arr) - 1, 3).astype(int), 0] if len(arr) >= 3 else []
    for i, tp in enumerate(picks):
        ax = fig.add_subplot(gs[0, i])
        a, b = int(tp - 60_000), int(tp + 90_000)
        tm, xm, _ = _read(s, s.mov, a - US, b + US, fs=fs_mov)
        tr, xr, _ = _read(s, s.ref, a - US, b + US)
        km, kr = (tm >= a) & (tm < b), (tr >= a) & (tr < b)
        ym, yr = _scale_pair(xm[km], xr[kr])
        ax.plot(_rel_s(tm[km], tp) * 1e3, ym / (np.nanmax(np.abs(ym)) or 1), color=MOV_C, lw=0.8,
                label=f"{s.mov_name} {fs_mov:.0f} Hz")
        ax.plot(_rel_s(tr[kr], tp) * 1e3, yr / (np.nanmax(np.abs(yr)) or 1), color=REF_C, marker="o",
                ms=3, lw=0.9, label=f"{s.ref_name} samples")
        ax.axvline(0, color="k", lw=0.5, ls=":")
        ax.set_title(f"stim artifact at {_clock(tp)}")
        ax.set_xlabel("ms from the moving file's pulse onset")
        if i == 0:
            ax.legend(loc="lower right", fontsize=7)
    ax = fig.add_subplot(gs[1, :2])
    if arr.size:
        ax.scatter(_dts(arr[:, 0]), arr[:, 1], s=10, color=MAP_C)  # type: ignore[arg-type]
        ax.axhline(0, color="k", lw=0.6)
        _time_axis(ax, (arr[-1, 0] - arr[0, 0]) / US)
    ax.set_ylabel("reference step − moving onset (ms)")
    ax.set_title(f"{p['n']} pulses across the recording: median {p['median_ms']:+.2f} ms, IQR {p['iqr_ms']:.2f} ms")
    ax2 = fig.add_subplot(gs[1, 2])
    if arr.size:
        ax2.hist(arr[:, 1], bins=30, color=MAP_C)
    ax2.set_xlabel("ms")
    ax2.set_title("distribution")
    fig.suptitle("Stimulation artifacts as sync markers — both systems record every pulse; after "
                 "coregistration they should coincide to within a reference sample")
    fig.tight_layout()
    return _save(fig, out, "6_stim_pulses"), (
        "The sharpest shared feature there is. Each pulse's onset is found in the moving recording at "
        "its native rate and the steepest step within ±30 ms in the reference; the difference is the "
        "map's timing error there, bounded below by the reference's sample period "
        f"({1e3 / s.rd.fs(s.ch_ref, s.ref):.1f} ms) and its anti-alias filter delay. A constant bias "
        "across the recording is a filter delay, not drift. Isolated points at ±20–30 ms are pairing "
        "errors (the stimulation program fires sub-pulses tens of ms apart), not map error — read the "
        "median and IQR.")


def fig_unified(s: Setup, out: Path) -> tuple[str, str, dict]:
    rec = CoregisteredRecording(s.rd, channels=[])
    name = _spec_label(s.ch_ref)
    src = {s.ref: s.ch_ref, s.mov: s.ch_mov}
    kw: dict[str, float] = {f"scale_{s.mov.lower()}": s.sign_mov}
    rec.set_channel(name, a=src["A"], b=src["B"], prefer=s.mov, **kw)
    ag = rec.agreement(name)
    note = ""
    if ag.status == "scale" and np.isfinite(ag.gain_b_over_a) and ag.gain_b_over_a > 0:
        # express the preferred (moving) file in the reference's units scale — a deliberate choice
        g = ag.gain_b_over_a if s.mov == "A" else 1 / ag.gain_b_over_a
        kw[f"scale_{s.mov.lower()}"] = s.sign_mov * g
        rec.set_channel(name, a=src["A"], b=src["B"], prefer=s.mov, **kw)
        note = f"gain corrected ×{g:.3g} on the moving file"
        ag = rec.agreement(name)
    mb = s.rd.time_bounds(s.ch_mov, s.mov)
    fig, axes = plt.subplots(2, 1, figsize=(14, 5.6))
    for ax, (edge, span) in zip(axes, ((mb[0], 40.0), (mb[1], 40.0)), strict=True):
        a, b = int(edge - span / 2 * US), int(edge + span / 2 * US)
        u = rec.get_data(name, a, b)
        assert isinstance(u, UnifiedRead)
        x = np.asarray(u.data, dtype=float)
        origin = np.asarray(u.source)
        y = _band(x, u.fs, 0.5, 40.0)
        for code, c, lab in ((1, REF_C if s.ref == "A" else MOV_C, "file A"),
                             (2, REF_C if s.ref == "B" else MOV_C, "file B")):
            yy = np.where(origin == code, y, np.nan)
            ax.plot(_rel_s(u.ts, edge), yy, color=c, lw=0.7,
                    label=f"{lab} ({s.ref_name if (code == 1) == (s.ref == 'A') else s.mov_name})")
        ax.axvline(0, color="k", ls=":", lw=0.8)
        ax.set_title(f"{name} as one channel, preferring {s.mov_name}, around the "
                     f"{'start' if edge == mb[0] else 'end'} of {s.mov_name} ({_clock(edge)})"
                     + (f" — {u.merge_note}" if u.merge_note else ""))
        ax.set_xlabel("s from the edge")
        ax.legend(loc="upper right")
    fig.tight_layout()
    return _save(fig, out, "7_unified"), (
        f"The two files served as one channel: {s.mov_name} where it has data, {s.ref_name} before "
        f"and after it. Source agreement: {ag.status} (r = {ag.corr:.2f}, gain B/A = "
        f"{ag.gain_b_over_a:.3g}){'; ' + note if note else ''}. Colour = which file the samples "
        "came from. For display only — analyse within one source."), {"agreement": ag.status,
                                                                     "corr": ag.corr, "note": note}


# --------------------------------------------------------------------------- report ----
def build(result_path: Path, out: Path, channel: str | None,
          file_a: str | Path | None = None, file_b: str | Path | None = None) -> Path:
    """Write the review report for a saved result: the figures, ``index.html`` and ``summary.json``.

    Parameters
    ----------
    result_path : pathlib.Path
        The saved coregistration result (``.json``).
    out : pathlib.Path
        The folder to write into (created if missing).
    channel : str or None
        The channel to review, e.g. ``"LTP1-LTP3"``; ``None`` uses the channels Stage 2 aligned on.
    file_a, file_b : str or pathlib.Path, optional
        Where the recordings are now, if they have moved (checked by time span, as when opening
        a result).

    Returns
    -------
    pathlib.Path
        The written ``index.html``.
    """
    res = CoregResult.load(result_path)
    rd = CoregisteredReader.from_result(result_path, file_a, file_b)
    ref: FileRole = "A" if res.reference_role == "A" else "B"
    mov: FileRole = "B" if ref == "A" else "A"
    names = {"A": Path(res.file_a).stem, "B": Path(res.file_b).stem}
    stage = (res.stage_channels or {}).get("fine", {})
    if channel:
        ch_ref, _ = _parse_label(channel, rd.channels(ref))
        ch_mov, sign = _parse_label(channel, rd.channels(mov))
        sign = -1.0 if stage.get("moving", "").startswith("-(") else 1.0
    else:
        ch_ref, s_ref = _parse_label(stage["reference"], rd.channels(ref))
        ch_mov, s_mov = _parse_label(stage["moving"], rd.channels(mov))
        sign = s_ref * s_mov
    mov_b = rd.time_bounds(ch_mov, mov)
    ref_b = rd.time_bounds(ch_ref, ref)
    overlap = (max(mov_b[0], ref_b[0]), min(mov_b[1], ref_b[1]))
    m_span = rd.clock_map.metadata.get("measured_span_uutc", overlap)
    measured = (int(m_span[0]), int(m_span[1]))
    s = Setup(res, rd, ref, mov, names[ref], names[mov], ch_ref, ch_mov, sign, overlap, measured)
    out.mkdir(parents=True, exist_ok=True)

    figs = []
    print("overview…")
    figs.append(("Overview", *fig_overview(s, out)))
    print("clock map…")
    figs.append(("Clock map", *fig_clock_map(s, out)))
    print("multi-scale…")
    figs.append(("Multi-scale overlay", *fig_multiscale(s, out)))
    print("residual lag…")
    resid = measure_residual(s)
    figs.append(("Before / after drift correction", *fig_before_after(s, out, resid)))
    figs.append(("Residual lag (independent check)", *fig_residual(s, out, resid)))
    print("stim pulses…")
    pulses = measure_pulses(s)
    if pulses["n"] >= 3:
        figs.append(("Stimulation pulses as sync markers", *fig_pulses(s, out, pulses)))
    print("unified…")
    uni = fig_unified(s, out)
    figs.append(("Both files as one channel", uni[0], uni[1]))

    qc = res.qc or {}
    q = np.linspace(measured[0], measured[1], 500)
    off = (np.asarray(rd.clock_map.forward(q)) - q) / US
    summary = [
        ("reference", f"{names[ref]} (file {ref}) · {_spec_label(ch_ref)} · {rd.fs(ch_ref, ref):.0f} Hz"),
        ("moving", f"{names[mov]} (file {mov}) · {'−' if sign < 0 else ''}{_spec_label(ch_mov)} · "
                   f"{rd.fs(ch_mov, mov):.0f} Hz"),
        ("overlap", f"{_clock(overlap[0])} → {_clock(overlap[1])} ({(overlap[1] - overlap[0]) / US / 3600:.2f} h)"),
        ("offset", f"{off.min():.3f} … {off.max():.3f} s (drift range {np.ptp(off) * 1e3:.0f} ms)"),
        ("map verified", f"{qc.get('map_verified', '—')} (NCC {qc.get('map_verification_ncc', float('nan')):.2f}, "
                         f"SNR {qc.get('map_verification_snr', float('nan')):.0f})"),
        ("drift windows", f"{qc.get('n_accepted', '—')} of {qc.get('n_windows', '—')} accepted"),
        ("max LOO residual", f"{qc.get('max_abs_loo_residual_ms', float('nan')):.1f} ms"),
        ("residual lag (this report)", f"median {resid['median_ms']:+.1f} ms, 95% within ±{resid['p95_abs_ms']:.1f} ms "
                                       f"({resid['n_good']} of {resid['n']} windows)"),
        ("stim pulse timing (this report)", f"median {pulses['median_ms']:+.2f} ms, IQR {pulses['iqr_ms']:.2f} ms "
                                            f"({pulses['n']} pulses)" if pulses["n"] else "no pulses found"),
        ("unified channel", f"agreement {uni[2]['agreement']} (r = {uni[2]['corr']:.2f}) {uni[2]['note']}"),
    ]
    warnings = qc.get("warnings") or []
    e = html.escape
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Coregistration review</title>
<style>
 :root {{ --bg:#fbfbfa; --fg:#1d232b; --muted:#5d6672; --line:#e3e5e8; --card:#fff; --accent:#1b9e8f; }}
 @media (prefers-color-scheme: dark) {{ :root {{ --bg:#14181d; --fg:#e6eaf0; --muted:#9aa4b1; --line:#2a313a; --card:#1b2027; }} img {{ background:#fff; }} }}
 body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 system-ui, sans-serif; }}
 main {{ max-width:1400px; margin:0 auto; padding:28px 16px 60px; }}
 h1 {{ font-size:22px; margin:0 0 4px; }} h2 {{ font-size:17px; margin:34px 0 6px; }}
 .sub {{ color:var(--muted); margin:0 0 18px; }}
 table {{ border-collapse:collapse; width:100%; max-width:900px; background:var(--card); }}
 td {{ border-bottom:1px solid var(--line); padding:6px 10px; vertical-align:top; }}
 td:first-child {{ color:var(--muted); white-space:nowrap; width:1%; }}
 figure {{ margin:10px 0 0; background:var(--card); border:1px solid var(--line); border-radius:6px; padding:10px; }}
 figure img {{ width:100%; height:auto; display:block; }}
 figcaption {{ color:var(--muted); font-size:14px; margin-top:8px; max-width:1000px; }}
 .warn {{ color:#b5542b; }}
 nav a {{ color:var(--accent); margin-right:14px; font-size:14px; }}
</style></head><body><main>
<h1>Coregistration review</h1>
<p class="sub">{e(str(result_path))} · created {e(str(res.created_utc or ''))}</p>
<table>{''.join(f'<tr><td>{e(k)}</td><td>{e(str(v))}</td></tr>' for k, v in summary)}</table>
{''.join(f'<p class="warn">⚠ {e(w)}</p>' for w in warnings)}
<nav>{''.join(f'<a href="#f{i}">{e(t)}</a>' for i, (t, _, _) in enumerate(figs))}</nav>
{''.join(f'<h2 id="f{i}">{e(t)}</h2><figure><img src="{e(img)}" alt="{e(t)}"><figcaption>{e(cap)}</figcaption></figure>' for i, (t, img, cap) in enumerate(figs))}
</main></body></html>"""
    index = out / "index.html"
    index.write_text(page)
    (out / "summary.json").write_text(json.dumps(dict(summary), indent=2))
    rd.close()
    return index


def main(argv: list[str] | None = None) -> None:
    """The ``sigcoreg-review`` command: ``sigcoreg-review RESULT [--channel C] [--out DIR]
    [--file-a PATH] [--file-b PATH]``.

    Parameters
    ----------
    argv : list of str, optional
        The command-line arguments (default: ``sys.argv[1:]``). The report goes to ``--out``, or
        ``<result>_review`` next to the result.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result", type=Path, help="saved coregistration result (.json)")
    ap.add_argument("--channel", help='review channel, e.g. "LTP1-LTP3" (default: the Fine stage channels)')
    ap.add_argument("--out", type=Path, help="output folder (default: <result>_review next to it)")
    ap.add_argument("--file-a", help="file A's path now (if the recordings have moved)")
    ap.add_argument("--file-b", help="file B's path now")
    a = ap.parse_args(argv)
    out = a.out or a.result.with_name(a.result.stem + "_review")
    index = build(a.result, out, a.channel, a.file_a, a.file_b)
    print(f"report: {index}")


if __name__ == "__main__":
    main()
