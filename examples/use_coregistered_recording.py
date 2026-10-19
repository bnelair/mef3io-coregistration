"""How to use a saved coregistration (.json) to read the two recordings as one.

    python examples/use_coregistered_recording.py path/to/result.coreg.json

Walks through the Python API step by step, printing what each call returns, and saves a figure
of the reads (``<result>_usage.png``). Each ``# %%`` block runs on its own in an IDE / Jupyter.

The classes:

* ``CoregisteredSession`` / ``CoregisteredMefReader`` — ONE file, read like a MEF session
  (``mef3io.Reader`` calls, or the legacy ``mef_tools`` ``MefReader`` calls 1:1), every time on the
  reference clock. Any channel, unipolar or bipolar, partial windows.
* ``CoregisteredRecording`` — the two files as ONE recording on the reference clock: logical
  channels, each served from a preferred file with the other filling where it has no data.
* ``CoregisteredReader`` — lower level: read either file, natively or coregistered onto the
  reference clock (e.g. the moving file sample-aligned to the reference file's grid).
"""
# %% 0 — imports and the result file
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from mef3io_coregistration.reader import (
    CoregisteredMefReader,
    CoregisteredRecording,
    CoregisteredSession,
)

RESULT = Path(sys.argv[1] if len(sys.argv) > 1 else "result.coreg.json")


def clock(t_uutc: float) -> str:
    """uUTC (microseconds since the epoch, as in MEF3) → readable UTC time."""
    return datetime.fromtimestamp(t_uutc / 1e6, UTC).strftime("%H:%M:%S.%f")[:-3]


# %% 1 — open the result: the two MEF3 files + the clock map between them
# The result names both files and which one is the reference (never resampled). It is only the
# transformation: if the recordings have moved, pass where they are now —
#     CoregisteredRecording.open(RESULT, file_a="/new/cadence.mefd", file_b="/new/natus.mefd")
# (without it, recordings missing from their stored paths are looked for next to the result).
# Opening checks each file's start and end against the result's (within 1 s; verify=False to skip).
rec = CoregisteredRecording.open(RESULT)
rd = rec.reader  # the underlying CoregisteredReader (CoregisteredReader.from_result(RESULT) also works)
ref, mov = rd.reference_file, rd.moving_file
print(f"reference file {ref}: {rd.reader(ref).path}")
print(f"moving file    {mov}: {rd.reader(mov).path}")
d0, d1 = rd.clock_map.domain
print(f"clock map covers {clock(d0)} → {clock(d1)} (reference clock)")

# %% 1b — read ONE file like a MEF session, on the reference clock
# Same calls as mef3io.Reader. Any channel of the file — the drift is the same for all of them.
mov_s = CoregisteredSession(rd, file="moving")  # reuse the open files; or pass the result path
lo, _ = mov_s.time_bounds("LTP1")
w0, w1 = lo + 900_000_000, lo + 902_000_000  # 2 s, 15 min in (reference-clock uUTC)
x_uni = mov_s.read("LTP1", w0, w1)  # only this window is read from disk
x_bip = mov_s.read("LTP1-LTP3", w0, w1)  # anode − cathode on the native samples, then resampled
print(f"\nmef3io-style: {mov_s.path.split('/')[-1]} LTP1 {x_uni.size} samples at {mov_s.fs('LTP1'):g} Hz, "
      f"LTP1-LTP3 {x_bip.size}; first sample at {clock(mov_s.times('LTP1', w0, w1)[0])} (reference clock)")
# …and the legacy mef_tools calls, unchanged apart from the constructor:
mr = CoregisteredMefReader(RESULT, file="moving")  # was: MefReader(path)
arrs = mr.get_data(["LTP1", "LTP3", "LTP1-LTP3"], w0, w1)
print(f"mef_tools-style: get_data → {[a.shape for a in arrs]}, fsamp {mr.get_property('fsamp', 'LTP1')}, "
      f"start_time on the reference clock {clock(mr.get_property('start_time', 'LTP1'))}")
mr.close()

# %% 2 — the logical channels you get by default
# Every channel name becomes a logical channel; names found in both files are linked, and the
# reference file is preferred. Nothing to configure for the common case.
print(f"\n{len(rec.channels)} logical channels:", ", ".join(rec.channels))
for lc in rec.logical_channels()[:3]:
    print(f"  {lc.name}: A={lc.a and lc.a.label}  B={lc.b and lc.b.label}  prefer {lc.prefer}")

# %% 3 — read one channel, on the reference clock
# get_data(channel, start, stop) with start/stop in reference-clock uUTC. Where the preferred file
# has data you get its samples; where it does not (before/after it, or in a gap) the other file
# fills in — through the clock map. Every sample says where it came from.
name = rec.channels[0]
mov_lo, mov_hi = rd.time_bounds(rec.channel(name).source(mov), mov)  # moving span, ref clock
t0 = mov_lo + 600_000_000  # 10 min into the moving recording
sig = rec.get_data(name, t0, t0 + 10_000_000)  # 10 s
print(f"\n{name}: {sig.n} samples at {sig.fs:g} Hz from {clock(sig.ts[0])}")
print("  data  ", sig.data[:5].round(2), "…  (physical units)")
print("  ts    ", sig.ts[:3], "…  (reference-clock uUTC per sample)")
print("  source", np.unique(sig.source, return_counts=True), " (1 = file A, 2 = file B, 0 = none)")
print("  extrapolated samples:", int(sig.extrapolated.sum()))

# %% 4 — your own logical channel: montage per file, preferred file, fallback
# Here: the bipolar LTP1 − LTP3 on both files, preferring the moving file (8 kHz Cadence) and
# falling back to the reference (256 Hz Natus) outside it. Names and montage may differ per file:
# a=("LA1", None) on one side and b=("EEG LA1-Ref", "Fz") on the other works the same way.
rec.set_channel("LTP1-3", a=("LTP1", "LTP3"), b=("LTP1", "LTP3"), prefer=mov)

# Before one file may fill the other's gaps, the two must agree (sign, gain, units):
ag = rec.agreement("LTP1-3")
print(f"\nagreement: {ag.status}  r={ag.corr:.2f}  gain B/A={ag.gain_b_over_a:.3g}  — {ag.message}")

# They are recorded with opposite polarity and ~16x different gain, so the gap filling is refused
# until you say how the two relate: a per-source scale (−1 inverts) and offset.
if ag.status != "ok":
    g = ag.gain_b_over_a  # corrected B/A amplitude ratio
    scale = -1.0 if ag.status == "inverted" else 1.0
    kw = {f"scale_{mov.lower()}": scale * (g if mov == "A" else 1 / g) if np.isfinite(g) else scale}
    rec.set_channel("LTP1-3", a=("LTP1", "LTP3"), b=("LTP1", "LTP3"), prefer=mov, **kw)
    ag = rec.agreement("LTP1-3")
    print(f"after correction {kw}: {ag.status}  r={ag.corr:.2f}  gain B/A={ag.gain_b_over_a:.3g}")

# Offsets (offset_a / offset_b) exist for amplifiers with a FIXED DC difference. Not here: the
# Cadence amplifier is DC-coupled — its baseline settles over the first ~10 s and then wanders by
# hundreds of units — while Natus is AC-coupled, so no constant offset fits (the agreement's
# offset_b_minus_a is only a whole-overlap median). The robust answer is §5: high-pass each run.

# %% 5 — read across the start of the moving recording: one continuous channel
edge = rec.get_data("LTP1-3", mov_lo - 5_000_000, mov_lo + 5_000_000)  # ±5 s around the edge
print(f"\nLTP1-3 across the edge: {edge.n} samples at {edge.fs:g} Hz"
      + (f" — note: {edge.merge_note}" if edge.merge_note else ""))
for src, a, b in edge.runs():  # stretches from ONE file — analyse each on its own
    print(f"  run: file {'?AB'[src]}  {clock(a)} → {clock(b)}")

# Filter within each run, never across a switch: one filter spanning two amplifiers would smear the
# DC step between them into the signal. Here a 0.5 Hz high-pass per run removes the DC difference.
from scipy.signal import butter, sosfiltfilt  # noqa: E402

hp = butter(2, 0.5 / (edge.fs / 2), "highpass", output="sos")
edge_hp = np.full(edge.n, np.nan)
for _src, a, b in edge.runs():
    k = (edge.ts >= a) & (edge.ts < b) & np.isfinite(edge.data)
    edge_hp[k] = sosfiltfilt(hp, edge.data[k])

# %% 6 — sample-aligned arrays for analysis (CoregisteredReader)
# Read the reference natively and the moving file coregistered ONTO THE SAME GRID: same length,
# same timestamps, so the two can be stacked, subtracted, correlated sample for sample.
spec = ("LTP1", "LTP3")
r_ref = rd.get_data(spec, t0, t0 + 10_000_000, file=ref)                 # native reference samples
r_mov = rd.get_data(spec, t0, t0 + 10_000_000, file=mov, coregister=True)  # resampled onto them
assert np.array_equal(r_ref.ts, r_mov.ts)
both = np.vstack([r_ref.data, r_mov.data])
print(f"\nsample-aligned: {both.shape} array at {r_ref.fs:g} Hz, timestamps identical")
# …or the moving file at its own rate, on the reference clock:
r_mov_hi = rd.get_data(spec, t0, t0 + 1_000_000, file=mov, coregister=True, fs=rd.fs(spec, mov))
print(f"moving at full rate on the reference clock: {r_mov_hi.n} samples at {r_mov_hi.fs:g} Hz")
# …or natively, on its own clock (no transform) — times here are the MOVING file's clock:
t_mov = int(rd.clock_map.forward(t0))
r_mov_native = rd.get_data(spec, t_mov, t_mov + 1_000_000, file=mov, coregister=False)
print(f"moving natively: {r_mov_native.n} samples from {clock(r_mov_native.ts[0])} (its own clock)")

# %% 7 — converting event times between the clocks
# e.g. an event marked in the moving file → where to look in the reference file, and back.
t_event_mov = int(rd.clock_map.forward(t0))  # the moving file's clock at our read start
print(f"\nmoving-file time {clock(t_event_mov)} → reference {clock(rec.to_reference_time(t_event_mov, mov))}")
print(f"reference time  {clock(t0)} → moving file {clock(rec.to_file_time(t0, mov))}")

# %% 8 — save the mapping, reopen it later with one line
cfg = RESULT.with_name(RESULT.stem + "_recording.json")
rec.save_config(cfg, result_path=RESULT)
rec2 = CoregisteredRecording.from_config(cfg)
print(f"\nsaved mapping → {cfg}; reopened: {len(rec2.channels)} channels, "
      f"LTP1-3 prefers {rec2.channel('LTP1-3').prefer}")
rec2.close()

# %% 9 — a picture of what was read
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    s = (edge.ts - mov_lo) / 1e6
    for code, c, lab in ((1, "#d9822b", "file A"), (2, "#1b9e8f", "file B")):
        axes[0].plot(s, np.where(edge.source == code, edge.data, np.nan), color=c, lw=0.4,
                     alpha=0.35, label=f"{lab} as read")
        axes[0].plot(s, np.where(edge.source == code, edge_hp, np.nan), color=c, lw=0.7,
                     label=f"{lab}, high-passed within its run")
    axes[0].axvline(0, color="k", ls=":")
    axes[0].set_title("§5 LTP1−3 as one channel across the start of the moving file "
                      "(colour = source file; faint = as read, with the DC step between systems)")
    axes[0].set_xlabel("s from the edge")
    axes[0].legend()
    z = (r_ref.ts - t0) / 1e6
    for x, c, lab in ((r_ref.data, "#1b9e8f", f"reference ({ref})"), (r_mov.data * np.sign(kw.get(f"scale_{mov.lower()}", 1)), "#d9822b", f"moving ({mov}), coregistered")):
        k = np.isfinite(x)
        med, mad = np.median(x[k]), np.median(np.abs(x[k] - np.median(x[k]))) or 1
        axes[1].plot(z, np.clip((x - med) / mad, -20, 20), color=c, lw=0.7, label=lab)
    axes[1].set_xlim(4, 6)
    axes[1].set_title("§6 sample-aligned arrays (2 s shown, robust-scaled, polarity matched)")
    axes[1].legend()
    offs = np.linspace(d0, d1, 400)
    axes[2].plot((offs - d0) / 3.6e9, (np.asarray(rd.clock_map.forward(offs)) - offs) / 1e6, color="#3355aa")
    axes[2].set_title("the clock map: offset t_moving − t_reference")
    axes[2].set_xlabel("hours into the map")
    axes[2].set_ylabel("s")
    fig.tight_layout()
    png = RESULT.with_name(RESULT.stem + "_usage.png")
    fig.savefig(png, dpi=120)
    print(f"\nfigure → {png}")
except ImportError:
    print("\n(install matplotlib for the figure: pip install -e '.[review]')")

rec.close()
