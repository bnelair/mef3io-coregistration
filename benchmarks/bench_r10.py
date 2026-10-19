"""R10 performance benchmarks on a 24 h synthetic recording.

    python benchmarks/bench_r10.py [--fs 256] [--hours 24]

Budgets: Stage 1 gross over 24 h within an interactive budget (< 2 s target);
Stage 3 drift over 24 h in < a few seconds at moderate Fs. The pair (10-minute offset, ±5 s swing
over the day) is rendered in memory; rendering is not timed. Also reports recovery accuracy.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from mef3io_coregistration.align import estimate_anchor, estimate_drift, estimate_gross_offset
from mef3io_coregistration.align.fine import local_seed
from mef3io_coregistration.config import US_PER_S
from mef3io_coregistration.io.derived import Decimated
from mef3io_coregistration.synthetic import WarpSpec, render_pair_long

_DAY = 86_400.0
_START = 1_700_000_000 * US_PER_S


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs", type=float, default=256.0, help="rate of both traces (Hz)")
    ap.add_argument("--hours", type=float, default=24.0)
    args = ap.parse_args()
    dur = args.hours * 3600.0
    spec = WarpSpec("swing", offset0_s=600.0, swing_amp_s=5.0, swing_period_s=_DAY)

    t0 = time.perf_counter()
    rp = render_pair_long(spec, dur, args.fs, args.fs, _START, seed=1)
    print(f"render {dur / 3600:.0f} h @ {args.fs:g} Hz: {time.perf_counter() - t0:.1f} s (not budgeted)")
    ref = Decimated(rp.ref, args.fs, rp.ref_start_uutc)
    mov = Decimated(rp.mov, args.fs, rp.mov_start_uutc)

    t0 = time.perf_counter()
    gross = estimate_gross_offset(ref, mov)
    t_gross = time.perf_counter() - t0
    print(f"R10.1 gross  : {t_gross:6.2f} s  (target < 2 s)  offset {gross.offset_s:.3f} s, "
          f"locked={gross.locked}")

    anchors = []
    t0 = time.perf_counter()
    for h in (args.hours / 4, 3 * args.hours / 4):  # Stage 2 as the session runs it
        roi = _START + int(h * 3600 * US_PER_S)
        seed = local_seed(ref, mov, roi, gross.offset_s).offset_s
        anchors.append(estimate_anchor(ref, mov, roi, seed, half_window_s=150.0, max_residual_s=0.5))
    t_fine = time.perf_counter() - t0
    print(f"      fine   : {t_fine:6.2f} s  for {len(anchors)} anchors (local seed + refine)")

    t0 = time.perf_counter()
    drift = estimate_drift(ref, mov, anchors)
    t_drift = time.perf_counter() - t0
    tr = (drift.t_ref_uutc - _START) / US_PER_S
    ok = drift.accepted
    err = (drift.smoothed_offset_s[ok] - spec.offset_s(tr[ok])) * args.fs
    print(
        f"R10.2 drift  : {t_drift:6.2f} s  (target < a few s)  {ok.sum()}/{ok.size} windows, "
        f"RMS {np.sqrt(np.mean(err**2)):.3f} samples"
    )


if __name__ == "__main__":
    main()
