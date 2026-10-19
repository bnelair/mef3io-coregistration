"""Align two recordings end to end, save the result, and read both files through it.

    python examples/01_align_synthetic_pair.py

Self-contained: it renders a synthetic pair of MEF 3.0 recordings of the same signal on two
clocks (the second clock runs two minutes ahead and drifts by 5 s per 12 h; the second system
is recorded at another sampling rate and starts three minutes later), writes them under a temporary directory, runs the three alignment
stages with ``CoregSession``, saves the result as JSON, and reads both recordings back through
it on the reference clock. Times are uUTC (microseconds since the Unix epoch), as in MEF 3.0.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import mef3io
import numpy as np

from mef3io_coregistration.reader import CoregisteredRecording, CoregisteredSession
from mef3io_coregistration.session import CoregSession
from mef3io_coregistration.synthetic import WarpSpec, render_pair

US = 1_000_000
START = 1_700_000_000 * US  # reference-clock start of the recording (uUTC)
DURATION_S = 1200.0  # 20 minutes
FS_A, FS_B = 1024.0, 256.0  # file A (the reference) and file B (the moving file)
LATE_S = 180  # file B starts this much later

# The ground truth: offset(t) = t_moving - t_reference, in seconds, t from the reference start.
TRUTH = WarpSpec("linear", offset0_s=120.0, rate_s_per_s=5.0 / (12 * 3600))


def write_session(path: Path, channel: str, data: np.ndarray, start_uutc: int, fs: float) -> None:
    with mef3io.Writer(str(path), overwrite=True, units="uV") as w:
        w.write(channel, data, int(start_uutc), fs, precision=5)


def main(workdir: Path) -> Path:
    # 1. Two recordings of one signal on two clocks (the synthetic harness; real data is just
    #    two .mefd sessions).
    pair = render_pair(TRUTH, DURATION_S, FS_A, FS_B, START, seed=5)
    file_a, file_b = workdir / "system_a.mefd", workdir / "system_b.mefd"
    write_session(file_a, "C3", pair.ref, pair.ref_start_uutc, pair.fs_ref)
    late = int(LATE_S * FS_B)  # system B was switched on three minutes after system A
    write_session(file_b, "C3", pair.mov[late:], pair.mov_start_uutc + LATE_S * US, pair.fs_moving)
    print(f"wrote {file_a.name} ({FS_A:g} Hz) and {file_b.name} ({FS_B:g} Hz) in {workdir}")

    # 2. Align: Stage 1 (gross, whole recording) -> Stage 2 (fine, one anchor) -> Stage 3 (drift).
    #    background=False loads in this process (simplest for a script).
    with CoregSession(background=False, chunk_s=120.0) as sess:
        sess.open(str(file_a), str(file_b), reference="A")
        sess.set_channels("gross", "C3")  # the channel to align on, both files
        sess.set_channels("fine", "C3")

        gross = sess.run_gross()
        print(f"Stage 1: offset {gross.offset_s:.2f} s, locked={gross.locked} "
              f"({gross.n_agree}/{gross.n_templates} templates agree)")

        roi = START + 600 * US  # 10 minutes in, on the reference clock
        anchor = sess.run_fine(roi)
        print(f"Stage 2: anchor at 10 min: {anchor.offset_s:.4f} s "
              f"(truth {float(TRUTH.offset_s(600.0)):.4f} s), confidence {anchor.confidence:.2f}")

        drift = sess.run_drift()
        print(f"Stage 3: {int(drift.accepted.sum())}/{drift.accepted.size} drift windows accepted")

        # 3. Save the result: the clock map plus provenance, as JSON.
        result_path = sess.result().save(workdir / "pair.coreg.json")
    print(f"saved {result_path.name}")

    # 4. Read file B like a mef3io.Reader, with every time on the reference clock.
    t0, t1 = START + 300 * US, START + 310 * US  # 10 s, reference-clock uUTC
    with CoregisteredSession(result_path, file="B") as b, \
            CoregisteredSession(result_path, file="A") as a:
        xb = b.read("C3", t0, t1, fs=FS_A)  # resampled onto the reference clock at 1024 Hz
        xa = a.read("C3", t0, t1)  # the reference file, exactly as mef3io returns it
        n = min(xa.size, xb.size)
        ok = np.isfinite(xa[:n]) & np.isfinite(xb[:n])
        r = np.corrcoef(xa[:n][ok], xb[:n][ok])[0, 1]
        print(f"file B read on the reference clock matches file A: r = {r:.4f}")

    # 5. Both files as one recording: the preferred file where it has data, the other elsewhere.
    #    File B starts three minutes after file A, so before that the samples come from file A.
    rec = CoregisteredRecording.open(result_path)
    rec.set_channel("C3", a="C3", b="C3", prefer="B")
    sig = rec.get_data("C3", START + 120 * US, START + 240 * US)  # minutes 2 to 4
    print(f"C3 as one channel, minutes 2-4: {sig.n} samples at {sig.fs:g} Hz")
    for src, s0, s1 in sig.runs():
        print(f"  {(s0 - START) / US:6.1f} s to {(s1 - START) / US:6.1f} s from file {'-AB'[src]}")

    # Event times between the clocks: an event marked in file B, at its own clock's time.
    t_b = START + 620 * US
    t_ref = rec.to_reference_time(t_b, "B")
    print(f"an event at START + 620 s on file B's clock is START + {(t_ref - START) / US:.3f} s "
          f"on the reference clock")
    rec.close()
    return result_path


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        main(Path(tmp))
