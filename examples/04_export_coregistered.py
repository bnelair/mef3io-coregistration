"""Write a new MEF 3.0 session with both recordings on the reference clock.

    python examples/04_export_coregistered.py

Self-contained: it writes a small synthetic pair (the second recorder's clock is 12 s ahead and
drifts by 100 us per s), builds the clock map from that known truth instead of aligning, and exports
a brand-new session: the reference channels unchanged, the moving channels resampled onto the
reference clock at their own rate, with a progress line and a thread count. With your own data the
call is the same, with the saved result and the two recordings:

    export_coregistered("subject01.coreg.json", "subject01_merged.mefd",
                        "system_a.mefd", "system_b.mefd", moving_prefix="B_", n_threads=4)
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import mef3io
import numpy as np

from mef3io_coregistration.export import export_coregistered
from mef3io_coregistration.io.mef_reader import fingerprint
from mef3io_coregistration.result import CoregResult
from mef3io_coregistration.transform import ClockMap

US = 1_000_000
START = 1_700_000_000 * US
OFFSET_S, DRIFT = 12.0, 1e-4


def signal(t_true_s: np.ndarray) -> np.ndarray:
    return 50 * np.sin(2 * np.pi * 1.3 * t_true_s) + 20 * np.sin(2 * np.pi * 4.7 * t_true_s)


def main(workdir: Path) -> Path:
    # Recording A (the reference clock): 256 Hz. Recording B: 200 Hz, on a clock that runs ahead.
    t = np.arange(int(60 * 256.0)) / 256.0
    t0_b = OFFSET_S - 5.0 * (1 + DRIFT)  # B starts 5 s before A, on its own clock
    tb = t0_b + np.arange(int(70 * 200.0)) / 200.0
    with mef3io.Writer(str(workdir / "system_a.mefd"), overwrite=True, units="uV") as w:
        w.write("C3", signal(t), START, 256.0, precision=3)
    with mef3io.Writer(str(workdir / "system_b.mefd"), overwrite=True, units="uV") as w:
        w.write("C3", signal((tb - OFFSET_S) / (1 + DRIFT)), START + int(t0_b * US), 200.0, precision=3)
        w.write("P1", signal((tb - OFFSET_S) / (1 + DRIFT)) * 0.5, START + int(t0_b * US), 200.0, precision=3)

    # The result (normally from CoregSession): reference clock -> B's clock.
    xs = np.linspace(-10, 70, 40)
    clock_map = ClockMap.from_reference_offset((START + xs * US).astype(np.int64), OFFSET_S + DRIFT * xs)
    result = CoregResult(clock_map, str(workdir / "system_a.mefd"), str(workdir / "system_b.mefd"),
                         reference_role="A",
                         file_fingerprints={"A": fingerprint(workdir / "system_a.mefd"),
                                            "B": fingerprint(workdir / "system_b.mefd")})
    result.save(workdir / "pair.coreg.json")

    # Export: B's channels get a prefix because both recordings have a C3.
    report = export_coregistered(
        workdir / "pair.coreg.json", workdir / "merged.mefd", workdir / "system_a.mefd",
        workdir / "system_b.mefd", moving_prefix="B_", n_threads=2, chunk_seconds=20.0,
        progress=lambda done, total: print(f"  {100 * done / total:3.0f} %", end="\r"),
    )
    print()
    print(f"wrote {Path(report.path).name}: " + ", ".join(
        f"{c.name} ({c.source}, {c.fs:g} Hz, {c.n_samples} samples)" for c in report.channels))

    with mef3io.Reader(str(workdir / "merged.mefd")) as r:
        x = r.read("B_C3")
        start = r.info("B_C3")["start_time"]
    ts = (start + np.rint(np.arange(x.size) * US / 200.0) - START) / US
    err = np.nanmax(np.abs(x - signal(ts)))
    print(f"B_C3 on the reference clock: max error against the true signal {err:.3f} uV")
    return workdir / "merged.mefd"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        main(Path(d))
