"""The examples run, and the examples README lists every one of them.

Self-contained examples are run end to end in a subprocess (as a user would run them). Examples
that need the user's own recordings are only compiled: there is no data in the repository to run
them on.
"""
from __future__ import annotations

import os
import py_compile
import re
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SCRIPTS = sorted(EXAMPLES.glob("[0-9][0-9]_*.py"))
SELF_CONTAINED = {"01_align_synthetic_pair.py"}  # the rest need a user's result and recordings
# Examples print non-ASCII (arrows, dashes); a Windows console pipe is not UTF-8 by default.
_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def test_example_names_follow_the_convention():
    assert SCRIPTS, "no examples found"
    for p in EXAMPLES.glob("*.py"):
        assert re.fullmatch(r"\d\d_[a-z0-9_]+\.py", p.name), p.name


def test_readme_lists_every_example():
    readme = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    for p in SCRIPTS:
        assert f"`{p.name}`" in readme, f"{p.name} missing from examples/README.md"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_example_compiles(script, tmp_path):
    py_compile.compile(str(script), cfile=str(tmp_path / "x.pyc"), doraise=True)


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(SELF_CONTAINED))
def test_self_contained_example_runs(name):
    proc = subprocess.run(
        [sys.executable, str(EXAMPLES / name)], capture_output=True, text=True,
        encoding="utf-8", env=_ENV, timeout=600,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    assert "locked=True" in out
    assert "saved pair.coreg.json" in out
    r = float(re.search(r"r = ([0-9.]+)", out).group(1))
    assert r > 0.99, out
    assert "from file A" in out and "from file B" in out, out


@pytest.mark.slow
def test_review_report_on_the_example_result(tmp_path):
    """``03_review_report.py`` / ``sigcoreg-review`` on the result example 01 produces: the
    figures and the index page are written (needs the ``report`` extra)."""
    pytest.importorskip("matplotlib")
    import importlib.util

    spec = importlib.util.spec_from_file_location("ex01", EXAMPLES / "01_align_synthetic_pair.py")
    ex01 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex01)
    result = ex01.main(tmp_path)

    from mef3io_coregistration.review import main as review_main

    review_main([str(result), "--out", str(tmp_path / "review")])
    out = tmp_path / "review"
    assert (out / "index.html").is_file()
    assert (out / "2_clock_map.png").is_file()
    assert (out / "summary.json").is_file()


def _synthetic_result_with_two_channels(workdir: Path) -> Path:
    """A saved result over a synthetic pair whose files both carry ``ch1`` and ``ch2`` (what
    ``02_read_coregistered_recording.py`` expects); file B starts three minutes after file A."""
    import mef3io
    import numpy as np

    from mef3io_coregistration.session import CoregSession
    from mef3io_coregistration.synthetic import WarpSpec, render_pair

    us = 1_000_000
    start = 1_700_000_000 * us
    pair = render_pair(WarpSpec("linear", offset0_s=120.0, rate_s_per_s=5.0 / 43200), 1200.0,
                       1024.0, 256.0, start, seed=7)
    late = int(180 * pair.fs_moving)
    files = {
        "A": (workdir / "recording_a.mefd", pair.ref, pair.ref_start_uutc, pair.fs_ref),
        "B": (workdir / "recording_b.mefd", pair.mov[late:], pair.mov_start_uutc + 180 * us,
              pair.fs_moving),
    }
    for path, x, t0, fs in files.values():
        with mef3io.Writer(str(path), overwrite=True, units="uV") as w:
            w.write("ch1", x, int(t0), fs, precision=5)
            w.write("ch2", 0.5 * np.roll(x, int(0.05 * fs)), int(t0), fs, precision=5)
    with CoregSession(background=False, chunk_s=120.0) as sess:
        sess.open(str(files["A"][0]), str(files["B"][0]), reference="A")
        sess.set_channels("gross", "ch1")
        sess.set_channels("fine", "ch1")
        assert sess.run_gross().locked
        sess.run_fine(start + 600 * us)
        sess.run_drift()
        return sess.result().save(workdir / "pair.coreg.json")


@pytest.mark.slow
def test_read_example_runs_on_a_synthetic_result(tmp_path):
    """``02_read_coregistered_recording.py`` end to end on a synthetic result with its channels."""
    result = _synthetic_result_with_two_channels(tmp_path)
    proc = subprocess.run(
        [sys.executable, str(EXAMPLES / "02_read_coregistered_recording.py"), str(result)],
        capture_output=True, text=True, encoding="utf-8", env=_ENV, timeout=600, cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "sample-aligned" in proc.stdout and "saved mapping" in proc.stdout, proc.stdout
    assert (tmp_path / "pair.coreg_recording.json").is_file()
