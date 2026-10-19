"""The examples run, and the examples README lists every one of them.

Self-contained examples are run end to end in a subprocess (as a user would run them). Examples
that need the user's own recordings are only compiled: there is no data in the repository to run
them on.
"""
from __future__ import annotations

import py_compile
import re
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SCRIPTS = sorted(EXAMPLES.glob("[0-9][0-9]_*.py"))
SELF_CONTAINED = {"01_align_synthetic_pair.py"}  # the rest need a user's result and recordings


def test_example_names_follow_the_convention():
    assert SCRIPTS, "no examples found"
    for p in EXAMPLES.glob("*.py"):
        assert re.fullmatch(r"\d\d_[a-z0-9_]+\.py", p.name), p.name


def test_readme_lists_every_example():
    readme = (EXAMPLES / "README.md").read_text()
    for p in SCRIPTS:
        assert f"`{p.name}`" in readme, f"{p.name} missing from examples/README.md"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_example_compiles(script, tmp_path):
    py_compile.compile(str(script), cfile=str(tmp_path / "x.pyc"), doraise=True)


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(SELF_CONTAINED))
def test_self_contained_example_runs(name):
    proc = subprocess.run(
        [sys.executable, str(EXAMPLES / name)], capture_output=True, text=True, timeout=600
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
