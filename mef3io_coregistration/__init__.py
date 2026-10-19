"""Coregistration of two MEF 3.0 recordings captured on independent, drifting clocks.

The engine: it aligns the two recordings (:mod:`~mef3io_coregistration.session` and
:mod:`~mef3io_coregistration.align`: a gross offset, sample-accurate anchors, the drift in
between), saves the clock map with its provenance (:mod:`~mef3io_coregistration.result`), and
reads either recording, or both as one, through a saved result on the reference clock
(:mod:`~mef3io_coregistration.reader`). All MEF 3.0 access goes through mef3io.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mef3io-coregistration")  # the VERSION file, via pyproject.toml
except PackageNotFoundError:  # a source checkout that is not installed
    __version__ = "0+unknown"
