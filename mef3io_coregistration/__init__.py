"""iEEG (and generic) signal coregistration.

Re-align two recordings captured on independent, drifting clocks onto one timeline.
This package is the reader: it opens a saved coregistration result and reads the recordings
through it, on the reference clock.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mef3io-coregistration")  # the VERSION file, via pyproject.toml
except PackageNotFoundError:  # a source checkout that is not installed
    __version__ = "0+unknown"
