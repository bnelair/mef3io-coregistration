"""Reader defaults: time units, display envelopes, and the unified recording's merge check."""
from __future__ import annotations

# --- Timescale -------------------------------------------------------------
US_PER_S: int = 1_000_000  # MEF3 timestamps are absolute uUTC (microseconds).

# --- Derived signals --------------------------------------------------------
# The min-max envelope is for DISPLAY ONLY: a 20 ms bin of an 8 kHz channel and of a 256 Hz
# channel summarise different physical quantities, so it is never compared across files.
ENVELOPE_FS_HZ: float = 50.0

# --- Unified recording: may two sources be merged into one channel? ----------
# Before a logical channel serves file B's samples where file A has none (or vice versa), the two
# sources are compared over their overlap, band-passed alike (MERGE_BAND_HZ) so that DC and
# bandwidth differences do not decide it. A merge is refused — unless the user overrides —
# when the correlation is below MERGE_MIN_CORR (negative: opposite polarity), or the robust gain
# ratio lies outside MERGE_GAIN_RANGE (the two amplifiers can differ ~250× in scale), or the
# units differ. Per-source scale/offset in the mapping correct scale and sign explicitly.
MERGE_BAND_HZ: tuple[float, float] = (0.5, 20.0)
MERGE_MIN_CORR: float = 0.2
MERGE_GAIN_RANGE: tuple[float, float] = (0.5, 2.0)
MERGE_CHECK_CHUNKS: int = 8
MERGE_CHECK_CHUNK_S: float = 20.0
