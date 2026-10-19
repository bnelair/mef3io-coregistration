"""Defaults: time units, the data folder, display envelopes, the unified recording's merge check,
and the alignment's conditioning bands, lock criteria and stage parameters. Deliberately
conservative and overridable; nothing in the algorithm hard-codes a domain assumption.
"""
from __future__ import annotations

import os
from pathlib import Path

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
# ratio lies outside MERGE_GAIN_RANGE (two amplifiers can differ by orders of magnitude in scale), or the
# units differ. Per-source scale/offset in the mapping correct scale and sign explicitly.
MERGE_BAND_HZ: tuple[float, float] = (0.5, 20.0)
MERGE_MIN_CORR: float = 0.2
MERGE_GAIN_RANGE: tuple[float, float] = (0.5, 2.0)
MERGE_CHECK_CHUNKS: int = 8
MERGE_CHECK_CHUNK_S: float = 20.0


# --- Data / fixtures -------------------------------------------------------
def _default_data_dir() -> Path:
    """Folder for generated/cached data (the synthetic fixtures).

    ``SIGCOREG_DATA_DIR`` overrides. In a source checkout (``pyproject.toml`` or ``.git`` next to
    the package) it is ``data/`` in the checkout; an installed package never writes next to
    itself, and uses the user cache folder instead: ``%LOCALAPPDATA%\\mef3io-coregistration`` on
    Windows, ``$XDG_CACHE_HOME/mef3io-coregistration`` (default ``~/.cache/...``) elsewhere.
    """
    env = os.environ.get("SIGCOREG_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    root = Path(__file__).resolve().parent.parent
    if (root / "pyproject.toml").is_file() or (root / ".git").exists():
        return (root / "data").resolve()
    return _user_cache_dir() / "mef3io-coregistration"


def _user_cache_dir() -> Path:
    """The platform's per-user cache folder."""
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA")
        return Path(local) if local else Path.home() / "AppData" / "Local"
    xdg = os.environ.get("XDG_CACHE_HOME")
    return Path(xdg).expanduser() if xdg else Path.home() / ".cache"


DATA_DIR: Path = _default_data_dir()
SYNTHETIC_DIR: Path = DATA_DIR / "synthetic"


# --- Conditioning (alignment path only; display is never conditioned) ------
HIGHPASS_HZ: float = 0.5  # removes the per-amplifier DC/slow component
CLIP_MAD: float = 8.0     # clip at N robust sigmas: bounds stim/saturation excursions

# Detection and estimation want DIFFERENT bands, and conflating them is why Stage 1 failed.
#
# DETECTION (Stage 1) searches the whole recording and must produce ONE unambiguous peak. Two
# amplifiers with different references share their LOW-frequency physiology; above ~10 Hz the
# content is reference noise, amplifier noise and stim, which is uncorrelated between the two
# systems and only dilutes the match. On a clinical test pair, narrowing to 0.5–8 Hz moved
# template agreement from 0/4 to 4/4.
#
# ESTIMATION (Stages 2–3) already knows roughly where the peak is — it searches a narrow seeded
# range — so ambiguity is not the risk; timing precision is. It can afford a wider band, but
# NOT an unlimited one: above ~20 Hz the two amplifiers share almost nothing, so extra
# bandwidth adds uncorrelated noise rather than timing information. On a clinical test pair the
# anchor spread across the recording was smallest for 0.5-10 to 0.5-40 Hz, about three times
# larger unfiltered, and grew again above ~40 Hz; 0.5-20 Hz is the default.
LOWPASS_HZ: float | None = 8.0        # detection band (Stage 1)
FINE_LOWPASS_HZ: float | None = 20.0  # estimation band (Stages 2-3)

# --- Lock / confidence -----------------------------------------------------
# `confidence` is a PEAK-DISTINCTIVENESS ratio mapped to [0, 1] — 1 - (runner_up / best) —
# not the raw NCC. Raw NCC falls as the window grows (more independent content between two
# amplifiers) even as the lag estimate improves, so thresholding it rejects the *good*
# windows. Distinctiveness is what "did I lock on?" actually means, and it is comparable
# across window lengths. Calibrated on clinical test pairs.
MIN_LOCK_CONFIDENCE: float = 0.20  # below this a stage reports "no lock" and refuses to advance
PEAK_GUARD_S: float = 1.0          # runner-up must lie at least this far from the best peak

# --- Stage 1 gross (stacked multi-template matched filter) -----------------
GROSS_TEMPLATE_S: float = 600.0  # length of each matched-filter template
GROSS_N_TEMPLATES: int = 4       # independent templates, drawn from different hours
GROSS_CONSENSUS_S: float = 5.0   # templates corroborate if they land within this of the peak
# The lock criterion is TEMPLATE AGREEMENT, not peak height. Measured over true and
# deliberately false pairs, the peak heights of true and false locks overlap heavily, so no SNR threshold can separate them. Agreement separates
# them perfectly (true: 4/4; false: 0/4).
#
# GROSS_MIN_SNR is therefore NOT a gate. It only decides whether a successful lock is annotated
# "locked, but weakly — confirm by eye", which is advice, not a veto.
GROSS_MIN_SNR: float = 5.0

# --- Offset verification -------------------------------------
# "Is the offset I was HANDED right at all?" — a different question from "where is the best
# offset?", and the one every later stage silently assumed the answer to. Stage 2/3
# distinctiveness cannot answer it: on a clinical test pair it was HIGHER at wrong
# offsets than at the true one, because
# a wrong offset throws up one sharp random blip while the true offset has a broad genuine ridge.
#
# Verification asks instead whether independent templates place their BEST match at the candidate
# offset — the lock criterion, seeded. It has no threshold, so there is nothing here to tune;
# GROSS_CONSENSUS_S is the only tolerance, and it exists to absorb real drift.
#
# A leave-one-out residual above this means the fitted map does not predict its own windows;
# on a clinical test pair a correct map gave tens of ms and an hour-wrong map several hundred.
MAX_LOO_RESIDUAL_MS: float = 200.0

# --- Stage 2 fine ----------------------------------------------------------
# 120 s minimum: a 40 s window of 256 Hz iEEG is not distinctive enough — its top correlation
# peaks sit within a few percent of each other, and the estimate can land seconds off with
# HIGH raw-NCC confidence.
FINE_WINDOW_MIN_S: float = 120.0
FINE_WINDOW_MAX_S: float = 600.0
FINE_WINDOW_DEFAULT_S: float = 300.0
# Local seed for Stage 2 (see align.fine.local_seed): the Stage 1 lock is one offset for the whole
# recording and can be off by the full drift range at a given ROI. Search twice the consensus
# tolerance — which is the drift the lock itself is designed to absorb — over ±10 min.
LOCAL_SEED_HALF_S: float = 600.0
LOCAL_SEED_LAG_S: float = 2 * GROSS_CONSENSUS_S

# --- Stage 3 drift ---------------------------------------------------------
DRIFT_WINDOW_S: float = 20.0  # sliding-window length for local time-delay estimation
DRIFT_GRID_S: float = 30.0    # spacing between window centers
DRIFT_MAX_GAP_FRACTION: float = 0.2  # reject a window with more than this much gap
# Physical bound on the drift rate the walk may extrapolate (s/s). Real clock pairs drift by
# ~5 s over 12–24 h (≤ ~4e-4 s/s at the steepest point of a ±5 s daily swing); 1e-3 (3.6 s/h)
# leaves a wide margin while stopping a spurious slope from being carried across a dead span.
DRIFT_MAX_SLOPE: float = 1e-3
