"""Cached synthetic MEF3 fixtures.

Fixtures are written to a designated folder (``config.SYNTHETIC_DIR`` by default) and
**generated only if absent** — a manifest records the exact parameters and a harness
version so a change in generation logic triggers regeneration, but ordinary test runs
reuse the cached data.

Each fixture is a pair of MEF3 sessions (``reference.mefd`` / ``moving.mefd``) plus a
``manifest.json`` holding the ground-truth WarpSpec and metadata.
"""
from __future__ import annotations

import json
import logging
import shutil
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from mef3io.compat import MefReader, MefWriter

from mef3io_coregistration.config import SYNTHETIC_DIR, US_PER_S

from .generate import render_pair
from .warp import WarpSpec

log = logging.getLogger(__name__)

# Bump when the generation logic changes so cached fixtures are rebuilt.
HARNESS_VERSION = 2  # 2: oscillations as narrowband noise (non-periodic source)

# A fixed, arbitrary absolute epoch for reproducibility (2023-11-14T22:13:20Z).
DEFAULT_START_UUTC = 1_700_000_000 * US_PER_S

CHANNEL = "ch1"
_MEF_PRECISION = 5

# Profiles are intentionally short with *exaggerated* drift so Stage-2/3 behaviour is
# exercised without generating a full 24 h recording.
PROFILES: dict[str, dict] = {
    # 2-minute absolute offset, no drift — exercises gross alignment.
    "short_constant": dict(
        spec=WarpSpec("constant", offset0_s=120.0),
        duration_s=30.0,
        fs_ref=2048.0,
        fs_moving=512.0,
        seed=1,
    ),
    # Constant offset + slow linear drift.
    "short_linear": dict(
        spec=WarpSpec("linear", offset0_s=120.0, rate_s_per_s=5.0 / (12 * 3600)),
        duration_s=60.0,
        fs_ref=2048.0,
        fs_moving=512.0,
        seed=2,
    ),
    # A full +5 s -> -5 s -> +0 s swing across the recording (period == duration).
    "drift_swing": dict(
        spec=WarpSpec("swing", offset0_s=120.0, swing_amp_s=5.0, swing_period_s=600.0),
        duration_s=600.0,
        fs_ref=2048.0,
        fs_moving=512.0,
        seed=3,
    ),
}

# Signal read back from a MEF3 session, with reconstructed absolute timestamps.
Signal = namedtuple("Signal", ["data", "ts", "fs", "start_uutc"])


@dataclass
class FixturePair:
    """Handle to a cached fixture pair + its ground-truth clock map."""

    name: str
    spec: WarpSpec
    fs_ref: float
    fs_moving: float
    duration_s: float
    start_uutc: int
    reference_path: Path
    moving_path: Path
    channel: str
    generated: bool  # True if this call generated it, False if reused from cache

    # --- ground truth ---
    def offset_s(self, t_rel_s: np.ndarray | float) -> np.ndarray:
        return self.spec.offset_s(t_rel_s)

    def forward_uutc(self, t_ref_uutc: np.ndarray | float) -> np.ndarray:
        """Ground-truth clock map: reference uUTC -> moving uUTC."""
        t = np.asarray(t_ref_uutc, dtype=float)
        t_rel = (t - self.start_uutc) / US_PER_S
        return t + self.spec.offset_s(t_rel) * US_PER_S

    # --- data access ---
    def load_reference(self) -> Signal:
        return _read_mef(self.reference_path, self.channel)

    def load_moving(self) -> Signal:
        return _read_mef(self.moving_path, self.channel)


def build_or_load(
    name: str,
    data_dir: Path | str | None = None,
    force: bool = False,
) -> FixturePair:
    """Return the fixture ``name``, generating it only if absent (or ``force``)."""
    if name not in PROFILES:
        raise KeyError(f"unknown profile {name!r}; choices: {sorted(PROFILES)}")
    profile = PROFILES[name]
    root = Path(data_dir) if data_dir is not None else SYNTHETIC_DIR
    fixture_dir = root / name
    manifest_path = fixture_dir / "manifest.json"

    if not force and _manifest_is_current(manifest_path, name, profile):
        log.debug("reusing cached fixture %s", name)
        return _pair_from_profile(name, profile, fixture_dir, generated=False)

    log.info("generating fixture %s at %s", name, fixture_dir)
    _generate(name, profile, fixture_dir)
    return _pair_from_profile(name, profile, fixture_dir, generated=True)


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #
def _pair_from_profile(name: str, profile: dict, fixture_dir: Path, generated: bool) -> FixturePair:
    return FixturePair(
        name=name,
        spec=profile["spec"],
        fs_ref=profile["fs_ref"],
        fs_moving=profile["fs_moving"],
        duration_s=profile["duration_s"],
        start_uutc=DEFAULT_START_UUTC,
        reference_path=fixture_dir / "reference.mefd",
        moving_path=fixture_dir / "moving.mefd",
        channel=CHANNEL,
        generated=generated,
    )


def _manifest_dict(name: str, profile: dict) -> dict:
    return {
        "harness_version": HARNESS_VERSION,
        "name": name,
        "spec": profile["spec"].to_dict(),
        "duration_s": profile["duration_s"],
        "fs_ref": profile["fs_ref"],
        "fs_moving": profile["fs_moving"],
        "seed": profile["seed"],
        "start_uutc": DEFAULT_START_UUTC,
        "channel": CHANNEL,
    }


def _manifest_is_current(manifest_path: Path, name: str, profile: dict) -> bool:
    if not manifest_path.exists():
        return False
    try:
        stored = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    expected = _manifest_dict(name, profile)
    if stored != expected:
        return False
    # Manifest matches; make sure the data actually exists.
    ref = manifest_path.parent / "reference.mefd"
    mov = manifest_path.parent / "moving.mefd"
    return ref.exists() and mov.exists()


def _generate(name: str, profile: dict, fixture_dir: Path) -> None:
    if fixture_dir.exists():
        shutil.rmtree(fixture_dir)
    fixture_dir.mkdir(parents=True, exist_ok=True)

    rendered = render_pair(
        spec=profile["spec"],
        duration_s=profile["duration_s"],
        fs_ref=profile["fs_ref"],
        fs_moving=profile["fs_moving"],
        start_uutc=DEFAULT_START_UUTC,
        seed=profile["seed"],
    )
    _write_mef(fixture_dir / "reference.mefd", rendered.ref, rendered.fs_ref, rendered.ref_start_uutc)
    _write_mef(fixture_dir / "moving.mefd", rendered.mov, rendered.fs_moving, rendered.mov_start_uutc)
    (fixture_dir / "manifest.json").write_text(json.dumps(_manifest_dict(name, profile), indent=2))


def _write_mef(path: Path, data: np.ndarray, fs: float, start_uutc: int) -> None:
    if path.exists():
        shutil.rmtree(path)
    writer = MefWriter(str(path), overwrite=True)
    writer.write_data(np.asarray(data, dtype=np.float64), CHANNEL, int(start_uutc), float(fs),
                      precision=_MEF_PRECISION)
    del writer  # flush/close the session


def _read_mef(path: Path, channel: str) -> Signal:
    reader = MefReader(str(path))
    fs = float(reader.get_property("fsamp", channel))
    start = int(reader.get_property("start_time", channel))
    data = np.asarray(reader.get_data(channel), dtype=np.float64)
    reader.close()
    ts = start + np.arange(data.shape[0]) / fs * US_PER_S
    return Signal(data=data, ts=ts, fs=fs, start_uutc=start)
