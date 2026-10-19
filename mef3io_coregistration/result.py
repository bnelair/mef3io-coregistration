"""Serializable coregistration result.

Part of the reader core: loading a result needs only this module and the clock map — never the
alignment pipeline or the console.

The saved clock map plus full provenance: the two files and which is the reference (A/B
roles), the channels and filters used per file, anchors, the drift estimate, a QC report,
a creation timestamp, and the tool version — as recorded by the tool that made it. Persisted as
sorted-key JSON with integer clock-map anchors, so save → load → evaluate is reproducible and
re-saving is byte-stable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from . import __version__
from .transform.clock_map import ClockMap


@dataclass(frozen=True)
class Anchor:
    """A precise ``(t_ref, t_moving)`` correspondence with a confidence in [0, 1]."""

    t_ref_uutc: int
    t_moving_uutc: int
    offset_s: float
    confidence: float
    ncc: float = 0.0
    window_s: float = 0.0


_FORMAT = "signal_coregistration.coreg_result"
_FORMAT_VERSION = 2


@dataclass
class CoregResult:
    """A completed coregistration: the clock map plus provenance for reproducibility."""

    clock_map: ClockMap
    reference_file: str
    moving_file: str
    stage_channels: dict = field(default_factory=dict)  # {stage: {"reference": lbl, "moving": lbl}}
    anchors: list[Anchor] = field(default_factory=list)
    drift: dict | None = None  # {window_s, grid_s, t_ref_uutc[], offset_s[], confidence[], accepted[]}
    created_utc: str = ""
    tool_version: str = __version__
    reference_role: str = "A"  # which of file A / file B is the reference
    stage_filters: dict = field(default_factory=dict)  # {stage: {"reference": [..], "moving": [..]}}
    qc: dict | None = None  # QC report: verification, residuals, extrapolation
    file_fingerprints: dict = field(default_factory=dict)  # {"A": fingerprint, "B": fingerprint}

    def __post_init__(self) -> None:
        if not self.created_utc:
            self.created_utc = datetime.now(UTC).isoformat()
        if self.reference_role not in ("A", "B"):
            raise ValueError('reference_role must be "A" or "B"')

    @property
    def file_a(self) -> str:
        return self.reference_file if self.reference_role == "A" else self.moving_file

    @property
    def file_b(self) -> str:
        return self.moving_file if self.reference_role == "A" else self.reference_file

    # -- persistence ------------------------------------------
    def to_dict(self) -> dict:
        return {
            "format": _FORMAT,
            "format_version": _FORMAT_VERSION,
            "tool_version": self.tool_version,
            "created_utc": self.created_utc,
            "reference_file": self.reference_file,
            "moving_file": self.moving_file,
            "reference_role": self.reference_role,
            "file_a": self.file_a,
            "file_b": self.file_b,
            "stage_channels": self.stage_channels,
            "stage_filters": self.stage_filters,
            "qc": self.qc,
            "file_fingerprints": self.file_fingerprints,
            "anchors": [
                {
                    "t_ref_uutc": int(a.t_ref_uutc),
                    "t_moving_uutc": int(a.t_moving_uutc),
                    "offset_s": a.offset_s,
                    "confidence": a.confidence,
                }
                for a in self.anchors
            ],
            "drift": self.drift,
            "clock_map": self.clock_map.to_dict(),
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return path

    @classmethod
    def from_dict(cls, d: dict) -> CoregResult:
        if d.get("format") != _FORMAT:
            raise ValueError(f"not a coreg result file (format={d.get('format')!r})")
        anchors = [
            Anchor(
                t_ref_uutc=int(a["t_ref_uutc"]),
                t_moving_uutc=int(a["t_moving_uutc"]),
                offset_s=a["offset_s"],
                confidence=a["confidence"],
            )
            for a in d.get("anchors", [])
        ]
        return cls(
            clock_map=ClockMap.from_dict(d["clock_map"]),
            reference_file=d["reference_file"],
            moving_file=d["moving_file"],
            stage_channels=d.get("stage_channels", {}),
            anchors=anchors,
            drift=d.get("drift"),
            created_utc=d.get("created_utc", ""),
            tool_version=d.get("tool_version", ""),
            reference_role=d.get("reference_role", "A"),  # v1 files: A was the reference
            stage_filters=d.get("stage_filters", {}),
            qc=d.get("qc"),
            file_fingerprints=d.get("file_fingerprints", {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> CoregResult:
        return cls.from_dict(json.loads(Path(path).read_text()))
