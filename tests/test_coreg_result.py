"""Tests for CoregResult persistence.

Full provenance is stored; save → load → evaluate is reproducible and re-saving is byte-stable.
"""
from __future__ import annotations

import numpy as np
import pytest

from mef3io_coregistration.align.fine import Anchor
from mef3io_coregistration.config import US_PER_S
from mef3io_coregistration.session import CoregResult
from mef3io_coregistration.transform import ClockMap

_EPOCH = 1_700_000_000 * US_PER_S


def _result():
    x = (_EPOCH + np.arange(6) * 60 * US_PER_S).astype(np.int64)
    cm = ClockMap.from_reference_offset(x, 120.0 + np.linspace(0, 0.01, 6))
    return CoregResult(
        clock_map=cm,
        reference_file="/data/reference.mefd",
        moving_file="/data/moving.mefd",
        stage_channels={"gross": {"reference": "a-c", "moving": "a-c"}},
        anchors=[Anchor(int(x[2]), int(x[2]) + 120_000_000, 120.005, 0.98)],
        drift={"window_s": 20.0, "grid_s": 30.0, "confidence": [0.9, 0.95], "accepted": [True, True]},
    )


def test_save_load_roundtrip_preserves_provenance(tmp_path):
    res = _result()
    path = res.save(tmp_path / "res.json")
    loaded = CoregResult.load(path)

    assert loaded.reference_file == res.reference_file
    assert loaded.moving_file == res.moving_file
    assert loaded.stage_channels == res.stage_channels
    assert loaded.created_utc == res.created_utc
    assert loaded.tool_version == res.tool_version
    assert len(loaded.anchors) == 1
    assert loaded.anchors[0].confidence == pytest.approx(0.98)
    assert loaded.drift["confidence"] == [0.9, 0.95]


def test_reload_evaluates_identical_map(tmp_path):
    """R7.5: the reloaded clock map produces identical outputs."""
    res = _result()
    loaded = CoregResult.load(res.save(tmp_path / "res.json"))
    q = np.linspace(*res.clock_map.domain, 500).astype(np.int64)
    np.testing.assert_array_equal(
        np.asarray(res.clock_map.forward(q)), np.asarray(loaded.clock_map.forward(q))
    )


def test_resave_is_byte_stable(tmp_path):
    res = _result()
    first = res.save(tmp_path / "a.json")
    second = CoregResult.load(first).save(tmp_path / "b.json")
    assert first.read_text() == second.read_text()


def test_rejects_foreign_file(tmp_path):
    p = tmp_path / "foreign.json"
    p.write_text('{"format": "nope"}')
    with pytest.raises(ValueError, match="not a coreg result"):
        CoregResult.load(p)
