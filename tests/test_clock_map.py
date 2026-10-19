"""Tests for the clock map.

canonical (x_ref, y_moving) + method + policy; exact forward/inverse;
flagged hold-end-slope extrapolation; bit-stable save/load. Property tests
(hypothesis) assert monotonicity and invertibility over random monotone maps.
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mef3io_coregistration.config import US_PER_S
from mef3io_coregistration.transform import ClockMap

_EPOCH = 1_700_000_000 * US_PER_S


def _swing_map(interp_method="linear", n=21, span_s=600.0, amp_s=5.0, base_s=120.0):
    x = (_EPOCH + np.linspace(0, span_s, n) * US_PER_S).astype(np.int64)
    off = base_s + amp_s * np.sin(2 * np.pi * np.linspace(0, span_s, n) / span_s)
    return ClockMap.from_reference_offset(x, off, interp_method=interp_method)


# --------------------------------------------------------------------------- #
# representation, forward/inverse
# --------------------------------------------------------------------------- #
def test_anchors_are_fixed_points():
    cm = _swing_map()
    np.testing.assert_array_equal(
        np.rint(np.asarray(cm.forward(cm.x_ref))).astype(np.int64), cm.y_moving
    )


@pytest.mark.parametrize("method", ["linear", "pchip"])
def test_forward_inverse_roundtrip(method):
    """inverse(forward(t)) == t within numerical tolerance across the domain."""
    cm = _swing_map(interp_method=method)
    q = np.linspace(cm.x_ref[0], cm.x_ref[-1], 1000).astype(np.int64)
    back = np.asarray(cm.inverse(cm.forward(q)))
    assert np.max(np.abs(back - q)) <= 2.0  # µs


def test_scalar_and_offset():
    cm = _swing_map()
    mid = int((cm.x_ref[0] + cm.x_ref[-1]) // 2)
    assert isinstance(cm.forward(mid), float)
    assert cm.offset_s(mid) == pytest.approx(120.0, abs=1e-3)  # swing midpoint


# --------------------------------------------------------------------------- #
# extrapolation policy, flagged
# --------------------------------------------------------------------------- #
def test_extrapolation_holds_end_slope_and_is_flagged():
    cm = _swing_map()
    x0, x1 = cm.domain
    # end slopes from the terminal anchor segments
    s1 = (cm.y_moving[-1] - cm.y_moving[-2]) / (cm.x_ref[-1] - cm.x_ref[-2])
    beyond = x1 + 5 * US_PER_S
    expected = cm.y_moving[-1] + s1 * (beyond - x1)
    assert cm.forward(beyond) == pytest.approx(expected, abs=2.0)

    flags = cm.is_extrapolated(np.array([x0 - 1, x0, x1, x1 + 1]))
    np.testing.assert_array_equal(flags, [True, False, False, True])
    assert cm.is_extrapolated(x0 - 1) is True
    assert cm.is_extrapolated((x0 + x1) // 2) is False


def test_forward_monotone_including_extrapolation():
    cm = _swing_map()
    x0, x1 = cm.domain
    q = np.linspace(x0 - 10 * US_PER_S, x1 + 10 * US_PER_S, 500).astype(np.int64)
    assert np.all(np.diff(np.asarray(cm.forward(q))) > 0)


# --------------------------------------------------------------------------- #
# construction validation
# --------------------------------------------------------------------------- #
def test_rejects_non_monotone_moving():
    x = np.array([0, 1, 2, 3], dtype=np.int64) + _EPOCH
    y = np.array([0, 5, 4, 9], dtype=np.int64) + _EPOCH  # dips at index 2
    with pytest.raises(ValueError, match="monotone"):
        ClockMap(x, y)


def test_rejects_too_few_anchors():
    with pytest.raises(ValueError, match="two anchors"):
        ClockMap(np.array([_EPOCH]), np.array([_EPOCH + 1]))


def test_rejects_unknown_method():
    with pytest.raises(ValueError):
        _swing_map(interp_method="quadratic")


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
def test_save_load_is_bit_stable(tmp_path):
    cm = _swing_map(interp_method="pchip")
    cm = ClockMap(cm.x_ref, cm.y_moving, interp_method="pchip", metadata={"reference": "A"})
    path = cm.save(tmp_path / "map.json")
    reloaded = ClockMap.load(path)

    q = np.linspace(cm.x_ref[0], cm.x_ref[-1], 500).astype(np.int64)
    np.testing.assert_array_equal(np.asarray(cm.forward(q)), np.asarray(reloaded.forward(q)))
    np.testing.assert_array_equal(cm.x_ref, reloaded.x_ref)
    np.testing.assert_array_equal(cm.y_moving, reloaded.y_moving)
    assert reloaded.metadata == {"reference": "A"}
    assert reloaded.interp_method == "pchip"
    # Re-saving the reloaded map yields byte-identical bytes.
    again = reloaded.save(tmp_path / "map2.json")
    assert path.read_text() == again.read_text()


def test_rejects_foreign_file(tmp_path):
    p = tmp_path / "foreign.json"
    p.write_text('{"format": "something_else"}')
    with pytest.raises(ValueError, match="not a clock map"):
        ClockMap.load(p)


# --------------------------------------------------------------------------- #
# property tests (hypothesis) — monotonicity & invertibility
# --------------------------------------------------------------------------- #
@st.composite
def _monotone_maps(draw):
    # Physical clock maps have slope dy/dx = 1 + drift_rate ≈ 1 (|drift_rate| ≪ 1). We span a
    # generous ±10 % so monotonicity/invertibility are stressed without exploring degenerate
    # near-flat maps whose inverse would amplify the unavoidable µs float quantization.
    n = draw(st.integers(min_value=2, max_value=25))
    x_gaps = draw(st.lists(st.integers(100_000, 2_000_000), min_size=n, max_size=n))
    factors = draw(st.lists(st.floats(0.9, 1.1), min_size=n, max_size=n))
    y_gaps = [max(1, int(g * f)) for g, f in zip(x_gaps, factors, strict=False)]
    x = _EPOCH + np.cumsum(np.asarray(x_gaps, dtype=np.int64))
    y = _EPOCH + np.cumsum(np.asarray(y_gaps, dtype=np.int64))
    return ClockMap(x, y)


@settings(max_examples=200, deadline=None)
@given(_monotone_maps())
def test_property_roundtrip_and_monotone(cm):
    x0, x1 = cm.domain
    q = np.unique(np.linspace(x0, x1, 50).astype(np.int64))
    fwd = np.asarray(cm.forward(q))
    assert np.all(np.diff(fwd) > 0)  # strictly monotone map
    back = np.asarray(cm.inverse(fwd))
    assert np.max(np.abs(back - q)) <= 4.0  # µs, invertible
