"""Fixtures for the reader's own tests (self-contained: numpy, mef3io and the reader only)."""
from __future__ import annotations

import numpy as np
import pytest
from mef3io.compat import MefWriter


@pytest.fixture
def write_mef(tmp_path):
    """Factory writing a MEF3 session under ``tmp_path``.

    ``channels`` maps a channel name to a list of ``(data, start_uutc, fs)`` segments;
    multiple segments on one channel (with a time gap between them) create a
    discontinuity. Returns the session path.
    """

    def _write(name: str, channels: dict[str, list[tuple]]):
        path = tmp_path / name
        writer = MefWriter(str(path), overwrite=True)
        for channel, segments in channels.items():
            for data, start_uutc, fs in segments:
                writer.write_data(
                    np.asarray(data, dtype=np.float64),
                    channel,
                    int(start_uutc),
                    float(fs),
                    precision=5,
                )
        del writer  # flush/close the session
        return path

    return _write
