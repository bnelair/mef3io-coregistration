"""Shared pytest fixtures.

``shared_data_dir`` points at the persistent fixture folder (``config.SYNTHETIC_DIR``; override
with ``SIGCOREG_DATA_DIR``): tests that use it reuse cached synthetic MEF3 recordings across runs,
regenerated only when the harness version or parameters change.
"""
from __future__ import annotations

import numpy as np
import pytest
from mef3io.compat import MefWriter

from mef3io_coregistration.config import SYNTHETIC_DIR


@pytest.fixture(scope="session")
def shared_data_dir():
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    return SYNTHETIC_DIR


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
