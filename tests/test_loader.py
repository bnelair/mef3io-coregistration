"""Tests for the background loader.

Streaming products match a full-length pass, progress is monotonic, loads are
cancellable, and distinct specs run concurrently. The multiprocessing paths
are marked ``slow``.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from mef3io_coregistration.config import US_PER_S
from mef3io_coregistration.io import MefChannelReader
from mef3io_coregistration.io.derived import decimate_bandlimited, minmax_envelope
from mef3io_coregistration.montage import bipolar
from mef3io_coregistration.session.loader import (
    BackgroundLoad,
    Cancelled,
    LoadSpec,
    load_alignment_signal,
)

_EPOCH = 1_700_000_000 * US_PER_S


def _bandlimited(n, fs, cutoff, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    spec = np.fft.rfft(x)
    spec[np.fft.rfftfreq(n, 1 / fs) > cutoff] = 0.0
    return np.fft.irfft(spec, n) * 50.0


def _two_channel_file(write_mef, fs=2048.0, seconds=400.0):
    n = int(seconds * fs)
    return write_mef(
        "load.mefd",
        {"a": [(_bandlimited(n, fs, 200.0, 1), _EPOCH, fs)],
         "c": [(_bandlimited(n, fs, 200.0, 2), _EPOCH, fs)]},
    )


class _CancelAfter:
    def __init__(self, n):
        self.n, self.calls = n, 0

    def is_set(self):
        self.calls += 1
        return self.calls > self.n


def test_streaming_products_match_full_pass(write_mef):
    """Chunked decimation + envelope reproduce a full-length pass."""
    path = _two_channel_file(write_mef)
    spec = LoadSpec(str(path), "a", "c", target_fs=512.0, chunk_s=30.0)
    product = load_alignment_signal(spec)

    with MefChannelReader(path) as reader:
        bip = bipolar(reader.read("a"), reader.read("c"))
    full_dec = decimate_bandlimited(bip.data, 2048.0, 512.0).data
    full_env = minmax_envelope(bip.data, 2048.0)

    assert product.decimated.fs == pytest.approx(512.0)
    assert product.decimated.start_uutc == _EPOCH
    m = min(product.decimated.data.shape[0], full_dec.shape[0])
    core = slice(int(0.02 * m), int(0.98 * m))
    assert np.max(np.abs(product.decimated.data[:m][core] - full_dec[:m][core])) < 1e-6
    me = min(product.envelope.mins.shape[0], full_env.mins.shape[0])
    np.testing.assert_allclose(product.envelope.mins[:me], full_env.mins[:me])
    np.testing.assert_allclose(product.envelope.maxs[:me], full_env.maxs[:me])


def test_progress_is_monotonic_and_completes(write_mef):
    path = _two_channel_file(write_mef, seconds=300.0)
    seen: list[float] = []
    load_alignment_signal(
        LoadSpec(str(path), "a", "c", target_fs=512.0, chunk_s=30.0),
        progress=lambda f, m: seen.append(f),
    )
    assert seen == sorted(seen)
    assert seen[-1] == pytest.approx(1.0)
    assert len(seen) >= 10  # ~300 s / 30 s chunks (chunk rounded to a bin/decimation multiple)


def test_single_channel_without_bipolar(write_mef):
    path = _two_channel_file(write_mef, seconds=120.0)
    product = load_alignment_signal(LoadSpec(str(path), "a", target_fs=512.0, chunk_s=30.0))
    with MefChannelReader(path) as reader:
        full = decimate_bandlimited(reader.read("a").data, 2048.0, 512.0).data
    m = min(product.decimated.data.shape[0], full.shape[0])
    core = slice(int(0.05 * m), int(0.95 * m))
    assert np.max(np.abs(product.decimated.data[:m][core] - full[:m][core])) < 1e-6


def test_cancel_token_aborts_load(write_mef):
    path = _two_channel_file(write_mef, seconds=300.0)
    with pytest.raises(Cancelled):
        load_alignment_signal(
            LoadSpec(str(path), "a", "c", target_fs=512.0, chunk_s=30.0), cancel=_CancelAfter(2)
        )


# --------------------------------------------------------------------------- #
# multiprocessing paths
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_background_load_completes_and_cancels(write_mef):
    path = _two_channel_file(write_mef, seconds=300.0)
    spec = LoadSpec(str(path), "a", "c", target_fs=512.0, chunk_s=30.0)

    done = BackgroundLoad(spec).start()
    assert done.wait(timeout=120) == "done"
    assert done.product is not None and done.product.decimated.data.shape[0] > 0

    cancelled = BackgroundLoad(LoadSpec(str(path), "a", "c", target_fs=512.0, chunk_s=5.0)).start()
    time.sleep(0.05)
    cancelled.cancel()
    assert cancelled.wait(timeout=120) == "cancelled"
    assert cancelled.product is None


@pytest.mark.slow
def test_distinct_loads_run_concurrently(write_mef):
    """Two distinct loads make progress at the same time, not one after the other.

    Checked structurally — both workers part-way through at one moment — rather than by timing,
    which depends on how many cores the machine has (a 2-core CI runner is not ~2x faster). This
    asserts concurrency (no serialization between loads), not a parallel speed-up. Small chunks
    keep the work phase (~80 chunks) long against the workers' start-up jitter.
    """
    path = _two_channel_file(write_mef, seconds=400.0)
    spec_a = LoadSpec(str(path), "a", "c", target_fs=512.0, chunk_s=5.0)
    spec_c = LoadSpec(str(path), "c", "a", target_fs=512.0, chunk_s=5.0)
    b1, b2 = BackgroundLoad(spec_a).start(), BackgroundLoad(spec_c).start()
    try:
        overlapped = False
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            s1, s2 = b1.poll(), b2.poll()
            if s1 == s2 == "running" and 0 < b1.progress < 1 and 0 < b2.progress < 1:
                overlapped = True
            if s1 != "running" and s2 != "running":
                break
            time.sleep(0.005)
        statuses = (b1.wait(5), b2.wait(5))
        assert statuses == ("done", "done"), statuses
        assert overlapped, "the second load only progressed once the first had finished"
    finally:
        for b in (b1, b2):
            b.cancel()
            b.terminate()


@pytest.mark.slow
def test_dead_worker_is_reported_not_hung(write_mef):
    """A worker that dies without reporting (here: killed) surfaces as an error instead of
    leaving the load 'running' forever."""
    path = _two_channel_file(write_mef, seconds=300.0)
    bl = BackgroundLoad(LoadSpec(str(path), "a", "c", target_fs=512.0, chunk_s=5.0)).start()
    time.sleep(0.5)
    bl._proc.kill()
    assert bl.wait(timeout=30) == "error"
    assert "exited unexpectedly" in (bl.error or "")
