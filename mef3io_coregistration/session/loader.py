"""Background full-length loader.

Reads a file's alignment channel(s) across the whole recording in ~5-minute chunks and
produces the two in-memory products the session holds: a band-limited decimated
full-length trace (Stage 2/3) and a ~50 Hz min-max envelope (Stage 1/display). The raw
full-Fs full-length signal is **never** held — only one chunk at a time is in memory.

The synchronous core :func:`load_alignment_signal` takes optional ``progress`` and ``cancel``
hooks so it is testable in-process; :class:`BackgroundLoad` wraps it in a cancellable
subprocess with progress reporting, and distinct specs run concurrently.

Decimation is streamed with overlap-save: each chunk is read with a margin on both sides so
the polyphase FIR's edge transient falls outside the retained core, matching a full-length
decimation in the interior.
"""
from __future__ import annotations

import multiprocessing as mp
import queue as queue_mod
import time

from mef3io_coregistration.io.chunked import (  # noqa: F401  (re-exported: the loader's public names)
    Cancelled,
    CancelToken,
    LoadProduct,
    LoadSpec,
    ProgressHook,
    load_alignment_signal,
)


class BackgroundLoad:
    """A cancellable subprocess running :func:`load_alignment_signal`.

    Progress events and the final product are delivered over a queue. Multiple instances run
    concurrently (each is its own process), so distinct channel sets load in parallel. A worker
    that exits without reporting (crash, OOM kill, import failure in the child) is reported as
    an error — never left "running" forever.
    """

    def __init__(self, spec: LoadSpec) -> None:
        self.spec = spec
        ctx = mp.get_context("spawn")
        self._cancel = ctx.Event()
        self._queue: mp.Queue = ctx.Queue()
        self._proc = ctx.Process(target=_worker, args=(spec, self._queue, self._cancel))
        self.progress: float = 0.0
        self._product: LoadProduct | None = None
        self._status: str = "pending"
        self.error: str | None = None

    def start(self) -> BackgroundLoad:
        self._proc.start()
        self._status = "running"
        return self

    def cancel(self) -> None:
        self._cancel.set()

    def terminate(self) -> None:
        """Stop the worker (after :meth:`cancel`, or when it must not outlive its caller)."""
        if self._proc.is_alive():
            self._proc.join(timeout=2.0)
        if self._proc.is_alive():
            self._proc.terminate()

    def poll(self) -> str:
        """Drain pending events; return the current status."""
        exited = self._status == "running" and not self._proc.is_alive()
        while True:
            try:
                kind, payload = self._queue.get_nowait()
            except queue_mod.Empty:
                break
            if kind == "progress":
                self.progress = payload
            elif kind == "started":
                self._status = "running"
            else:  # done | cancelled | error
                self._status = kind
                self._product = payload if kind == "done" else None
                self.error = payload if kind == "error" else None
        if exited and self._status == "running":
            self._status = "error"
            self.error = (
                f"loader process exited unexpectedly (exit code {self._proc.exitcode}). Loads run "
                "in spawned worker processes, which re-import the calling script: if this came "
                "from a script, put the calling code under `if __name__ == \"__main__\":`, or "
                "use CoregSession(background=False) to load in this process."
            )
        return self._status

    def wait(self, timeout: float | None = None) -> str:
        """Block until the load finishes (or is cancelled/errors); return the status.

        With ``timeout`` the status is returned when it expires (possibly still ``"running"``).
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.poll() in ("running", "pending"):
            if deadline is not None and time.monotonic() > deadline:
                return self._status
            time.sleep(0.01)
        self._proc.join(timeout=5.0)
        return self._status

    @property
    def product(self) -> LoadProduct | None:
        return self._product

    @property
    def status(self) -> str:
        return self._status


def _worker(spec: LoadSpec, queue: mp.Queue, cancel: CancelToken) -> None:
    queue.put(("started", None))
    try:
        product = load_alignment_signal(
            spec, progress=lambda f, m: queue.put(("progress", f)), cancel=cancel
        )
        queue.put(("done", product))
    except Cancelled:
        queue.put(("cancelled", None))
    except Exception as exc:  # surface worker failures to the parent
        queue.put(("error", repr(exc)))
