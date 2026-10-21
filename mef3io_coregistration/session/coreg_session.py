"""Coregistration session.

Holds in-memory products from file/channel selection until explicitly closed,
orchestrates the three stages, and drives the background loaders. Channels are selected per
stage (default: matching channel names across both files). Loading a stage reads both
files' alignment channels full-length in ~5-minute chunks into the decimated +
envelope products; if a stage's channels match an already-loaded stage the product is
**reused**, otherwise the two files load **concurrently**.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from mef3io_coregistration.config import (
    ENVELOPE_FS_HZ,
    FINE_WINDOW_DEFAULT_S,
    GROSS_CONSENSUS_S,
    US_PER_S,
)
from mef3io_coregistration.io.derived import Decimated, Envelope
from mef3io_coregistration.io.mef_reader import MefChannelReader, fingerprint
from mef3io_coregistration.reader.coregistered_reader import CoregisteredReader, FileRole
from mef3io_coregistration.result import CoregResult
from mef3io_coregistration.transform.clock_map import ClockMap

from ..align.conditioning import FilterSpec
from ..align.drift import DriftEstimate, estimate_drift, refit_drift
from ..align.fine import Anchor, CostCurve, cost_surface, estimate_anchor, local_seed
from ..align.gross import (
    GrossOffset,
    OffsetVerification,
    estimate_gross_offset,
    verify_offset,
)
from .loader import BackgroundLoad, LoadProduct, LoadSpec, load_alignment_signal

Channels = tuple[str, str | None]  # (anode, cathode); cathode None -> single channel


def _serialized(method):
    """Run a session method that reads-then-writes the alignment state under the state lock."""
    import functools

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._state_lock:
            return method(self, *args, **kwargs)

    return wrapper


class NotLocked(RuntimeError):
    """Stage 1 did not lock, so later stages must not run on its offset."""


@dataclass
class _Stage:
    reference: Channels
    moving: Channels
    invert_reference: bool = False
    invert_moving: bool = False


class CoregSession:
    """Interactive, stateful coregistration of two MEF3 files."""

    def __init__(
        self, envelope_fs: float = ENVELOPE_FS_HZ, chunk_s: float = 300.0, background: bool = True
    ) -> None:
        self.envelope_fs = envelope_fs
        self.chunk_s = chunk_s
        self.background = background
        self._files: dict[FileRole, str] = {}
        self.reference_role: FileRole = "A"
        self._stages: dict[str, _Stage] = {}
        self._display_envelopes: dict = {}  # recording-view envelopes, keyed by (path, channel)
        self._load_lock = threading.RLock()
        # One writer at a time for the alignment state (offset, anchors, map). The API serves
        # requests concurrently; without this, a drift fit finishing after an anchor was added
        # would pair the new anchor list with a map fitted from the old one. Viewer reads
        # (get_product) do not take it, so the display stays live during a long computation.
        self._state_lock = threading.RLock()
        self._products: dict[tuple, LoadProduct] = {}
        self._fs_cache: dict[tuple[str, str], float] = {}

        self.gross_offset: GrossOffset | None = None
        self.moving_inverted: bool = False
        self.offset_verification: OffsetVerification | None = None
        self.map_verification: OffsetVerification | None = None
        self.anchors: list[Anchor] = []
        # Provenance of the filters each estimate used (saved with the result): per anchor, by its
        # reference time, and the ones the drift fit used.
        self._anchor_filters: dict[int, dict] = {}
        self._drift_filters: dict | None = None
        self.drift_estimate: DriftEstimate | None = None
        self.clock_map: ClockMap | None = None
        # A result reopened with open_result(): returned as saved while its clock map is current.
        self._opened: CoregResult | None = None

    # -- lifecycle --------------------------------------------------
    def open(self, file_a: str, file_b: str, reference: FileRole = "A") -> CoregSession:
        """Choose the two recordings (``.mefd`` or ``.mefd.tar``) and which is the reference.

        Nothing is read yet; loading happens per stage. Returns the session, so
        ``CoregSession().open(a, b)`` chains.
        """
        if reference not in ("A", "B"):
            raise ValueError('reference must be "A" or "B"')
        if self._files:  # new files: nothing aligned on the old ones carries over
            self._clear()
        self._files = {"A": str(file_a), "B": str(file_b)}
        self.reference_role = reference
        return self

    def _clear(self) -> None:
        """Forget everything about the current pair: alignment, channels, reopened result and the
        in-memory products."""
        self.reset_alignment()
        self.moving_inverted = False
        self._stages = {}
        self._anchor_filters = {}
        self._drift_filters = None
        self._opened = None
        self.close()

    @_serialized
    def open_result(
        self,
        result: CoregResult | str | Path,
        file_a: str | Path | None = None,
        file_b: str | Path | None = None,
        verify: bool = True,
    ) -> CoregSession:
        """Reopen a saved result: its two recordings, its reference role and its clock map, ready
        to read (:meth:`open_reader`, :meth:`open_recording`) or to save again.

        ``result`` is a :class:`CoregResult` or a saved file (``.json`` or ``.json.gz``).
        ``file_a``/``file_b`` override the stored paths; each recording is checked by its time
        span, as :meth:`CoregisteredReader.from_result` does (``verify=False`` skips that). The
        stages are not re-run: until a new clock map is made, :meth:`result` returns the result
        as saved (anchors, drift, filters, QC), with the recordings' current paths. Returns the
        session, so ``CoregSession().open_result(path)`` chains.

        To continue aligning from it, select channels again (:meth:`set_channels`); the anchors
        are restored (without their correlation values, which a result does not store), but there
        is no Stage 1 lock, so give one (:meth:`manual_offset`) before new anchors are sought.
        A :class:`CoregResult` with relative recording paths resolves them against the working
        directory (a saved file: against its folder).
        """
        from ..reader.coregistered_reader import CoregisteredReader

        res = result if isinstance(result, CoregResult) else CoregResult.load(result)
        near = None if isinstance(result, CoregResult) else Path(result).parent
        rd = CoregisteredReader.from_result(res, file_a, file_b, verify=verify, near=near)
        paths = dict(rd.paths)
        rd.close()
        self._clear()
        self.open(paths["A"], paths["B"], reference=res.reference_role)  # type: ignore[arg-type]
        self.clock_map = res.clock_map
        self.anchors = list(res.anchors)
        # each anchor keeps the filters it was estimated with, so a map built from them later
        # saves the right provenance
        for f in (res.stage_filters or {}).get("fine", []) or []:
            if isinstance(f, dict) and "t_ref_uutc" in f:
                self._anchor_filters[int(f["t_ref_uutc"])] = f.get("filters")
        self._opened = res
        return self

    @property
    def reopened(self) -> CoregResult | None:
        """The result :meth:`open_result` reopened, while its clock map is still the current one
        (``None`` otherwise): a front end shows its saved quality report instead of rebuilding
        one from stages that were not run here."""
        o = self._opened
        if o is not None and self.clock_map is o.clock_map and self.reference_role == o.reference_role:
            return o
        return None

    def close(self) -> None:
        """Release the in-memory products (they live until the session is closed)."""
        self._products.clear()
        self._fs_cache.clear()
        self._display_envelopes.clear()

    def __enter__(self) -> CoregSession:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def reference_file(self) -> str:
        """The reference recording's path (its clock is the output clock)."""
        return self._files[self.reference_role]

    @property
    def moving_file(self) -> str:
        """The moving recording's path (resampled onto the reference clock when read)."""
        return self._files["B" if self.reference_role == "A" else "A"]

    # -- channel selection -----------------------------------------
    @_serialized
    def set_channels(
        self,
        stage: str,
        anode: str | None = None,
        cathode: str | None = None,
        reference: Channels | None = None,
        moving: Channels | None = None,
        invert_reference: bool = False,
        invert_moving: bool = False,
    ) -> None:
        """Select ``stage`` alignment channels, independently per file.

        By default the same ``(anode, cathode)`` names are used on both files. Pass
        ``reference``/``moving`` — each an ``(anode, cathode)`` pair — to choose different
        channels on each: the two systems may name the same electrode differently, may not carry
        the same subset at all, and stim can saturate one amplifier's channel while leaving the
        other's usable, so the best channel to align on genuinely differs per file and per stage.

        ``invert_reference`` / ``invert_moving`` multiply that file's derivation by −1, for when
        the two systems record the same physiology with opposite polarity. Correlation then peaks
        *negatively* at the true offset, and since every estimator takes the maximum of a
        correlation, an un-flipped inverted channel does not just score badly — it locks onto the
        wrong lag, or does not lock at all.
        """
        ref = reference or (anode, cathode)
        mov = moving or (anode, cathode)
        if not ref[0] or not mov[0]:
            raise ValueError("a channel must be selected for both files")
        ref_sel: Channels = (str(ref[0]), ref[1] or None)
        mov_sel: Channels = (str(mov[0]), mov[1] or None)
        for sel, which in ((ref_sel, "reference"), (mov_sel, "moving")):
            if sel[1] is not None and sel[0] == sel[1]:
                raise ValueError(
                    f"{which} file: anode and cathode are the same channel ({sel[0]}) — a bipolar "
                    "pair needs two different channels (leave the cathode empty for a single channel)"
                )
        # Fail now, with the list of what exists, rather than at load time.
        for path, sel in ((self.reference_file, ref_sel), (self.moving_file, mov_sel)):
            with MefChannelReader(path) as reader:
                for ch in (c for c in sel if c):
                    if not reader.has_channel(ch):
                        raise KeyError(
                            f"channel {ch!r} is not in {path}; available: {sorted(reader.channels)}"
                        )
        new = _Stage(ref_sel, mov_sel, bool(invert_reference), bool(invert_moving))
        if self._stages.get(stage) != new and self._stages.get(stage) is not None:
            # Different alignment signal: everything measured on the old one starts over.
            self.gross_offset = None
            self.anchors = []
            self._invalidate_map()
        self._stages[stage] = new

    # -- loading ---------------------------------------------
    def load(self, stage: str, on_progress=None) -> dict[str, LoadProduct]:
        """Load both files' products for ``stage`` (reusing cached, else concurrent).

        Serialized: the API serves requests concurrently, and a second request for the same
        stage must wait and reuse the products rather than start a duplicate full-length load.
        """
        with self._load_lock:
            return self._load(stage, on_progress)

    def _load(self, stage: str, on_progress=None) -> dict[str, LoadProduct]:
        specs = self._stage_specs(stage)
        pending = {role: s for role, s in specs.items() if s.key not in self._products}

        if pending:
            if self.background:
                self._run_concurrent(pending, on_progress)
            else:
                pending_specs = list(pending.values())
                for i, s in enumerate(pending_specs):
                    # scale each file's 0→1 into its slice so global progress is monotonic
                    hook = (
                        (lambda f, m, i=i: on_progress((i + f) / len(pending_specs)))
                        if on_progress is not None
                        else None
                    )
                    self._products[s.key] = load_alignment_signal(s, progress=hook)
        return {role: self._products[s.key] for role, s in specs.items()}

    def get_product(self, stage: str, role: str) -> LoadProduct:
        """The loaded product for a stage/role (``"reference"`` | ``"moving"``); load first."""
        key = self._stage_specs(stage)[role].key
        if key not in self._products:
            raise KeyError(f"stage {stage!r} role {role!r} not loaded")
        return self._products[key]

    def _run_concurrent(self, pending: dict[str, LoadSpec], on_progress) -> None:
        loads = {role: BackgroundLoad(s).start() for role, s in pending.items()}
        active = dict(loads)
        try:
            while active:
                for role, bl in list(active.items()):
                    status = bl.poll()
                    if status in ("done", "cancelled", "error"):
                        if status == "error":
                            raise RuntimeError(f"load of {bl.spec.label} failed: {bl.error}")
                        if bl.product is not None:
                            self._products[bl.spec.key] = bl.product
                        del active[role]
                if on_progress is not None:
                    on_progress(sum(bl.progress for bl in loads.values()) / len(loads))
                time.sleep(0.01)
        finally:
            for bl in active.values():  # on error, stop the sibling loads too
                bl.cancel()
                bl.terminate()

    def _stage_specs(self, stage: str) -> dict[str, LoadSpec]:
        if stage not in self._stages:
            raise KeyError(f"no channels selected for stage {stage!r}; call set_channels first")
        st = self._stages[stage]
        target = min(
            self._fs(self.reference_file, st.reference[0]),
            self._fs(self.moving_file, st.moving[0]),
        )
        return {
            "reference": self._spec(
                self.reference_file, st.reference, target, st.invert_reference
            ),
            "moving": self._spec(self.moving_file, st.moving, target, st.invert_moving),
        }

    def _spec(
        self, file_path: str, channels: Channels, target_fs: float, invert: bool = False
    ) -> LoadSpec:
        return LoadSpec(
            file_path=file_path,
            anode=channels[0],
            cathode=channels[1],
            target_fs=target_fs,
            envelope_fs=self.envelope_fs,
            chunk_s=self.chunk_s,
            invert=invert,
        )

    def _fs(self, file_path: str, channel: str) -> float:
        key = (file_path, channel)
        if key not in self._fs_cache:
            with MefChannelReader(file_path) as reader:
                self._fs_cache[key] = reader.fs(channel)
        return self._fs_cache[key]

    # -- polarity -----------------------------------------------------------
    @_serialized
    def set_invert(self, role: str, invert: bool, stages: tuple[str, ...] = ("gross", "fine")):
        """Flip a file's alignment derivation by −1, across the given stages.

        The already-loaded products are negated **in memory** rather than re-read: inverting a
        signal is exact, so re-reading and re-decimating the file to get the same numbers with
        the opposite sign would only cost the user half a minute for nothing.

        Polarity is something the user tries, looks at, and tries again — it must not send them
        back to loading. So the Stage 1 offset and the anchors are **kept** (they are measurements
        the user can re-run or delete; an offset does not stop being an offset because the sign
        of a trace changed); only the clock map is dropped, because it was fitted from windows
        correlated with the old sign.
        """
        changed = False
        for stage in stages:
            st = self._stages.get(stage)
            if st is None:
                continue
            old_key = self._stage_specs(stage)[role].key
            current = st.invert_reference if role == "reference" else st.invert_moving
            changed |= current != invert
            if role == "reference":
                st.invert_reference = invert
            else:
                st.invert_moving = invert
            new_spec = self._stage_specs(stage)[role]
            if new_spec.key not in self._products and old_key in self._products:
                self._products[new_spec.key] = _negate(self._products[old_key], new_spec)
        if changed:
            self._invalidate_map()

    @_serialized
    def set_polarity(self, role: str, invert: bool, stage: str) -> list[str]:
        """Invert ``role``'s signal in ``stage`` — and in every other stage aligning on the same
        channel of that file, so one electrode never has two signs. Returns the stages changed."""
        if role not in ("reference", "moving"):
            raise ValueError('role must be "reference" or "moving"')
        if stage not in self._stages:
            raise KeyError(f"no channels selected for stage {stage!r}")
        def channels(st: _Stage) -> Channels:
            return st.reference if role == "reference" else st.moving

        target = channels(self._stages[stage])
        stages = tuple(k for k, st in self._stages.items() if channels(st) == target)
        self.set_invert(role, bool(invert), stages)
        return list(stages)

    def polarity(self) -> dict[str, dict[str, bool]]:
        """``{stage: {"reference": inverted, "moving": inverted}}``."""
        return {
            k: {"reference": st.invert_reference, "moving": st.invert_moving}
            for k, st in self._stages.items()
        }

    # -- stages -----------------------------------------------------
    @_serialized
    def run_gross(self, on_progress=None, **kwargs) -> GrossOffset:
        """Stage 1 on the **decimated waveform** — never the min-max envelope.

        The envelope is a display product: it summarises each file's raw signal at that file's
        own bandwidth, so the same 20 ms bin means different things on an 8 kHz amplifier and a
        256 Hz one, and correlating the two is meaningless (see align/gross.py).
        """
        products = self.load("gross", on_progress=on_progress)
        g = estimate_gross_offset(
            products["reference"].decimated, products["moving"].decimated, **kwargs
        )
        self._set_gross(g)
        return g

    def _set_gross(self, g: GrossOffset) -> None:
        """Adopt a new Stage 1 result and invalidate what was built on the old one.

        The clock map always goes: it was fitted from the previous alignment, and saving or
        reading through it after Stage 1 changed would silently use a result the user has moved
        on from. Anchors are independent measurements and are kept — unless the offset moved by
        more than the consensus tolerance, in which case they belong to a different alignment
        hypothesis and are discarded too.
        """
        old = self.gross_offset
        if (
            old is not None
            and self.anchors
            and abs(float(g.offset_s) - float(old.offset_s)) > GROSS_CONSENSUS_S
        ):
            self.anchors = []
        self.gross_offset = g
        self._invalidate_map()

    @_serialized
    def reset_alignment(self) -> None:
        """Forget the alignment (lock, anchors, map) but keep the files and loaded products — a
        new run from Stage 1."""
        self.gross_offset = None
        self.offset_verification = None
        self.anchors = []
        self._invalidate_map()

    def _invalidate_map(self) -> None:
        self.drift_estimate = None
        self.clock_map = None
        self.map_verification = None

    def verify_offset(self, offset_s: float) -> OffsetVerification:
        """Does the shared signal actually support ``offset_s``?"""
        products = self.load("gross")
        return verify_offset(
            products["reference"].decimated, products["moving"].decimated, float(offset_s)
        )

    @_serialized
    def manual_offset(self, offset_s: float, force: bool = False) -> GrossOffset:
        """Accept a human-supplied offset as the Stage 1 result — **after checking it**.

        The escape hatch that makes the tool human-in-the-loop: when the automatic estimate does
        not lock, the user drags the moving trace onto the reference. But a human's say-so is not
        evidence, and this used to mint ``locked=True, confidence=1.0`` on the strength of a drag,
        walking straight past the only check the pipeline had. Hand-place the offset an hour wrong
        and Stage 3 would accept 195 of 258 windows, fit a smooth curve, and raise no warning at
        all.

        So the drag is *verified*: we ask whether the shared signal is actually present at that
        offset. If it is not, this is not a lock, and the later stages will refuse it.
        ``force=True`` overrides — a human may know something the correlation does not — but the
        override is recorded in the reason and travels into the saved map's provenance.

        **The offset the user placed is the offset that is used.** Verification decides whether to
        accept it; it does not get to move it. An earlier version "snapped" the accepted value onto
        the templates' own estimate whenever the two were within the tolerance window — which meant
        that a user who deliberately hand-aligned anywhere near the automatic answer watched the
        tool quietly discard their placement and put its own back. That is not a human-in-the-loop
        tool, it is a tool that argues with you. Where the templates disagree with the placement,
        the difference is *reported* (``OffsetVerification.refined_offset_s``) so the user can
        adopt it if they want to — deliberately, and by choice.
        """
        v = self.verify_offset(offset_s)
        self.offset_verification = v
        locked = v.verified or force
        accepted_s = float(offset_s)

        if v.verified:
            drift_s = abs(float(v.refined_offset_s) - accepted_s)
            note = (
                f" (the templates put it at {v.refined_offset_s:+.3f}s, {drift_s:.3f}s away — "
                "yours is what will be used)"
                if drift_s > 0.05
                else ""
            )
            reason = f"set by hand from the overlay — {v.reason}{note}"
        elif force:
            reason = (
                f"set by hand and accepted without template support — {v.reason} "
                "Stimulation or artifact can hide the shared signal from the templates; the Fine "
                "anchors are what confirm or refute this offset."
            )
        else:
            reason = f"NOT ACCEPTED — {v.reason}"

        g = GrossOffset(
            offset_s=accepted_s,
            confidence=1.0 if v.verified else 0.0,
            locked=locked,
            spread_s=abs(float(v.refined_offset_s) - accepted_s),
            n_agree=v.n_support,
            n_templates=v.n_templates,
            matches=[],
            reason=reason,
            snr=v.snr,
            accepted_by="hand" if v.verified else "user",
        )
        self._set_gross(g)
        return g

    @_serialized
    def accept_gross(self) -> GrossOffset:
        """Continue from the automatic Stage 1 estimate although the templates did not agree.

        Never a dead end: stimulation, saturation on one amplifier and blanking on the other, or a
        long artifact can leave too few clean templates to corroborate an offset that is in fact
        right. The user may take the best estimate and let Stage 2 confirm it at a clean ROI. The
        decision is recorded (``accepted_by="user"``) and reported with the map's QC.
        """
        g = self.gross_offset
        if g is None:
            raise RuntimeError("No Stage 1 estimate yet — run run_gross() first.")
        if g.locked:
            return g
        accepted = replace(
            g,
            locked=True,
            accepted_by="user",
            reason=(
                f"accepted by the user without template agreement ({g.n_agree} of "
                f"{g.n_templates} agree). Confirm it with Fine anchors at clean stretches."
            ),
        )
        self.gross_offset = accepted  # same offset: anchors and loaded products stay valid
        self._invalidate_map()
        return accepted

    @_serialized
    def run_fine(
        self,
        roi_center_uutc: int,
        seed_offset_s: float | None = None,
        half_window_s: float = FINE_WINDOW_DEFAULT_S / 2,
        # Wide enough to correct a hand-placed seed. A human aligning by eye in the overlay is
        # accurate to a fraction of a second, but not to 0.25 s, and a Stage 2 that cannot reach
        # past its own seed's error is not refining anything — it just inherits it.
        max_residual_s: float = 0.5,
        filters: FilterSpec | None = None,
        on_progress=None,
        refine_seed: bool = True,
    ) -> Anchor:
        """Stage 2 at an ROI. ``refine_seed``: first refine the seed from the ROI's
        neighbourhood (:func:`~mef3io_coregistration.align.fine.local_seed`); pass False when the
        seed was placed by hand and must be used as-is."""
        products = self.load("fine", on_progress=on_progress)
        if seed_offset_s is None:
            seed_offset_s = self._require_lock().offset_s
        if refine_seed:
            seed_offset_s = local_seed(
                products["reference"].decimated,
                products["moving"].decimated,
                roi_center_uutc,
                seed_offset_s,
            ).offset_s
        anchor = estimate_anchor(
            products["reference"].decimated,
            products["moving"].decimated,
            roi_center_uutc,
            seed_offset_s,
            half_window_s=half_window_s,
            max_residual_s=max_residual_s,
            filters=filters,
        )
        # A re-run at the same ROI replaces the earlier anchor there.
        self.anchors = [a for a in self.anchors if a.t_ref_uutc != anchor.t_ref_uutc]
        self.anchors.append(anchor)
        self._anchor_filters[int(anchor.t_ref_uutc)] = _filters_dict(filters)
        # The map was fitted from the previous anchor set: saving or reading through it now would
        # pair an anchor list with a map it did not produce. Drift (or no-drift) must be re-run.
        self._invalidate_map()
        return anchor

    @_serialized
    def remove_anchor(self, index: int) -> Anchor:
        """Delete anchor ``index`` (in :attr:`anchors` order); the clock map is dropped with it."""
        if index < 0 or index >= len(self.anchors):
            raise IndexError(f"no anchor {index}; there are {len(self.anchors)}")
        removed = self.anchors.pop(index)
        self._anchor_filters.pop(int(removed.t_ref_uutc), None)
        self._invalidate_map()
        return removed

    def cost_at_roi(
        self,
        roi_center_uutc: int,
        seed_offset_s: float | None = None,
        half_window_s: float = FINE_WINDOW_DEFAULT_S / 2,
        max_residual_s: float = 0.5,
        filters: FilterSpec | None = None,
        step: int = 1,
        on_progress=None,
    ) -> CostCurve:
        """The correlation-vs-offset curve for an ROI, without committing an anchor.

        Same windows, conditioning and estimator as :meth:`run_fine`, so its peak is the anchor
        Refine would produce — this is the evidence the user inspects before (or instead of)
        accepting it.
        """
        products = self.load("fine", on_progress=on_progress)
        if seed_offset_s is None:
            seed_offset_s = self._require_lock().offset_s
        return cost_surface(
            products["reference"].decimated,
            products["moving"].decimated,
            roi_center_uutc,
            seed_offset_s,
            half_window_s=half_window_s,
            max_residual_s=max_residual_s,
            filters=filters,
            step=step,
        )

    def _require_lock(self) -> GrossOffset:
        """Refuse to build on a failed Stage 1.

        The old pipeline reported confidence 0.000, marked the stage done, and let Stage 2
        "refine" an offset that was three hours wrong into an anchor. A tool that fails is
        annoying; a tool that fails, knows it failed, and lets you build on the failure is
        dangerous.
        """
        if self.gross_offset is None:
            raise RuntimeError(
                "No Stage 1 offset yet — run run_gross() (or give one with manual_offset()) first."
            )
        if not self.gross_offset.locked:
            raise NotLocked(self.gross_offset.reason)
        return self.gross_offset

    @_serialized
    def build_map_without_drift(self) -> ClockMap:
        """Produce a clock map WITHOUT Stage 3 — a deliberate bypass, not a shortcut.

        Stage 3 exists to track a clock that wanders over many hours. When it has nothing to
        track, running it anyway is not conservative — it is fitting a curve to noise. That is
        the case whenever the recordings are short, or the overlap between them is: the true
        offset has no room to drift, and a sliding-window estimate will happily produce a wiggle
        of its own measurement scatter and present it as a clock.

        So the map is built from what is actually known:

        * **no anchors** — a *constant* offset, taken from Stage 1. The two clocks are assumed
          not to have moved relative to each other across the recording. State that assumption
          and it is a perfectly good map.
        * **one anchor** — a constant offset at that anchor's (more precise) value.
        * **two or more anchors** — a straight line *through the anchors*, which absorbs a
          constant difference in clock RATE without ever estimating drift.

        Outside the outermost anchor the offset is held, not projected, and the map records that
        it was built this way. The result is a map the tool can defend: everything in it was
        either measured at an anchor or is an assumption the user made on purpose.
        """
        products = self.load("fine")
        ref = products["reference"].decimated
        t0 = int(ref.start_uutc or 0)
        t1 = t0 + int(round((ref.data.shape[0] - 1) * US_PER_S / ref.fs))

        if len(self.anchors) >= 2:
            order = sorted(self.anchors, key=lambda a: a.t_ref_uutc)
            xs = np.array([a.t_ref_uutc for a in order], dtype=np.float64)
            ys = np.array([a.offset_s for a in order], dtype=np.float64)
            centers = np.unique(np.concatenate(([t0], xs, [t1]))).astype(np.int64)
            # np.interp holds the end VALUE outside the anchors — it does not project the trend.
            offsets = np.interp(centers.astype(np.float64), xs, ys)
            method = f"line through {len(order)} anchors"
        else:
            if self.anchors:
                off = float(self.anchors[0].offset_s)
                method = "constant offset from the Stage 2 anchor"
            else:
                off = float(self._require_lock().offset_s)
                method = "constant offset from Stage 1"
            centers = np.array([t0, t1], dtype=np.int64)
            offsets = np.array([off, off], dtype=np.float64)

        self.clock_map = ClockMap.from_reference_offset(
            centers,
            offsets,
            metadata={
                "stage": "no-drift",
                "method": method,
                "n_anchors": len(self.anchors),
                "assumption": (
                    "the two clocks do not drift relative to each other over this recording; "
                    "Stage 3 was deliberately bypassed"
                ),
                "measured_span_uutc": [
                    int(min(a.t_ref_uutc for a in self.anchors)) if self.anchors else int(t0),
                    int(max(a.t_ref_uutc for a in self.anchors)) if self.anchors else int(t1),
                ],
            },
        )
        self.drift_estimate = None

        # The map still gets checked. Bypassing Stage 3 is a statement about drift, not a licence
        # to skip asking whether the offset is in the right place at all.
        self.map_verification = verify_offset(
            ref, products["moving"].decimated, float(np.median(offsets))
        )
        return self.clock_map

    @_serialized
    def run_drift(
        self, anchor: Anchor | list[Anchor] | None = None, **kwargs
    ) -> DriftEstimate:
        """Stage 3 over the shared span, walking outward from every anchor (default: all)."""
        products = self.load("fine")  # Stage 3 reuses the fine decimated traces
        if anchor is None:
            if not self.anchors:
                raise RuntimeError(
                    "Place at least one anchor with run_fine() (Stage 2) before estimating drift — "
                    "or build the map with build_map_without_drift()."
                )
            anchor = list(self.anchors)
        self._drift_filters = _filters_dict(kwargs.get("filters"))
        self.drift_estimate = estimate_drift(
            products["reference"].decimated, products["moving"].decimated, anchor, **kwargs
        )
        self.clock_map = self.drift_estimate.clock_map

        # Verify the MAP, not just the seed. Every per-window statistic in Stage 3 is computed on
        # a ±0.25 s surface around wherever the seed pointed, so a globally wrong seed produces
        # windows that look excellent — 195 of 258 accepted, a smooth curve, no warnings — while
        # the map is an hour out. The only way to catch that is to ask, on the full-length
        # correlation, whether the shared signal is really where the map says it is.
        mid = int(np.median(self.drift_estimate.t_ref_uutc))
        fitted = float(np.interp(
            mid, self.drift_estimate.t_ref_uutc, self.drift_estimate.smoothed_offset_s
        ))
        self.map_verification = verify_offset(
            products["reference"].decimated, products["moving"].decimated, fitted
        )
        return self.drift_estimate

    @_serialized
    def review_drift(self, keep: list[bool]) -> DriftEstimate:
        """Rebuild the map from a user-edited keep set (manual window review).

        The correlations are untouched — this only changes which windows the spline is fit
        through — so it is cheap. The map is re-verified afterwards, exactly as a fresh drift
        would be: discarding good windows (or keeping bad ones) can move the fitted offset, and
        the full-length check must run again on the result.
        """
        if self.drift_estimate is None:
            raise RuntimeError("Estimate drift in Stage 3 first.")
        products = self.load("fine")
        self.drift_estimate = refit_drift(self.drift_estimate, np.asarray(keep, dtype=bool))
        self.clock_map = self.drift_estimate.clock_map

        mid = int(np.median(self.drift_estimate.t_ref_uutc))
        fitted = float(np.interp(
            mid, self.drift_estimate.t_ref_uutc, self.drift_estimate.smoothed_offset_s
        ))
        self.map_verification = verify_offset(
            products["reference"].decimated, products["moving"].decimated, fitted
        )
        return self.drift_estimate

    # -- result & reader ----------------------------------------------------
    @_serialized
    def result(self, qc: dict | None = None) -> CoregResult:
        """The saved artefact: map + provenance, the files' fingerprints and the QC report
        (``qc``; a front end may pass its fuller report)."""
        if self.clock_map is None:
            raise RuntimeError(
                "No clock map yet — estimate drift in Stage 3, or build the map without drift."
            )
        if self.reopened is not None:
            # A reopened result, unchanged: as saved, with the recordings where they are now.
            return replace(
                self.reopened,
                reference_file=self.reference_file,
                moving_file=self.moving_file,
                qc=qc if qc is not None else self.reopened.qc,
            )
        return CoregResult(
            clock_map=self.clock_map,
            reference_file=self.reference_file,
            moving_file=self.moving_file,
            stage_channels={
                stage: {
                    "reference": _label(st.reference, st.invert_reference),
                    "moving": _label(st.moving, st.invert_moving),
                }
                for stage, st in self._stages.items()
            },
            anchors=list(self.anchors),
            stage_filters=self._stage_filters(),
            drift=self._drift_qc(),
            reference_role=self.reference_role,
            qc=qc if qc is not None else self._verification_qc(),
            file_fingerprints={
                role: fingerprint(path) for role, path in self._files.items()
            },
        )

    def _stage_filters(self) -> dict:
        """The filters behind the saved map: Stage 1's fixed detection band, each anchor's
        (by reference time, in anchor order) and the drift fit's — so the result can be audited
        and reproduced."""
        from ..align.conditioning import DETECTION_FILTERS

        out: dict = {
            "gross": _filters_dict(DETECTION_FILTERS),
            "fine": [
                {"t_ref_uutc": int(a.t_ref_uutc),
                 "filters": self._anchor_filters.get(int(a.t_ref_uutc), _filters_dict(None))}
                for a in sorted(self.anchors, key=lambda a: a.t_ref_uutc)
            ],
        }
        if self.drift_estimate is not None:
            out["drift"] = self._drift_filters or _filters_dict(None)
        return out

    def _verification_qc(self) -> dict | None:
        """The minimum QC a saved map carries even outside the API: was it verified?"""
        out: dict = {}
        v = self.map_verification
        if v is not None:
            out.update(map_verified=bool(v.verified), verification=v.reason)
        g = self.gross_offset
        if g is not None:
            out.update(stage1_accepted_by=g.accepted_by, stage1_reason=g.reason)
        return out or None

    def open_reader(self) -> CoregisteredReader:
        """A :class:`~mef3io_coregistration.reader.CoregisteredReader` on the current clock map,
        without saving a result first."""
        if self.clock_map is None:
            raise RuntimeError(
                "No clock map yet — estimate drift in Stage 3, or build the map without drift."
            )
        return CoregisteredReader(
            self.clock_map, self._files["A"], self._files["B"], reference=self.reference_role
        )

    def open_recording(self, channels=None, default_prefer: FileRole | None = None):
        """The two files as one recording through the current clock map.

        Display envelopes are cached on the session, so re-mapping channels or refitting the map
        does not re-read every channel."""
        from mef3io_coregistration.reader.unified import CoregisteredRecording

        return CoregisteredRecording(
            self.open_reader(), channels, default_prefer, envelope_cache=self._display_envelopes
        )

    def _drift_qc(self) -> dict | None:
        d = self.drift_estimate
        if d is None:
            return None
        return {
            "window_s": float(d.clock_map.metadata.get("window_s", 0.0)),
            "grid_s": float(d.clock_map.metadata.get("grid_s", 0.0)),
            "t_ref_uutc": [int(v) for v in d.t_ref_uutc],
            "offset_s": [float(v) for v in d.offset_s],
            "smoothed_offset_s": [float(v) for v in d.smoothed_offset_s],
            "confidence": [float(v) for v in d.confidence],
            "accepted": [bool(v) for v in d.accepted],
            "gapped": [bool(v) for v in d.gapped],
            "measured_span_uutc": [int(v) for v in d.measured_span_uutc],
        }


def _negate(product: LoadProduct, spec: LoadSpec) -> LoadProduct:
    """The same product with the signal multiplied by -1, without touching the file.

    The envelope's min and max SWAP under negation — the most negative sample becomes the most
    positive one. Negating them in place without swapping would silently produce an envelope
    whose "min" lies above its "max", which draws as an inverted band and is not a signal at all.
    """
    dec = product.decimated
    env = product.envelope
    return LoadProduct(
        decimated=Decimated(-dec.data, dec.fs, dec.start_uutc),
        envelope=Envelope(-env.maxs, -env.mins, env.fs, env.bin_size, env.start_uutc),
        spec=spec,
    )


def _label(channels: Channels, invert: bool = False) -> str:
    base = f"{channels[0]}-{channels[1]}" if channels[1] else channels[0]
    return f"-({base})" if invert else base


def _filters_dict(filters: FilterSpec | None) -> dict:
    """A filter set as saved in a result; ``None`` means the estimation-band default."""
    from dataclasses import asdict

    from ..align.conditioning import ESTIMATION_FILTERS

    return asdict(filters if filters is not None else ESTIMATION_FILTERS)
