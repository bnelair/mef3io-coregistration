"""Stage 1 — gross alignment.

Recovers a coarse global offset that can be **minutes to hours** (the isolated, no-DST cadence
clock), by matched-filtering several independent templates from the shorter recording into the
longer one and requiring them to **agree**.

Why not the min-max envelope, as originally designed? Because the envelope is not a shared
physical quantity. It is computed from each file's *raw* signal at that file's native rate, so
a 20 ms bin of an 8 kHz cadence channel summarises 4 kHz of bandwidth while the same bin of a
256 Hz natus channel summarises 128 Hz. On the reference dataset the median bin range is
1246 uV against 49 uV. Correlating those two features produces a flat surface whose argmax sits
**2.96 hours** from the truth, with the correlation at the *true* offset only 10–53 % of the
spurious maximum. No amount of whitening or tuning rescues it: there is no peak to find. The
min-max envelope remains what R2.3 says it is — a display product, not time-accurate, not an
alignment feature.

The band-limited **decimated waveform** does lock, cheaply: a 5-minute cadence template at
256 Hz matches into the 24 h natus trace with NCC 0.43–0.52 against a 0.25 background, from
four independent starting points, agreeing to within 0.12 s.

**Consensus is the confidence.** One template's argmax is a guess; four templates drawn from
different hours of the recording landing on the same offset is a measurement. If they disagree,
the estimate is reported as **not locked** — and a stage that has not locked must never be
marked done (REVIEW B1: the old code reported confidence 0.000, drew it in a stat box, and let
the user build Stage 2 on top of it).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft

from mef3io_coregistration.config import (
    GROSS_CONSENSUS_S,
    GROSS_MIN_SNR,
    GROSS_N_TEMPLATES,
    GROSS_TEMPLATE_S,
    LOWPASS_HZ,
    PEAK_GUARD_S,
    US_PER_S,
)

from .conditioning import Conditioned, condition
from .fine import Trace
from .tde import peak_quality, peak_snr

_DETECT_OVERSAMPLE = 4  # detection-rate samples per detection-band edge (x LOWPASS_HZ)
_MIN_VALID_FRACTION = 0.6  # a match window must be at least this un-gapped
# A template may be mostly masked (hours of stimulation) and still carry minutes of clean signal:
# it is correlated over its valid samples only, so it needs only this much of them.
_MIN_TEMPLATE_VALID = 0.25
# A run of identical samples this long carries no signal: an amplifier pinned at its rail
# (saturation) or blanked by the stimulator. Physiology is never exactly constant for 0.1 s.
_FLAT_RUN_S = 0.1
_FLAT_PAD_S = 0.25  # plus the recovery transient either side of it
_SLOT_TRIES = 8  # alternative positions tried per template slot when the nominal one is unusable


@dataclass(frozen=True)
class TemplateMatch:
    """One template's independent estimate of the global offset."""

    t_ref_uutc: int  # centre of the template, in the reference clock
    offset_s: float
    ncc: float  # peak normalized cross-correlation (how well it matched)
    confidence: float  # peak distinctiveness in [0, 1] (how unambiguous the match was)
    in_consensus: bool = False


@dataclass(frozen=True)
class GrossOffset:
    """Coarse global offset ``t_moving - t_ref`` (seconds), with an explicit lock verdict.

    ``locked`` is the field that matters. When it is False the offset is a guess, and the later
    stages will not build on it until someone accepts it (R4, REVIEW B1). ``accepted_by`` says
    who did: ``"templates"`` (they agreed), ``"hand"`` (a hand placement the shared signal
    supports) or ``"user"`` (accepted without that support — recorded, and reported in QC).
    """

    offset_s: float
    confidence: float  # display mapping of `snr` onto [0, 1]; the lock threshold sits at 0.5
    locked: bool
    spread_s: float  # spread of the agreeing templates; the estimate's own error bar
    n_agree: int
    n_templates: int
    matches: list[TemplateMatch] = field(default_factory=list)
    reason: str = ""
    snr: float = 0.0  # stacked peak height in robust sigmas — the actual lock statistic
    accepted_by: str = "templates"


def estimate_gross_offset(
    ref: Trace,
    mov: Trace,
    n_templates: int = GROSS_N_TEMPLATES,
    template_s: float = GROSS_TEMPLATE_S,
    consensus_s: float = GROSS_CONSENSUS_S,
    min_snr: float = GROSS_MIN_SNR,
    guard_s: float = PEAK_GUARD_S,
) -> GrossOffset:
    """Estimate the coarse offset by multi-template consensus (R4.1, R4.2).

    ``ref`` and ``mov`` are harmonized decimated traces (same Fs) with absolute ``start_uutc``.
    Templates are drawn from whichever recording is **shorter** — a template from the long file
    usually has no counterpart in the short one — and matched into the longer. The reported
    offset is always ``t_moving - t_ref`` regardless of which way round that was.
    """
    if ref.start_uutc is None or mov.start_uutc is None:
        raise ValueError("gross alignment needs traces with absolute start_uutc")
    if abs(ref.fs - mov.fs) > 1e-6:
        raise ValueError("reference and moving traces must be harmonized to the same Fs")

    ref_start, mov_start = int(ref.start_uutc), int(mov.start_uutc)
    ref, mov = _at_detection_rate(ref), _at_detection_rate(mov)
    fs = ref.fs
    c_ref = condition(ref.data, fs, ref_start)
    c_mov = condition(mov.data, fs, mov_start)

    # Draw templates from the shorter trace; search the longer one.
    ref_is_short = c_ref.data.shape[0] <= c_mov.data.shape[0]
    short, long_ = (c_ref, c_mov) if ref_is_short else (c_mov, c_ref)

    # Templates must be independent, so they cannot each span the whole recording: cap the
    # template at a quarter of the shorter trace and shrink rather than fail on short pairs.
    n_short = short.data.shape[0]
    m = min(int(round(template_s * fs)), n_short // 4)
    if m < int(round(fs)) or long_.data.shape[0] < m:
        raise ValueError(
            f"recordings too short for gross alignment: the shorter is "
            f"{n_short / fs:.0f}s, which leaves no room for independent templates"
        )

    starts = _pick_template_starts(short, m, n_templates)
    if not starts:
        return GrossOffset(
            0.0, 0.0, False, 0.0, 0, 0, [],
            "no usable template: the alignment channel is flat or all gap",
        )

    long_fft, nfft = _prepare(long_.data, m)
    long_stats = _long_stats(long_, m)
    n_pos = long_.data.shape[0] - m + 1

    # Every template's correlation surface is expressed against the SAME offset axis
    # (d = match index - template index), so they can be stacked. The true offset receives a
    # contribution from every template and adds coherently; a template's spurious peaks sit at
    # unrelated offsets and do not. Stacking is what converts four ambiguous surfaces into one
    # unambiguous peak — voting alone leaves the confidence near zero even when the votes are
    # unanimous, because over 24 h of self-similar iEEG each individual surface really is
    # ambiguous.
    #
    # Templates reach different offsets (a template late in the short recording cannot match
    # near the start of the long one), so the stack is accumulated with a per-offset COVERAGE
    # count and averaged by it. Requiring every template to reach an offset would collapse the
    # searchable range to a single point whenever the two recordings are of similar length.
    d_lo, d_hi = -max(starts), (n_pos - 1) - min(starts)
    stack = np.zeros(d_hi - d_lo + 1, dtype=np.float32)
    cover = np.zeros(d_hi - d_lo + 1, dtype=np.int16)

    guard = int(round(guard_s * fs))
    matches: list[TemplateMatch] = []
    n_stacked = 0

    surfaces = _ncc_surfaces(short, starts, m, long_, long_fft, nfft, long_stats)
    for i0, ncc in zip(starts, surfaces, strict=True):
        if ncc is None:
            continue
        # This template covers offsets d = k - i0 for k in [0, n_pos).
        # Only POSITIVE evidence is stacked: a template that happens to anti-correlate near the
        # true offset (inverted polarity on one derivation, say) would otherwise drag the summed
        # peak off it. The surface itself stays signed — the negative half is the null
        # distribution every statistic downstream is measured against.
        lo = (0 - i0) - d_lo
        stack[lo : lo + n_pos] += np.maximum(ncc, 0.0).astype(np.float32)
        cover[lo : lo + n_pos] += 1
        n_stacked += 1

        k = int(np.argmax(ncc))
        conf, _ratio = peak_quality(ncc, k, guard)
        matches.append(
            TemplateMatch(
                t_ref_uutc=_t_ref_of(ref, i0, k, m, fs, ref_is_short),
                offset_s=_offset_of(ref, mov, i0, k, fs, ref_is_short),
                ncc=float(ncc[k]),
                confidence=conf,
            )
        )

    if n_stacked == 0:
        return GrossOffset(0.0, 0.0, False, 0.0, 0, 0, [], "no template produced a match")

    # Only offsets corroborated by a majority of templates are candidates: an offset just one
    # template can even reach cannot be a consensus.
    min_cover = max(2, (n_stacked + 1) // 2) if n_stacked >= 2 else 1
    usable = np.flatnonzero(cover >= min_cover)
    if usable.size < 3:
        return GrossOffset(
            0.0, 0.0, False, 0.0, 0, n_stacked, matches,
            "NO LOCK — the recordings overlap too little for independent templates to agree",
        )

    lo_i, hi_i = int(usable[0]), int(usable[-1])
    c = np.maximum(cover[lo_i : hi_i + 1], 1).astype(np.float32)
    # Normalise by sqrt(coverage), not coverage. At a wrong offset each template contributes
    # independent noise, so a sum over c templates has noise ~sigma*sqrt(c) — dividing by c
    # would leave low-coverage offsets NOISIER than high-coverage ones and let a spurious peak
    # reachable by only two templates out-score the true one reachable by all four. Dividing by
    # sqrt(c) makes the background uniform across the search and lets the true offset, where
    # every template adds coherently, grow as sqrt(c).
    stack = stack[lo_i : hi_i + 1] / np.sqrt(c)
    base = d_lo + lo_i

    d_best = int(np.argmax(stack)) + base
    snr = peak_snr(stack, d_best - base)
    offset_s = _offset_of(ref, mov, 0, d_best, fs, ref_is_short)

    return _verdict(offset_s, snr, matches, consensus_s, min_snr, float(stack.max()))


@dataclass(frozen=True)
class OffsetVerification:
    """Does the shared signal actually support THIS offset? (REVIEW-2 N1)

    Distinct from :class:`GrossOffset`, which asks "where is the best offset?". This asks the
    question every later stage silently assumed the answer to: *is the offset I was handed
    right at all?*

    The distinctiveness confidence used by Stages 2 and 3 cannot answer it. That statistic is
    computed on a correlation surface only ±0.25 s wide, so it measures whether the peak is
    sharp within a quarter-second of wherever it was pointed — not whether it was pointed in the
    right place. Measured on the reference pair, it is *higher* at every wrong offset than at the
    true one (0.83 median across 60 random offsets, versus 0.66 at the truth): at a wrong offset
    two unrelated traces throw up one random dominant blip, which is very "distinct", while the
    true offset has a broad genuine ridge that is less distinct against its own shoulders.

    So we ask a different question, on the full-length surface where it is answerable: how far
    does the correlation AT THIS OFFSET stand above the background of correlations everywhere
    else? A globally wrong offset sits in the noise, and several independent templates all say so.
    """

    offset_s: float  # the offset that was asked about
    verified: bool
    snr: float  # mean height at this offset, in robust sigmas above each surface's background
    ncc: float  # mean raw NCC at this offset
    n_support: int  # templates whose correlation here stands clear of their own background
    n_templates: int
    reason: str = ""
    # Verification tolerates +/-`tol_s` around the candidate, because the true offset genuinely
    # drifts by seconds across a recording. That slack would otherwise be a trap: a hand-drag
    # 5 s off passes verification, and is then handed to a Stage 2 that only searches +/-0.5 s
    # and can never correct it. So verification also reports WHERE inside that window the shared
    # signal actually is, and the caller adopts it. The drag only has to be close.
    refined_offset_s: float = 0.0


def verify_offset(
    ref: Trace,
    mov: Trace,
    offset_s: float,
    n_templates: int = GROSS_N_TEMPLATES,
    template_s: float = GROSS_TEMPLATE_S,
    tol_s: float = GROSS_CONSENSUS_S,
) -> OffsetVerification:
    """Check whether the shared signal supports ``offset_s`` (R12, REVIEW-2 N1).

    Cuts the same independent templates as Stage 1, but instead of taking each surface's argmax
    it reads the surface **at the offset in question** (allowing ±``tol_s`` of slack for drift)
    and scores that value against the surface's own background. Templates that stand clear of
    their background *support* the offset.

    This is what makes a hand-placed offset trustworthy — and what stops an hour-wrong seed from
    producing 195 accepted windows, a smooth drift curve and no warnings at all.
    """
    if ref.start_uutc is None or mov.start_uutc is None:
        raise ValueError("offset verification needs traces with absolute start_uutc")
    if abs(ref.fs - mov.fs) > 1e-6:
        raise ValueError("reference and moving traces must be harmonized to the same Fs")

    ref_start, mov_start = int(ref.start_uutc), int(mov.start_uutc)
    ref, mov = _at_detection_rate(ref), _at_detection_rate(mov)
    fs = ref.fs
    c_ref = condition(ref.data, fs, ref_start)
    c_mov = condition(mov.data, fs, mov_start)
    ref_is_short = c_ref.data.shape[0] <= c_mov.data.shape[0]
    short, long_ = (c_ref, c_mov) if ref_is_short else (c_mov, c_ref)

    n_short = short.data.shape[0]
    m = min(int(round(template_s * fs)), n_short // 4)
    if m < int(round(fs)) or long_.data.shape[0] < m:
        return OffsetVerification(
            offset_s, False, 0.0, 0.0, 0, 0, "recordings too short to verify an offset",
            refined_offset_s=offset_s,
        )

    starts = _pick_template_starts(short, m, n_templates)
    if not starts:
        return OffsetVerification(
            offset_s, False, 0.0, 0.0, 0, 0,
            "no usable template: the alignment channel is flat or all gap",
            refined_offset_s=offset_s,
        )

    long_fft, nfft = _prepare(long_.data, m)
    long_stats = _long_stats(long_, m)
    n_pos = long_.data.shape[0] - m + 1
    tol = max(1, int(round(tol_s * fs)))

    snrs: list[float] = []
    nccs: list[float] = []
    supported_offsets: list[float] = []
    n_support = 0

    surfaces = _ncc_surfaces(short, starts, m, long_, long_fft, nfft, long_stats)
    for i0, ncc in zip(starts, surfaces, strict=True):
        if ncc is None:
            continue
        # Where in THIS template's surface does the candidate offset land?
        k = _index_of_offset(ref, mov, offset_s, i0, fs, ref_is_short)
        lo, hi = max(0, k - tol), min(n_pos, k + tol + 1)
        if hi <= lo:
            snrs.append(0.0)
            nccs.append(0.0)
            continue

        local = int(lo + np.argmax(ncc[lo:hi]))
        here = float(ncc[local])
        snrs.append(peak_snr(ncc, local))
        nccs.append(here)

        # A template SUPPORTS the offset when its own best match — the single place in the whole
        # recording where it correlates most — lands at that offset. Nothing else. This is exactly
        # the criterion that decides the lock in the first place, simply asked about a candidate
        # rather than about the argmax, and it inherits that criterion's one virtue: independent
        # templates cut from different hours cannot be fooled into the same wrong answer.
        #
        # Two statistics were tried here first and both lie, for the same underlying reason —
        # they are absolute, and the surfaces they are measured on are not comparable across
        # datasets:
        #   * Height above background. The true offset stands 14σ clear on the real pair, but
        #     barely 1σ on the synthetic pair, whose source is so self-similar that the entire
        #     surface is elevated — while an hour-wrong offset still reaches 6σ. No threshold
        #     separates those.
        #   * The ratio of the correlation here to the template's best anywhere. Better, and
        #     scale-free, but it still passes a zero offset on a source containing pure tones:
        #     the 2 Hz component alone correlates at 74% of the peak, at any lag you like.
        #
        # "The best match is here" has no threshold to tune, and no scale to be wrong about.
        best_anywhere_k = int(np.argmax(ncc))
        best_offset = _offset_of(ref, mov, i0, best_anywhere_k, fs, ref_is_short)
        if abs(best_offset - offset_s) <= tol_s:
            n_support += 1
            supported_offsets.append(best_offset)

    n = len(snrs)
    if n == 0:
        return OffsetVerification(
            offset_s, False, 0.0, 0.0, 0, 0,
            "no template could be evaluated at this offset",
            refined_offset_s=offset_s,
        )

    mean_snr = float(np.mean(snrs))
    mean_ncc = float(np.mean(nccs))
    # STRICT majority. A bare half (2 of 4) still let a 60 s error through; requiring more than
    # half rejects it. Verification is the last thing standing between a wrong seed and a
    # confident clock map, so it fails towards "no".
    verified = n_support >= 2 and n_support > n / 2
    refined = float(np.median(supported_offsets)) if supported_offsets else float(offset_s)

    if verified:
        moved = abs(refined - offset_s)
        snap = f", snapped {moved:.2f}s to the shared signal" if moved > 0.05 else ""
        reason = (
            f"verified — {n_support}/{n} templates find real shared signal at {refined:+.2f}s "
            f"({mean_snr:.0f}σ above background, NCC {mean_ncc:.2f}){snap}"
        )
    else:
        reason = (
            f"NOT SUPPORTED — at {offset_s:+.2f}s the correlation is only {mean_snr:.1f}σ above "
            f"background (NCC {mean_ncc:.2f}); {n_support}/{n} templates find shared signal "
            "there. The recordings do not overlap at this offset. Anything built on it — "
            "anchors, drift, the clock map — will be confidently wrong."
        )
    return OffsetVerification(
        offset_s, verified, mean_snr, mean_ncc, n_support, n, reason, refined_offset_s=refined
    )


def _index_of_offset(
    ref: Trace, mov: Trace, offset_s: float, i0: int, fs: float, ref_is_short: bool
) -> int:
    """Inverse of :func:`_offset_of`: which long-trace index does this offset correspond to?"""
    ref_start, mov_start = ref.start_uutc, mov.start_uutc
    if ref_start is None or mov_start is None:
        raise ValueError("both traces need an absolute start_uutc")
    base = (int(mov_start) - int(ref_start)) / US_PER_S
    d = (offset_s - base) * fs
    return int(round(i0 + d)) if ref_is_short else int(round(i0 - d))


def _offset_of(ref, mov, i0: int, k: int, fs: float, ref_is_short: bool) -> float:
    """``t_moving - t_ref`` for a template at short-index ``i0`` matching at long-index ``k``."""
    base = (int(mov.start_uutc) - int(ref.start_uutc)) / US_PER_S
    d = (k - i0) / fs
    return base + (d if ref_is_short else -d)


def _t_ref_of(ref, i0: int, k: int, m: int, fs: float, ref_is_short: bool) -> int:
    """Reference-clock centre of the template (or of its match, if templates ran the other way)."""
    idx = i0 if ref_is_short else k
    return int(ref.start_uutc + (idx + m / 2) * US_PER_S / fs)


def _verdict(
    offset_s: float,
    snr: float,
    matches: list[TemplateMatch],
    consensus_s: float,
    min_snr: float,
    peak_ncc: float,
) -> GrossOffset:
    """Decide whether Stage 1 locked. **Template agreement is the criterion; SNR is not.**

    This is measured, not assumed. Across true pairs (real and synthetic) and deliberately
    false ones (unrelated noise, two different recordings):

                          detection SNR        template agreement
        true  locks       min  4.8 sigma       min  4/4
        false locks       max 72.8 sigma       max  0/4

    The peak's height above its background — the obvious statistic, and the one the correlation
    literature reaches for — **does not separate the two cases at all**. A false match between
    two unrelated recordings reached 72 sigma, while a true match on drifting data sat at 4.8.
    Gating on SNR would therefore reject good locks and wave through catastrophic ones. Peak
    height answers "is there structure here?", which is not the question; iEEG always has
    structure, and over a day of it there is always *somewhere* that resembles anywhere else.

    Agreement separates the cases perfectly, and for a reason that survives contact with new
    data: several 10-minute templates cut from different hours of the recording carry
    independent physiology. They can each be fooled, but not fooled into the *same* wrong
    answer. That is corroboration, and corroboration is what a lock means.

    SNR is still computed and reported — it is genuinely informative once you already know the
    offset is right, and a very low value alongside agreement is worth telling the user about —
    but it does not get a vote.
    """
    n = len(matches)
    agree_idx = [j for j, mt in enumerate(matches) if abs(mt.offset_s - offset_s) <= consensus_s]
    n_agree = len(agree_idx)

    agreeing = np.array([matches[j].offset_s for j in agree_idx], dtype=np.float64)
    spread_s = float(agreeing.max() - agreeing.min()) if agreeing.size > 1 else 0.0

    locked = n_agree >= 2 and n_agree >= (n + 1) // 2
    confidence = (n_agree / n) if n else 0.0

    if not locked:
        landed = ", ".join(f"{mt.offset_s:+.1f}s" for mt in matches)
        reason = (
            f"NO LOCK — the best offset is {offset_s:+.2f}s, but only {n_agree} of {n} "
            f"independent templates corroborate it; they landed on {landed}. Two templates from "
            "different hours do not disagree like that when the alignment is real — unless "
            "stimulation or artifact hides the shared signal. Continue with this estimate and "
            "confirm it with Fine anchors, or place the offset by hand."
        )
    elif snr < min_snr:
        reason = (
            f"locked, but weakly — {n_agree}/{n} templates agree within {spread_s:.2f}s, yet the "
            f"correlation peak is only {snr:.1f}σ above background. The shared signal is faint. "
            "Zoom in and confirm the traces really do line up before you trust the map."
        )
    else:
        reason = (
            f"locked — {n_agree}/{n} templates agree within {spread_s:.2f}s; stacked peak "
            f"{snr:.0f}σ above background, peak NCC {peak_ncc:.2f}"
        )

    marked = [
        TemplateMatch(mt.t_ref_uutc, mt.offset_s, mt.ncc, mt.confidence, in_consensus=(j in agree_idx))
        for j, mt in enumerate(matches)
    ]
    return GrossOffset(
        offset_s, confidence, locked, spread_s, n_agree, n, marked, reason, snr=snr
    )


def _pick_template_starts(short: Conditioned, m: int, n_templates: int) -> list[int]:
    """Evenly spaced template starts, moved off stretches that are mostly gap or flat.

    A slot whose nominal position is unusable (a gap, a disconnected electrode, a stimulation
    block masked by :func:`mask_flat`) tries nearby positions within its share of the recording
    before giving up, so a few bad hours do not cost the search its independent templates.
    """
    n = short.data.shape[0]
    n_slots = max(n_templates, 1)
    spacing = (n - m) / max(n_slots - 1, 1)
    # Spread the candidates across the recording so they sample independent content.
    nominal = [int(round(i * spacing)) for i in range(n_slots)]
    step = max(1, int(spacing / (2 * _SLOT_TRIES)))

    def usable(i0: int) -> float:
        valid = short.valid[i0 : i0 + m]
        frac = float(valid.mean())
        if frac < _MIN_TEMPLATE_VALID or float(np.std(short.data[i0 : i0 + m])) < 1e-6:
            return 0.0
        return frac

    picks: list[int] = []
    for i0 in nominal:
        tries = [i0] + [i0 + sgn * k * step for k in range(1, _SLOT_TRIES + 1) for sgn in (1, -1)]
        tries = [t for t in tries if 0 <= t <= n - m]
        # The nominal position unless it is noticeably gapped; then the cleanest alternative
        # (max keeps the first of equals, so a tie still goes to the nominal position).
        if usable(i0) >= 0.9:
            picks.append(i0)
            continue
        score, best = max(((usable(t), t) for t in tries), key=lambda st: st[0])
        if score > 0:
            picks.append(best)
    return sorted(set(picks))


def mask_flat(x: np.ndarray, fs: float) -> np.ndarray:
    """``x`` with every run of identical samples ≥ ``_FLAT_RUN_S`` (plus ``_FLAT_PAD_S`` either
    side) set to NaN. Returns ``x`` itself when there is nothing to mask."""
    min_run = max(3, int(round(_FLAT_RUN_S * fs)))
    if x.shape[0] < min_run:
        return x
    same = np.diff(x) == 0  # NaN != NaN, so existing gaps are not "flat"
    if not same.any():
        return x
    d = np.diff(np.concatenate(([0], same.astype(np.int8), [0])))
    starts, stops = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    long_ = (stops - starts + 1) >= min_run
    if not long_.any():
        return x
    pad = int(round(_FLAT_PAD_S * fs))
    mask = np.zeros(x.shape[0] + 1, dtype=np.int32)
    for a, b in zip(starts[long_], stops[long_], strict=True):
        mask[max(0, a - pad)] += 1
        mask[min(x.shape[0], b + 1 + pad)] -= 1
    out = x.copy()
    out[np.cumsum(mask[:-1]) > 0] = np.nan
    return out


@dataclass(frozen=True)
class _Detect:
    data: np.ndarray
    fs: float
    start_uutc: int | None


def _at_detection_rate(tr: Trace) -> Trace:
    """The trace at the rate the DETECTION band needs (R10.1).

    Stage 1 correlates the detection band (≤ ``LOWPASS_HZ``), so anything beyond a few times
    that edge is oversampling: at 256 Hz the matched filter ran 16× more samples than the band
    carries, which put a day-long search at over a minute. Resampling to
    ``_DETECT_OVERSAMPLE × LOWPASS_HZ`` (anti-aliased, zero-phase) changes nothing the band
    contains; the offset resolution becomes 1/fs (31 ms at 32 Hz), far inside Stage 1's ±1 s
    remit — Stage 2 refines to the sample. Gaps are bridged before resampling (as conditioning
    does) and restored as NaN wherever a block is mostly gap.

    Flat stretches (:func:`mask_flat`) are treated as gap first, so a stimulation period — rail
    saturation on one amplifier, blanking on the other — contributes nothing to either side of
    the correlation instead of a large, system-specific transient.
    """
    x = mask_flat(np.asarray(tr.data, dtype=np.float64), tr.fs)
    if LOWPASS_HZ is None:
        return _Detect(x, tr.fs, tr.start_uutc)
    q = int(tr.fs // (_DETECT_OVERSAMPLE * LOWPASS_HZ))
    if q < 2:
        return _Detect(x, tr.fs, tr.start_uutc)
    valid = np.isfinite(x)
    if not valid.any():
        return tr
    if not valid.all():
        idx = np.arange(x.shape[0])
        x = np.interp(idx, idx[valid], x[valid])
    # Band-limit and resample in the frequency domain: one multithreaded FFT, the spectrum cut at
    # the new Nyquist (an ideal anti-alias), one small inverse. Output sample j sits at input
    # sample j·q, as a polyphase resampler's would. The circular wrap touches only the first and
    # last few samples of a recording hours long — immaterial to a whole-recording template search.
    n = (x.shape[0] // q) * q
    n_out = n // q
    spec = rfft(x[:n], workers=-1)
    y = irfft(spec[: n_out // 2 + 1], n_out, workers=-1) * (n_out / n)
    nb = y.shape[0]
    v = np.zeros(nb * q, dtype=np.float64)
    v[: min(nb * q, valid.shape[0])] = valid[: nb * q]
    gap = v.reshape(nb, q).mean(axis=1) < 0.5
    y[gap] = np.nan
    return _Detect(y, tr.fs / q, tr.start_uutc)


def _prepare(long_data: np.ndarray, m: int) -> tuple[np.ndarray, int]:
    """One rFFT of the long trace, reused by every template (R10.1 interactive budget)."""
    nfft = next_fast_len(long_data.shape[0] + m - 1)
    return rfft(long_data, nfft, workers=-1), nfft


def _long_stats(long_: Conditioned, m: int) -> tuple[np.ndarray, np.ndarray]:
    """``(sqrt of the local variance, valid fraction)`` of the long trace under an ``m``-sample
    window at every shift. Template-independent: computed once, shared by every template."""
    s = long_.data
    c1 = np.concatenate(([0.0], np.cumsum(s)))
    c2 = np.concatenate(([0.0], np.cumsum(s * s)))
    s1 = c1[m:] - c1[:-m]
    s2 = c2[m:] - c2[:-m]
    var = s2 - (s1 * s1) / m
    np.maximum(var, 0.0, out=var)
    cv = np.concatenate(([0.0], np.cumsum(long_.valid.astype(np.float64))))
    valid_frac = (cv[m:] - cv[:-m]) / m
    return np.sqrt(var), valid_frac


def _ncc_surfaces(
    short: Conditioned,
    starts: list[int],
    m: int,
    long_: Conditioned,
    long_fft: np.ndarray,
    nfft: int,
    long_stats: tuple[np.ndarray, np.ndarray],
) -> list[np.ndarray | None]:
    """:func:`_ncc_surface` for every template, with the FFTs batched into one multithreaded
    2-D transform (identical results; the templates are independent)."""
    n_pos = long_.data.shape[0] - m + 1
    if n_pos < 1:
        return [None] * len(starts)
    rows, norms, keep, masks = [], [], [], []
    for i0 in starts:
        w = short.valid[i0 : i0 + m]
        t = short.data[i0 : i0 + m].astype(np.float64)
        gapped = bool(w.mean() < 0.999)
        # Mean over the valid samples only; masked samples stay exactly 0 (no contribution).
        t = np.where(w, t - t[w].mean(), 0.0) if gapped else t - t.mean()
        nrm = float(np.linalg.norm(t))
        keep.append(nrm > 0.0)
        if nrm > 0.0:
            rows.append(t[::-1])
            norms.append(nrm)
            masks.append(w[::-1].astype(np.float64) if gapped else None)
    out: list[np.ndarray | None] = [None] * len(starts)
    if not rows:
        return out
    T = rfft(np.vstack(rows), nfft, axis=1, workers=-1)
    T *= long_fft[None, :]
    corr = irfft(T, nfft, axis=1, workers=-1)[:, m - 1 : m - 1 + n_pos]
    sd, valid_frac = long_stats
    bad = valid_frac < _MIN_VALID_FRACTION
    masked = _masked_denominators(rows, masks, long_, m, nfft, n_pos)
    j = 0
    for k, ok in enumerate(keep):
        if not ok:
            continue
        if masks[j] is None:
            denom, reject = norms[j] * sd, bad
        else:
            denom, reject = masked[j]
        with np.errstate(invalid="ignore", divide="ignore"):
            ncc = np.where(denom > 1e-12, corr[j] / denom, 0.0)
        ncc[reject] = 0.0
        out[k] = ncc
        j += 1
    return out


def _masked_denominators(
    rows: list[np.ndarray],
    masks: list[np.ndarray | None],
    long_: Conditioned,
    m: int,
    nfft: int,
    n_pos: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """NCC denominators for templates with masked samples (gap, flat, stimulation).

    A gapped template is correlated over the samples valid in BOTH traces at each shift — the
    masked normalised cross-correlation — so its energy and the long trace's energy are summed
    over that overlap only, and shifts whose overlap is under ``_MIN_VALID_FRACTION`` of the
    template's valid samples are rejected. Four extra correlations per gapped template; a clean
    template keeps the exact sliding-window denominator.
    """
    idx = [j for j, w in enumerate(masks) if w is not None]
    if not idx:
        return {}
    lv = long_.valid.astype(np.float64)
    lfft = rfft(np.vstack([lv, long_.data * long_.data]), nfft, axis=1, workers=-1)
    t2 = rfft(np.vstack([rows[j] ** 2 for j in idx]), nfft, axis=1, workers=-1)
    wf = rfft(np.vstack([masks[j] for j in idx]), nfft, axis=1, workers=-1)  # type: ignore[misc]
    sl = slice(m - 1, m - 1 + n_pos)
    e_t = irfft(t2 * lfft[0][None, :], nfft, axis=1, workers=-1)[:, sl]   # template energy
    e_l = irfft(wf * lfft[1][None, :], nfft, axis=1, workers=-1)[:, sl]   # long energy
    n_ov = irfft(wf * lfft[0][None, :], nfft, axis=1, workers=-1)[:, sl]  # overlap count
    out = {}
    for r, j in enumerate(idx):
        n_valid = float(masks[j].sum())  # type: ignore[union-attr]
        denom = np.sqrt(np.maximum(e_t[r], 0.0) * np.maximum(e_l[r], 0.0))
        out[j] = (denom, n_ov[r] < _MIN_VALID_FRACTION * n_valid)
    return out


def _ncc_surface(
    short: Conditioned,
    i0: int,
    m: int,
    long_: Conditioned,
    long_fft: np.ndarray,
    nfft: int,
    long_stats: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray | None:
    """Normalized cross-correlation of one template against every position in the long trace.

    Exact NCC, not raw correlation: the local energy of the long trace under the template is
    divided out at every shift (via cumulative sums), so a loud stretch of the recording cannot
    win simply by being loud. Positions where the long trace is mostly gap score 0.
    """
    t = short.data[i0 : i0 + m].astype(np.float64)
    t = t - t.mean()
    t_norm = float(np.linalg.norm(t))
    if t_norm <= 0.0:
        return None

    n = long_.data.shape[0]
    n_pos = n - m + 1
    if n_pos < 1:
        return None

    # corr[k] = sum_p t[p] * s[k + p], via one multiply in the frequency domain.
    T = rfft(t[::-1], nfft, workers=-1)
    corr = irfft(long_fft * T, nfft, workers=-1)[m - 1 : m - 1 + n_pos]

    # Local energy of s under the window at every shift (shared across templates).
    sd, valid_frac = long_stats if long_stats is not None else _long_stats(long_, m)
    denom = t_norm * sd

    with np.errstate(invalid="ignore", divide="ignore"):
        ncc = np.where(denom > 1e-12, corr / denom, 0.0)

    # Zero out positions where the long trace is mostly gap — a window of zeroed gap samples
    # has low energy and could otherwise produce a spuriously high normalized score.
    ncc[valid_frac < _MIN_VALID_FRACTION] = 0.0

    # NOT clipped to >= 0. Negative correlations are real, and they are half of the null
    # distribution: clipping them to zero made 50% of the surface *exactly* 0, which drove its
    # median and MAD to ~1e-5 and turned "height above background" into thousands of sigma for
    # any value at all. That silently disabled the check meant to catch a globally wrong offset.
    return ncc
