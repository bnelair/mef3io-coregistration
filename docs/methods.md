---
description: "What mef3io-coregistration computes at each alignment stage, the number each stage reports, the rule it decides by, and how it fails."
---

# Methods — what is computed at each step, and what every number means

A walk through the pipeline in the order you meet it. For each stage: the signal it runs on, the
algorithm, the number it reports, the rule it uses to decide, and how it fails.

Every threshold quoted here was measured, not chosen: on clinical test pairs and on synthetic
pairs with a known clock offset (`mef3io_coregistration.synthetic`). Figures from synthetic pairs
are given as measured; results on clinical pairs are described without their specifics.

Every step is a method of
[`CoregSession`](api/python.md#mef3io_coregistration.session.CoregSession)
(see [Align two recordings in Python](howto/align_python.md)). A front end
(an interactive viewer) can drive the same methods; where this page mentions
one, the Python call is named alongside.

**Sign convention, used everywhere without exception:**

> `offset = t_moving − t_ref`

A positive offset means an event happens *later* on the moving clock. The reference recording is
never resampled or transformed; only the moving one is warped onto it.

---

## 0. Loading — the two derived products

Alignment never runs on the raw full-rate signal across full length. Each selected channel is
read once, in ~5-minute chunks, into two products held in memory:

| product | rate | built from | used for |
|---|---|---|---|
| **min-max envelope** | ~50 Hz | the **raw** signal | **display only** |
| **decimated waveform** | the lower of the two files' Fs | anti-aliased polyphase decimation | **every alignment stage** |

The envelope keeps each bin's true minimum and maximum, so spikes and saturation stay visible
when you are zoomed out. It is **not time-accurate within a bin** (the min and max are drawn at
the bin boundary, not where they happened), so a front end should say so when it draws one.

**The envelope is not an alignment feature and must never be used as one.** It summarises each
file's raw signal at *that file's own bandwidth*, so a 20 ms bin of an 8 kHz channel means
something physically different from the same bin of a 256 Hz channel: on a clinical test pair
the two bin ranges differed by more than an order of magnitude, and correlating them produced a
Stage 1 estimate **hours** from the truth.

The decimation is **group-delay compensated** (zero-phase), so sample timing is preserved. If it
were not, every lag this tool measures would be biased by the filter.

---

## 1. Conditioning — applied before every correlation

The two amplifiers agree on nothing: a high-rate DC-coupled recorder can sit at a large DC offset
with stimulation excursions orders of magnitude above the physiology, while the same derivation
on an AC-coupled 256 Hz clinical system is centred near zero — a large amplitude difference plus
a large slow component on one system only. Correlating that
directly lets a DC ramp on one machine dominate the physiology on the other.

In order:

1. **Bridge gaps.** `NaN` samples are linearly interpolated *before* filtering, so the IIR does
   not ring off a step edge, then re-zeroed afterwards. A `valid` mask records where the signal
   was real.
2. **Filter** (zero-phase `filtfilt` throughout — a causal filter's group delay would shift the
   trace and bias the very lag being measured):
   - high-pass, low-pass, and an optional **notch** which also removes that frequency's
     harmonics below Nyquist (a stim comb is not a single line).
3. **Robust scale** by the MAD, not the standard deviation. A few large stimulation excursions would
   set σ and crush the physiology to nothing.
4. **Clip** at `clip_mad` robust sigmas. Bounds saturation without discarding the window.

The output is dimensionless (units of robust sigma), so the two systems are directly comparable.

**The filters are yours, and they travel with each request** — the view you are looking at and
the calculation you launch use the same spec. They are deliberately not a session-wide mode: a
stretch buried in stim artifact may only be alignable with a tight low-pass and a notch, while the
same recording an hour later aligns best wide open.

### Detection band vs estimation band

Two different jobs, two different bands, and conflating them is what broke Stage 1 originally:

| | band | why |
|---|---|---|
| **Detection** (Stage 1) | **0.5–8 Hz** | Searches the whole recording and must produce ONE unambiguous peak. Two amplifiers with different references share their *low-frequency* physiology; above ~10 Hz the content is reference noise, amplifier noise and stim, uncorrelated between systems, and it only dilutes the match. |
| **Estimation** (Stages 2–3) | **0.5–20 Hz** (default) | Searches a narrow seeded range, so ambiguity is not the risk — timing precision is, and an 8 Hz-limited trace cannot resolve a 4 ms sample. |

Measured effect of the band on Stage 1 (4 templates, a clinical test pair):

| band | recovered offset | templates agreeing |
|---|---|---|
| 0.5–40 Hz | **hours wrong** | 0/4 |
| 0.5–20 Hz | **hours wrong** | 0/4 |
| **0.5–8 Hz** | **within 0.1 s of the truth** | **4/4** |

---

## 2. Stage 1 — Gross: *where is the offset, to within a second?*

**Input:** the decimated waveform, full length, conditioned on the **detection** band — and
first brought to the rate that band needs: 4 × `LOWPASS_HZ` (32 Hz for the 8 Hz edge), by FFT
resampling with gaps bridged first and restored after. Everything the detection band contains
survives; the offset resolution becomes 1/32 s, far inside Stage 1's ±1 s remit (Stage 2 refines
to the sample). At 256 Hz the search otherwise ran 8× more samples than the band carries: a
day-long gross took **66 s**; now **~3 s** (4 templates batched in one FFT, the long trace's
local energy computed once instead of per template).

**Algorithm — stacked multi-template matched filter.**

0. **Mask what carries no signal.** A run of identical samples ≥ 0.1 s (plus 0.25 s either side
   for the recovery transient) is treated as gap on both traces: an amplifier pinned at its rail
   by stimulation, or a stimulator blanking the input. Physiology is never exactly constant for
   0.1 s. Saturation on one system against blanking on the other then contributes nothing to
   either side of the correlation, instead of a large system-specific transient.
1. Cut **4 templates of 600 s** from the **shorter** recording, evenly spaced. A slot whose
   position is mostly gap, flat or masked moves to the cleanest nearby position within its share
   of the recording before it is given up. (Templates come from the shorter file because a
   template from the long one usually has no counterpart in the short one at all.)
2. Matched-filter each template against the **whole** of the longer recording — an exact
   normalized cross-correlation, with the long trace's local energy divided out at every shift, so
   a merely *loud* stretch cannot win by being loud. A template with masked samples is correlated
   over the samples valid in **both** traces at each shift (masked NCC: both energies summed over
   that overlap), so a template may be three-quarters stimulation and still vote with its clean
   minutes; shifts where the overlap is under 60 % of its valid samples are rejected. Measured on
   a synthetic pair stimulated 8 of every 12 minutes (rail on one side, blanking on the other):
   4/4 templates agree with masking, 2/4 without.
3. Express every template's surface against the same offset axis and **stack** them. The true
   offset gets a contribution from every template and adds coherently; each template's spurious
   peaks sit at unrelated offsets and do not. Normalised by **√coverage**, not coverage, so the
   background noise is uniform across the search.

**The numbers it reports**

| number | what it is |
|---|---|
| `offset_s` | the argmax of the stacked surface |
| `n_agree / n_templates` | how many templates independently put their own best match within `GROSS_CONSENSUS_S` (5 s) of it |
| `snr` | height of the stacked peak in robust sigmas above the surface's own background |
| `ncc` (per template) | the raw normalized cross-correlation at that template's best match |
| `spread_s` | how far apart the agreeing templates landed — the estimate's own error bar |

**The decision rule: `locked` = a majority of templates agree (≥ 2, and more than half).**

**Peak height is not the criterion, and must never become it.** Measured across true and
deliberately false pairs, the detection SNR of true and false locks *overlaps*, and a false pair
can out-score a genuine lock. Peak
height is an absolute measure taken on surfaces that are not comparable between datasets. Template
agreement separates the cases perfectly (true 4/4, false 0/4), and for a reason that should
generalise: several 10-minute templates from different hours carry independent physiology; each
can be fooled, but not into the *same* wrong answer. `GROSS_MIN_SNR` only decides whether a
successful lock is annotated "locked, but weakly — confirm by eye". It has no vote.

**If it does not lock, the stage says so — and is not a dead end.** Stimulation, or long
artifact, can leave too few clean templates to corroborate an offset that is in fact right. The
user may **continue with the best estimate** (`CoregSession.accept_gross()`), or give the offset
by hand (`CoregSession.manual_offset()`, below). Either way the decision is recorded
(`accepted_by = "user"`) and carried in the saved result's QC (`stage1_accepted_by`), and the
Fine anchors are what confirm or correct it: Stage 2 searches the neighbourhood of the ROI, not
just the seed.

---

## 3. Verification — *is the offset I was handed right at all?*

Runs when you hand-place an offset, and again on the fitted map after Stage 3.

This is a **different question** from Stage 1's, and the one the pipeline used to assume the
answer to. Every statistic in Stages 2 and 3 is computed on a correlation surface only ±0.25 s
wide — the seeded search range — so it measures whether the peak is sharp *near where it was
pointed*, and says nothing about whether it was pointed in the right place. Measured on a
clinical test pair, that confidence was **higher at wrong offsets (a minute, an hour) than at the
true one**.

At a wrong offset two unrelated traces throw up one sharp random blip, which is very "distinct";
the true offset has a broad genuine ridge, which is less distinct against its own shoulders. So an
hour-wrong hand placement used to sail through: most drift windows accepted, a smooth drift
curve, and no warning anywhere.

**`verify_offset` asks the question those statistics cannot, on the full-length surface where it
is answerable: do independent templates put their best match at this offset?** That is the lock
criterion, seeded — and it has no threshold to tune.

| candidate | templates supporting | verified |
|---|---|---|
| true offset | 4/4 | ✓ |
| 1 minute wrong | 0/4 | ✗ |
| **1 hour wrong** | **0/4** | **✗** |

`CoregSession.manual_offset(offset_s)` runs this check. A placement the templates support is a
lock; one they do not support is recorded as unconfirmed and is not a lock, so the later stages
refuse to build on it — unless `force=True`, which makes the override explicit and records it in
the reason. A front end may choose to always accept a hand placement and show the verdict
instead; the verdict travels in the saved QC either way.

**Your placement is what is used.** Verification decides whether to accept it; it does not move
it. Where the templates disagree, their own estimate is reported
(`OffsetVerification.refined_offset_s`) for you to adopt deliberately, or not.

---

## 4. Stage 2 — Fine: *the offset at one place, to a sample*

**Input:** one ROI of the decimated waveform, conditioned on **your** filters.

**Algorithm.** Cut a window of `half_window_s` either side of the ROI centre from the reference,
and the same window from the moving trace at the **seeded** position (`roi_centre + seed`).
Normalized cross-correlation between them, bounded to ±`max_residual_s`, with **parabolic
sub-sample peak interpolation**. The anchor's offset is `seed + measured residual`.

**The seed is the offset you pass** (`run_fine(roi, seed_offset_s=...)`; by default the current
Stage 1 offset) — the offset at that ROI as you see it, not a stale stored value. A refinement stage
whose search is bounded cannot reach past its own seed's error: seed it 3 s off with a ±0.5 s
search and it returns a confidently wrong anchor 3 s away, rather than correcting anything.

**Unless you placed it by hand, the seed is first refined locally.** After an automatic lock the
default seed is Stage 1's *single* offset for the whole recording — and over a day the clocks
drift by seconds (the consensus tolerance `GROSS_CONSENSUS_S` exists to absorb exactly that). At
an ROI 6 h into a ±5 s day the lock was 0.5 s from the truth there, and Stage 2 returned the best
peak *inside* its ±0.5 s search: **604.51 s for a true 605.00 s, confidence 0.45** — confident and
wrong (a synthetic pair). So before refining, `local_seed` correlates ±10 min around the ROI on the **detection** band,
searching ±2×`GROSS_CONSENSUS_S` around the seed, and moves the seed there only if that peak is
distinctive and not on the search boundary; otherwise the seed is used as given. A seed you placed
by hand is used as-is when you say so: `run_fine(roi, seed_offset_s=..., refine_seed=False)`
(a front end that lets you drag the traces into place should pass exactly that).

**Window length is yours.** Long windows suit ongoing background; a *short* one (a few seconds) is
right when the shared feature is a discrete event — a seizure, a stim train, a large discharge —
where the information lives in those seconds and the surrounding minutes are uncorrelated noise
that only dilutes the correlation. On a clinical test pair, anchors across the recording agreed
to within a few tens of milliseconds even at a 5 s window, close to what 300 s windows gave. Only the upper bound
is enforced (~30 min), and for an unrelated reason: past that the offset drifts measurably *within*
the window and smears the peak the window exists to sharpen.

**The numbers it reports**

| number | what it is | how to read it |
|---|---|---|
| `offset_s` | `seed + residual lag` | the anchor |
| `confidence` | **peak distinctiveness**: `1 − (best rival outside a guard band) / (best peak)` | "was there one clear winner?" — high when the lock is unambiguous |
| `ncc` | raw normalized cross-correlation at the chosen lag | "how similar are these two traces?" — informational |

**`confidence` is distinctiveness, not similarity, and the distinction is load-bearing.** The raw
NCC *falls* as a window lengthens — two different amplifiers accumulate more independent content —
even while the lag estimate becomes far more reliable. The original tool thresholded on raw NCC,
so it rejected precisely the estimates that were best: at a 40 s window it returned an answer
**5.0 s wrong with the highest confidence of any window length**.

Note the limit stated in §3: distinctiveness is computed on the ±0.25 s seeded surface, so it
cannot tell you the seed was globally wrong. That is verification's job, not this number's.

---

## 5. Stage 3 — Drift: *the offset everywhere*

**Input:** the full decimated waveform, conditioned on **your** filters.

**Algorithm.** A grid of window centres (default 20 s windows every 30 s) is walked **outward from
every anchor**: each window belongs to its nearest anchor, and the walk runs forward and backward
from it. Each window is searched narrowly (±`max_residual_s`) around a **prediction** — a Theil–Sen
line (median slope, median level) through the recent accepted windows, its slope clamped to a
physical bound (`DRIFT_MAX_SLOPE`, 3.6 s/h) — so it locks onto the *nearby* peak and cannot
track-jump. A window advances the walk only if it lands near that prediction (within a tolerance
scaled to the walk's own scatter), and a peak **on the search boundary** is rejected outright: the
true peak lies outside the search.

*Why from the anchors, not from the start of the recording.* The walk used to start at the first
window with the anchor's offset as its seed. When the anchor sat far from the start, the true
offset there was already further from that seed than the search reaches; the walk locked onto a
side peak and followed it — **a constant 0.5 s error with 229 of 240 windows accepted** (2 h,
5 s/12 h drift, anchor near the end). Every anchor is ground truth for its neighbourhood, so the
walk starts there. All anchors are used; a later anchor at the same time supersedes an earlier one.

Then:

1. **Reject gap-overlapping windows** outright (>20 % gap). Never correlate against interpolated
   zeros.
2. **Reject windows whose peak is not distinctive** (`confidence < min_confidence`). These do not
   advance the seed either.
3. **Reject local outliers** — windows that disagree with their neighbours by more than 4 robust
   sigmas. Drift is smooth by assumption, so a window that disagrees with its neighbours is wrong
   *no matter how confident it is*, and this is what stops a seeded chain from walking off.
4. **Fit a weighted smoothing spline** through the survivors, weighted `√confidence / σ`, where σ
   is the **measured** per-window scatter. That makes the objective a chi-square, so the smoothing
   budget `s = n` is principled rather than arbitrary.

The old default asked the spline to pass within 0.3 *samples* (≈1 ms) of every window. On real
data a single 20 s window locates the lag far less precisely than that, so the spline chased the
noise: on a clinical test pair it swung several times further than the true drift.

**Map anchors are built only from ACCEPTED windows.** Building them from every window centre — and
then reporting the full span as "measured" — is extrapolation dressed up as measurement.

**Stage 2 anchors seed the walk but are not fitted.** Their residual against the fitted map is
reported (`anchor_residual_s`): a wrong anchor shows up there instead of bending the map.

**Outputs:** a monotone clock map (`dy/dx > 0` is *enforced*, so `inverse()` is always valid), plus
per-window arrays for QC. Every array is finite; a window that could not be estimated carries its
seed and is marked rejected, never `NaN`.

---

## 6. Bypassing Stage 3

On short recordings — or a short overlap — the clocks have no room to drift, and a sliding-window
fit does not measure drift: it fits the estimator's own scatter and presents the wiggle as a clock.
`CoregSession.build_map_without_drift()` builds it from what is actually known:

- **no anchors** → a constant offset from Stage 1
- **one anchor** → a constant offset at that anchor's more precise value
- **two or more** → a straight line *through the anchors*, which absorbs a constant difference in
  clock **rate** without ever estimating drift

Outside the outermost anchor the offset is **held, not projected**. The assumption is written into
the saved map's provenance, and everything outside the anchored span is flagged **extrapolated**
(`ClockMap.measured_span`) even though the map's knots reach the ends of the recording. The map is
still verified (§3).

---

## 7. QC — what to look at before you trust the map

What a saved result carries, from `CoregSession.result()`:

| where | number | what it means |
|---|---|---|
| `qc` | **`map_verified`** | **the one that matters.** Do independent templates find the shared signal where the map says it is (§3, run on the fitted map)? If false, everything else describes a map that is in the wrong place. |
| `qc` | `verification` | the verification verdict in words: how many templates support the map, and how clearly |
| `qc` | `stage1_accepted_by`, `stage1_reason` | `"templates"` when Stage 1 locked by agreement, `"user"` when someone continued or placed the offset by hand, and why |
| `clock_map.metadata` | `measured_span_uutc` | the span between the first and last **accepted** window (or the anchors, without Stage 3). Not the recording: outside it the map is extrapolated, and reads flag those samples (`UnifiedRead.extrapolated`, `ClockMap.is_extrapolated`). |
| `clock_map.metadata` | `end_offset_slopes` | beyond the measured span a Stage 3 map continues with **one drift rate per end**, fitted by least squares over the first/last fifth of the accepted windows (2–20): the map applies it and the plotted curve follows it. |
| `clock_map.metadata` | `n_accepted / n_windows`, `n_gapped` | how much of Stage 3 survived, and how much was lost to gaps |
| `drift` | `offset_s[]`, `smoothed_offset_s[]`, `confidence[]`, `accepted[]` | every drift window, for plotting the fit against what it was fitted to |

`DriftEstimate.anchor_residual_s` (not saved) is each Stage 2 anchor's distance from the fitted
map: a wrong anchor shows up there. It is self-consistency only — the anchors seeded the walk —
and can look excellent on a map that is globally wrong; that is what `map_verified` is for.

A front end may pass a fuller QC report to `CoregSession.result(qc=...)`, for example a
**leave-one-out residual** (refit without each anchor, then predict it; `MAX_LOO_RESIDUAL_MS` in
`config` is a suggested bound for it). The engine does not compute one itself; the
[review report](howto/review_report.md) shows it when the result carries it, and makes its own
independent check instead: the **residual lag** re-measured in fresh windows after
coregistration (figure 5).

The reader adds one more: `Gap.grid_offset_us` — where a recording resumes after a gap a
fraction of a sample off its sampling grid, the samples after it are stamped on the grid, off by
this much (at most half a sample; warned above 1 %).

**And then look at the signal.** Read a few seconds of both files on the reference clock
(`CoregisteredReader.get_data(..., coregister=True)`, or the review report's multi-scale figure),
filtered to the alignment band, and check that the two traces sit on top of each other. Every number above can be satisfied by a map that is
subtly wrong; the overlay is the only check that cannot.

---

## 8. Using the two files as one recording

Once the map exists, `CoregisteredRecording` (`CoregisteredRecording.open(result)`, or
`CoregSession.open_recording()` before saving) serves both files as a single recording on the
reference clock.

- **Logical channels.** Each names its signal in file A and/or file B — a channel, or an
  `anode − cathode` pair, chosen *per file*: the two systems need not share names or montage. By
  default every channel name becomes a logical channel, linked across the files when both carry it
  — exactly, or else after normalisation (system prefixes such as `EEG `, reference suffixes such
  as `-Ref`, case and spacing dropped: `EEG LA1-Ref` ↔ `LA1`), and only when the normalised name
  is unique in both files. Anything else is linked by hand.
- **Per-source correction.** Each source carries `scale` and `offset` (value′ = scale·value +
  offset; scale −1 inverts). The two amplifiers can differ in gain, sign and units; the correction
  is how the user states that they are the same signal.
- **Merge gate — may the two sources stand in for each other?** Before a channel falls back from
  one file to the other, the corrected sources are compared over their overlap
  (`MERGE_CHECK_CHUNKS` × `MERGE_CHECK_CHUNK_S` spread across it, both band-passed
  `HIGHPASS_HZ`–`FINE_LOWPASS_HZ` so DC and bandwidth do not decide). The fallback is refused
  unless the correlation is at least `MERGE_MIN_CORR` (a clearly negative one is reported as
  *opposite polarity*), the robust gain ratio lies within `MERGE_GAIN_RANGE`, and the units match.
  A refused channel serves its preferred file only, with a note saying why and what to change;
  `allow_mismatch` overrides the gate deliberately. Single-source channels are not gated.
  **The baseline (DC) is reported, not gated** (`offset_b_minus_a`, `baseline_step`): a
  DC-coupled amplifier's baseline can wander by more than its signal from one minute to the
  next, so no constant offset fits and gating on it would
  refuse sources that agree. When the baselines differ by more than the signal's spread the
  verdict says so ("agree · step"): expect a step at each switch, and filter each run
  (`UnifiedRead.runs()`) on its own.
- **Preferred source, with fallback.** A read serves the preferred file wherever it has data, and
  the other file where it has none (outside its recording, or in a gap). Every sample reports which
  file it came from; samples drawn from the moving file where the map is not measured are flagged
  **extrapolated**.
- **The grid** is the preferred source's own sampling grid on the reference clock; the reference
  file at its native rate is returned untransformed. Moving-file samples are resampled through the
  map (anti-alias decimation when needed, ×4 band-limited oversampling, cubic interpolation; gaps
  are never interpolated across).
- **Display.** `CoregisteredRecording.display()` gives a front end min/max buckets per
  source: zoomed out, from a cached ~50 Hz min–max envelope (extremes right, timing within a bin
  not — it should say so); zoomed in, from the exact resampled samples. A front end should give
  each source its own display gain (two amplifiers can differ by orders of magnitude, and one gain
  flattens one side into a line), and mark each switch between files. Display is a **navigation
  aid**: analysis reads the data with `get_data`.
- **Runs.** `UnifiedRead.runs()` returns the contiguous single-source stretches. Filter, detect
  or measure each run on its own: a statistic spanning a switch mixes two amplifiers (bandwidth,
  hardware filters, reference, stim handling).
- **Time conversion.** `to_file_time(t, file)` and `to_reference_time(t, file)` convert event
  times between the reference clock and either file's own clock, through the map.
- **Provenance.** The saved result records each MEF3 file's fingerprint (SHA-256, channel count,
  start and end). Opening a result checks each file only by its time span — start and end within
  1 s of the stored ones — and refuses a file that matches the other role's span better (A/B
  swapped); `verify=False` overrides. Recordings with identical spans cannot be told apart.
- **Persisted** as a small JSON config next to the saved clock map
  (`CoregisteredRecording.from_config`). The config stores the result file's SHA-256, so a
  mapping is never silently reapplied to a clock map that has since been overwritten.

---

## Failure modes, and what they look like

| symptom | cause | what to do |
|---|---|---|
| Stage 1 will not lock | templates disagree — genuinely weak shared signal, stimulation, wrong channel, or opposite polarity | flip a file's polarity (`set_polarity("moving", True, "gross")`) and re-run; try another channel for Stage 1; or continue with the best estimate / align by hand and let Fine anchors confirm it |
| Stage 1 locks somewhere absurd | should not happen — agreement is what prevents it. If it does, the templates all matched the same artifact | narrow the band; verify by eye |
| Refine jumps somewhere unreasonable | the seed was too far from the truth for the bounded search | pass a better seed (`run_fine(roi, seed_offset_s=...)`, e.g. read off an overlay), or widen `max_residual_s` |
| Anchor has high `ncc` but low `confidence` | the two traces are similar *everywhere* in the window — no unique lag | pick an ROI with a distinctive event |
| Drift rejects nearly every window | shared signal too weak on this channel, or the anchor is wrong | check the anchor; change channel or filters |
| Drift curve wiggles far more than the physical drift | you are fitting measurement noise | there is probably nothing to fit — bypass Stage 3 |
| Everything green, overlay clearly misaligned | trust the overlay | re-align by hand (`manual_offset`); the QC numbers cannot see a globally wrong offset that verification did not catch |
