---
description: "How mef3io-coregistration is put together: the layers, the invariants they keep, and why."
---

# Design & internals

## Layers

The package is three layers, each usable without the ones above it.

| Layer | Modules | Needs | Does |
|---|---|---|---|
| **Reader core** | `result`, `transform`, `io`, `montage`, `reader`, `config` | numpy, scipy, mef3io | load a result, read either file or both through its clock map |
| **Alignment** | `align`, `session`, `synthetic` | the reader core | estimate the clock map from two recordings, produce a result |
| **Review** | `review` | the above, matplotlib (`report` extra) | the plotted report |

The reader core never imports the alignment pipeline, so an analysis that
only reads coregistered data loads none of it. All MEF 3.0 access goes
through `io.mef_reader.MefChannelReader`, a thin wrapper over
`mef3io.Reader`; nothing in this package parses MEF bytes.

## Time

- Every time is **absolute uUTC** as an integer (microseconds since the
  Unix epoch), as MEF 3.0 stores it. Arithmetic on the clock map runs on
  offsets from its first knot, so interpolation keeps microsecond precision
  at ~1.7 × 10¹⁵.
- **Sign convention, everywhere:** `offset = t_moving − t_reference`.
- Reads are **half-open** `[t0, t1)` and snapped to the channel's sample
  grid first, so consecutive reads tile without overlap or loss, on either
  clock.

## The clock map

`transform.ClockMap` is a pair of strictly increasing integer arrays,
reference-clock knots `x_ref` and moving-clock knots `y_moving`, with linear
(default) or monotone-cubic interpolation and an end-slope extrapolation
policy. Strict monotonicity is enforced on construction, so `inverse()` is
always defined. The map knows its **measured span**; anything outside it is
reported as extrapolated. One map serves every channel of a file: drift is a
property of the two recording systems, not of a channel.

## Reading

- The **reference file is never transformed**: read on the reference clock it
  returns mef3io's samples as they are, unless a different output rate is
  asked for.
- The **moving file is resampled**, not re-stamped: each output time on the
  reference clock is mapped to the moving clock and the moving signal is
  evaluated there, band-limited (anti-alias decimation when the output rate
  is lower, ×4 oversampling, cubic interpolation). Gap samples are never
  interpolated across; an output sample whose filter support touches a gap
  is `NaN`.
- A **bipolar** channel is always derived on the file's own samples first and
  resampled after, never the other way round.
- Reads fetch only the blocks the window needs, plus a short margin for the
  interpolation filter.
- `CoregisteredRecording` serves logical channels from a preferred file and
  fills from the other only when the two sources agree over their overlap
  (correlation, robust gain ratio and units; the baseline is reported, not
  gated).

## Opening a result

The result is only the transformation; the recordings are found by path
(overrides first, then the stored path, then next to the result), and each is
matched to its role **only by its time span**, within 1 s. A file that
matches the other role's span is refused as swapped. The reference role
always comes from the result.

## Alignment

- Alignment runs on a **decimated, group-delay-compensated** copy of one
  channel per file, loaded full length in chunks (`io.chunked`), never on the
  full-rate signal and never on the display envelope.
- Every correlation runs on a **conditioned** signal: gaps bridged, zero-phase
  filtered, scaled by the MAD, clipped. Detection (Stage 1) and estimation
  (Stages 2 and 3) use different bands.
- **Confidence is peak distinctiveness**, `1 − runner-up / best`, not raw
  correlation height.
- **Stage 1 locks by agreement** of independent templates, not by peak
  height; a stage that has not locked refuses to let later stages build on it,
  unless the user explicitly continues, which is recorded.
- **Stage 3 walks outward from every anchor**, each window seeded by a
  slope-clamped prediction, and fits a weighted smoothing spline through the
  accepted windows only.

[Methods](methods.md) gives the measurements behind each of these choices.

## Persistence

Results, clock maps and recording mappings are sorted-key JSON with integer
times, so save → load → save is byte-stable. Their format identifiers are
kept stable across versions; see [Result format](result_format.md).
