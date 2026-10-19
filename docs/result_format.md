---
description: "The JSON a mef3io-coregistration result, its clock map and a recording mapping are saved as: every field, its units, and the compatibility rules."
---

# Result format

A coregistration is saved as one JSON file (by convention `*.coreg.json`),
written by [`CoregResult.save`](api/python.md#mef3io_coregistration.result.CoregResult)
and read by `CoregResult.load` and every reader. It holds no signal: only the
clock map, the times and metadata of the two recordings, and how the map was
made. Keys are written sorted, with 2-space indentation, and every absolute
time is an **integer uUTC** (microseconds since the Unix epoch), so saving,
loading and saving again gives the same bytes and the same map.

The `format` identifiers below are historical and kept for compatibility:
every result and mapping saved since they were introduced opens unchanged.
They are identifiers, not package names.

## The result: `signal_coregistration.coreg_result`

| Key | Type | Meaning |
|---|---|---|
| `format` | string | `"signal_coregistration.coreg_result"`; anything else is refused |
| `format_version` | int | `2` (version 1 files, which had no `reference_role`, load with file A as the reference) |
| `tool_version` | string | the mef3io-coregistration version that made it |
| `created_utc` | string | ISO 8601 creation time, UTC |
| `reference_role` | `"A"` or `"B"` | which file is the reference (its clock is the output clock) |
| `file_a`, `file_b` | string | the two recordings' paths when the result was made |
| `reference_file`, `moving_file` | string | the same two paths, by role |
| `file_fingerprints` | object | per role (`"A"`, `"B"`): `start_uutc`, `end_uutc` (earliest channel start, latest channel end), `n_channels`, `sha256` (every channel's name, rate, start, end and sample count), `content_sha256` (stored samples at each channel's start, middle and end) |
| `clock_map` | object | the map, below |
| `anchors` | list | the Stage 2 anchors: `t_ref_uutc`, `t_moving_uutc` (int), `offset_s` (= moving − reference, s), `confidence` (peak distinctiveness, 0–1) |
| `drift` | object or null | the Stage 3 windows: `t_ref_uutc[]`, `offset_s[]`, `smoothed_offset_s[]`, `confidence[]`, `accepted[]`, `gapped[]`, `window_s`, `grid_s`, `measured_span_uutc` |
| `stage_channels` | object | per stage (`gross`, `fine`): the channel label used in the `reference` and `moving` file (an inverted one is marked) |
| `stage_filters` | object | the filters behind the map: `gross` (the detection band), `fine` (a list of `{t_ref_uutc, filters}`, one per anchor), `drift`; each filter set has `highpass_hz`, `lowpass_hz`, `notch_hz`, `notch_q`, `clip_mad`, `feature` |
| `qc` | object or null | the quality report `CoregSession.result()` writes: `map_verified` and `verification` (once the map has been verified), `stage1_accepted_by` (`"templates"` or `"user"`) and `stage1_reason`; a front end may store a fuller report with `result(qc=...)` (see [Methods §7](methods.md#7-qc-what-to-look-at-before-you-trust-the-map)) |

**Only `file_fingerprints.<role>.start_uutc` and `end_uutc` are used to
match a file** to the result (within 1 s; see
[Open a result from a pipeline](howto/pipeline.md)). The hashes are a record
of exactly which recording the map was made on; they are not checked when
opening. A result without fingerprints opens with a warning and no check.

## The clock map: `signal_coregistration.clock_map`

Embedded in the result under `clock_map`, or saved on its own by
`ClockMap.save`.

| Key | Type | Meaning |
|---|---|---|
| `format` | string | `"signal_coregistration.clock_map"` |
| `format_version` | int | `1` |
| `tool_version` | string | the version that wrote it |
| `x_ref` | list of int | knots on the reference clock, uUTC, strictly increasing |
| `y_moving` | list of int | the same instants on the moving clock, uUTC, strictly increasing |
| `interp_method` | string | `"linear"` (default) or `"pchip"` (monotone cubic) between knots |
| `extrapolation` | string | `"hold_end_slope"`: beyond the knots the map continues with the end slope |
| `metadata` | object | provenance from the stage that built the map: `stage` (`"drift"` or `"no-drift"`), `measured_span_uutc` (the span actually measured: first to last accepted drift window, or first to last anchor; without it, the knots' span). A drift map adds `end_offset_slopes` (the drift rate used beyond each end), its parameters (`window_s`, `grid_s`, `max_residual_s`, `min_confidence`, `max_gap_fraction`), the window counts (`n_windows`, `n_accepted`, `n_gapped`) and the Stage 2 anchors it was seeded from (`anchors`: a list of `{t_ref_uutc, offset_s}`, and `anchor`: the first of them, kept for files written before several anchors were supported); a no-drift map adds `method`, `n_anchors` and the `assumption` it rests on |

The map is the function *t*<sub>moving</sub> = *f*(*t*<sub>reference</sub>);
the offset at a reference time is `y − x`. Because both arrays increase
strictly, *f* is invertible everywhere, which is how moving-clock times are
brought back to the reference clock. Times outside the measured span are
reported as extrapolated (`ClockMap.is_extrapolated`).

## A recording mapping: `signal_coregistration.recording`

Written by `CoregisteredRecording.save_config`, read by
`CoregisteredRecording.from_config`: how the two files are served as one.

| Key | Type | Meaning |
|---|---|---|
| `format` | string | `"signal_coregistration.recording"` |
| `format_version` | int | `1` |
| `result` | string or null | the result file it applies to, as passed to `save_config(result_path=...)`. On opening, a relative path is first looked for relative to the mapping's folder, and a path that no longer exists is looked for by its file name next to the mapping |
| `result_sha256` | string or null | that result file's SHA-256; a result changed since is refused unless `verify=False` |
| `default_prefer` | `"A"` or `"B"` | the preferred file for channels that do not set one |
| `channels` | list | the logical channels: `name`; `a` and `b` (each `{anode, cathode}` or null: the signal in that file, bipolar when `cathode` is set); `prefer`; `fs` (output rate, null = the preferred file's); `scale_a`, `offset_a`, `scale_b`, `offset_b` (value′ = scale · value + offset); `allow_mismatch` |

## Compatibility

Results and mappings written by any earlier version keep loading: a change
that would stop an existing file from opening comes with a new
`format_version` and a reader for the old one, and is listed under
"Upgrading" in the [release notes](release_notes/index.md). New keys may be
added; readers ignore keys they do not know.
