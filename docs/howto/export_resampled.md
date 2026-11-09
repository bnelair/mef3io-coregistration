---
description: "Write a brand-new MEF 3.0 session from a coregistration result and the two recordings: chosen channels, the moving ones resampled onto the reference clock, with a thread count, progress and cancel."
---

# Write a coregistered session

You have a saved result (`pair.coreg.json` or `.json.gz`) and the two recordings it was made on, and you
want ONE new `.mefd` in which the channels you choose from either recording share the reference
clock.

```python
from mef3io_coregistration.export import export_coregistered

report = export_coregistered(
    "pair.coreg.json", "pair_merged.mefd", "system_a.mefd", "system_b.mefd",
    moving_prefix="B_",          # both recordings have a channel called C3
    n_threads=4,
    progress=lambda done, total: print(f"{100 * done / total:.0f} %"),
)
```

## What is written

- **Reference channels** keep their own samples, times and sampling rate. Nothing changes in the
  values.
- **Moving channels** are resampled with the band-limited evaluation the readers use, onto a
  uniform grid on the reference clock at **each channel's own sampling rate**. The grid starts at
  the reference-clock time of the channel's first usable sample, so the new times are regular and
  comparable with the reference channels'.
- **Gaps stay gaps.** A gap in a source channel is `NaN` after resampling, with the resampling
  filter's support around it (a few samples); `NaN` is stored as a discontinuity. Nothing is
  interpolated across a gap or invented.
- The first and last few samples of each moving channel (the resampler's filter support, 11
  source samples) are not written: there the filter has no data and the values would be distorted.
- **Units and precision.** Each channel keeps its source's sampling rate and units conversion
  factor. A factor that is a power of ten stores decimal counts of it (mef3io's float path); any
  other factor, such as an amplifier's step, is kept exactly and the values are rounded to whole
  counts of it. A session has one units label: the channels' common one, or the `units=` you pass
  (the call is refused when the selected channels' units differ and `units=` is missing).
  Subject metadata is not copied.
- **Provenance.** A session-level record holds what was done: the recordings' folder names (no
  paths), the result's SHA-256, which recording was the reference and in which direction the map was
  used, the channel lists, and the versions of this package and of mef3io.

## Choosing the reference

By default the result's own reference keeps its clock. `reference="A"` or `"B"` chooses either
recording; when it is the opposite of the result's, the **inverse** of the clock map moves the other
recording onto it. Both directions are tested against a known drift.

## Channels and names

`reference_channels=[...]` and `moving_channels=[...]` list the channels (names in the recording they
belong to) to include; `None` is all of a recording's channels and `[]` none. Two recordings often use the
same channel names, and a session cannot hold a name twice, so a collision is **refused**, naming
the channel, until you give `reference_prefix=` and/or `moving_prefix=` (for example `"B_"`) or select
different channels. Names are never renamed silently.

## Beyond the measured span

The clock map is measured over a span; outside it, it can only extrapolate. A moving channel whose
mapped extent reaches more than 1 s (the engine's span tolerance) outside `ClockMap.measured_span` is
**refused** by default, naming the channel and by how much. `out_of_span="trim"` writes only the
measured part (the report says how much was cut); `out_of_span="extrapolate"` follows the map's held
end slope, as the readers do (the report says how much is extrapolated).

## Resources

`n_threads` is mef3io's one thread rule (`0` automatic, `1` no extra threads). The same count goes
to the mef3io `Reader` (open, file access, decoding), the `Writer` (encoding) and the resampling
pool, and channels that share a grid are read and written together (`Reader.read([...])`,
`Writer.write_many`). The stored values and times do not depend on `n_threads` (the files are
byte-identical apart from their random header ids) nor on `chunk_seconds`. Memory is bounded by
`chunk_seconds`: the recordings are processed a chunk at a time, with the resampling filter's margin
fetched around each chunk, never loaded whole. (The RED block layout inside the files follows the
chunking; the decoded samples do not.)

## Safe by construction

- The output must not exist; neither must `<output>.partial`. Nothing existing is overwritten or
  deleted, and the inputs are only read.
- The session is written inside `<output>.partial/`, read back (channel set, rates, sample counts,
  start and end times, the first and last samples of every channel) and only then renamed to the
  output. Any failure or a cancel removes only that folder.
- `progress(done, total)` is called after each chunk (output samples, gaps included). `cancel` is a
  `threading.Event` or a callable returning `True`; it is checked between chunks and raises
  `ExportCancelled`.
- Encrypted recordings: pass `password_a=` / `password_b=` to read them. The output is not
  encrypted unless you pass `output_passwords=(level1, level2)`.

See the [API reference](../api/python.md) for every argument and the returned `ExportReport`.
