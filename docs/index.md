---
description: "Align two MEF 3.0 recordings captured on independent, drifting clocks, such as EEG and iEEG from two systems, and read them as one recording on one clock, in Python."
---

# mef3io-coregistration

**Two MEF 3.0 recordings on two drifting clocks, read as one recording on one
clock.**

When two systems record the same subject at the same time (for example a
clinical EEG system and a second, implanted or research system), their clocks
do not agree: one can be minutes off, and the difference changes by seconds
over a day. mef3io-coregistration measures that difference from the signals
themselves and saves it as a **clock map**, a monotone function from the
reference recording's clock to the other ("moving") recording's clock, in a
small JSON result with a record of how it was made. It then reads either
file, or both together, through that result: every time you pass in or get
back is on the reference clock. The recordings are never changed or copied.
All MEF 3.0 access goes through [mef3io](https://github.com/bnelair/mef3io).

!!! tip "Check before you trust"
    Every stage reports whether it locked, and the saved result says what was
    verified and how much of the map is extrapolated. The
    [review report](howto/review_report.md) plots the alignment from the whole
    recording down to single samples; the [Methods](methods.md) guide explains
    every number.

## Why mef3io-coregistration

- **Drift, not just an offset** — three stages: a **gross** offset over the
  whole recording (minutes to hours apart), sample-accurate **anchors** at
  chosen places, and the **drift** in between, fitted as a smooth monotone
  map. Outside the measured span the map is held, and flagged as
  extrapolated.
- **Honest verdicts** — Stage 1 locks only when independent templates from
  different hours agree on the offset; a hand-placed offset is checked the
  same way; confidence is peak distinctiveness, not raw correlation. A stage
  that did not lock says so, and you can still continue and have that choice
  recorded.
- **Two different systems** — different sampling rates, gains, polarity,
  references and channel names; the channel to align on is chosen per file,
  unipolar or bipolar.
- **Drop-in reading** — `CoregisteredSession` has the `mef3io.Reader` calls,
  `CoregisteredMefReader` the legacy `mef_tools` `MefReader` calls, and
  `CoregisteredRecording` serves both files as one, each channel from a
  preferred file with the other filling its gaps.
- **Large recordings** — reads fetch only the window asked for; alignment
  works on a decimated copy loaded in chunks, never the full-rate signal.
- **A result that travels** — the result is separate from the data: pass the
  recordings' current paths, and each file is matched to its role by its
  start and end time ([how](howto/pipeline.md)).

## Quick taste (Python)

```python
from mef3io_coregistration.session import CoregSession
from mef3io_coregistration.reader import CoregisteredSession

with CoregSession() as sess:                       # align
    sess.open("system_a.mefd", "system_b.mefd", reference="A")
    sess.set_channels("gross", "C3")
    sess.set_channels("fine", "C3")
    sess.run_gross()
    sess.run_fine(t_roi)                           # an anchor at t_roi (reference-clock uUTC)
    sess.run_drift()
    sess.result().save("pair.coreg.json")

with CoregisteredSession("pair.coreg.json", file="B") as r:   # read file B on A's clock
    x = r.read("C3", t0, t1)
```

A complete version that makes its own recordings is
[`01_align_synthetic_pair.py`](examples.md).

## Where to go

| | |
|---|---|
| [Install](install.md) | pip, the optional extras, a development install |
| [Examples](examples.md) | runnable scripts, one of them self-contained |
| [Read a coregistration in Python](howto/read_python.md) | one file on the reference clock, both files as one, event times between the clocks |
| [Align two recordings in Python](howto/align_python.md) | `CoregSession` from two `.mefd` paths to a saved result |
| [Open a result from a pipeline](howto/pipeline.md) | passing the recordings' current paths; how files are matched to a result by time span |
| [Review a coregistration](howto/review_report.md) | the plotted report, `sigcoreg-review` |
| [Methods](methods.md) | what each stage computes, the numbers it reports and how it fails |
| [Result format](result_format.md) | the JSON a result and a recording mapping are saved as |
| [Python API](api/python.md) | the reference, generated from the docstrings |
| [Release notes](release_notes/index.md) | what changed in each version, and what an upgrade asks of you |

## Scope and limitations

- One result maps one moving recording onto one reference recording.
- The two recordings must overlap in time and share signal on at least one
  channel; where they share little, the stages say they did not lock.
- One clock relation serves every channel of a file, and the drift is
  assumed smooth and monotone; a clock that jumps needs one result per
  stretch.
- Files are matched to a result only by their start and end (within 1 s);
  two recordings with the same span cannot be told apart.
- Coregistered reads return arrays; writing an aligned copy as a new MEF 3.0
  session is left to the caller (with mef3io).

What changed in each version is in the
[release notes](release_notes/index.md).

## Related projects

- [mef3io](https://github.com/bnelair/mef3io): MEF 3.0 reading and writing
  for Python, MATLAB and C++, which this package reads through.
- [mef3io-studio](https://github.com/bnelair/mef3io-studio): a desktop app for
  exploring and anonymizing MEF 3.0 recordings.
- [mef3io-server](https://github.com/bnelair/mef3io-server): a gRPC server
  that streams MEF 3.0 data to viewers on a local network.

Maintained by Filip Mivalt, Bioelectronics Neurophysiology and Engineering
Laboratory (BNEL) at Mayo Clinic.
