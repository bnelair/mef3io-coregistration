# mef3io-coregistration — the reader

Read two MEF3 recordings that were coregistered (aligned onto one clock) as if they were one
recording. This package is only the reader: it opens a
saved coregistration result (`*.coreg.json`) and reads the recordings through it. Results are made with
the BNEL coregistration console, which is not part of this package.

Dependencies: numpy, scipy, [mef3io](https://pypi.org/project/mef3io/). Python ≥ 3.11.

## Install

```bash
pip install mef3io-coregistration
```

Development: see [CONTRIBUTING.md](CONTRIBUTING.md). Releases: see [RELEASING.md](RELEASING.md).

## The idea

A coregistration result is only the **transformation** between the two recordings' clocks — a
clock map from the reference recording's time to the other ("moving") recording's time. The
recordings themselves stay where they are, and can be moved or shared: pass their current paths
when opening. Everything you read comes out on the **reference clock**, in absolute uUTC
(microseconds since the epoch, as MEF3 stores time). The drift is a property of the two recording
systems, so one map serves every channel of either file.

## Read either file like a MEF session

```python
from mef3io_coregistration.reader import CoregisteredSession

with CoregisteredSession("patient.coreg.json", file="moving",          # or "reference", "A", "B"
                         file_a="/data/cadence.mefd", file_b="/data/natus.mefd") as r:
    r.channels                              # this file's channels
    x = r.read("LTP1", t0, t1)              # [t0, t1) reference-clock uUTC → float64, NaN in gaps
    y = r.read("LTP1-LTP3", t0, t1)         # bipolar: subtracted on the native samples, then resampled
    ts = r.times("LTP1", t0, t1)            # the uUTC of each sample
    r.info("LTP1")                          # metadata; start/end on the reference clock
    r.records()                             # annotations: times on the reference clock, durations on the file's
```

Same calls as `mef3io.Reader`. Reads fetch only the window asked for. The moving file comes back
at its own sampling rate (`fs=` to choose another), on a uniform grid on the reference clock;
consecutive reads tile. The reference file comes back exactly as mef3io returns it.

Code written for the legacy `mef_tools` `MefReader` works with one changed line:

```python
from mef3io_coregistration.reader import CoregisteredMefReader

mr = CoregisteredMefReader("patient.coreg.json", file="moving")   # was: MefReader(path, password2=…)
x = mr.get_data(["LTP1", "LTP1-LTP3"], t0, t1)
fs = mr.get_property("fsamp", "LTP1")
```

## Both files as one recording

```python
from mef3io_coregistration.reader import CoregisteredRecording

rec = CoregisteredRecording.open("patient.coreg.json")            # or .from_config("mapping.json")
rec.set_channel("LTP1-3", a=("LTP1", "LTP3"), b=("LTP1", "LTP3"), prefer="A", scale_a=-0.06)
sig = rec.get_data("LTP1-3", t0, t1)     # the preferred file where it has data, the other elsewhere
sig.data, sig.ts, sig.source             # source: 1 = file A, 2 = file B, 0 = no data
sig.runs()                               # single-file stretches — analyse (and filter) each on its own
rec.agreement("LTP1-3")                  # do the two sources agree (sign, gain, units)?
rec.to_file_time(t_ref, "A")             # event times between the clocks
```

A gap is only filled from the other file when the two sources agree over their overlap; set a
per-source `scale_*`/`offset_*` to make them agree, or `allow_mismatch=True` to override.

## Moved, shared, encrypted

* `file_a=` / `file_b=` override the recordings' stored paths. Without them, a recording that is not
  at its stored path is looked for next to the result file (`.mefd` or `.mefd.tar`).
* A mapping saved next to its result (the console's *Download result* and *Download mapping*)
  reopens from that folder: `CoregisteredRecording.from_config("mapping.json")`.
* Each recording is checked only by its time span: its start and end must be within 1 s of the
  ones saved in the result. Paths, channels, rates and signal do not matter — you assign the two
  files, so a moved, downsampled or re-exported copy passes, and a recording with another span is
  refused with both spans in the message (`verify=False` to override).
* `password_a=` / `password_b=` open encrypted sessions.
