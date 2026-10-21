---
description: "Read one or both MEF 3.0 recordings of a saved coregistration in Python, with every time on the reference clock."
---

# Read a coregistration in Python

You have a saved result (`pair.coreg.json`) and the two recordings it was
made on. Every time below, passed in or returned, is **reference-clock uUTC**
(microseconds since the Unix epoch, on the reference recording's clock),
whichever file you read.

## One file, like a `mef3io.Reader`

```python
from mef3io_coregistration.reader import CoregisteredSession

with CoregisteredSession("pair.coreg.json", file="B") as r:   # "A", "B", "reference" or "moving"
    r.channels                        # this file's channels
    x = r.read("C3", t0, t1)          # float64 over [t0, t1); NaN in gaps and outside the file
    y = r.read("C3-C4", t0, t1)       # bipolar: subtracted on the file's own samples, then resampled
    ts = r.times("C3", t0, t1)        # the reference-clock time of each sample in x
    r.info("C3")                      # metadata; start_time/end_time on the reference clock
    r.records()                       # annotations, their times on the reference clock
```

The reference file comes back exactly as mef3io returns it. The moving file
is resampled through the clock map onto a uniform grid on the reference
clock, at its own sampling rate (`fs=` for another); only the window you ask
for is read, and consecutive reads tile without overlap or loss. A channel
can be a name, an `"anode-cathode"` string or an `(anode, cathode)` tuple.

Code written for the legacy `mef_tools` reader needs only the constructor
changed:

```python
from mef3io_coregistration.reader import CoregisteredMefReader

mr = CoregisteredMefReader("pair.coreg.json", file="moving")   # was: MefReader(path)
x = mr.get_data(["C3", "C3-C4"], t0, t1)
fs = mr.get_property("fsamp", "C3")
mr.close()
```

## Both files as one recording

```python
from mef3io_coregistration.reader import CoregisteredRecording

with CoregisteredRecording.open("pair.coreg.json") as rec:
    rec.channels                                   # one logical channel per name; shared names linked
    rec.set_channel("C3", a="C3", b="C3", prefer="B")
    sig = rec.get_data("C3", t0, t1)
    sig.data, sig.ts, sig.fs                       # samples on the preferred file's grid
    sig.source                                     # per sample: 1 = file A, 2 = file B, 0 = no data
    sig.extrapolated                               # True where the map is held beyond its data
    for source, s0, s1 in sig.runs():              # single-file stretches: analyse each on its own
        ...
```

The preferred file serves the channel wherever it has data and the other
file fills in where it has none. The two files may name and montage a channel
differently (`a="LA1"`, `b=("EEG LA1-Ref", "Fz")`). Filling is allowed only
when the two sources agree over their overlap (`rec.agreement("C3")`: sign,
gain, units); state how they relate with `scale_a`/`offset_a`/`scale_b`/
`offset_b` (`scale_b=-1` inverts), or override with `allow_mismatch=True`.
[Methods §8](../methods.md#8-using-the-two-files-as-one-recording) explains
the check.

## Compressed results

A result, a clock map or a mapping saved as `….json.gz` is gzip-compressed; every
call above opens it the same way (compression is detected from the content):

```python
from mef3io_coregistration.result import CoregResult

res = CoregResult.load("subject01.coreg.json")
res.save("subject01.coreg.json.gz")            # the same result, compressed
with CoregisteredRecording.open("subject01.coreg.json.gz") as rec:
    ...
```

## Reopen a result to save or change it

`CoregSession().open_result(path)` reopens a saved result (plain or `.gz`) with
its recordings and clock map, without re-running the stages: read through it,
save it again (for example compressed), or continue aligning from it (select the
channels again with `set_channels`, and give a Stage 1 offset with
`manual_offset` before seeking new anchors).

```python
from mef3io_coregistration.session import CoregSession

sess = CoregSession().open_result("subject01.coreg.json", file_b="/new/place/recording_b.mefd")
sess.result().save("subject01.coreg.json.gz")
```

## Event times between the clocks

```python
t_in_b = rec.to_file_time(t_ref, "B")         # reference clock -> file B's own clock
t_ref = rec.to_reference_time(t_in_b, "B")    # and back
```

## Sample-aligned arrays

`CoregisteredReader` reads the moving file onto exactly the reference file's
sample grid, so the two arrays can be stacked or subtracted sample by sample:

```python
from mef3io_coregistration.reader import CoregisteredReader

rd = CoregisteredReader.from_result("pair.coreg.json")
a = rd.get_data("C3", t0, t1, file="A")                    # native reference samples
b = rd.get_data("C3", t0, t1, file="B", coregister=True)   # resampled onto them
assert (a.ts == b.ts).all()
rd.close()
```

If the recordings are no longer where the result says, pass their paths:
see [Open a result from a pipeline](pipeline.md). A longer walk through all
of this is
[`02_read_coregistered_recording.py`](https://github.com/bnelair/mef3io-coregistration/blob/main/examples/02_read_coregistered_recording.py);
the classes are in the [API reference](../api/python.md#reading-through-a-result).
