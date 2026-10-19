---
description: "Align two MEF 3.0 recordings on drifting clocks in Python with CoregSession, check each stage, and save the result."
---

# Align two recordings in Python

`CoregSession` runs the three stages on two `.mefd` sessions (or `.mefd.tar`
archives) and produces the result. Choose which file is the **reference**:
its clock is the one everything is expressed on, and it is never resampled.

```python
from mef3io_coregistration.session import CoregSession

with CoregSession() as sess:
    sess.open("system_a.mefd", "system_b.mefd", reference="A")

    # The channel to align on, for each stage. The same name in both files, or
    # per file: reference=("LA1", None), moving=("EEG LA1-Ref", "Fz").
    sess.set_channels("gross", "C3")
    sess.set_channels("fine", "C3")

    gross = sess.run_gross()                      # Stage 1: whole recording
    print(gross.offset_s, gross.locked, gross.reason)

    anchor = sess.run_fine(t_roi)                 # Stage 2: at t_roi (reference-clock uUTC)
    print(anchor.offset_s, anchor.confidence)     # call again at other places for more anchors

    drift = sess.run_drift()                      # Stage 3: the drift between and around anchors
    print(drift.accepted.sum(), "of", drift.accepted.size, "windows accepted")

    sess.result().save("pair.coreg.json")
```

The offset is always `t_moving - t_reference`, in seconds. Pick `t_roi`
where both recordings carry clear shared signal (an event, a stretch without
artifact).

## When a stage does not lock

- **Stage 1 not locked** (`gross.locked` is false): the templates disagree.
  Try another channel, flip one file's polarity
  (`sess.set_polarity("moving", True, "gross")`), continue with the best
  estimate anyway (`sess.accept_gross()`), or give the offset yourself
  (`sess.manual_offset(offset_s)`). A given offset is checked against the
  signal and used exactly as given; one the signal does not support does not
  count as a lock, so the later stages refuse to build on it, unless you pass
  `force=True`. Each choice is recorded in the result.
- **Short recordings**: with no room to drift, skip Stage 3 and build the map
  from the anchors: `sess.build_map_without_drift()`.
- **Check a candidate offset** without adopting it: `sess.verify_offset(offset_s)`.

What each number means, and the rule each stage decides by, is in
[Methods](../methods.md). Before relying on the result, look at it:
[Review a coregistration](review_report.md).

`CoregSession()` loads the full-length alignment signals in worker processes;
`CoregSession(background=False)` loads in the calling process, which is
simpler in scripts and tests. A complete runnable version on synthetic data is
[`01_align_synthetic_pair.py`](https://github.com/bnelair/mef3io-coregistration/blob/main/examples/01_align_synthetic_pair.py).
