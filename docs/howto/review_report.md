---
description: "Review a saved coregistration with plots, from the whole recording down to single samples, with sigcoreg-review."
---

# Review a coregistration

Every number a stage reports can be satisfied by a map that is subtly wrong;
looking at the two signals on top of each other cannot. The review report
plots that for you.

```bash
pip install "mef3io-coregistration[report]"      # matplotlib

sigcoreg-review pair.coreg.json
sigcoreg-review pair.coreg.json --channel C3-C4 --out review/ --file-a /new/a.mefd --file-b /new/b.mefd
```

It writes PNG figures and an `index.html` that shows them with the numbers
behind them (by default into `<result>_review/` next to the result):

1. **overview**: both recordings on the reference clock, their overlap, the
   measured span of the map;
2. **clock map**: offset over time, the drift windows, the fitted map, the
   anchors and their confidence;
3. **multi-scale**: the two signals overlaid at 2 min, 10 s, 2 s and 250 ms,
   at three times;
4. **before/after**: a constant offset against the fitted map, early and late
   in the recording;
5. **residual lag**: an independent re-measurement of the lag left after
   coregistration;
6. **stimulation pulses**: sharp artifacts present in both systems, used as
   sub-millisecond markers (when there are any);
7. **unified**: the two files as one channel across the edge of the moving
   recording.

`--channel` chooses the channel to review (default: the channels Stage 2
aligned on). `--file-a` / `--file-b` point it at recordings that have moved;
files are checked as described in
[Open a result from a pipeline](pipeline.md). From Python:

```python
from mef3io_coregistration.review import main
main(["pair.coreg.json", "--out", "review/"])
```

What to look for in each number is in [Methods §7](../methods.md#7-qc-what-to-look-at-before-you-trust-the-map).
