---
description: "Runnable mef3io-coregistration scripts: aligning a synthetic pair end to end, reading through a saved result, and the review report."
---

# Examples

Runnable Python scripts live in
[`examples/`](https://github.com/bnelair/mef3io-coregistration/tree/main/examples).
The first is self-contained: it writes a synthetic pair of recordings under a
temporary directory and prints what it does. The others work on your own saved
result and the two recordings it names. Times are uUTC (microseconds since
the Unix epoch), as in MEF 3.0.

| Script | Shows |
|---|---|
| [01_align_synthetic_pair.py](https://github.com/bnelair/mef3io-coregistration/blob/main/examples/01_align_synthetic_pair.py) | The whole workflow on a synthetic pair with a known clock offset and drift: Stage 1, 2 and 3 with `CoregSession`, saving the result, reading file B on the reference clock with `CoregisteredSession`, both files as one channel with `CoregisteredRecording` (one file filling in before the other starts), event times between the clocks ([how-to](howto/align_python.md)) |
| [02_read_coregistered_recording.py](https://github.com/bnelair/mef3io-coregistration/blob/main/examples/02_read_coregistered_recording.py) | Every way to read through a saved result: `CoregisteredSession` and the legacy `CoregisteredMefReader`, logical channels, a bipolar channel per file, the agreement check and per-source scale, runs across a file edge, sample-aligned arrays with `CoregisteredReader`, time conversion, saving and reopening a mapping (needs your result; [how-to](howto/read_python.md)) |
| [03_review_report.py](https://github.com/bnelair/mef3io-coregistration/blob/main/examples/03_review_report.py) | The plotted review report, the same as `sigcoreg-review` (needs your result and the `report` extra; [how-to](howto/review_report.md)) |

```bash
python examples/01_align_synthetic_pair.py
python examples/02_read_coregistered_recording.py path/to/result.coreg.json
```

All three run in the test suite (`tests/test_examples.py`): `01` as it is,
`02` on a synthetic result whose recordings carry `ch1` and `ch2`, and the
review report on `01`'s result.
