# mef3io-coregistration examples

Small scripts, each runnable directly:

```bash
python examples/01_align_synthetic_pair.py
```

`01` is self-contained: it writes a synthetic pair of recordings under a
temporary directory and prints what it does. `02` and `03` work on your own
saved result and the two recordings it names. Times in MEF 3.0 are **uUTC** —
microseconds since the Unix epoch.

| Script | Shows |
| --- | --- |
| `01_align_synthetic_pair.py` | The whole workflow on a synthetic pair with a known clock offset and drift: `CoregSession` Stage 1, 2 and 3, saving the result, reading file B on the reference clock with `CoregisteredSession`, both files as one channel with `CoregisteredRecording` (one file filling in before the other starts), event times between the clocks |
| `02_read_coregistered_recording.py` | Every way to read through a saved result: `CoregisteredSession` and the legacy `CoregisteredMefReader`, logical channels, a bipolar channel per file, the agreement check and per-source scale, runs across a file edge, sample-aligned arrays with `CoregisteredReader`, time conversion, saving and reopening a mapping (needs your result: `python examples/02_read_coregistered_recording.py result.coreg.json`; the figure needs `pip install "mef3io-coregistration[report]"`) |
| `03_review_report.py` | The plotted review report, the same as `sigcoreg-review` (needs your result and `pip install "mef3io-coregistration[report]"`) |
