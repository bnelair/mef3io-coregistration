# mef3io-coregistration

Align two **MEF 3.0** recordings of the same subject that were captured on
**independent, drifting clocks** (for example a clinical EEG system and a
second, implanted or research system), and read them as **one recording on
one clock**. It finds the offset between the two clocks and how it drifts
over the recording, saves that as a small JSON result, and then reads either
file, or both together, through it: every time you pass in or get back is on
the reference recording's clock. It is built on
[mef3io](https://github.com/bnelair/mef3io).

[![PyPI](https://img.shields.io/pypi/v/mef3io-coregistration)](https://pypi.org/project/mef3io-coregistration/)
[![Python](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fbnelair%2Fmef3io-coregistration%2Fmain%2Fpyproject.toml)](https://pypi.org/project/mef3io-coregistration/)
[![CI](https://img.shields.io/github/actions/workflow/status/bnelair/mef3io-coregistration/ci.yml?branch=main&label=CI)](https://github.com/bnelair/mef3io-coregistration/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/bnelair/mef3io-coregistration/docs.yml?branch=main&label=docs)](https://bnelair.github.io/mef3io-coregistration/)
[![License](https://img.shields.io/github/license/bnelair/mef3io-coregistration)](https://github.com/bnelair/mef3io-coregistration/blob/main/LICENSE)

**Documentation: <https://bnelair.github.io/mef3io-coregistration/>**

## Two recordings, two clocks

When two systems record the same subject at the same time, their clocks do
not agree: one can be minutes off, and the difference changes by seconds over
a day. An event at 10:00:00 in one file is not at 10:00:00 in the other.
mef3io-coregistration measures that difference from the signals themselves,
in three stages: a **gross** offset over the whole recording, a **fine**,
sample-accurate offset at chosen places, and the **drift** in between. The
result is a monotone **clock map** from the reference recording's clock to
the other ("moving") recording's clock, plus a record of how it was made and
how well it is supported. The recordings themselves are never changed or
copied. The [Methods](https://bnelair.github.io/mef3io-coregistration/methods/)
guide explains each stage and every number it reports.

## Install

```bash
pip install mef3io-coregistration
```

Python 3.11+; depends on numpy, scipy and
[mef3io](https://pypi.org/project/mef3io/). The plotted review report needs
matplotlib: `pip install "mef3io-coregistration[report]"`. Details:
[Install](https://bnelair.github.io/mef3io-coregistration/install/).

## Usage

Align two recordings and save the result:

```python
from mef3io_coregistration.session import CoregSession

with CoregSession(background=False) as sess:   # load in this process (no worker processes)
    sess.open("system_a.mefd", "system_b.mefd", reference="A")
    sess.set_channels("gross", "C3")          # the channel to align on, in both files
    sess.set_channels("fine", "C3")
    gross = sess.run_gross()                  # Stage 1: offset over the whole recording
    print(gross.offset_s, gross.locked)
    sess.run_fine(t_roi)                      # Stage 2: a sample-accurate anchor at t_roi (uUTC)
    sess.run_drift()                          # Stage 3: the drift everywhere
    sess.result().save("pair.coreg.json")
```

Read either file like a `mef3io.Reader`, with every time on the reference
clock:

```python
from mef3io_coregistration.reader import CoregisteredSession

with CoregisteredSession("pair.coreg.json", file="B") as r:   # or "A", "reference", "moving"
    x = r.read("C3", t0, t1)          # [t0, t1) in reference-clock uUTC; float64, NaN in gaps
    y = r.read("C3-C4", t0, t1)       # bipolar: subtracted on the file's own samples, then resampled
    ts = r.times("C3", t0, t1)        # the reference-clock time of each sample
```

Or both files as one recording, each channel served from a preferred file and
filled from the other where it has no data:

```python
from mef3io_coregistration.reader import CoregisteredRecording

with CoregisteredRecording.open("pair.coreg.json") as rec:
    rec.set_channel("C3", a="C3", b="C3", prefer="B")
    sig = rec.get_data("C3", t0, t1)
    sig.data, sig.ts, sig.source      # source per sample: 1 = file A, 2 = file B, 0 = no data
```

Times are microseconds since the Unix epoch (uUTC), as MEF 3.0 stores them.
`CoregSession()` without `background=False` loads both files in parallel worker
processes; a script using it must keep its code under
`if __name__ == "__main__":`
([how-to](https://bnelair.github.io/mef3io-coregistration/howto/align_python/)).
More: a runnable
[end-to-end example](https://github.com/bnelair/mef3io-coregistration/blob/main/examples/01_align_synthetic_pair.py)
that needs no data of your own, the
[examples](https://bnelair.github.io/mef3io-coregistration/examples/), and
short how-to pages such as
[reading a coregistration in Python](https://bnelair.github.io/mef3io-coregistration/howto/read_python/).

## What it is built for

- **Clocks that drift, not just differ.** The map follows the offset as it
  changes over the recording, to a fraction of a sample where the signals
  share structure, and marks where it is measured and where it is held
  beyond the data
  ([Methods](https://bnelair.github.io/mef3io-coregistration/methods/)).
- **Two systems that see the signal differently.** Different sampling rates,
  gains, polarity, references and channel names are expected. The channel to
  align on is chosen per file, and the signals are filtered and scaled before
  any correlation so that one amplifier's baseline cannot dominate.
- **Answers that say when they are unsure.** Stage 1 locks only when
  independent stretches of the recording agree on the same offset; an offset
  placed by hand is checked the same way; the saved result records what was
  verified, who accepted it, and how much of the map is extrapolated.
- **Existing MEF code.** `CoregisteredSession` has the `mef3io.Reader` calls
  and `CoregisteredMefReader` the legacy `mef_tools` `MefReader` calls, so
  existing analysis code needs only a different constructor.
- **Recordings far larger than memory.** Reads fetch only the window asked
  for; the alignment works on a decimated copy loaded in chunks.
- **A result that travels.** The result is a small JSON file, separate from
  the recordings. Recordings that were moved or shared are passed by their
  new paths, and each is matched to its role by its start and end time
  ([open a result from a pipeline](https://bnelair.github.io/mef3io-coregistration/howto/pipeline/);
  the [result format](https://bnelair.github.io/mef3io-coregistration/result_format/)).
- **Review before trust.** `sigcoreg-review result.json` writes a plotted
  report, from the whole recording down to single samples
  ([review report](https://bnelair.github.io/mef3io-coregistration/howto/review_report/)).

## Scope and limitations

- **Two recordings per result.** A result maps one moving recording onto one
  reference recording; more systems need one result per pair.
- **Shared signal is required.** The clocks are found from the signals, so
  the two recordings must overlap in time and share at least one channel of
  the same physiology (or a common artifact). Where the signals share little,
  a stage reports that it did not lock instead of guessing.
- **The map is one function of time.** It assumes one clock relation for all
  channels of a file, and a smooth, monotone drift; a clock that jumps during
  the recording needs one result per stretch.
- **Files are matched by time span only.** A file passes when its start and
  end are within 1 s of the ones stored in the result; two recordings with
  the same span cannot be told apart.
- **Reading, not writing.** Coregistered reads return arrays; writing an
  aligned copy as a new MEF 3.0 session is left to the caller (with mef3io).

## Contributing and support

Bug reports, feature requests and questions go to
[issues](https://github.com/bnelair/mef3io-coregistration/issues). To build
and test from source, see
[CONTRIBUTING.md](https://github.com/bnelair/mef3io-coregistration/blob/main/CONTRIBUTING.md);
the [design](https://bnelair.github.io/mef3io-coregistration/design/) is on the
documentation site. Report vulnerabilities privately as described in
[SECURITY.md](https://github.com/bnelair/mef3io-coregistration/blob/main/SECURITY.md).

```
mef3io_coregistration/   the Python package
  reader/                read one file or both through a saved result
  align/                 Stage 1 gross, Stage 2 fine, Stage 3 drift
  session/               CoregSession: runs the stages, produces the result
  transform/             the clock map and resampling onto the reference clock
  io/, montage/          MEF 3.0 access over mef3io, derived signals, channel specs
  synthetic/             synthetic recording pairs with a known clock offset
  review/                the plotted review report (sigcoreg-review)
examples/                runnable scripts
tests/                   pytest suite
benchmarks/              performance checks on day-long synthetic recordings
docs/                    documentation site source
```

## Related projects

- [mef3io](https://github.com/bnelair/mef3io): MEF 3.0 reading and writing
  for Python, MATLAB and C++, which this package reads through.
- [mef3io-studio](https://github.com/bnelair/mef3io-studio): a desktop app for
  exploring and anonymizing MEF 3.0 recordings.
- [mef3io-server](https://github.com/bnelair/mef3io-server): a gRPC server
  that streams MEF 3.0 data to viewers on a local network.

## Citing and license

If you use mef3io-coregistration in published work, please cite it via
[CITATION.cff](https://github.com/bnelair/mef3io-coregistration/blob/main/CITATION.cff).
Licensed under [Apache 2.0](https://github.com/bnelair/mef3io-coregistration/blob/main/LICENSE).
Maintained by Filip Mivalt, Bioelectronics Neurophysiology and Engineering
Laboratory (BNEL) at Mayo Clinic.

## Contributors and credits

**mef3io-coregistration:** Filip Mivalt (maintainer). It reads MEF 3.0
through [mef3io](https://github.com/bnelair/mef3io) and builds on the work of
everyone credited there. See
[CONTRIBUTORS.md](https://github.com/bnelair/mef3io-coregistration/blob/main/CONTRIBUTORS.md).

This project follows a
[code of conduct](https://github.com/bnelair/mef3io-coregistration/blob/main/CODE_OF_CONDUCT.md).
