---
description: "Open a saved coregistration result from an analysis pipeline: pass the recordings' current paths; files are matched to the result by their start and end time."
---

# Open a result from a pipeline

A result is only the transformation between two clocks; it does not contain
the recordings. It stores where they were when it was made, but a pipeline
usually finds them somewhere else (another machine, a copy, a `.mefd.tar`
archive). Pass their current paths:

```python
from mef3io_coregistration.reader import CoregisteredRecording, CoregisteredSession

rec = CoregisteredRecording.open(
    "results/subject01.coreg.json",
    file_a="/data/subject01/recording_a.mefd",
    file_b="/data/subject01/recording_b.mefd.tar",
)

with CoregisteredSession("results/subject01.coreg.json", file="B",
                         file_a="/data/subject01/recording_a.mefd",
                         file_b="/data/subject01/recording_b.mefd.tar") as r:
    ...
```

Without `file_a=` / `file_b=`, a recording missing from its stored path is
looked for next to the result file (the same name, as a `.mefd` folder or a
`.mefd.tar` archive), so a result kept in the same folder as its recordings
opens as it is.

## How a file is matched to the result

**Only by its time span.** The result stores each file's start and end; a
file you pass must start and end within **1 s** of the ones stored for its
role (A or B). Paths, file names, channels, sampling rates and the signal
are not compared, so a moved copy, a downsampled or re-exported copy, or one
with other channels passes, and a different recording is refused with both
spans in the message.

- **A and B swapped** (or one file given for both) is refused: a file that
  matches the *other* role's span is reported as swapped, because applying
  the map to the wrong file would shift every time by the offset.
- **Two recordings with the same span** cannot be told apart by this check.
- **`verify=False`** skips the check, for when you know better.
- **Encrypted sessions**: `password_a=` / `password_b=` (used for the check
  too).

The roles themselves come from the result: which file is the reference can
never be inverted by how you pass the paths.

## A saved channel mapping

The channel set-up of a `CoregisteredRecording` (logical channels,
preferences, corrections) can be saved next to the result and reopened in one
line, with the same path overrides:

```python
rec.save_config("results/subject01.mapping.json", result_path="results/subject01.coreg.json")

rec = CoregisteredRecording.from_config(
    "results/subject01.mapping.json",
    file_a="/data/subject01/recording_a.mefd", file_b="/data/subject01/recording_b.mefd.tar",
)
```

The mapping stores the result file's SHA-256: a result overwritten since is
refused (`verify=False` overrides). Both formats are described in
[Result format](../result_format.md).
