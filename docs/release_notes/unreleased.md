# mef3io-coregistration (unreleased)

*Draft.*

## Highlights

- **Write a coregistered session.** New `export_coregistered(result, out, file_a, file_b, ...)`
  (`mef3io_coregistration.export`) writes a brand-new MEF 3.0 session from a saved result and the
  two recordings. Reference channels keep their samples and times; moving channels are resampled
  (the band-limited path the readers use) onto a uniform grid on the reference clock at their own
  rate; gaps stay gaps. You choose which recording is the reference (the opposite of the result's
  own direction uses the inverse clock map), which channels of either recording to include, and
  how many threads to use (`n_threads`, mef3io's one thread rule, passed to the reader, the writer
  and the resampling). Memory is bounded (chunks with the filter margins); the stored values and
  times do not depend on the chunk size or the thread count. Progress and cancel are supported;
  the session is written next to the output, read back and verified, then renamed into place. An
  existing output is never touched, and a cancelled or failed run removes only what it created.
  ([How-to](https://bnelair.github.io/mef3io-coregistration/howto/export_resampled/))
- **Export length comes from the block index.** `export_coregistered` sizes each channel from the
  recording's block index, never from the start and end times in the file headers (which some
  recorders write short or late), so no block is dropped from such files.
- **mef3io 2.3.** The dependency is now `mef3io>=2.3.0,<3` (multi-channel `Reader.read` and
  `Writer.write_many`).

## Upgrading

- Install mef3io 2.3 or newer (`pip install -U mef3io-coregistration` brings it).
