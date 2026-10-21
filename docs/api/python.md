---
description: "Python API reference for mef3io-coregistration, generated from the docstrings."
---

# Python API reference

Auto-generated from the source docstrings. For task-oriented walkthroughs see
the how-to pages: [read a coregistration](../howto/read_python.md) and
[align two recordings](../howto/align_python.md). Every time is uUTC
(microseconds since the Unix epoch); in the readers, on the reference clock
unless a method says otherwise.

## Reading through a result

See [Read a coregistration in Python](../howto/read_python.md) and
[Open a result from a pipeline](../howto/pipeline.md).

::: mef3io_coregistration.reader.CoregisteredSession

::: mef3io_coregistration.reader.CoregisteredMefReader

::: mef3io_coregistration.reader.CoregisteredRecording

::: mef3io_coregistration.reader.LogicalChannel

::: mef3io_coregistration.reader.UnifiedRead

::: mef3io_coregistration.reader.unified.SourceAgreement

::: mef3io_coregistration.reader.CoregisteredReader

## The result and the clock map

The JSON they are saved as is described in [Result format](../result_format.md).

::: mef3io_coregistration.result.CoregResult

::: mef3io_coregistration.result.Anchor

::: mef3io_coregistration.transform.ClockMap

::: mef3io_coregistration.transform.resample_onto_reference

## Aligning two recordings

See [Align two recordings in Python](../howto/align_python.md) for the
workflow and [Methods](../methods.md) for what each stage computes.

::: mef3io_coregistration.session.CoregSession

::: mef3io_coregistration.session.coreg_session.NotLocked

## The alignment stages

The functions `CoregSession` calls, for use on your own arrays.

::: mef3io_coregistration.align.estimate_gross_offset

::: mef3io_coregistration.align.GrossOffset

::: mef3io_coregistration.align.gross.verify_offset

::: mef3io_coregistration.align.gross.OffsetVerification

::: mef3io_coregistration.align.estimate_anchor

::: mef3io_coregistration.align.fine.local_seed

::: mef3io_coregistration.align.estimate_drift

::: mef3io_coregistration.align.DriftEstimate

### Conditioning and time-delay estimation

::: mef3io_coregistration.align.conditioning.FilterSpec

::: mef3io_coregistration.align.conditioning.condition

::: mef3io_coregistration.align.gcc_phat

::: mef3io_coregistration.align.tde.normxcorr_delay

::: mef3io_coregistration.align.Delay

::: mef3io_coregistration.align.parabolic_peak

## JSON files, plain or compressed

::: mef3io_coregistration.io.jsonfile

## Loading the alignment signals

::: mef3io_coregistration.session.LoadSpec

::: mef3io_coregistration.session.LoadProduct

::: mef3io_coregistration.session.load_alignment_signal

::: mef3io_coregistration.session.BackgroundLoad

::: mef3io_coregistration.session.Cancelled

## MEF 3.0 access and channels

A thin layer over [mef3io](https://bnelair.github.io/mef3io/): reads by
absolute time, gaps as records, and channels written as unipolar or bipolar
specs per file.

::: mef3io_coregistration.io.MefChannelReader

::: mef3io_coregistration.io.ChannelRead

::: mef3io_coregistration.io.Gap

::: mef3io_coregistration.io.Decimated

::: mef3io_coregistration.io.Envelope

::: mef3io_coregistration.io.decimate_bandlimited

::: mef3io_coregistration.io.minmax_envelope

::: mef3io_coregistration.montage.ChannelSpec

::: mef3io_coregistration.montage.read_montage

::: mef3io_coregistration.montage.bipolar

::: mef3io_coregistration.montage.BipolarSignal

::: mef3io_coregistration.montage.spec_bounds

::: mef3io_coregistration.montage.spec_fs

::: mef3io_coregistration.montage.spec_gaps

## Synthetic recording pairs

Pairs with a known clock offset, for testing and for trying the workflow
without data ([example](../examples.md)).

::: mef3io_coregistration.synthetic.WarpSpec

::: mef3io_coregistration.synthetic.render_pair

::: mef3io_coregistration.synthetic.render_pair_long

::: mef3io_coregistration.synthetic.make_source

::: mef3io_coregistration.synthetic.build_or_load

::: mef3io_coregistration.synthetic.FixturePair

## Review report

Needs the `report` extra; see [Review a coregistration](../howto/review_report.md).

::: mef3io_coregistration.review.main

::: mef3io_coregistration.review.build
