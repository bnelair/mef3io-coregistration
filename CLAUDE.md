# mef3io-coregistration — assistant context

Pure-Python engine for **coregistering two MEF 3.0 recordings captured on
independent, drifting clocks** (e.g. a clinical EEG system and a second,
implanted or research recorder), and for reading them through the saved result as
one recording on the reference clock. Depends on numpy, scipy and
[mef3io](https://github.com/bnelair/mef3io) only (matplotlib for the `report`
extra). All MEF 3.0 access goes through mef3io's public API — never parse MEF
bytes here; a missing MEF capability is a mef3io issue first.

Status: 0.3.0, first public release in preparation (VERSION = 0.3.0, release
notes `docs/release_notes/0.3.0.md` still a draft). Complete: Stage 1 gross
(stacked multi-template matched filter, lock by agreement), offset
verification, Stage 2 fine (seeded NCC + local seed refinement), Stage 3 drift
(outward walk from every anchor, weighted smoothing spline), no-drift map
build, `CoregSession` orchestration with background loaders, `CoregResult`
JSON, the readers (`CoregisteredSession`, `CoregisteredMefReader`,
`CoregisteredRecording`, `CoregisteredReader`), the synthetic harness and the
review report. A web console built on this engine is a separate, private
project; nothing here may depend on it or name it.

## Build & test (use the active conda env for everything)

```bash
# the project's conda env, with the engine installed editable (pip install -e ".[dev]")
ruff check .                                  # lint; CI runs the same (ruff 0.16.x, rules in pyproject)
SIGCOREG_DATA_DIR=/some/scratch python -m pytest tests     # full suite, ~10 min incl. slow tests
python -m pytest tests -m "not slow"          # quick subset
python -m build && twine check dist/*         # sdist + wheel (wheel = package + py.typed only)
mkdocs build --strict                         # docs; tooling from the [docs] extra, in its own venv
```

`SIGCOREG_DATA_DIR` sets where cached synthetic fixtures go (default: `./data/`
in a source checkout, gitignored; the user cache folder for an installed
package; `config.DATA_DIR`, pinned by `tests/test_config.py`).
`CoregSession()` loads in "spawn" worker processes: scripts need an
`if __name__ == "__main__":` guard, or `CoregSession(background=False)`. The
`shared_data_dir` fixture (tests/conftest.py) reuses them across runs; they
regenerate when the harness version or profile changes.

## Module map

- `result.py` — `CoregResult` (+ `Anchor`): the saved artefact; sorted-key
  JSON, format id `signal_coregistration.coreg_result` v2 (v1 = no
  `reference_role`, A was the reference). `session/coreg_result.py` re-exports
  it for old imports.
- `transform/` — `clock_map.ClockMap` (monotone int64 uUTC knots, linear or
  pchip, `hold_end_slope`, `measured_span`, format id
  `signal_coregistration.clock_map` v1); `resample.py` band-limited evaluation
  of the moving signal at mapped times; `filters.py` zero-phase filters shared
  by reader and alignment.
- `io/` — `mef_reader.MefChannelReader` (thin wrapper over `mef3io.Reader`:
  uUTC reads snapped to the sample grid, gaps as `Gap` records,
  `time_span`, `fingerprint`, `SPAN_TOLERANCE_US` = 1 s); `derived.py`
  decimation (group-delay compensated) and the min-max envelope;
  `chunked.py` full-length chunked loading (overlap-save decimation);
  `jsonfile.py` the package's JSON files, plain or gzip (compressed when the
  name ends in `.gz`, detected by content on read, bounded decompression,
  `content_sha256` of the decompressed JSON — what a mapping stores).
- `montage/` — `ChannelSpec` (unipolar / `anode - cathode`, per file, string
  forms resolved against the file's channel list), `bipolar`, `read_montage`.
- `reader/` — `coregistered_reader.CoregisteredReader` (raw or coregistered
  per file; `from_result` path lookup + span check), `mef_api`
  (`CoregisteredSession` = mef3io.Reader calls, `CoregisteredMefReader` =
  legacy mef_tools MefReader 1:1), `unified.CoregisteredRecording` (logical
  channels, preferred source + gap fill gated by `SourceAgreement`, config
  format id `signal_coregistration.recording` v1, display envelopes).
- `align/` — `conditioning` (gap bridge, filters, MAD scale, clip;
  DETECTION vs ESTIMATION bands), `tde` (NCC/GCC-PHAT, distinctiveness
  confidence, parabolic peak), `gross` (Stage 1 + `verify_offset`), `fine`
  (Stage 2 + `local_seed`), `drift` (Stage 3).
- `session/` — `CoregSession` (stage state, per-stage channels and polarity,
  product reuse, `result()`), `loader` (`BackgroundLoad` subprocess,
  cancel/progress).
- `synthetic/` — `WarpSpec` analytic offsets, `render_pair(_long)`, cached
  fixture profiles (`build_or_load`).
- `review/` — `report.build/main` = `sigcoreg-review` (matplotlib, Agg).
- `config.py` — every threshold, with the measurement behind it in comments.

The reader core (`result`, `transform`, `io`, `montage`, `reader`, `config`)
NEVER imports `align`, `session`, `synthetic` or `review`.

## Invariants (each pinned by a test)

- Sign convention `offset = t_moving − t_ref`, everywhere; times are int uUTC.
- The REFERENCE file is NEVER transformed: read on the reference clock it
  returns mef3io's samples as is (`test_reader.py::test_reference_never_transformed`,
  `test_mef_api.py::test_reference_file_reads_exactly_as_mef3io`,
  `test_unified.py::test_prefer_reference_returns_native_samples`).
- The moving file is resampled, not re-stamped, and lands on the reference
  clock within a sample (`test_reader.py::test_coregistered_moving_timing_within_one_lower_fs_sample`,
  `test_open_and_merge.py::test_moving_file_matches_the_true_signal_on_the_reference_clock`).
- Bipolar is derived on native samples, THEN resampled
  (`test_mef_api.py::test_bipolar_is_derived_natively_then_resampled`).
- Reads are half-open on the sample grid and consecutive reads tile
  (`test_generic_channels.py::test_read_bounds_are_half_open_on_grid_and_tile`,
  `::test_consecutive_coregistered_reads_tile`, `test_mef_api.py::test_partial_reads_tile`,
  `test_edge_cases.py::test_half_open_bounds_at_rounding_ties`).
- Gaps are never filled or interpolated across
  (`test_mef_reader.py::test_gap_is_reported_and_not_filled`,
  `test_edge_cases.py::test_decimating_resample_masks_the_filter_support_around_gaps`).
- Clock map strictly monotone, inverse valid, extrapolation flagged
  (`test_clock_map.py::test_forward_monotone_including_extrapolation`,
  `::test_property_roundtrip_and_monotone` (hypothesis), `::test_extrapolation_holds_end_slope_and_is_flagged`).
- Save → load → save is byte-stable for results and maps; foreign JSON is
  refused; v1 results still load (`test_coreg_result.py`, `test_clock_map.py::test_save_load_is_bit_stable`,
  `test_generic_channels.py::test_v1_result_files_still_load`). The three
  `signal_coregistration.*` format ids are KEPT for compatibility — never
  rename them; a breaking change needs a new `format_version` + old reader.
- Files are matched to a result ONLY by time span (start/end within 1 s);
  swapped A/B refused; roles always from the result
  (`test_open_and_merge.py::test_swapped_files_are_refused`, `::test_a_recording_with_another_span_is_refused`,
  `test_mef_api.py::test_recordings_can_move`, `::test_a_different_recording_is_refused`,
  `test_generic_channels.py::test_reader_from_result_takes_roles_from_result`).
  A recording mapping is bound to its result's SHA-256
  (`test_unified.py::test_recording_config_is_bound_to_its_result`).
- Gap fill between files only when the sources agree; `allow_mismatch` is an
  explicit override (`test_unified.py::test_inverted_source_is_not_merged_until_corrected`,
  `::test_scale_mismatch_is_not_merged_until_corrected`, `::test_allow_mismatch_is_an_explicit_override`).
- Confidence = peak distinctiveness, NOT raw NCC
  (`test_tde.py::test_confidence_is_distinctiveness_not_match_height`).
- Stage 1 locks by template AGREEMENT, never by peak height
  (`test_gross.py::test_peak_height_cannot_separate_true_locks_from_false_ones`,
  `::test_reports_no_lock_on_unrelated_signals`); verification catches a
  globally wrong offset (`::test_verification_rejects_a_globally_wrong_offset`).
- Flag, don't block: later stages refuse an unlocked Stage 1
  (`test_coreg_session.py::test_refuses_to_build_on_a_failed_lock`), but the
  user can always continue and that is RECORDED
  (`::test_an_unlocked_estimate_can_be_accepted_and_is_recorded`); a hand
  offset is used exactly as placed (`::test_manual_offset_uses_the_offset_the_user_placed`).
- Stage 2 seed is what the caller passes (`test_fine.py::test_seed_is_what_the_caller_passes_not_a_stale_offset`);
  an ROI outside a recording is refused, not clamped (`::test_roi_outside_a_recording_is_refused_not_silently_clamped`).
- Stage 3 walks from every anchor; a search-edge peak is not a measurement;
  a bad anchor shows in its residual, not in the map (`test_drift.py::test_anchor_far_from_start_realistic_rate`,
  `::test_search_edge_peak_is_not_a_measurement`, `::test_bad_anchor_shows_in_residual_not_in_map`).
- Streaming load equals a full pass; loads cancel; a dead worker is reported,
  not hung (`test_loader.py`).
- End to end: a synthetic drift is recovered to < 1 sample RMS and the
  result drives the reader (`test_coreg_session.py::test_end_to_end_recovers_drift_and_reader_reconstructs`);
  `examples/01_align_synthetic_pair.py` runs, `02_read_coregistered_recording.py`
  runs on a synthetic two-channel result, and the review report renders on
  01's result (`test_examples.py`, slow).
- Text I/O is always `encoding="utf-8"`, and saved JSON is written with
  `newline="\n"`, so results are byte-identical on Windows.

## Docs, examples, distribution

`examples/` = `NN_snake_case.py` listed in examples/README.md (01 =
self-contained synthetic end to end; 02 = every read path, run in the tests on
a synthetic result; 03 = review report; all smoke-tested by
`tests/test_examples.py`). `benchmarks/bench_day_long.py` = stage timings on a
day-long synthetic pair. Docs site: MkDocs Material
from `docs/` (`mkdocs.yml` holds the nav: getting started, how-to, guides
(methods, result format), API reference, release notes, development),
deployed to GitHub Pages by `.github/workflows/docs.yml` on push to main —
keep `mkdocs build --strict` green. The API reference is mkdocstrings
(`docs/api/python.md`, grouped `:::` directives, NumPy-style docstrings,
`paths: [.]`). `docs/methods.md` = the method walkthrough with the
measurements behind every threshold; `docs/design.md` = layers and
invariants; `docs/result_format.md` = the JSON formats.

Distribution: **version single source of truth = repo-root `VERSION`**
(setuptools `version = {file = "VERSION"}` → wheel metadata;
`mef3io_coregistration.__version__` from installed metadata, `0+unknown`
uninstalled; also written as `tool_version` into every result). Release flow:
manually run the `bump-version` workflow on main (patch/minor/major or an
explicit version) → commits VERSION as "Release vX.Y.Z", tags vX.Y.Z,
dispatches `release.yml` → `python -m build`, twine check, the suite against
the installed wheel, twine upload with the `PYPI_Token_General` secret, GitHub
release with `docs/release_notes/X.Y.Z.md` as its text (refused while that
page still has its `*Draft` line). A release.yml dispatch
on a branch is a dry run. `ci.yml` = ruff + pytest matrix (ubuntu
3.11/3.12/3.13, macOS 3.13, windows 3.13) + version sanity + 3.14 smoke on
push (main, dev) and PRs. Before every release, a full docs revision PR
(docs/releasing.md); notes collect in docs/release_notes/unreleased.md between
releases. Published on PyPI as `mef3io-coregistration`. requires-python stays
>= 3.11 (datetime.UTC).

Public text: `tests/test_public_text.py` (and the docs workflow's check of the
built site) scans EVERY tracked text file, tests and this file included, and
fails on any of the private product names it holds (base64-encoded there —
never spell them out anywhere in this repo), on recording-device vendor names,
and on the channel labels of the real dataset the thresholds were calibrated
on. Describe real-data results generically ("on a clinical test pair ..."),
without the pair's measurements; use `ch1`/`ch2`, `subject01`, `recording_a`
in examples. Requirement/review numbering of the private project (R-numbers,
review ids) does not belong in public code or comments either. The JSON format ids `signal_coregistration.*` are identifiers, not
product names, and are allowed; the old private import name appears once, in
the 0.3.0 release note's upgrading line, and nowhere else. Institution
mentions allowed: the affiliation line ("Maintained by Filip Mivalt,
Bioelectronics Neurophysiology and Engineering Laboratory (BNEL) at Mayo
Clinic.") in README, docs/index.md and CONTRIBUTORS.md, the CITATION.cff
author `affiliation` field, and the code-of-conduct contact address. Nothing
else. The private web console is "a web console" in public text, never by
name.
