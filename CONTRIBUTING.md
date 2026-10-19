# Contributing to mef3io-coregistration

Thanks for your interest in improving mef3io-coregistration. It is a pure
Python package on numpy, scipy and [mef3io](https://github.com/bnelair/mef3io):
it aligns two MEF 3.0 recordings on drifting clocks and reads them through the
saved result. All MEF 3.0 access goes through mef3io's public API; a missing
MEF capability is a mef3io issue first.

Everyone taking part is expected to follow our
[code of conduct](CODE_OF_CONDUCT.md). Contributors are credited in
[CONTRIBUTORS.md](CONTRIBUTORS.md).

## Ways to help

- **Report a bug** or **request a feature** via
  [GitHub issues](https://github.com/bnelair/mef3io-coregistration/issues) —
  please use the templates.
- **Ask a question** via
  [GitHub issues](https://github.com/bnelair/mef3io-coregistration/issues).
- **Send a pull request** for fixes and features (see below).

## Development setup

Requires Python 3.11+.

```bash
git clone https://github.com/bnelair/mef3io-coregistration && cd mef3io-coregistration
pip install -e ".[dev]"          # the package + pytest, hypothesis, matplotlib, ruff, build, twine
ruff check .                     # lint (CI runs the same)
python -m pytest tests           # full suite, including the slow end-to-end tests
python -m pytest tests -m "not slow"   # the quick subset while iterating
```

Synthetic test recordings are cached between runs in `data/synthetic/` next
to the package (set `SIGCOREG_DATA_DIR` to put them elsewhere); they are
regenerated when the harness changes.

Docs: `pip install -e ".[docs]"` and `mkdocs serve` (or
`mkdocs build --strict`, as CI does).

More context: [`CLAUDE.md`](CLAUDE.md) (architecture and invariants),
[`docs/`](https://bnelair.github.io/mef3io-coregistration/) (guides, methods,
design).

## Pull requests

- Branch from `main` (`feat/…`, `fix/…`, `docs/…`); open the PR against `main`.
- **Add or update tests.** New behavior needs coverage; the alignment stages
  are tested against synthetic pairs with a known clock offset
  (`mef3io_coregistration.synthetic`), so a change in accuracy shows up as a
  number, not an impression.
- **Keep old results loading.** The JSON a coregistration is saved as (format
  ids `signal_coregistration.coreg_result`, `signal_coregistration.clock_map`
  and `signal_coregistration.recording`) is read by other tools: a change
  that would stop an existing file from opening needs a new
  `format_version` and a reader for the old one.
- **Keep the dependencies small**: numpy, scipy and mef3io at run time;
  matplotlib only for the `report` extra.
- Run the full suite and `ruff check .`, and keep them green.
- CI must pass: it tests on Linux, macOS and Windows across Python 3.11–3.14.

## Documentation rules

The documentation is part of the product: a wrong sentence in it costs a user
as much as a bug. When you change behavior, change the docs in the same pull
request.

- **Revised against the code before every release.** The docs, the README,
  the examples and the release notes are checked against the code before each
  release; the checklist is in
  [Versioning & releases](https://bnelair.github.io/mef3io-coregistration/releasing/)
  (`docs/releasing.md`). Every example must run, and
  `mkdocs build --strict` must stay green.
- **No facts that go stale in the README.** It is also the PyPI page, frozen
  at each release. Leave out test counts, version histories and "new in"
  lists; those belong in the release notes.
- **One name for each thing, everywhere.** "MEF 3.0" for the format,
  "mef3io-coregistration" for this package (`mef3io_coregistration` when it is
  the import), "reference" and "moving" for the two roles, "file A" and
  "file B" for the two files, and one docs address,
  https://bnelair.github.io/mef3io-coregistration/.
- **No private product names.** Public text never names a private product;
  `tests/test_public_text.py` and the docs workflow check this.
- **No institution branding.** A plain affiliation line for the maintainers
  is the only mention.
- **Say what it does for the user.** Describe a feature by what the reader
  gains or must do, in plain words, with runnable code first where it fits.
  The how-to pages (`docs/howto/`) answer one question each and link to the
  full guides instead of repeating them.

## Reporting good bugs

The most useful reports include the version
(`mef3io_coregistration.__version__` and `mef3io.__version__`), OS and Python
version, a minimal snippet, the full error/traceback and, for an alignment
that went wrong, the saved result JSON (it holds no signal, only times and
file metadata; remove the file paths if they identify anyone). Where possible,
reproduce it on a synthetic pair (`mef3io_coregistration.synthetic`). Never
attach clinical or subject-identifying data.

## Reporting security issues

Please follow [`SECURITY.md`](SECURITY.md) rather than opening a public issue.

## License

By contributing you agree that your contributions are licensed under the
project's [Apache 2.0 license](LICENSE).
