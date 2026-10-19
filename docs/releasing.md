# Versioning & releases

## One source of truth

The repo-root **`VERSION`** file is the only place the version lives:

| Consumer | How it gets the version |
|---|---|
| PyPI wheel / sdist | `pyproject.toml` reads `VERSION` via setuptools' dynamic `version = { file = "VERSION" }` |
| Python runtime | `mef3io_coregistration.__version__` (installed metadata; `0+unknown` in an uninstalled source tree) |
| Saved results | `tool_version` in every result and clock map (= `__version__`) |
| Git tags | `vX.Y.Z`, created by the release trigger |

CI asserts `mef3io_coregistration.__version__ == VERSION`, and git tags
mirror the file — they are created by the release trigger, so they cannot
drift.

## Releasing

### Before the release: revise the documentation (required)

Every release starts with a **complete documentation revision**, done as its
own reviewed pull request and **merged before the tag is created**. Check each
of these against the code that is about to ship, not against the previous
release:

- [ ] `README.md` (it is also the PyPI project page, so links must be absolute)
- [ ] the documentation site (`docs/`, built with `mkdocs build --strict`)
- [ ] `examples/README.md` and the example scripts it lists
- [ ] `pyproject.toml`: `description`, `keywords`, `classifiers`
- [ ] `CITATION.cff`
- [ ] `CLAUDE.md` (status, module map, invariants)
- [ ] the release notes: draft what changed and what an upgrading user must
      do in `docs/release_notes/X.Y.Z.md`, and add it to the release-notes
      index and the `mkdocs.yml` nav (newest first). Between releases, pull
      requests collect their notes in `docs/release_notes/unreleased.md`
      (listed in the index and the nav as "Unreleased"); the revision renames
      it to `X.Y.Z.md`, rewrites it for the reader (highlights, fixes,
      upgrading), changes its "Draft" line to `*Released YYYY-MM-DD.*` and
      removes the "Unreleased" entries. The next pull request that needs a
      note creates the file again. Links in these pages are absolute, so they
      work on GitHub too: the release workflow uses
      `docs/release_notes/X.Y.Z.md` as the GitHub release text when it
      exists (GitHub's generated notes otherwise).

Nothing in these may state a fact that goes stale on its own (test counts,
"currently", dates of past work); link to the release notes instead.

### Cutting the release

One button: run the **bump-version** workflow (GitHub → Actions →
bump-version → Run workflow) on `main`, choosing `patch`, `minor`, `major`,
or an explicit `X.Y.Z` (pre-releases such as `1.3.0rc1` too). An explicit
version equal to `VERSION` releases the current commit as it is. It commits
the bumped `VERSION` as `Release vX.Y.Z`, tags, and dispatches the
**release** workflow, which:

1. checks that the tag matches `VERSION`;
2. builds the sdist and the wheel (`python -m build`) and checks them with
   `twine check`;
3. installs the built wheel and runs the test suite against it;
4. publishes both to PyPI (`twine`, `PYPI_Token_General` secret);
5. creates the GitHub release with the release notes page as its text and
   the two files attached (marked as a pre-release for `aN`/`bN`/`rcN`).

Dispatching the release workflow manually on a branch is a **dry run**: it
builds and tests, and publishes nothing.

If `main` is ever protected (pull requests required), give **GitHub
Actions** a bypass in that rule: bump-version pushes the `VERSION` commit
and the tag to `main` itself.

## CI

`ci.yml` runs on pushes to `main` and `dev` and on every pull request: ruff,
the full pytest matrix (Ubuntu 3.11/3.12/3.13, macOS 3.13, Windows 3.13), a
Python 3.14 smoke test, and the version-consistency check. The docs site
(`docs.yml`) builds with `mkdocs build --strict`, checks the built site for
private names, and deploys to GitHub Pages from `main`.
