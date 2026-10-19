# Releasing

`VERSION` is the single source of truth (the package reads it at build time).

1. Merge the changes to `main` (CI green).
2. **Actions → bump-version → Run workflow** on `main`: `patch`, `minor`, `major`, or an explicit
   version (`1.2.3`, `1.3.0rc1`). It commits `VERSION`, tags `vX.Y.Z` and starts **release**.
   An explicit version equal to `VERSION` releases the current commit as it is (no bump).
3. **release** builds the sdist and wheel, tests the wheel, uploads both to
   [PyPI](https://pypi.org/project/mef3io-coregistration/) (organisation secret
   `PYPI_Token_General`) and creates the GitHub Release.

`release` run by hand on a branch is a dry run: it builds and tests, and publishes nothing.

If `main` is ever protected (pull requests required), give **GitHub Actions** a bypass in that
rule: bump-version pushes the `VERSION` commit and the tag to `main` itself.
