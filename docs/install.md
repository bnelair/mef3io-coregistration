---
description: "Install mef3io-coregistration from PyPI, with the optional review report, or from source for development."
---

# Install

```bash
pip install mef3io-coregistration
```

Python 3.11 or newer on Linux, macOS or Windows. It is pure Python; the
dependencies are numpy, scipy and [mef3io](https://pypi.org/project/mef3io/)
(which brings the compiled MEF 3.0 core as a wheel for every platform).

## Optional extras

| Extra | Adds | For |
|---|---|---|
| `report` | matplotlib | the plotted [review report](howto/review_report.md) (`sigcoreg-review`) |
| `test` | pytest, hypothesis, matplotlib | running the test suite |
| `docs` | MkDocs Material, mkdocstrings | building this site |
| `dev` | the test tools, ruff, build, twine | development |

```bash
pip install "mef3io-coregistration[report]"
```

## From source

```bash
git clone https://github.com/bnelair/mef3io-coregistration && cd mef3io-coregistration
pip install -e ".[dev]"
python -m pytest tests
```

See [CONTRIBUTING.md](https://github.com/bnelair/mef3io-coregistration/blob/main/CONTRIBUTING.md)
for the development workflow.

## Check the install

```bash
python -c "import mef3io_coregistration, mef3io; print(mef3io_coregistration.__version__, mef3io.__version__)"
python examples/01_align_synthetic_pair.py      # from a source checkout: the whole workflow on synthetic data
```
