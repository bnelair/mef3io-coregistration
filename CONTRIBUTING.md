# Contributing

```bash
git clone https://github.com/bnelair/mef3io-coregistration && cd mef3io-coregistration
pip install -e ".[dev]"
ruff check . && pytest
```

Work on a branch and open a pull request against `main`; CI (lint + tests on Linux, Windows and
macOS) must pass. Keep `mef3io_coregistration` free of anything beyond numpy, scipy and mef3io.

The JSON a coregistration is saved as (`signal_coregistration.coreg_result`, `…clock_map`,
`…recording`) is read by other tools: keep old files loading.
