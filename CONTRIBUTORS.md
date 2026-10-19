# Contributors and credits

## mef3io-coregistration

- **Filip Mivalt** ([@xmival00](https://github.com/xmival00)), maintainer

Maintained by Filip Mivalt, Bioelectronics Neurophysiology and Engineering Laboratory (BNEL) at Mayo Clinic.

Contributions are welcome; see [CONTRIBUTING.md](https://github.com/bnelair/mef3io-coregistration/blob/main/CONTRIBUTING.md).

## Built on

- [mef3io](https://github.com/bnelair/mef3io): every MEF 3.0 read goes
  through it, including the legacy `mef_tools` API that
  `CoregisteredMefReader` mirrors. mef3io's own
  [contributors and credits](https://github.com/bnelair/mef3io/blob/main/CONTRIBUTORS.md)
  list the earlier MEF libraries (meflib, pymef, mef_tools and others) and the
  people behind them; we thank them all.
- [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/): the signal
  processing (filters, resampling, correlation, splines).

The MEF 3.0 format's design is described in Brinkmann et al., *Large-scale
electrophysiology: acquisition, compression, encryption, and storage of big
data*, J Neurosci Methods 180(1):185-192, 2009,
[doi:10.1016/j.jneumeth.2009.03.022](https://doi.org/10.1016/j.jneumeth.2009.03.022).

If you contributed and are missing or misnamed, please open an issue.
