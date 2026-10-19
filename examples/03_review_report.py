"""Plotted review of a saved coregistration; the same as the ``sigcoreg-review`` command.

    python examples/03_review_report.py path/to/result.coreg.json
    python examples/03_review_report.py result.coreg.json --channel ch1-ch2 --out review/

Writes PNG figures and an ``index.html`` next to the result (or in ``--out``). Needs your own
result and the two recordings it names, and the ``report`` extra (matplotlib):
``pip install "mef3io-coregistration[report]"``.
"""
from mef3io_coregistration.review import main

if __name__ == "__main__":
    main()
