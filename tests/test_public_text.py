"""Public text names no private product, and the institution only in the affiliation line.

Scanned: everything that is published (README, the docs site source, the examples, the package,
the root community files, the workflows and templates). The private names are stored encoded,
so this file does not spell them out; the docs workflow checks the built site with the same
pattern. Tests and CLAUDE.md are exempt from the private-name scan only because they hold the
rule itself — keep them clean all the same.
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_PRIVATE = re.compile(
    base64.b64decode(
        "Ym5lbFtfLV1zaWduYWxbXy1dY29yZWdpc3RyYXRpb258c2lnbmFsX2NvcmVnaXN0cmF0aW9uX2FwcA=="
    ).decode(),
    re.IGNORECASE,
)
# The one allowed mention of the institution (README, docs/index.md, CONTRIBUTORS.md), the
# CITATION.cff affiliation field, and the code-of-conduct enforcement contact.
_AFFILIATION = (
    "Maintained by Filip Mivalt, Bioelectronics Neurophysiology and Engineering Laboratory "
    "(BNEL) at Mayo Clinic."
)
_CITATION_AFFILIATION = 'affiliation: "Bioelectronics Neurophysiology and Engineering Laboratory (BNEL), Mayo Clinic"'
_INSTITUTION = re.compile(r"\bBNEL\b|Mayo|Bioelectronics Neurophysiology")

_SUFFIXES = {".md", ".py", ".yml", ".yaml", ".toml", ".txt", ".cff", ".in", ".cfg", ".json"}


def _public_files() -> list[Path]:
    files = [p for p in ROOT.iterdir() if p.is_file() and p.suffix in _SUFFIXES]
    for d in ("docs", "examples", "mef3io_coregistration", "benchmarks", ".github"):
        files += [p for p in (ROOT / d).rglob("*") if p.is_file() and p.suffix in _SUFFIXES]
    return sorted(p for p in files if p.name != "CLAUDE.md" and "__pycache__" not in p.parts)


FILES = _public_files()


def test_scan_covers_the_published_text():
    names = {p.relative_to(ROOT).as_posix() for p in FILES}
    for must in ("README.md", "pyproject.toml", "mkdocs.yml", "docs/index.md", "CITATION.cff"):
        assert must in names


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_no_private_product_name(path):
    text = path.read_text(encoding="utf-8")
    m = _PRIVATE.search(text)
    assert m is None, f"{path.relative_to(ROOT)} names a private product: {m.group(0)!r}"


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_institution_only_in_the_affiliation_line(path):
    text = " ".join(path.read_text(encoding="utf-8").split())  # wrapped lines join
    text = text.replace(_AFFILIATION, "").replace(_CITATION_AFFILIATION, "")
    m = _INSTITUTION.search(text)
    assert m is None, (
        f"{path.relative_to(ROOT)} mentions the institution outside the affiliation line: "
        f"...{text[max(0, m.start() - 60):m.end() + 60]}..."
    )
