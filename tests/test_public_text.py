"""Public text names no private product, no recording device or dataset detail, and the
institution only in the affiliation line.

Scanned: every tracked text file (README, the docs site source, the examples, the package, the
tests, CLAUDE.md, the root community files, the workflows and templates). The private names are
stored encoded, so this file does not spell them out; the docs workflow checks the built site
with the same patterns.
"""
from __future__ import annotations

import base64
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_PRIVATE = re.compile(
    base64.b64decode(
        "Ym5lbFtfLV1zaWduYWxbXy1dY29yZWdpc3RyYXRpb258c2lnbmFsX2NvcmVnaXN0cmF0aW9uX2FwcA=="
    ).decode(),
    re.IGNORECASE,
)
# Recording devices and channel labels of a real dataset: public text says "a clinical system",
# "a second recorder", "ch1". Written in pieces so this file does not match itself.
_DEVICES = re.compile("na" + "tus|ca" + "dence", re.IGNORECASE)
_CHANNEL_LABELS = re.compile(r"\b(?:L" + "TP|L" + r"D)[0-9]")
# The one allowed mention of the institution (README, docs/index.md, CONTRIBUTORS.md, CLAUDE.md),
# the CITATION.cff affiliation field, and the code-of-conduct enforcement contact.
_ALLOWED = (
    "Maintained by Filip Mivalt, Bioelectronics Neurophysiology and Engineering Laboratory "
    "(BNEL) at Mayo Clinic.",
    'affiliation: "Bioelectronics Neurophysiology and Engineering Laboratory (BNEL), Mayo Clinic"',
    "mivalt.filip@mayo.edu",
)
_INSTITUTION = re.compile(r"\bbnel\b|mayo|bioelectronics neurophysiology", re.IGNORECASE)

_SUFFIXES = {".md", ".py", ".yml", ".yaml", ".toml", ".txt", ".cff", ".in", ".cfg", ".json"}
_THIS = Path(__file__).resolve()


def _tracked_files() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.split()
        paths = [ROOT / p for p in out]
    except (OSError, subprocess.CalledProcessError):  # no git (e.g. an unpacked sdist)
        skip = {".git", ".hypothesis", ".ruff_cache", ".pytest_cache", "build", "dist", "data", "site"}
        paths = [p for p in ROOT.rglob("*") if not skip & set(p.relative_to(ROOT).parts)]
    return sorted(
        p for p in paths
        if p.is_file() and p.suffix in _SUFFIXES and "__pycache__" not in p.parts
    )


FILES = _tracked_files()
_IDS = [p.relative_to(ROOT).as_posix() for p in FILES]
_OTHERS = [p for p in FILES if p.resolve() != _THIS]  # this file holds the institution rule


def _hit(text: str, pattern: re.Pattern) -> str | None:
    m = pattern.search(text)
    return None if m is None else f"...{text[max(0, m.start() - 60):m.end() + 60]}..."


def test_scan_covers_the_published_text():
    names = set(_IDS)
    for must in ("README.md", "pyproject.toml", "mkdocs.yml", "docs/index.md", "CITATION.cff",
                 "CLAUDE.md", "tests/conftest.py"):
        assert must in names


@pytest.mark.parametrize("path", FILES, ids=_IDS)
def test_no_private_product_name(path):
    hit = _hit(path.read_text(encoding="utf-8"), _PRIVATE)
    assert hit is None, f"{path.relative_to(ROOT)} names a private product: {hit}"


@pytest.mark.parametrize("path", FILES, ids=_IDS)
def test_no_device_names_or_dataset_channel_labels(path):
    text = path.read_text(encoding="utf-8")
    for pattern in (_DEVICES, _CHANNEL_LABELS):
        hit = _hit(text, pattern)
        assert hit is None, f"{path.relative_to(ROOT)} names a device or dataset label: {hit}"


@pytest.mark.parametrize("path", _OTHERS, ids=[p.relative_to(ROOT).as_posix() for p in _OTHERS])
def test_institution_only_in_the_affiliation_line(path):
    text = " ".join(path.read_text(encoding="utf-8").split())  # wrapped lines join
    for allowed in _ALLOWED:
        text = text.replace(allowed, "")
    hit = _hit(text, _INSTITUTION)
    assert hit is None, (
        f"{path.relative_to(ROOT)} mentions the institution outside the affiliation line: {hit}"
    )
