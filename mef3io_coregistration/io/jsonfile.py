"""JSON files of this package — results, clock maps, recording mappings — plain or gzip-compressed.

A file is written compressed when its name ends in ``.gz`` (e.g. ``result.coreg.json.gz``), and
read compressed whenever its content is gzip, whatever its name — so a renamed or downloaded copy
always opens. Compressed files are byte-stable (no timestamp or name in the gzip header), so saving
the same document twice gives the same bytes, and the same hash.

>>> from mef3io_coregistration.io.jsonfile import read_json, write_json
>>> write_json("result.coreg.json.gz", {"format": "…"})      # compressed: the name says so
>>> read_json("result.coreg.json.gz")                           # either kind, detected
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path

#: The first two bytes of every gzip stream (RFC 1952).
GZIP_MAGIC = b"\x1f\x8b"


def is_gzip(data: bytes) -> bool:
    """Whether ``data`` is a gzip stream (by its magic bytes, not by a file name)."""
    return data[:2] == GZIP_MAGIC


def dumps(doc: dict, compress: bool = False) -> bytes:
    """``doc`` as this package writes it: sorted keys, two-space indent, ``\\n`` line ends,
    UTF-8 — gzip-compressed (byte-stable) when ``compress``."""
    text = json.dumps(doc, indent=2, sort_keys=True).encode("utf-8")
    return gzip_bytes(text) if compress else text


def gzip_bytes(data: bytes) -> bytes:
    """``data`` gzip-compressed with no timestamp or file name in the header (byte-stable)."""
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0, compresslevel=9) as gz:
        gz.write(data)
    return buf.getvalue()


def plain_bytes(data: bytes) -> bytes:
    """``data`` decompressed if it is gzip, unchanged otherwise."""
    return gzip.decompress(data) if is_gzip(data) else data


def loads(data: bytes | str) -> dict:
    """A document from plain or gzip-compressed JSON bytes (or a JSON string)."""
    if isinstance(data, str):
        return json.loads(data)
    return json.loads(plain_bytes(data).decode("utf-8"))


def read_json(path: str | Path) -> dict:
    """Read a plain or gzip-compressed JSON file (detected from its content)."""
    return loads(Path(path).read_bytes())


def write_json(path: str | Path, doc: dict) -> Path:
    """Write ``doc``; compressed when the file name ends in ``.gz``. Returns the path."""
    path = Path(path)
    path.write_bytes(dumps(doc, compress=path.name.lower().endswith(".gz")))
    return path


def content_sha256(source: str | Path | bytes) -> str:
    """SHA-256 of the JSON *content* — the same for a file and its compressed copy, so a mapping
    that names its result by hash accepts either form."""
    data = source if isinstance(source, bytes) else Path(source).read_bytes()
    return hashlib.sha256(plain_bytes(data)).hexdigest()


def other_form(path: str | Path) -> Path:
    """The same file in the other form: ``x.json`` ↔ ``x.json.gz``."""
    p = Path(path)
    return p.with_name(p.name[:-3]) if p.name.lower().endswith(".gz") else p.with_name(p.name + ".gz")
