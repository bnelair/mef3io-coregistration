"""JSON files of this package — results, clock maps, recording mappings — plain or gzip-compressed.

A file is written compressed when its name ends in ``.gz`` (e.g. ``result.coreg.json.gz``), and
read compressed whenever its content is gzip, whatever its name — so a renamed or downloaded copy
always opens. The gzip header carries no timestamp or name, so re-saving gives the same bytes with
the same zlib; hashes and a mapping's check of its result use the decompressed content, which is
the same everywhere. Decompression is bounded (``max_bytes``): a result is a few MB even for days
of recording, so a far larger stream is refused rather than allowed to exhaust memory.

Usage::

    from mef3io_coregistration.io.jsonfile import read_json, write_json
    write_json("result.coreg.json.gz", doc)     # compressed: the name says so
    doc = read_json("result.coreg.json.gz")     # either kind, detected from the content
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import zlib
from pathlib import Path, PurePath, PureWindowsPath

#: The first two bytes of every gzip stream (RFC 1952).
GZIP_MAGIC = b"\x1f\x8b"
#: The most a file of this package may decompress to (1 GiB; a result is a few MB).
MAX_BYTES = 1024 * 1024 * 1024


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


def plain_bytes(data: bytes, max_bytes: int = MAX_BYTES) -> bytes:
    """``data`` decompressed if it is gzip, unchanged otherwise. Raises :class:`ValueError` for a
    damaged gzip stream, or one that decompresses to more than ``max_bytes``."""
    if not is_gzip(data):
        return data
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:
            out = gz.read(max_bytes + 1)
    except (EOFError, OSError, zlib.error) as exc:
        raise ValueError(f"not a valid gzip stream ({exc})") from exc
    if len(out) > max_bytes:
        raise ValueError(f"the file decompresses to more than {max_bytes} bytes; not a result")
    return out


def loads(data: bytes | str, max_bytes: int = MAX_BYTES) -> dict:
    """A document from plain or gzip-compressed JSON bytes (or a JSON string). Raises
    :class:`ValueError` for anything that is not a JSON object (a damaged or foreign file)."""
    if isinstance(data, bytes):
        data = plain_bytes(data, max_bytes).decode("utf-8-sig")  # a BOM from an editor is fine
    doc = json.loads(data)
    if not isinstance(doc, dict):
        raise ValueError("not a file of this package (the JSON is not an object)")
    return doc


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


def file_name(path: str | Path) -> str:
    """The file name of a stored path, also when it was written on Windows (``C:\\d\\x.json``)."""
    s = str(path)
    return PureWindowsPath(s).name if "\\" in s else PurePath(s).name


def other_form(path: str | Path) -> Path:
    """The same file in the other form: ``x.json`` ↔ ``x.json.gz``."""
    p = Path(path)
    return p.with_name(p.name[:-3]) if p.name.lower().endswith(".gz") else p.with_name(p.name + ".gz")
