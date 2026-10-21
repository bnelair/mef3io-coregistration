"""Results, clock maps and mappings as plain or gzip-compressed JSON, and reopening a saved result."""
from __future__ import annotations

import gzip
import shutil

import numpy as np
import pytest
from test_open_and_merge import pair  # noqa: F401  (the synthetic pair fixture)

from mef3io_coregistration.io.jsonfile import (
    content_sha256,
    is_gzip,
    other_form,
    read_json,
    write_json,
)
from mef3io_coregistration.reader import CoregisteredRecording
from mef3io_coregistration.result import CoregResult
from mef3io_coregistration.session import CoregSession
from mef3io_coregistration.transform import ClockMap


def test_json_files_are_written_compressed_by_name_and_read_by_content(tmp_path):
    doc = {"b": 1, "a": [1, 2, 3], "t": "µs"}
    plain, packed = tmp_path / "x.json", tmp_path / "x.json.gz"
    write_json(plain, doc)
    write_json(packed, doc)
    assert not is_gzip(plain.read_bytes()) and is_gzip(packed.read_bytes())
    assert read_json(plain) == read_json(packed) == doc
    renamed = tmp_path / "renamed.json"  # a downloaded copy that lost its .gz
    shutil.copy(packed, renamed)
    assert read_json(renamed) == doc
    assert content_sha256(plain) == content_sha256(packed) == content_sha256(renamed)
    first = packed.read_bytes()
    write_json(packed, doc)
    assert packed.read_bytes() == first  # byte-stable: no timestamp in the gzip header
    assert gzip.decompress(first) == plain.read_bytes()
    assert other_form(plain) == packed and other_form(packed) == plain


def test_result_and_clock_map_round_trip_compressed(pair, tmp_path):  # noqa: F811
    res = CoregResult.load(pair[0])
    gz = res.save(tmp_path / "r.coreg.json.gz")
    assert is_gzip(gz.read_bytes())
    again = CoregResult.load(gz)
    assert again.to_dict() == res.to_dict()
    assert CoregResult.from_bytes(res.to_bytes(compress=True)).to_dict() == res.to_dict()
    assert CoregResult.from_bytes(res.to_bytes()).to_dict() == res.to_dict()
    cm = res.clock_map.save(tmp_path / "map.json.gz")
    np.testing.assert_array_equal(ClockMap.load(cm).y_moving, res.clock_map.y_moving)


def test_a_mapping_finds_and_accepts_its_result_in_either_form(pair, tmp_path):  # noqa: F811
    res_p = pair[0]
    folder = tmp_path / "shared"
    folder.mkdir()
    plain = shutil.copy(res_p, folder / "pair.coreg.json")
    rec = CoregisteredRecording.open(plain)
    rec.set_channel("x", a="x", b="x", prefer="A")
    cfg = rec.save_config(folder / "pair.recording.json.gz", result_path=plain)
    assert is_gzip(cfg.read_bytes())
    # the result is later kept compressed only: the mapping still finds and accepts it
    CoregResult.load(plain).save(folder / "pair.coreg.json.gz")
    plain.unlink()
    again = CoregisteredRecording.from_config(cfg)
    assert list(again.channels) == ["x"]


def test_a_saved_result_reopens_in_a_session(pair, tmp_path):  # noqa: F811
    res_p, a_p, _, other_p = pair
    gz = CoregResult.load(res_p).save(tmp_path / "r.coreg.json.gz")
    sess = CoregSession(background=False).open_result(gz)
    assert sess.reference_role == "A" and sess.clock_map is not None
    out = sess.result()
    saved = CoregResult.load(res_p)
    assert out.to_dict()["clock_map"] == saved.to_dict()["clock_map"]
    assert out.anchors == saved.anchors and out.created_utc == saved.created_utc
    with sess.open_reader() as rd:
        assert rd.paths["A"] == str(a_p)
    assert sess.reopened is not None
    sess.clock_map = None  # any new map (or none) ends it
    assert sess.reopened is None
    with pytest.raises(ValueError, match="spans"):  # the time-span check applies
        CoregSession().open_result(gz, file_b=other_p)
