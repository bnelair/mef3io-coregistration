"""Where generated data goes: the override, a source checkout, an installed package."""
from __future__ import annotations

from pathlib import Path

from mef3io_coregistration import config


def _fake_package(root: Path, checkout: bool) -> str:
    pkg = root / "mef3io_coregistration"
    pkg.mkdir(parents=True)
    if checkout:
        (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    return str(pkg / "config.py")


def test_env_variable_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("SIGCOREG_DATA_DIR", str(tmp_path / "mine"))
    assert config._default_data_dir() == (tmp_path / "mine").resolve()


def test_source_checkout_uses_data_next_to_it(monkeypatch, tmp_path):
    monkeypatch.delenv("SIGCOREG_DATA_DIR", raising=False)
    monkeypatch.setattr(config, "__file__", _fake_package(tmp_path / "repo", checkout=True))
    assert config._default_data_dir() == (tmp_path / "repo" / "data").resolve()


def test_installed_package_uses_the_user_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("SIGCOREG_DATA_DIR", raising=False)
    monkeypatch.setattr(config, "__file__", _fake_package(tmp_path / "site-packages", checkout=False))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    got = config._default_data_dir()
    assert got.name == "mef3io-coregistration"
    assert got.parent in {tmp_path / "xdg", tmp_path / "local"}
    assert "site-packages" not in got.parts
