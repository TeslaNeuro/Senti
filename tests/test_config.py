"""Configuration validation tests."""

import pytest

from app.config import AppConfig, find_duplicate_env_keys


def test_invalid_camera_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("CAMERA_WIDTH", "0")
  config = AppConfig.load()
  with pytest.raises(ValueError, match="Camera dimensions"):
    config.validate()


def test_invalid_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("LOG_LEVEL", "VERBOSE")
  config = AppConfig.load()
  with pytest.raises(ValueError, match="LOG_LEVEL"):
    config.validate()


def test_find_duplicate_env_keys(tmp_path) -> None:
  env_file = tmp_path / ".env"
  env_file.write_text("OCR_ENABLED=true\nOCR_ENABLED=false\n", encoding="utf-8")
  assert find_duplicate_env_keys(env_file) == ["OCR_ENABLED"]

