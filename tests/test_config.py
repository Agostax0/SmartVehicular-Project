"""Tests for src.core.config.load_config.

Verifies YAML loading for valid files, nested structures, missing files,
empty files, and the actual project config.
"""

from pathlib import Path

import pytest

from src.core.config import load_config


def test_load_valid_yaml(tmp_path):
    """load_config should parse a well-formed YAML file and return its contents."""
    config_file = tmp_path / "valid.yaml"
    config_file.write_text("server:\n  host: localhost\n  port: 8080\n")

    result = load_config(str(config_file))

    assert result == {"server": {"host": "localhost", "port": 8080}}


def test_load_config_returns_dict(tmp_path):
    """load_config should return a dict for a valid YAML mapping."""
    config_file = tmp_path / "simple.yaml"
    config_file.write_text("key: value\n")

    result = load_config(str(config_file))

    assert isinstance(result, dict)


def test_load_config_nested_keys(tmp_path):
    """Nested YAML mappings should be accessible via chained dict lookups."""
    config_file = tmp_path / "nested.yaml"
    config_file.write_text(
        "level1:\n"
        "  level2:\n"
        "    level3: deep_value\n"
    )

    result = load_config(str(config_file))

    assert result["level1"]["level2"]["level3"] == "deep_value"


def test_load_config_missing_file():
    """load_config should raise FileNotFoundError for a nonexistent path."""
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/to/config.yaml")


def test_load_config_empty_file(tmp_path):
    """load_config should return None for an empty YAML file."""
    config_file = tmp_path / "empty.yaml"
    config_file.write_text("")

    result = load_config(str(config_file))

    assert result is None


def test_load_real_config():
    """The actual project config.yaml should load and contain expected top-level keys."""
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config" / "config.yaml"

    if not config_path.exists():
        pytest.skip(f"Project config not found at {config_path}")

    result = load_config(str(config_path))

    assert isinstance(result, dict), "config.yaml should parse to a dict"
    assert len(result) > 0, "config.yaml should not be empty"
