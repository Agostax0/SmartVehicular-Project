"""Tests for utility helpers."""

from src.utils.logger import get_logger


def test_get_logger_returns_logger():
    logger = get_logger("test")
    assert logger is not None
    assert logger.name == "test"
