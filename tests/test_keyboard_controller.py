"""Tests for the KeyboardController in src/controllers/keyboard_controller.py."""

import pytest

pygame = pytest.importorskip("pygame", reason="pygame is required for keyboard controller tests")

from src.controllers.keyboard_controller import KeyboardController


@pytest.fixture
def controller():
    return KeyboardController()


def _make_keys(**pressed):
    """Create a mock key state where only the specified keys are pressed.

    Uses a size large enough to cover all pygame key constants (SDL2 can
    use constants above 1000 via scancode mapping).
    """
    size = max(getattr(pygame, k) for k in dir(pygame) if k.startswith("K_")) + 1
    keys = [False] * size
    for key_name, val in pressed.items():
        key_const = getattr(pygame, key_name)
        keys[key_const] = val
    return keys


# ── No keys pressed ──────────────────────────────────────────────────────


def test_no_keys_pressed(controller):
    """All outputs should be zero / False when no key is pressed."""
    keys = _make_keys()
    ctrl = controller.parse(keys)
    assert ctrl["throttle"] == 0.0
    assert ctrl["brake"] == 0.0
    assert ctrl["steer"] == 0.0
    assert ctrl["hand_brake"] is False


# ── Throttle ─────────────────────────────────────────────────────────────


def test_throttle_w(controller):
    ctrl = controller.parse(_make_keys(K_w=True))
    assert ctrl["throttle"] > 0.0
    assert ctrl["brake"] == 0.0


def test_throttle_up(controller):
    ctrl = controller.parse(_make_keys(K_UP=True))
    assert ctrl["throttle"] > 0.0


# ── Brake ────────────────────────────────────────────────────────────────


def test_brake_s(controller):
    ctrl = controller.parse(_make_keys(K_s=True))
    assert ctrl["brake"] > 0.0
    assert ctrl["throttle"] == 0.0


def test_brake_down(controller):
    ctrl = controller.parse(_make_keys(K_DOWN=True))
    assert ctrl["brake"] > 0.0


# ── Steer ────────────────────────────────────────────────────────────────


def test_steer_left_a(controller):
    ctrl = controller.parse(_make_keys(K_a=True))
    assert ctrl["steer"] < 0.0


def test_steer_left_arrow(controller):
    ctrl = controller.parse(_make_keys(K_LEFT=True))
    assert ctrl["steer"] < 0.0


def test_steer_right_d(controller):
    ctrl = controller.parse(_make_keys(K_d=True))
    assert ctrl["steer"] > 0.0


def test_steer_right_arrow(controller):
    ctrl = controller.parse(_make_keys(K_RIGHT=True))
    assert ctrl["steer"] > 0.0


# ── Hand brake ───────────────────────────────────────────────────────────


def test_hand_brake_space(controller):
    ctrl = controller.parse(_make_keys(K_SPACE=True))
    assert ctrl["hand_brake"] is True


# ── Combinations ─────────────────────────────────────────────────────────


def test_throttle_and_steer(controller):
    ctrl = controller.parse(_make_keys(K_w=True, K_d=True))
    assert ctrl["throttle"] > 0.0
    assert ctrl["steer"] > 0.0
    assert ctrl["brake"] == 0.0


def test_returns_dict_with_all_keys(controller):
    ctrl = controller.parse(_make_keys())
    assert set(ctrl.keys()) == {"throttle", "steer", "brake", "hand_brake"}
