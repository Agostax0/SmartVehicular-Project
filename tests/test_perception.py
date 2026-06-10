"""Tests for the PerceptionSystem class in src/core/perception.py.

Covers initialisation (including defaults), focal-length math,
depth-image processing, and the analyze_detections pipeline with
mocked Kalman filters and collision checks.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from src.core.perception import PerceptionSystem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(fps=20, width=800, height=600, fov=60):
    """Return a minimal config dict accepted by PerceptionSystem."""
    return {
        "simulation": {"fps": fps},
        "sensors": {"camera": {"width": width, "height": height, "fov": fov}},
    }


def _make_depth_image(height, width, r_val=0, g_val=0, b_val=128):
    """Create a mock CARLA-style depth image with known BGRA raw_data.

    Each pixel has B=b_val, G=g_val, R=r_val, A=255.
    Returns a mock object with .raw_data, .height, .width attributes.
    """
    pixel = np.array([b_val, g_val, r_val, 255], dtype=np.uint8)
    raw = np.tile(pixel, height * width)
    mock_img = MagicMock()
    mock_img.raw_data = raw.tobytes()
    mock_img.height = height
    mock_img.width = width
    return mock_img


def _make_state(results=None, depth_array=None, ego_speed=0.0):
    """Return a lightweight mock SimulationState."""
    state = MagicMock()
    state.results = results
    state.depth_array = depth_array
    state.ego_speed = ego_speed
    state.kalman_filters = {}
    state.kalman_predictions = {}
    state.brake_needed = False
    return state


def _make_scenario(prediction_future_frames=10):
    scenario = MagicMock()
    scenario.prediction_future_frames = prediction_future_frames
    return scenario


# ---------------------------------------------------------------------------
# Initialisation tests
# ---------------------------------------------------------------------------

def test_perception_system_init():
    """PerceptionSystem stores fps, img_w, img_h and computes f_length."""
    config = _make_config(fps=30, width=1024, height=768, fov=90)
    ps = PerceptionSystem(config)

    assert ps.fps == 30
    assert ps.img_w == 1024
    assert ps.img_h == 768

    expected_f = (1024 / 2.0) / np.tan(np.deg2rad(90) / 2.0)
    assert pytest.approx(ps.f_length, rel=1e-6) == expected_f


def test_perception_system_default_fps():
    """fps defaults to 20 when the config omits it."""
    config = {
        "simulation": {},
        "sensors": {"camera": {"width": 800, "height": 600, "fov": 60}},
    }
    ps = PerceptionSystem(config)
    assert ps.fps == 20


def test_focal_length_calculation():
    """Manually verify focal length for a 90° FOV on 800-px wide sensor.

    f = (width / 2) / tan(fov_rad / 2)
      = 400 / tan(45°)
      = 400 / 1.0
      = 400.0
    """
    config = _make_config(width=800, height=600, fov=90)
    ps = PerceptionSystem(config)
    assert pytest.approx(ps.f_length, abs=1e-6) == 400.0


# ---------------------------------------------------------------------------
# Depth processing tests
# ---------------------------------------------------------------------------

def test_process_depth_converts_raw_data():
    """process_depth should decode BGRA bytes into depth in metres.

    With R=0, G=0, B=128 the normalised depth is:
        (0 + 0*256 + 128*256^2) / (256^3 - 1)
    and the returned depth = 1000 * normalised_depth.
    """
    height, width = 4, 6
    b_val, g_val, r_val = 128, 0, 0
    mock_img = _make_depth_image(height, width, r_val=r_val, g_val=g_val, b_val=b_val)

    config = _make_config(width=width, height=height)
    ps = PerceptionSystem(config)
    depth = ps.process_depth(mock_img)

    assert depth.shape == (height, width)

    normalized = (r_val + g_val * 256.0 + b_val * 256.0 * 256.0) / (256.0**3 - 1.0)
    expected = 1000.0 * normalized
    np.testing.assert_allclose(depth, expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# analyze_detections tests
# ---------------------------------------------------------------------------

def test_analyze_detections_no_results():
    """When state.results is None, predictions should be empty and no brake."""
    ps = PerceptionSystem(_make_config())
    state = _make_state(results=None)
    scenario = _make_scenario()

    ps.analyze_detections(state, scenario)

    assert state.kalman_predictions == {}
    assert state.brake_needed is False


def test_analyze_detections_empty_boxes():
    """When results exist but boxes list is empty, predictions stay empty."""
    ps = PerceptionSystem(_make_config())

    results = MagicMock()
    results.boxes = []  # empty → falsy
    state = _make_state(results=results)
    scenario = _make_scenario()

    ps.analyze_detections(state, scenario)

    assert state.kalman_predictions == {}
    assert state.brake_needed is False


@patch("src.core.perception.check_collision_risk", return_value=(False, "", 999.0))
@patch("src.core.perception.TrajectoryKalmanFilter")
def test_analyze_detections_creates_kalman_filter(mock_kf_cls, mock_collision):
    """A new TrajectoryKalmanFilter should be created for an unseen obj_id."""
    ps = PerceptionSystem(_make_config())

    # Prepare a single detection box with id=7
    box = MagicMock()
    box.xyxy = [np.array([100.0, 150.0, 200.0, 250.0])]

    boxes = MagicMock()
    boxes.__iter__ = MagicMock(return_value=iter([box]))
    boxes.__bool__ = MagicMock(return_value=True)
    boxes.id = [7]

    results = MagicMock()
    results.boxes = boxes

    state = _make_state(results=results, depth_array=None, ego_speed=5.0)
    scenario = _make_scenario(prediction_future_frames=10)

    # Configure the mock Kalman filter instance
    kf_instance = MagicMock()
    kf_instance.predict_update.return_value = (0.0, 10.0, 0.0, 5.0)
    kf_instance.dt = 1.0 / 20
    mock_kf_cls.return_value = kf_instance

    ps.analyze_detections(state, scenario)

    # A filter should have been created for obj_id 7
    assert 7 in state.kalman_filters
    mock_kf_cls.assert_called_once_with(dt=1.0 / 20)


@patch("src.core.perception.check_collision_risk", return_value=(True, "Collision imminent", 1.5))
@patch("src.core.perception.TrajectoryKalmanFilter")
def test_analyze_detections_sets_brake_for_collision(mock_kf_cls, mock_collision):
    """brake_needed must be True when check_collision_risk returns risk=True."""
    ps = PerceptionSystem(_make_config())

    box = MagicMock()
    box.xyxy = [np.array([100.0, 150.0, 200.0, 250.0])]

    boxes = MagicMock()
    boxes.__iter__ = MagicMock(return_value=iter([box]))
    boxes.__bool__ = MagicMock(return_value=True)
    boxes.id = [42]

    results = MagicMock()
    results.boxes = boxes

    depth = np.full((600, 800), 15.0, dtype=np.float32)
    state = _make_state(results=results, depth_array=depth, ego_speed=10.0)
    scenario = _make_scenario(prediction_future_frames=10)

    kf_instance = MagicMock()
    kf_instance.predict_update.return_value = (0.0, 15.0, 0.0, 10.0)
    kf_instance.dt = 1.0 / 20
    mock_kf_cls.return_value = kf_instance

    ps.analyze_detections(state, scenario)

    assert state.brake_needed is True
    mock_collision.assert_called_once()
