"""Tests for src.core.state.SimulationState.

Verifies default attribute initialization, attribute mutation,
instance independence, and full state update cycles.
"""

import numpy as np

from src.core.state import SimulationState


def test_initial_frame_is_none():
    """Frame should default to None on a fresh SimulationState."""
    state = SimulationState()
    assert state.frame is None


def test_initial_depth_array_is_none():
    """Depth array should default to None on a fresh SimulationState."""
    state = SimulationState()
    assert state.depth_array is None


def test_initial_results_is_none():
    """Results should default to None on a fresh SimulationState."""
    state = SimulationState()
    assert state.results is None


def test_initial_kalman_predictions_is_none():
    """Kalman predictions should default to None on a fresh SimulationState."""
    state = SimulationState()
    assert state.kalman_predictions is None


def test_initial_kalman_filters_empty_dict():
    """Kalman filters should default to an empty dict on a fresh SimulationState."""
    state = SimulationState()
    assert state.kalman_filters == {}
    assert isinstance(state.kalman_filters, dict)


def test_initial_ego_speed_zero():
    """Ego speed should default to 0.0 on a fresh SimulationState."""
    state = SimulationState()
    assert state.ego_speed == 0.0


def test_initial_brake_needed_false():
    """Brake needed flag should default to False on a fresh SimulationState."""
    state = SimulationState()
    assert state.brake_needed is False


def test_set_frame():
    """Setting frame to a numpy array should persist the value."""
    state = SimulationState()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    state.frame = frame
    assert state.frame is frame
    assert state.frame.shape == (480, 640, 3)


def test_set_ego_speed():
    """Ego speed should be updatable to arbitrary float values."""
    state = SimulationState()
    state.ego_speed = 42.5
    assert state.ego_speed == 42.5


def test_set_brake_needed():
    """Brake needed flag should be togglable between True and False."""
    state = SimulationState()
    assert state.brake_needed is False

    state.brake_needed = True
    assert state.brake_needed is True

    state.brake_needed = False
    assert state.brake_needed is False


def test_kalman_filters_independent_instances():
    """Two SimulationState instances must not share the same kalman_filters dict."""
    state_a = SimulationState()
    state_b = SimulationState()

    state_a.kalman_filters["vehicle_1"] = "filter_a"

    assert "vehicle_1" not in state_b.kalman_filters
    assert state_a.kalman_filters is not state_b.kalman_filters


def test_full_state_update_cycle():
    """Setting multiple fields in one cycle should all persist independently."""
    state = SimulationState()

    frame = np.ones((720, 1280, 3), dtype=np.uint8)
    depth = np.random.rand(720, 1280).astype(np.float32)
    results = [{"label": "car", "confidence": 0.95}]
    predictions = {"vehicle_1": (10.0, 20.0)}

    state.frame = frame
    state.depth_array = depth
    state.results = results
    state.kalman_predictions = predictions
    state.kalman_filters["vehicle_1"] = "kf_object"
    state.ego_speed = 30.0
    state.brake_needed = True

    assert state.frame is frame
    np.testing.assert_array_equal(state.depth_array, depth)
    assert state.results == results
    assert state.kalman_predictions == predictions
    assert state.kalman_filters == {"vehicle_1": "kf_object"}
    assert state.ego_speed == 30.0
    assert state.brake_needed is True
