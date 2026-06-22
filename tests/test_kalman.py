"""
Tests for src.utils.kalman.TrajectoryKalmanFilter.

Covers initialisation, predict-update behaviour, velocity convergence,
ego-motion compensation, filter independence, and noise smoothing.
"""

import numpy as np
import pytest

from src.utils.kalman import TrajectoryKalmanFilter


# ── Initialisation ──────────────────────────────────────────────────────────


def test_initialization_default_dt():
    """Default dt should be 0.05 and filter should not be initialised."""
    kf = TrajectoryKalmanFilter()
    assert kf.dt == pytest.approx(0.05)
    assert kf.initialized is False


def test_initialization_custom_dt():
    """A custom dt value should be stored and used in the matrices."""
    kf = TrajectoryKalmanFilter(dt=0.1)
    assert kf.dt == pytest.approx(0.1)
    # Transition matrix should embed the custom dt for position-velocity coupling
    assert kf.kf.transitionMatrix[0, 2] == pytest.approx(0.1)
    assert kf.kf.transitionMatrix[1, 3] == pytest.approx(0.1)


def test_predict_update_refreshes_dt_dependent_matrices():
    """Real-time updates can pass the actual elapsed frame time."""
    kf = TrajectoryKalmanFilter(dt=0.05)
    kf.predict_update(0.0, 0.0, dt=0.08)

    assert kf.dt == pytest.approx(0.08)
    assert kf.kf.transitionMatrix[0, 2] == pytest.approx(0.08)
    assert kf.kf.controlMatrix[0, 0] == pytest.approx(-0.08)


# ── First call behaviour ───────────────────────────────────────────────────


def test_first_call_returns_measured_values():
    """The very first predict_update should echo the measurement with zero velocities."""
    kf = TrajectoryKalmanFilter()
    est_x, est_z, vel_x, vel_z = kf.predict_update(3.0, 7.0)

    assert est_x == pytest.approx(3.0)
    assert est_z == pytest.approx(7.0)
    assert vel_x == pytest.approx(0.0)
    assert vel_z == pytest.approx(0.0)


def test_first_call_sets_initialized():
    """After the first call the filter should be marked as initialised."""
    kf = TrajectoryKalmanFilter()
    assert kf.initialized is False
    kf.predict_update(1.0, 2.0)
    assert kf.initialized is True
    assert kf.update_count == 1


# ── Return types ────────────────────────────────────────────────────────────


def test_second_call_returns_four_floats():
    """Every element returned by predict_update should be a Python float."""
    kf = TrajectoryKalmanFilter()
    kf.predict_update(0.0, 0.0)  # initialise
    result = kf.predict_update(1.0, 1.0)

    assert len(result) == 4
    for value in result:
        assert isinstance(value, float), f"Expected float, got {type(value)}"
    assert kf.update_count == 2


# ── Stationary target ──────────────────────────────────────────────────────


def test_stationary_pedestrian_velocity_near_zero():
    """Feeding a constant position should keep velocity estimates close to zero."""
    kf = TrajectoryKalmanFilter()
    pos_x, pos_z = 5.0, 10.0

    for _ in range(50):
        _, _, vel_x, vel_z = kf.predict_update(pos_x, pos_z)

    assert vel_x == pytest.approx(0.0, abs=0.05)
    assert vel_z == pytest.approx(0.0, abs=0.05)


# ── Moving target ──────────────────────────────────────────────────────────


def test_moving_pedestrian_velocity_converges():
    """Linearly increasing X should cause vel_x to converge to a positive value."""
    kf = TrajectoryKalmanFilter(dt=0.05)
    speed = 2.0  # units per second → displacement per step = speed * dt

    for step in range(100):
        x = speed * step * kf.dt
        z = 0.0
        _, _, vel_x, _ = kf.predict_update(x, z)

    # vel_x should converge to the true speed in m/s (2.0 units/s)
    assert vel_x > 0.0, "vel_x should be positive for increasing X"
    assert vel_x == pytest.approx(speed, abs=0.1)


# ── Position tracking ──────────────────────────────────────────────────────


def test_position_estimate_tracks_measurement():
    """Estimated position should remain close to the true measurement."""
    kf = TrajectoryKalmanFilter()

    for step in range(60):
        true_x = 0.5 * step * kf.dt
        true_z = -0.3 * step * kf.dt
        est_x, est_z, _, _ = kf.predict_update(true_x, true_z)

    # After converging, the estimate should be very close to the measurement
    assert est_x == pytest.approx(true_x, abs=0.1)
    assert est_z == pytest.approx(true_z, abs=0.1)


# ── Ego-motion compensation ────────────────────────────────────────────────


def test_ego_motion_compensation():
    """Providing ego velocity should shift position estimates relative to no ego velocity."""
    kf_no_ego = TrajectoryKalmanFilter()
    kf_with_ego = TrajectoryKalmanFilter()

    pos_x, pos_z = 10.0, 10.0
    ego_vx, ego_vz = 1.0, 0.0

    # Initialise both
    kf_no_ego.predict_update(pos_x, pos_z)
    kf_with_ego.predict_update(pos_x, pos_z)

    # Run several steps with the same measurements but different ego velocities
    for _ in range(20):
        est_no = kf_no_ego.predict_update(pos_x, pos_z, ego_vx=0.0, ego_vz=0.0)
        est_ego = kf_with_ego.predict_update(pos_x, pos_z, ego_vx=ego_vx, ego_vz=ego_vz)

    assert est_no[2] == pytest.approx(0.0, abs=0.05)
    assert est_ego[2] == pytest.approx(ego_vx, abs=0.05)


# ── Filter independence ────────────────────────────────────────────────────


def test_multiple_independent_filters():
    """Two filter instances with different inputs must not interfere."""
    kf_a = TrajectoryKalmanFilter()
    kf_b = TrajectoryKalmanFilter()

    for step in range(30):
        xa = 1.0 * step * 0.05
        za = 0.0
        xb = 0.0
        zb = 2.0 * step * 0.05

        res_a = kf_a.predict_update(xa, za)
        res_b = kf_b.predict_update(xb, zb)

    # Filter A should have significant X motion, near-zero Z
    assert res_a[0] > 0.5, "Filter A should have positive est_x"
    assert res_a[1] == pytest.approx(0.0, abs=0.1)

    # Filter B should have near-zero X, significant Z motion
    assert res_b[0] == pytest.approx(0.0, abs=0.1)
    assert res_b[1] > 0.5, "Filter B should have positive est_z"


# ── Noise smoothing ────────────────────────────────────────────────────────


def test_noisy_measurements_smoothed():
    """Kalman-filtered estimates of noisy measurements should be smoother than raw input."""
    rng = np.random.default_rng(42)
    kf = TrajectoryKalmanFilter()

    true_positions = []
    measurements = []
    estimates = []

    noise_std = 0.5

    for step in range(100):
        true_x = 0.1 * step * kf.dt
        true_z = 0.0
        noisy_x = true_x + rng.normal(0.0, noise_std)
        noisy_z = true_z + rng.normal(0.0, noise_std)

        est_x, est_z, _, _ = kf.predict_update(noisy_x, noisy_z)

        true_positions.append(true_x)
        measurements.append(noisy_x)
        estimates.append(est_x)

    # Skip the first few steps where the filter is still converging
    true_arr = np.array(true_positions[10:])
    meas_arr = np.array(measurements[10:])
    est_arr = np.array(estimates[10:])

    meas_mse = float(np.mean((meas_arr - true_arr) ** 2))
    est_mse = float(np.mean((est_arr - true_arr) ** 2))

    assert est_mse < meas_mse, (
        f"Filtered MSE ({est_mse:.6f}) should be lower than raw measurement MSE ({meas_mse:.6f})"
    )
