"""
Edge-case tests for check_collision_risk.

These tests complement the main functional tests by exercising every
boundary condition in the collision-risk decision tree:
  • vel_z threshold at -0.5
  • TTC boundaries at 15.0 s and 5.0 s
  • Hitbox-limit boundary (car_half_width + safety_margin)
  • Custom car/margin parameters
  • Extreme and degenerate inputs
"""

import math

from src.utils.collision_module import check_collision_risk


# ---------------------------------------------------------------------------
# vel_z threshold boundary (>= -0.5 → no risk)
# ---------------------------------------------------------------------------

def test_vel_z_exactly_minus_half():
    """vel_z = -0.5 sits exactly on the >= -0.5 guard → no risk."""
    risk, msg, ttc = check_collision_risk(est_x=0, est_z=5, vel_x=0, vel_z=-0.5)
    assert risk is False
    assert "no risk" in msg.lower()
    assert ttc == float("inf")


def test_vel_z_just_below_threshold():
    """vel_z = -0.51 passes the guard, so TTC computation should proceed."""
    risk, msg, ttc = check_collision_risk(est_x=0, est_z=5, vel_x=0, vel_z=-0.51)
    # TTC = 5 / 0.51 ≈ 9.80 – within 15 s, in hitbox (x_at_impact=0) → risk depends on ttc<=5
    assert ttc != float("inf")
    expected_ttc = 5.0 / 0.51
    assert math.isclose(ttc, expected_ttc, rel_tol=1e-9)


def test_vel_z_zero():
    """vel_z = 0 (stationary in Z) satisfies >= -0.5 → no risk."""
    risk, msg, ttc = check_collision_risk(est_x=3, est_z=10, vel_x=1, vel_z=0)
    assert risk is False
    assert ttc == float("inf")


def test_vel_z_positive():
    """vel_z = 5.0 (pedestrian moving away) satisfies >= -0.5 → no risk."""
    risk, msg, ttc = check_collision_risk(est_x=0, est_z=5, vel_x=0, vel_z=5.0)
    assert risk is False
    assert ttc == float("inf")


# ---------------------------------------------------------------------------
# TTC = 15.0 boundary  (> 15.0 → "far away")
# ---------------------------------------------------------------------------

def test_ttc_exactly_15():
    """TTC = 15.0 exactly.  The check is strict (> 15.0) so this is NOT 'far away'.

    Setup: est_z=15, vel_z=-1  →  TTC = 15/1 = 15.0
    x_at_impact = 0 + 0*15 = 0  →  inside hitbox  →  but ttc > 5 → pre-warning.
    """
    risk, msg, ttc = check_collision_risk(est_x=0, est_z=15, vel_x=0, vel_z=-1.0)
    assert math.isclose(ttc, 15.0)
    assert "far away" not in msg.lower()
    # ttc > 5 and inside hitbox → risk False, trajectory warning
    assert risk is False
    assert "trajectory" in msg.lower()


def test_ttc_just_above_15():
    """TTC = 15.01 → exceeds 15.0 → 'far away'.

    Setup: est_z=15.01, vel_z=-1  →  TTC = 15.01
    """
    risk, msg, ttc = check_collision_risk(est_x=0, est_z=15.01, vel_x=0, vel_z=-1.0)
    assert math.isclose(ttc, 15.01)
    assert risk is False
    assert "far away" in msg.lower()


# ---------------------------------------------------------------------------
# TTC = 5.0 boundary  (<= 5.0 with hit → risk True)
# ---------------------------------------------------------------------------

def test_ttc_exactly_5():
    """TTC = 5.0 exactly, impact inside hitbox → risk is True.

    Setup: est_z=5, vel_z=-1  →  TTC = 5.0
    x_at_impact = 0  →  inside default hitbox (1.5 m)
    """
    risk, msg, ttc = check_collision_risk(est_x=0, est_z=5, vel_x=0, vel_z=-1.0)
    assert math.isclose(ttc, 5.0)
    assert risk is True
    assert "impact" in msg.lower()


def test_ttc_just_above_5():
    """TTC = 5.01, impact inside hitbox → risk False (pre-warning only).

    Setup: est_z=5.01, vel_z=-1  →  TTC = 5.01
    x_at_impact = 0  →  inside hitbox
    """
    risk, msg, ttc = check_collision_risk(est_x=0, est_z=5.01, vel_x=0, vel_z=-1.0)
    assert math.isclose(ttc, 5.01)
    assert risk is False
    assert "trajectory" in msg.lower()


# ---------------------------------------------------------------------------
# Hitbox boundary  (|x_at_impact| <= hitbox_limit)
# ---------------------------------------------------------------------------

def test_hitbox_boundary_exact():
    """x_at_impact exactly at hitbox_limit (1.5 m) → inside (<=) → risk True.

    Setup: TTC = 5 / 1 = 5.0
    Need x_at_impact = 1.5  →  est_x + vel_x * 5 = 1.5
    Use est_x=0, vel_x=0.3  →  0 + 0.3*5 = 1.5
    """
    risk, msg, ttc = check_collision_risk(est_x=0, est_z=5, vel_x=0.3, vel_z=-1.0)
    assert math.isclose(ttc, 5.0)
    assert risk is True
    assert "impact" in msg.lower()


def test_hitbox_boundary_just_outside():
    """x_at_impact = 1.501 → outside hitbox → no collision risk.

    Setup: TTC = 5.0
    Need x_at_impact = 1.501  →  vel_x = 1.501 / 5 = 0.3002
    """
    risk, msg, ttc = check_collision_risk(est_x=0, est_z=5, vel_x=0.3002, vel_z=-1.0)
    x_at_impact = 0 + 0.3002 * 5.0
    assert x_at_impact > 1.5  # confirm setup
    assert risk is False
    assert "walker" in msg.lower()


# ---------------------------------------------------------------------------
# Custom parameters
# ---------------------------------------------------------------------------

def test_custom_car_half_width():
    """Wider car (car_half_width=2.0) → hitbox_limit = 2.5.

    x_at_impact = 2.0 should be inside the wider hitbox.
    Setup: TTC=5, vel_x = 2.0/5 = 0.4
    """
    risk, msg, ttc = check_collision_risk(
        est_x=0, est_z=5, vel_x=0.4, vel_z=-1.0, car_half_width=2.0
    )
    assert risk is True
    assert math.isclose(ttc, 5.0)


def test_custom_safety_margin():
    """Larger safety_margin=1.0 → hitbox_limit = 2.0.

    x_at_impact = 1.8 should be inside (1.8 <= 2.0).
    Setup: TTC=5, vel_x = 1.8/5 = 0.36
    """
    risk, msg, ttc = check_collision_risk(
        est_x=0, est_z=5, vel_x=0.36, vel_z=-1.0, safety_margin=1.0
    )
    assert risk is True


def test_zero_car_width_zero_margin():
    """car_half_width=0, safety_margin=0 → only a direct centre-line hit counts.

    x_at_impact=0.0 → inside (0 <= 0), x_at_impact=0.01 → outside.
    """
    # Direct hit
    risk, _, _ = check_collision_risk(
        est_x=0, est_z=5, vel_x=0, vel_z=-1.0,
        car_half_width=0, safety_margin=0,
    )
    assert risk is True

    # Slight offset → miss
    risk2, _, _ = check_collision_risk(
        est_x=0, est_z=5, vel_x=0.002, vel_z=-1.0,
        car_half_width=0, safety_margin=0,
    )
    assert risk2 is False


# ---------------------------------------------------------------------------
# Special positions / velocities
# ---------------------------------------------------------------------------

def test_pedestrian_at_origin():
    """Pedestrian directly ahead at (0, 5), heading straight at us.

    est_x=0, vel_x=0  →  x_at_impact=0, TTC=1 s  →  risk True.
    """
    risk, msg, ttc = check_collision_risk(est_x=0, est_z=5, vel_x=0, vel_z=-5.0)
    assert risk is True
    assert math.isclose(ttc, 1.0)
    assert "impact" in msg.lower()


def test_negative_est_x():
    """Pedestrian to the left (negative X) walking into the car's path.

    est_x=-3, vel_x=0, est_z=5, vel_z=-1  →  TTC=5, x_at_impact=-3
    |-3| = 3 > 1.5 → miss.
    """
    risk, msg, ttc = check_collision_risk(est_x=-3, est_z=5, vel_x=0, vel_z=-1.0)
    assert risk is False
    assert math.isclose(ttc, 5.0)


def test_very_high_speed():
    """Very fast approach (vel_z=-100) → very small TTC.

    est_z=5  →  TTC = 5/100 = 0.05 s.  x_at_impact = 0 → risk True.
    """
    risk, msg, ttc = check_collision_risk(est_x=0, est_z=5, vel_x=0, vel_z=-100.0)
    assert risk is True
    assert math.isclose(ttc, 0.05)


def test_very_large_est_z():
    """Pedestrian far away: est_z=200, vel_z=-1 → TTC=200 → 'far away'."""
    risk, msg, ttc = check_collision_risk(est_x=0, est_z=200, vel_x=0, vel_z=-1.0)
    assert risk is False
    assert math.isclose(ttc, 200.0)
    assert "far away" in msg.lower()


def test_pedestrian_moving_into_path_from_left():
    """Pedestrian starts left but drifts right into the car's lane.

    est_x=-5, vel_x=1, est_z=5, vel_z=-1  →  TTC=5
    x_at_impact = -5 + 1*5 = 0  →  inside hitbox  →  risk True (ttc=5).
    """
    risk, msg, ttc = check_collision_risk(est_x=-5, est_z=5, vel_x=1, vel_z=-1.0)
    assert risk is True
    assert math.isclose(ttc, 5.0)
    assert "impact" in msg.lower()
