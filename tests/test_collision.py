"""Tests for collision risk logic."""

from src.utils.collision_module import check_collision_risk

def test_collision_imminent_crossing_pedestrian():
    """
    CASE 1: Pedestrian crossing and will hit the car head-on.
    The car goes at 10 m/s (vel_z = -10). The pedestrian is at 15m (est_z = 15).
    The pedestrian is 1.5m to the right (est_x = 1.5) but runs towards the center at 1 m/s (vel_x = -1.0).
    """
    est_x = 1.5
    est_z = 15.0
    vel_x = -1.0
    vel_z = -10.0
    
    risk, state_message, ttc = check_collision_risk(est_x, est_z, vel_x, vel_z)
    
    assert risk is True
    assert ttc == 1.5  # 15m / 10m/s
    assert "CRITICO!" in state_message
    assert "X=0.00m" in state_message


def test_safe_pedestrian_on_sidewalk():
    """
    CASE 2: Pedestrian very close but safe on the sidewalk.
    The car goes at 10 m/s (vel_z = -10). The pedestrian is at 11m (est_z = 11).
    The pedestrian is 3m to the right (est_x = 3.0) and walks straight or is still (vel_x = 0.0).
    """
    est_x = 3.0
    est_z = 11.0
    vel_x = 0.0
    vel_z = -10.0
    
    risk, state_message, ttc = check_collision_risk(est_x, est_z, vel_x, vel_z)
    
    assert risk is False
    assert ttc == 1.1  # 11m / 10m/s
    assert "Sicuro" in state_message
    assert "X=3.00m" in state_message


def test_pedestrian_warning_but_not_critical_yet():
    """
    CASE 3: Pedestrian crossing, trajectories intersect, but more than 1.5 seconds remain.
    Pre-warning situation to make the car slow down.
    """
    est_x = 2.1
    est_z = 21.0
    vel_x = -0.98
    vel_z = -9.95
    
    risk, state_message, ttc = check_collision_risk(est_x, est_z, vel_x, vel_z)
    
    assert risk is False
    assert ttc > 1.5
    assert ttc < 3.0
    assert "ATTENZIONE" in state_message


def test_no_danger_moving_away():
    """
    CASE 4: False alarm (the pedestrian walks faster than the car or the car is stopped).
    vel_z is positive, indicating departure.
    """
    est_x = 0.0
    est_z = 5.0
    vel_x = 0.0
    vel_z = 2.0 
    
    risk, state_message, ttc = check_collision_risk(est_x, est_z, vel_x, vel_z)
    
    assert risk is False
    assert ttc == float('inf')
    assert "Nessun pericolo" in state_message