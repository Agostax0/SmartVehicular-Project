"""Tests for the SensorManager class in src/sensors/sensor_manager.py.

Covers construction, adding sensors, destroying them, and the
add-after-destroy lifecycle.  All CARLA objects are replaced with
MagicMock instances so no simulator is required.
"""

from unittest.mock import MagicMock

from src.sensors.sensor_manager import SensorManager


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def test_init_stores_world_and_vehicle():
    """SensorManager should store the world and vehicle references."""
    world = MagicMock()
    vehicle = MagicMock()
    sm = SensorManager(world, vehicle)

    assert sm.world is world
    assert sm.vehicle is vehicle


def test_init_empty_sensors_list():
    """A freshly created SensorManager has no sensors."""
    sm = SensorManager(MagicMock(), MagicMock())
    assert sm._sensors == []


# ---------------------------------------------------------------------------
# add_sensor
# ---------------------------------------------------------------------------

def test_add_sensor_increases_count():
    """Adding one sensor should increase the internal list length to 1."""
    sm = SensorManager(MagicMock(), MagicMock())
    sm.add_sensor(MagicMock())
    assert len(sm._sensors) == 1


def test_add_multiple_sensors():
    """Adding several sensors should accumulate them in order."""
    sm = SensorManager(MagicMock(), MagicMock())
    sensors = [MagicMock(name=f"sensor_{i}") for i in range(5)]
    for s in sensors:
        sm.add_sensor(s)

    assert len(sm._sensors) == 5
    assert sm._sensors == sensors


# ---------------------------------------------------------------------------
# destroy
# ---------------------------------------------------------------------------

def test_destroy_calls_destroy_on_each_sensor():
    """destroy() must call .destroy() on every registered sensor."""
    sm = SensorManager(MagicMock(), MagicMock())
    sensors = [MagicMock() for _ in range(3)]
    for s in sensors:
        sm.add_sensor(s)

    sm.destroy()

    for s in sensors:
        s.destroy.assert_called_once()


def test_destroy_clears_sensor_list():
    """After destroy(), the internal sensor list should be empty."""
    sm = SensorManager(MagicMock(), MagicMock())
    sm.add_sensor(MagicMock())
    sm.add_sensor(MagicMock())

    sm.destroy()

    assert sm._sensors == []


def test_destroy_empty_list_no_error():
    """Calling destroy() with no sensors should not raise."""
    sm = SensorManager(MagicMock(), MagicMock())
    sm.destroy()  # should simply be a no-op
    assert sm._sensors == []


# ---------------------------------------------------------------------------
# Lifecycle: add after destroy
# ---------------------------------------------------------------------------

def test_add_after_destroy():
    """New sensors can be registered after a previous destroy() call."""
    sm = SensorManager(MagicMock(), MagicMock())
    sm.add_sensor(MagicMock())
    sm.destroy()

    new_sensor = MagicMock()
    sm.add_sensor(new_sensor)

    assert len(sm._sensors) == 1
    assert sm._sensors[0] is new_sensor
