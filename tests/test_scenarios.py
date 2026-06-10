"""Tests for scenario loading and validation (src.scenarios.moving)."""

import pytest

from src.scenarios.moving import (
    MovingScenario,
    WalkerParams,
    _parse_float,
    _parse_positive_int,
    get_scenario,
    list_scenario_names,
    load_scenarios,
)

# ---------------------------------------------------------------------------
# Helpers – build minimal valid config dicts
# ---------------------------------------------------------------------------


def _single_walker_scenario(overrides=None):
    """Return a single-scenario config using top-level walker fields."""
    scenario = {
        "description": "unit-test scenario",
        "target_speed_mps": 10.0,
        "prediction_future_frames": 5,
        "walker_distance_from_vehicle": 30.0,
        "walker_height": 1.8,
        "walker_horizontal_offset": 0.5,
        "walker_speed": 1.4,
    }
    if overrides:
        scenario.update(overrides)
    return {"scenarios": {"basic": scenario}}


def _multi_walker_scenario(walkers=None, scenario_overrides=None):
    """Return a single-scenario config using the ``walkers`` list format."""
    default_walkers = [
        {
            "walker_distance_from_vehicle": 25.0,
            "walker_height": 1.7,
            "walker_horizontal_offset": 0.3,
            "walker_speed": 1.2,
        },
        {
            "walker_distance_from_vehicle": 40.0,
            "walker_height": 1.9,
            "walker_horizontal_offset": -0.4,
            "walker_speed": 2.0,
            "walker_type": "cyclist",
        },
    ]
    scenario = {
        "description": "multi-walker scenario",
        "target_speed_mps": 12.0,
        "prediction_future_frames": 10,
        "walkers": walkers if walkers is not None else default_walkers,
    }
    if scenario_overrides:
        scenario.update(scenario_overrides)
    return {"scenarios": {"multi": scenario}}


def _two_scenario_config():
    """Return a config with two named scenarios (alpha-unordered)."""
    return {
        "scenarios": {
            "bravo": {
                "description": "second",
                "target_speed_mps": 8.0,
                "prediction_future_frames": 3,
                "walker_distance_from_vehicle": 20.0,
                "walker_height": 1.6,
                "walker_horizontal_offset": 0.0,
                "walker_speed": 1.0,
            },
            "alpha": {
                "description": "first",
                "target_speed_mps": 15.0,
                "prediction_future_frames": 7,
                "walker_distance_from_vehicle": 35.0,
                "walker_height": 1.8,
                "walker_horizontal_offset": 1.0,
                "walker_speed": 2.5,
            },
        }
    }


# ---------------------------------------------------------------------------
# WalkerParams dataclass
# ---------------------------------------------------------------------------


def test_walker_params_creation():
    """WalkerParams can be created with all required fields."""
    wp = WalkerParams(
        walker_distance_from_vehicle=30.0,
        walker_height=1.8,
        walker_horizontal_offset=0.5,
        walker_speed=1.4,
        walker_type="cyclist",
    )
    assert wp.walker_distance_from_vehicle == 30.0
    assert wp.walker_height == 1.8
    assert wp.walker_horizontal_offset == 0.5
    assert wp.walker_speed == 1.4
    assert wp.walker_type == "cyclist"


def test_walker_params_frozen():
    """WalkerParams is frozen – attribute assignment must raise."""
    wp = WalkerParams(
        walker_distance_from_vehicle=10.0,
        walker_height=1.7,
        walker_horizontal_offset=0.0,
        walker_speed=1.0,
    )
    with pytest.raises(AttributeError):
        wp.walker_speed = 2.0


def test_walker_params_default_type():
    """walker_type defaults to 'pedestrian' when not specified."""
    wp = WalkerParams(
        walker_distance_from_vehicle=10.0,
        walker_height=1.7,
        walker_horizontal_offset=0.0,
        walker_speed=1.0,
    )
    assert wp.walker_type == "pedestrian"


# ---------------------------------------------------------------------------
# MovingScenario dataclass
# ---------------------------------------------------------------------------


def test_moving_scenario_creation():
    """MovingScenario can be created with all explicit fields."""
    walkers = [
        WalkerParams(
            walker_distance_from_vehicle=20.0,
            walker_height=1.8,
            walker_horizontal_offset=0.0,
            walker_speed=1.5,
        )
    ]
    ms = MovingScenario(
        name="test",
        description="a test scenario",
        target_speed_mps=10.0,
        prediction_future_frames=5,
        enable_aeb=False,
        walkers=walkers,
    )
    assert ms.name == "test"
    assert ms.description == "a test scenario"
    assert ms.target_speed_mps == 10.0
    assert ms.prediction_future_frames == 5
    assert ms.enable_aeb is False
    assert ms.walkers == walkers


def test_moving_scenario_defaults():
    """MovingScenario defaults: enable_aeb=True, walkers=None."""
    ms = MovingScenario(
        name="defaults",
        description="",
        target_speed_mps=5.0,
        prediction_future_frames=1,
    )
    assert ms.enable_aeb is True
    assert ms.walkers is None


# ---------------------------------------------------------------------------
# _parse_float
# ---------------------------------------------------------------------------


def test_parse_float_valid_int():
    """_parse_float accepts a plain int and returns a float."""
    result = _parse_float(7, "field", "scenario")
    assert result == 7.0
    assert isinstance(result, float)


def test_parse_float_valid_float():
    """_parse_float accepts a float and returns it unchanged."""
    result = _parse_float(3.14, "field", "scenario")
    assert result == 3.14


def test_parse_float_rejects_bool():
    """_parse_float must reject bool even though bool is a subclass of int."""
    with pytest.raises(ValueError, match="expected a number"):
        _parse_float(True, "field", "scenario")


def test_parse_float_rejects_string():
    """_parse_float rejects string values."""
    with pytest.raises(ValueError, match="expected a number"):
        _parse_float("10.0", "field", "scenario")


# ---------------------------------------------------------------------------
# _parse_positive_int
# ---------------------------------------------------------------------------


def test_parse_positive_int_valid():
    """_parse_positive_int accepts a positive integer."""
    assert _parse_positive_int(5, "field", "scenario") == 5


def test_parse_positive_int_rejects_zero():
    """_parse_positive_int rejects zero."""
    with pytest.raises(ValueError, match="expected > 0"):
        _parse_positive_int(0, "field", "scenario")


def test_parse_positive_int_rejects_negative():
    """_parse_positive_int rejects negative values."""
    with pytest.raises(ValueError, match="expected > 0"):
        _parse_positive_int(-3, "field", "scenario")


def test_parse_positive_int_rejects_float():
    """_parse_positive_int rejects float values."""
    with pytest.raises(ValueError, match="expected an integer"):
        _parse_positive_int(1.5, "field", "scenario")


def test_parse_positive_int_rejects_bool():
    """_parse_positive_int must reject bool (True) even though bool is int."""
    with pytest.raises(ValueError, match="expected an integer"):
        _parse_positive_int(True, "field", "scenario")


# ---------------------------------------------------------------------------
# load_scenarios – valid configs
# ---------------------------------------------------------------------------


def test_load_scenarios_single_walker():
    """Load a minimal config with one scenario using top-level walker fields."""
    config = _single_walker_scenario()
    result = load_scenarios(config)

    assert "basic" in result
    sc = result["basic"]
    assert sc.name == "basic"
    assert sc.description == "unit-test scenario"
    assert sc.target_speed_mps == 10.0
    assert sc.prediction_future_frames == 5
    assert sc.enable_aeb is True
    assert len(sc.walkers) == 1
    assert sc.walkers[0].walker_speed == 1.4
    # Backward-compat fields should be populated from the walker values.
    assert sc.walker_distance_from_vehicle == 30.0
    assert sc.walker_height == 1.8
    assert sc.walker_horizontal_offset == 0.5
    assert sc.walker_speed == 1.4


def test_load_scenarios_multi_walker():
    """Load a config that uses the ``walkers`` list format."""
    config = _multi_walker_scenario()
    result = load_scenarios(config)

    sc = result["multi"]
    assert len(sc.walkers) == 2
    assert sc.walkers[0].walker_type == "pedestrian"
    assert sc.walkers[1].walker_type == "cyclist"
    # Backward-compat fields come from the first walker.
    assert sc.walker_distance_from_vehicle == 25.0
    assert sc.walker_speed == 1.2


def test_load_scenarios_cyclist_type():
    """walker_type='cyclist' is preserved through the single-walker path."""
    config = _single_walker_scenario({"walker_type": "cyclist"})
    result = load_scenarios(config)
    sc = result["basic"]
    assert sc.walker_type == "cyclist"
    assert sc.walkers[0].walker_type == "cyclist"


# ---------------------------------------------------------------------------
# load_scenarios – error paths
# ---------------------------------------------------------------------------


def test_load_scenarios_missing_section():
    """Raises ValueError when 'scenarios' key is absent."""
    with pytest.raises(ValueError, match="Missing or invalid 'scenarios'"):
        load_scenarios({})


def test_load_scenarios_empty_scenarios():
    """Raises ValueError when scenarios dict is empty."""
    with pytest.raises(ValueError, match="Missing or invalid 'scenarios'"):
        load_scenarios({"scenarios": {}})


def test_load_scenarios_invalid_scenario_type():
    """Raises ValueError when a scenario value is not a dict."""
    with pytest.raises(ValueError, match="expected a mapping of parameters"):
        load_scenarios({"scenarios": {"bad": "not-a-dict"}})


def test_load_scenarios_missing_walker_fields():
    """Raises ValueError for missing required walker fields (single-walker path)."""
    config = {
        "scenarios": {
            "incomplete": {
                "description": "oops",
                "target_speed_mps": 5.0,
                "prediction_future_frames": 2,
                "walker_speed": 1.0,
                # Missing walker_distance_from_vehicle, walker_height, walker_horizontal_offset
            }
        }
    }
    with pytest.raises(ValueError, match="missing required fields"):
        load_scenarios(config)


def test_load_scenarios_invalid_walker_speed_zero():
    """Raises ValueError when walker_speed is 0 (single-walker path)."""
    config = _single_walker_scenario({"walker_speed": 0})
    with pytest.raises(ValueError, match="walker_speed.*expected > 0"):
        load_scenarios(config)


def test_load_scenarios_invalid_target_speed():
    """Raises ValueError when target_speed_mps is 0 or negative."""
    config = _single_walker_scenario({"target_speed_mps": 0})
    with pytest.raises(ValueError, match="target_speed_mps.*expected > 0"):
        load_scenarios(config)

    config_neg = _single_walker_scenario({"target_speed_mps": -5.0})
    with pytest.raises(ValueError, match="target_speed_mps.*expected > 0"):
        load_scenarios(config_neg)


def test_load_scenarios_invalid_prediction_frames():
    """Raises ValueError when prediction_future_frames is a float."""
    config = _single_walker_scenario({"prediction_future_frames": 3.5})
    with pytest.raises(ValueError, match="expected an integer"):
        load_scenarios(config)


def test_load_scenarios_missing_target_speed():
    """Raises ValueError when target_speed_mps is missing entirely."""
    config = _single_walker_scenario()
    del config["scenarios"]["basic"]["target_speed_mps"]
    with pytest.raises(ValueError, match="missing required fields.*target_speed_mps"):
        load_scenarios(config)


# ---------------------------------------------------------------------------
# list_scenario_names
# ---------------------------------------------------------------------------


def test_list_scenario_names_sorted():
    """list_scenario_names returns names in sorted order."""
    config = _two_scenario_config()
    names = list_scenario_names(config)
    assert names == ["alpha", "bravo"]


# ---------------------------------------------------------------------------
# get_scenario
# ---------------------------------------------------------------------------


def test_get_scenario_by_name():
    """get_scenario returns the named scenario."""
    config = _two_scenario_config()
    sc = get_scenario(config, name="alpha")
    assert sc.name == "alpha"
    assert sc.target_speed_mps == 15.0


def test_get_scenario_default_from_config():
    """get_scenario falls back to simulation.default_scenario when name=None."""
    config = _two_scenario_config()
    config["simulation"] = {"default_scenario": "bravo"}

    sc = get_scenario(config, name=None)
    assert sc.name == "bravo"


def test_get_scenario_unknown_name():
    """Raises ValueError for an unknown scenario name."""
    config = _single_walker_scenario()
    with pytest.raises(ValueError, match="Unknown scenario 'nope'"):
        get_scenario(config, name="nope")


def test_get_scenario_no_name_no_default():
    """Raises ValueError when name=None and no default_scenario configured."""
    config = _single_walker_scenario()
    # No 'simulation' section at all.
    with pytest.raises(ValueError, match="Missing or invalid 'simulation'"):
        get_scenario(config, name=None)
