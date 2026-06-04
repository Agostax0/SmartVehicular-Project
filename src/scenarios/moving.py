"""Scenario definitions for the moving simulation entrypoint."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class MovingScenario:
    """Parameters that define a simulation scenario."""

    name: str
    description: str
    walker_distance_from_vehicle: float
    walker_height: float
    walker_horizontal_offset: float
    walker_speed: float
    target_speed_mps: float
    prediction_future_frames: int

def _parse_float(raw_value: Any, field_name: str, scenario_name: str) -> float:
    """Parse and validate a numeric scenario value."""
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise ValueError(
            f"Scenario '{scenario_name}' has invalid '{field_name}': expected a number."
        )
    return float(raw_value)


def _parse_positive_int(raw_value: Any, field_name: str, scenario_name: str) -> int:
    """Parse and validate a positive integer scenario value."""
    if isinstance(raw_value, bool) or not isinstance(raw_value, int):
        raise ValueError(
            f"Scenario '{scenario_name}' has invalid '{field_name}': expected an integer."
        )
    if raw_value <= 0:
        raise ValueError(
            f"Scenario '{scenario_name}' has invalid '{field_name}': expected > 0."
        )
    return raw_value


def load_scenarios(config: Dict[str, Any]) -> Dict[str, MovingScenario]:
    """Load and validate scenarios from config."""
    scenarios_cfg = config.get("scenarios")
    if not isinstance(scenarios_cfg, dict) or not scenarios_cfg:
        raise ValueError(
            "Missing or invalid 'scenarios' section in config. "
            "Expected a non-empty mapping of scenario names."
        )

    scenarios: Dict[str, MovingScenario] = {}
    for scenario_name, raw_scenario in scenarios_cfg.items():
        if not isinstance(scenario_name, str) or not scenario_name.strip():
            raise ValueError("Scenario names must be non-empty strings.")
        if not isinstance(raw_scenario, dict):
            raise ValueError(
                f"Scenario '{scenario_name}' is invalid: expected a mapping of parameters."
            )

        description = raw_scenario.get("description", "")
        if not isinstance(description, str):
            raise ValueError(
                f"Scenario '{scenario_name}' has invalid 'description': expected a string."
            )

        required_fields = [
            "walker_distance_from_vehicle",
            "walker_height",
            "walker_horizontal_offset",
            "walker_speed",
            "target_speed_mps",
            "prediction_future_frames",
        ]
        missing_fields = [field for field in required_fields if field not in raw_scenario]
        if missing_fields:
            raise ValueError(
                f"Scenario '{scenario_name}' is missing required fields: "
                f"{', '.join(sorted(missing_fields))}"
            )

        walker_speed = _parse_float(
            raw_scenario["walker_speed"], "walker_speed", scenario_name
        )
        if walker_speed <= 0.0:
            raise ValueError(
                f"Scenario '{scenario_name}' has invalid 'walker_speed': expected > 0."
            )

        target_speed_mps = _parse_float(
            raw_scenario["target_speed_mps"], "target_speed_mps", scenario_name
        )
        if target_speed_mps <= 0.0:
            raise ValueError(
                f"Scenario '{scenario_name}' has invalid 'target_speed_mps': expected > 0."
            )

        scenarios[scenario_name] = MovingScenario(
            name=scenario_name,
            description=description,
            walker_distance_from_vehicle=_parse_float(
                raw_scenario["walker_distance_from_vehicle"],
                "walker_distance_from_vehicle",
                scenario_name,
            ),
            walker_height=_parse_float(
                raw_scenario["walker_height"], "walker_height", scenario_name
            ),
            walker_horizontal_offset=_parse_float(
                raw_scenario["walker_horizontal_offset"],
                "walker_horizontal_offset",
                scenario_name,
            ),
            walker_speed=walker_speed,
            target_speed_mps=target_speed_mps,
            prediction_future_frames=_parse_positive_int(
                raw_scenario["prediction_future_frames"],
                "prediction_future_frames",
                scenario_name,
            ),
        )

    return scenarios


def list_scenario_names(config: Dict[str, Any]) -> List[str]:
    """Return all available scenario names from config."""
    return sorted(load_scenarios(config).keys())


def get_scenario(config: Dict[str, Any], name: Optional[str] = None) -> MovingScenario:
    """Get scenario by name (or default scenario from config)."""
    scenarios = load_scenarios(config)

    selected_name = name
    if selected_name is None:
        simulation_cfg = config.get("simulation")
        if not isinstance(simulation_cfg, dict):
            raise ValueError("Missing or invalid 'simulation' section in config.")
        selected_name = simulation_cfg.get("default_scenario")

    if not isinstance(selected_name, str) or not selected_name.strip():
        raise ValueError(
            "No scenario selected. Provide --scenario or set simulation.default_scenario."
        )

    scenario = scenarios.get(selected_name)
    if scenario is None:
        available = ", ".join(sorted(scenarios.keys()))
        raise ValueError(
            f"Unknown scenario '{selected_name}'. Available scenarios: {available}"
        )
    return scenario
