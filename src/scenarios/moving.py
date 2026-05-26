"""Scenario definitions for the moving simulation entrypoint."""

from dataclasses import dataclass
from typing import Dict, List


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


_SCENARIOS: Dict[str, MovingScenario] = {
    "non-colliding-pedestrian": MovingScenario(
        name="non-colliding-pedestrian",
        description="Pedestrian crosses with enough lateral offset to avoid collision.",
        walker_distance_from_vehicle=35.0,
        walker_height=1.0,
        walker_horizontal_offset=5.0,
        walker_speed=1.4,
        target_speed_mps=5.56,  # 20 km/h
        prediction_future_frames=100,
    ),
    "colliding-pedestrian": MovingScenario(
        name="colliding-pedestrian",
        description="Pedestrian crosses the direct path of the ego vehicle.",
        walker_distance_from_vehicle=25.0,
        walker_height=1.0,
        walker_horizontal_offset=0.0,
        walker_speed=1.7,
        target_speed_mps=5.56,  # 20 km/h
        prediction_future_frames=100,
    ),
}


def list_scenario_names() -> List[str]:
    """Return all available scenario names."""
    return sorted(_SCENARIOS.keys())


def get_scenario(name: str) -> MovingScenario:
    """Get scenario by name or raise with a helpful message."""
    scenario = _SCENARIOS.get(name)
    if scenario is None:
        available = ", ".join(list_scenario_names())
        raise ValueError(f"Unknown scenario '{name}'. Available scenarios: {available}")
    return scenario
