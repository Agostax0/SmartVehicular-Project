import typer
from typing import Optional

from core.config import load_config
from core.engine import SimulationEngine
from scenarios import get_scenario, list_scenario_names

def main(
    config: str = typer.Option(
        "config/config.yaml",
        "--config",
        help="Path to the YAML configuration file.",
    ),
    scenario: Optional[str] = typer.Option(
        None,
        "--scenario",
        help="Scenario name to run. Defaults to simulation.default_scenario in config.",
    ),
    list_scenarios: bool = typer.Option(
        False,
        "--list-scenarios",
        help="List all available scenarios and exit.",
    ),
) -> None:
    """CLI entrypoint for running simulation scenarios."""
    if list_scenarios:
        config_data = load_config(config)
        for name in list_scenario_names(config_data):
            scenario_config = get_scenario(config_data, name)
            typer.echo(f"{name}: {scenario_config.description}")
        raise typer.Exit()

    try:
        config_data = load_config(config)
        scenario_config = get_scenario(config_data, scenario)
        engine = SimulationEngine(config_data, scenario_config)
        engine.setup()
        engine.run()
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--scenario") from exc
    except RuntimeError as exc:
        typer.echo(f"Simulation error: {exc}", err=True)

if __name__ == "__main__":
    typer.run(main)
