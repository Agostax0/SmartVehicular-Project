# SmartVehicular Project

Autonomous driving simulation built on top of the [CARLA](https://carla.org/) simulator.

## Setup

```bash
conda create -n carla-env python=3.7
conda activate carla-env
pip install -r requirements.txt
```

## Project Structure

```
SmartVehicular-Project/
├── config/                  # YAML configuration files
│   └── config.yaml
├── data/
│   └── output/              # Simulation artefacts (images, logs)
├── src/
│   ├── scenarios/           # Scenario loading and validation utilities
│   │   └── moving.py
│   ├── agents/              # Autonomous driving agents
│   │   └── base_agent.py
│   ├── controllers/         # Vehicle control logic
│   │   └── vehicle_controller.py
│   ├── sensors/             # Sensor management (camera, lidar, …)
│   │   └── sensor_manager.py
│   ├── utils/               # Shared utilities (logging, …)
│   |   └── logger.py
│   └── run_simulation_moving.py  # Main CLI entry-point to launch moving scenarios
├── tests/                   # Unit tests (pytest)
│   └── test_utils.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Running a simulation

```bash
python src/run_simulation_moving.py --config config/config.yaml --scenario non-colliding-pedestrian
```

If `--scenario` is omitted, the CLI uses `simulation.default_scenario` from `config/config.yaml`.

List available scenarios:

```bash
python src/run_simulation_moving.py --list-scenarios
```

Scenario definitions are configured under the `scenarios` section in `config/config.yaml`.

## Running tests

```bash
pytest tests/
```
