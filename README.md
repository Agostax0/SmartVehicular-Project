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
├── src/
│   ├── core/                # Core simulation logic and orchestration
│   │   ├── config.py        # Configuration loader
│   │   ├── engine.py        # Main simulation loop and world management
│   │   ├── pedestrian.py    # Pedestrian spawning and control logic
│   │   ├── perception.py    # Perception, detection, and collision risk assessment
│   │   └── state.py         # Shared simulation state
│   ├── scenarios/           # Scenario loading and validation utilities
│   │   └── moving.py
│   ├── controllers/         # Vehicle control logic
│   │   └── vehicle_controller.py
│   ├── sensors/             # Sensor management (camera, lidar, …)
│   │   └── sensor_manager.py
│   ├── utils/               # Shared utilities (logging, …)
│   │   ├── carla_utils.py
│   │   ├── collision_module.py
│   │   ├── detector.py
│   │   ├── kalman.py
│   │   ├── logger.py
│   │   └── visualizer.py
│   └── run_simulation.py  # Main CLI entry-point to launch moving scenarios
├── tests/                   # Unit tests (pytest)
│   ├── test_collision.py
│   └── test_utils.py
├── .gitignore
├── README.md
└── requirements.txt
```

### Core Architecture (Refactoring)

The `src/core/` package was introduced to separate concerns from the main entrypoint:
- **`engine.py`**: Abstracts CARLA client connections, actor lifecycle (setup/teardown), and the main simulation loop.
- **`perception.py`**: Isolates object detection logic, depth processing, Kalman filtering, and future collision risk assessment.
- **`pedestrian.py`**: Extracts pedestrian walker spawning logic and offset calculations.
- **`state.py`**: Provides a unified `SimulationState` object to share variables across loops.
- **`config.py`**: Centralizes YAML configuration loading.


## Running a simulation

```bash
python src/run_simulation.py --config config/config.yaml --scenario non-colliding-pedestrian
```

If `--scenario` is omitted, the CLI uses `simulation.default_scenario` from `config/config.yaml`.

List available scenarios:

```bash
python src/run_simulation.py --list-scenarios
```

Scenario definitions are configured under the `scenarios` section in `config/config.yaml`.

## Running tests

To run the unit tests:

```bash
pytest tests/
```

To run the tests with code coverage and generate an HTML report:

```bash
pytest --cov=src --cov-report=html tests/
```

This will create an `htmlcov/` directory. Open `htmlcov/index.html` in your web browser to view the detailed line-by-line coverage report.
