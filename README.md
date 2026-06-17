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
│   │   ├── vehicle_controller.py
│   │   └── keyboard_controller.py
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
│   ├── test_collision_edge_cases.py
│   ├── test_config.py
│   ├── test_kalman.py
│   ├── test_keyboard_controller.py
│   ├── test_perception.py
│   ├── test_scenarios.py
│   ├── test_sensor_manager.py
│   ├── test_state.py
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

## Manual control

You can drive the vehicle manually using the keyboard with the `--manual` flag:

```bash
python src/run_simulation.py --scenario non-colliding-pedestrian --manual
```

| Key | Action |
|---|---|
| **W** / **↑** | Throttle |
| **S** / **↓** | Brake |
| **A** / **←** | Steer left |
| **D** / **→** | Steer right |
| **Space** | Hand-brake |

The AEB (automatic emergency braking) system remains active in manual mode.

## Live demo: interactive crowd

The `interactive-crowd` scenario populates Town03 with a large number of
AI-driven pedestrians that wander the streets and cross the roads at random,
while you drive the ego vehicle manually and the AEB system stays active to
protect pedestrians.

### Run it

1. Start the CARLA server (it loads `Town03` automatically via the config).

   ```bash
   # Example: packaged server
   ./CarlaUE4.sh -quality-level=Low
   # or Docker:
   docker run --privileged --gpus all --net=host -e CARLA_SERVER_ARGS="-quality-level=Low" carlasim/carla:0.9.13 /bin/bash ./CarlaUE4.sh
   ```

2. Activate the environment and launch the simulation with manual control:

   ```bash
   conda activate carla-env
   python src/run_simulation.py --scenario interactive-crowd --manual
   ```

### Controls

| Key | Action |
|---|---|
| **W** / **↑** | Throttle |
| **S** / **↓** | Brake |
| **A** / **←** | Steer left |
| **D** / **→** | Steer right |
| **Space** | Hand-brake |

While you drive, the perception pipeline (YOLOv8 + Kalman filter + depth
camera) continuously tracks pedestrians. If a pedestrian is predicted to
enter the vehicle's path (TTC ≤ 5 s with lateral overlap), the **AEB takes
priority over your input**, zeroing the throttle and applying full brake —
even if you are holding the throttle. Drive toward a crossing pedestrian to
see the system intervene in real time.

### Tuning the crowd

The scenario is configurable in `config/config.yaml`:

```yaml
interactive-crowd:
  crowd_mode: true
  crowd_size: 50        # number of AI pedestrians
  crowd_max_speed: 1.5  # walking speed in m/s
```

Increase `crowd_size` for a denser city, or lower it if the frame rate drops.

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
