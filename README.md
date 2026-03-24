# SmartVehicular Project

Autonomous driving simulation built on top of the [CARLA](https://carla.org/) simulator.

## Setup

```bash
conda create -n carla-env python=3.7
conda activate carla-env
pip install -r requirements.txt
pip install carla
```

## Project Structure

```
SmartVehicular-Project/
├── config/                  # YAML configuration files
│   └── config.yaml
├── data/
│   └── output/              # Simulation artefacts (images, logs)
├── scripts/
│   └── run_simulation.py    # Main entry-point to launch a simulation
├── src/
│   ├── agents/              # Autonomous driving agents
│   │   └── base_agent.py
│   ├── controllers/         # Vehicle control logic
│   │   └── vehicle_controller.py
│   ├── sensors/             # Sensor management (camera, lidar, …)
│   │   └── sensor_manager.py
│   └── utils/               # Shared utilities (logging, …)
│       └── logger.py
├── tests/                   # Unit tests (pytest)
│   └── test_utils.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Running a simulation

```bash
python scripts/run_simulation.py --config config/config.yaml
```

## Running tests

```bash
pytest tests/
```

