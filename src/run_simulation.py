#!/usr/bin/env python
"""Entry-point script to run a SmartVehicular simulation."""

import argparse
import yaml

from utils.logger import get_logger

logger = get_logger(__name__)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="SmartVehicular simulation runner")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info("Loaded config: %s", args.config)
    logger.info("Connecting to CARLA at %s:%d", config["carla"]["host"], config["carla"]["port"])

    # TODO: initialise CARLA client, spawn vehicle, attach sensors and run agent loop


if __name__ == "__main__":
    main()
