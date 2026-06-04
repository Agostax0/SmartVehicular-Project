import yaml

def load_config(path: str) -> dict:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Parsed configuration as a dictionary.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)
