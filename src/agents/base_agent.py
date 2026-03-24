class BaseAgent:
    """Base class for all autonomous driving agents."""

    def __init__(self, vehicle):
        self.vehicle = vehicle

    def run_step(self, sensor_data):
        """Execute one control step. Override in subclasses."""
        raise NotImplementedError
