class SimulationState:
    """Helper to share data between camera callback and main loop."""

    def __init__(self):
        """Initialize the shared simulation state.

        Args:
            None.

        Returns:
            None.
        """
        self.frame = None
        self.depth_array = None
        self.results = None
        self.kalman_predictions = None
        self.kalman_filters = {}
        self.ego_speed = 0.0
        self.brake_needed = False
