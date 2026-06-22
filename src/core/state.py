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
        # Ego velocity components in the camera frame: ego_vz is the forward
        # (depth) component used for ego-motion compensation, ego_vx the lateral
        # one (kept at 0 by default — see engine.run).
        self.ego_vx = 0.0
        self.ego_vz = 0.0
        self.sensor_dt = None
        self.last_sensor_timestamp = None
        self.brake_needed = False
