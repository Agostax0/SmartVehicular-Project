import carla


class VehicleController:
    """Applies throttle/steer/brake commands to a CARLA vehicle."""

    def __init__(self, vehicle):
        self.vehicle = vehicle

    def apply_control(self, throttle=0.0, steer=0.0, brake=0.0):
        control = carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
        )
        self.vehicle.apply_control(control)
