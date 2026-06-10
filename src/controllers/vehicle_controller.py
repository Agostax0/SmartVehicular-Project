import carla
import numpy as np


class VehicleController:
    """Applies throttle/steer/brake commands to a CARLA vehicle."""

    def __init__(self, vehicle):
        self.vehicle = vehicle

    def apply_control(self, throttle=0.0, steer=0.0, brake=0.0, hand_brake=False):
        control = carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
        )
        self.vehicle.apply_control(control)
        
        # Turn on/off brake lights programmatically
        current_lights = self.vehicle.get_light_state()
        if brake > 0.0:
            current_lights |= carla.VehicleLightState.Brake
        else:
            current_lights &= ~carla.VehicleLightState.Brake
        self.vehicle.set_light_state(carla.VehicleLightState(current_lights))

    def apply_throttle(self, target_speed_mps):
        current_vel = self.vehicle.get_velocity()
        current_speed = np.sqrt(current_vel.x**2 + current_vel.y**2 + current_vel.z**2)
        self.vehicle.ego_speed = current_speed
        target_speed = target_speed_mps
            
        error = target_speed - current_speed
        throttle = max(0.0, min(1.0, 0.4 * error + 0.1))
        self.apply_control(throttle=throttle)
