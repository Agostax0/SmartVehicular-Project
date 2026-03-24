class SensorManager:
    """Manages the lifecycle of CARLA sensors attached to a vehicle."""

    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self._sensors = []

    def add_sensor(self, sensor):
        self._sensors.append(sensor)

    def destroy(self):
        for sensor in self._sensors:
            sensor.destroy()
        self._sensors.clear()
