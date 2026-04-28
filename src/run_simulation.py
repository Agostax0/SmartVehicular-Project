import argparse
import yaml
import carla
import numpy as np
import time

from utils.logger import get_logger
from utils.detector import ObjectDetector
from utils.visualizer import Visualizer
from sensors.sensor_manager import SensorManager

logger = get_logger(__name__)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class SimulationState:
    """Helper to share data between camera callback and main loop."""
    def __init__(self):
        self.frame = None
        self.results = None

state = SimulationState()

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
    
    # 1. Connect to CARLA
    client = carla.Client(config["carla"]["host"], config["carla"]["port"])
    client.set_timeout(config["carla"]["timeout"])
    
    try:
        world = client.get_world()
        logger.info("Connected to CARLA: %s", world.get_map().name)
        
        # 2. Setup Components
        detector = ObjectDetector()
        visualizer = Visualizer(
            width=config["sensors"]["camera"]["width"],
            height=config["sensors"]["camera"]["height"]
        )
        
        # 3. Spawn Vehicle
        blueprint_library = world.get_blueprint_library()
        v_bp = blueprint_library.find(config["vehicle"]["blueprint"])
        spawn_point = world.get_map().get_spawn_points()[0]
        vehicle = world.spawn_actor(v_bp, spawn_point)
        logger.info("Vehicle spawned: %s", vehicle.type_id)
        
        # 4. Setup Sensors
        sensor_manager = SensorManager(world, vehicle)
        
        cam_bp = blueprint_library.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(config["sensors"]["camera"]["width"]))
        cam_bp.set_attribute('image_size_y', str(config["sensors"]["camera"]["height"]))
        cam_bp.set_attribute('fov', str(config["sensors"]["camera"]["fov"]))
        
        cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
        sensor_manager.add_sensor(camera)
        
        def camera_callback(image):
            # Convert raw data to numpy array
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            rgb_array = array[:, :, :3]
            
            # Run detection
            results = detector.detect(rgb_array)
            
            # Update shared state
            state.frame = rgb_array
            state.results = results

        camera.listen(lambda image: camera_callback(image))
        
        # 5. Simulation Loop
        logger.info("Starting simulation loop. Press Ctrl+C or close window to stop.")
        while True:
            world.wait_for_tick()
            
            # Update visualizer from main thread
            if state.frame is not None:
                if not visualizer.update(state.frame, state.results):
                    break
            
    except KeyboardInterrupt:
        logger.info("Simulation stopped by user.")
    finally:
        logger.info("Cleaning up actors...")
        if 'visualizer' in locals():
            visualizer.close()
        if 'sensor_manager' in locals():
            sensor_manager.destroy()
        if 'vehicle' in locals():
            vehicle.destroy()


if __name__ == "__main__":
    main()
