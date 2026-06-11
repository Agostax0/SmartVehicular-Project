import carla
import queue
import numpy as np

from utils.logger import get_logger
from utils.detector import ObjectDetector
from utils.visualizer import Visualizer
from sensors.sensor_manager import SensorManager
from utils.carla_utils import (
    image_to_bgr,
    safe_destroy,
    spawn_camera,
    spawn_depth_camera,
    spawn_vehicle,
)

from controllers.vehicle_controller import VehicleController
from controllers.keyboard_controller import KeyboardController
from core.state import SimulationState
from core.pedestrian import spawn_pedestrian
from core.perception import PerceptionSystem

logger = get_logger(__name__)

class SimulationEngine:
    def __init__(self, config, scenario, manual=False):
        self.config = config
        self.scenario = scenario
        self.manual = manual
        self.client = carla.Client(config["carla"]["host"], config["carla"]["port"])
        self.client.set_timeout(config["carla"]["timeout"])
        self.world = None
        self.vehicle = None
        self.walker = None
        self.walkers = []
        self.sensor_manager = None
        self.visualizer = None
        self.detector = ObjectDetector()
        self.perception = PerceptionSystem(config)
        self.state = SimulationState()
        
    def setup(self):
        self.world = self.client.load_world(self.config["carla"]["map"])
        self.world.set_weather(self.scenario.weather_effects.toCarlaWeather())

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / self.config["simulation"].get("fps", 20)
        self.world.apply_settings(settings)
        logger.info("Connected to CARLA and enabled sync mode: %s", self.world.get_map().name)
        
        self.visualizer = Visualizer(
            width=self.config["sensors"]["camera"]["width"],
            height=self.config["sensors"]["camera"]["height"],
            title=f"SmartVehicular Detection - {self.scenario.name}"
        )
        
        spawn_idx = self.config["vehicle"].get("spawn_point")
        if spawn_idx is None:
            spawn_idx = 0

        self.vehicle, spawn_transform = spawn_vehicle(
            self.world,
            spawn_index=spawn_idx,
            vehicle_filter=self.config["vehicle"]["blueprint"],
            autopilot=False,
        )
        logger.info("Vehicle spawned: %s at %s", self.vehicle.type_id, spawn_transform.location)
        
        self.walkers = []
        if getattr(self.scenario, "walkers", None) is not None:
            for w_params in self.scenario.walkers:
                w_actor = spawn_pedestrian(self.world, spawn_transform, w_params)
                if w_actor is not None:
                    self.walkers.append(w_actor)
            # Maintain self.walker referencing the first spawned walker if any
            if self.walkers:
                self.walker = self.walkers[0]
        else:
            self.walker = spawn_pedestrian(self.world, spawn_transform, self.scenario)
            if self.walker is not None:
                self.walkers.append(self.walker)
        
        self.sensor_manager = SensorManager(self.world, self.vehicle)
        
        cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = spawn_camera(
            self.world,
            self.vehicle,
            cam_transform,
            width=self.config["sensors"]["camera"]["width"],
            height=self.config["sensors"]["camera"]["height"],
            fov=self.config["sensors"]["camera"]["fov"],
            tick=0.0,
        )
        self.sensor_manager.add_sensor(self.camera)
        
        self.depth_camera = spawn_depth_camera(
            self.world,
            self.vehicle,
            cam_transform,
            width=self.config["sensors"]["camera"]["width"],
            height=self.config["sensors"]["camera"]["height"],
            fov=self.config["sensors"]["camera"]["fov"],
            tick=0.0,
        )
        self.sensor_manager.add_sensor(self.depth_camera)
        
        self.controller = VehicleController(self.vehicle)
        self.keyboard = KeyboardController() if self.manual else None
        if self.manual:
            logger.info("Manual control enabled — use W/A/S/D or arrow keys.")
        
    def run(self):
        image_queue = queue.Queue()
        depth_queue = queue.Queue()
        
        self.depth_camera.listen(depth_queue.put)
        self.camera.listen(image_queue.put)
        
        logger.info("Starting simulation loop. Press Ctrl+C or close window to stop.")
        
        try:
            while True:
                self.world.tick()
                
                try:
                    image = image_queue.get(timeout=2.0)
                    depth_image = depth_queue.get(timeout=2.0)
                except queue.Empty:
                    logger.warning("Timeout waiting for sensor data.")
                    continue

                self.state.depth_array = self.perception.process_depth(depth_image)

                current_vel = self.vehicle.get_velocity()
                self.state.ego_speed = np.sqrt(current_vel.x**2 + current_vel.y**2 + current_vel.z**2)

                bgr_array = image_to_bgr(image)
                rgb_array = bgr_array[:, :, ::-1]
                
                self.state.results = self.detector.detect(rgb_array)
                self.perception.analyze_detections(self.state, self.scenario)
                self.state.frame = rgb_array

                # Visualize and capture key state
                running = True
                keys = None
                if self.state.frame is not None:
                    running, keys = self.visualizer.update(
                        self.state.frame, self.state.results, self.state.kalman_predictions
                    )
                if not running:
                    break

                # Ego update
                if self.manual and keys is not None:
                    ctrl = self.keyboard.parse(keys)
                    # AEB override: automatic braking takes priority
                    if self.state.brake_needed:
                        ctrl["throttle"] = 0.0
                        ctrl["brake"] = 1.0
                    self.controller.apply_control(**ctrl)
                else:
                    if self.state.brake_needed:
                        self.controller.apply_control(throttle=0.0, steer=0.0, brake=1.0)
                    else:
                        self.controller.apply_throttle(self.scenario.target_speed_mps)
                        
        except KeyboardInterrupt:
            logger.info("Simulation stopped by user.")
        finally:
            self.cleanup()
            
    def cleanup(self):
        logger.info("Cleaning up actors...")
        if self.world is not None:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
            
        if self.visualizer is not None:
            self.visualizer.close()
        if self.sensor_manager is not None:
            self.sensor_manager.destroy()
        actors_to_destroy = list(self.walkers) if hasattr(self, "walkers") else []
        if self.vehicle is not None:
            actors_to_destroy.append(self.vehicle)
        safe_destroy(actors_to_destroy)
