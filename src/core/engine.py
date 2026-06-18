import carla
import queue
import numpy as np
import pygame

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
    move_spectator_to
)

from controllers.vehicle_controller import VehicleController
from controllers.keyboard_controller import KeyboardController
from controllers.steering_wheel_controller import SteeringWheelController
from core.state import SimulationState
from core.pedestrian import spawn_crowd, navigate_crowd, cleanup_crowd, spawn_pedestrian
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
        self.walker_controllers = []
        self.sensor_manager = None
        self.visualizer = None
        self.detector = ObjectDetector()
        self.perception = PerceptionSystem(config)
        self.state = SimulationState()
        
    def setup(self):
        self.world = self.client.load_world(self.config["carla"]["map"])
        self.world.set_weather(self.scenario.weather_effects.toCarlaWeather())
        spectator = self.world.get_spectator()



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
        move_spectator_to(spawn_transform, spectator)


        self.walkers = []
        self.walker_controllers = []
        if getattr(self.scenario, "crowd_mode", False):
            self.walkers, self.walker_controllers = spawn_crowd(
                self.world,
                self.client,
                num_walkers=self.scenario.crowd_size,
                max_speed=self.scenario.crowd_max_speed,
            )
        elif getattr(self.scenario, "walkers", None) is not None:
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
        self.steering_wheel = None
        if self.manual:
            # Try to connect a USB steering wheel; fall back to keyboard-only
            # if none is detected. pygame.init() has already been called by the
            # Visualizer at this point, so joystick API is available.
            wheel_cfg = self.config.get("controllers", {}).get("steering_wheel", {})
            self.steering_wheel = SteeringWheelController(
                steer_axis=wheel_cfg.get("steer_axis", 0),
                throttle_axis=wheel_cfg.get("throttle_axis", 2),
                brake_axis=wheel_cfg.get("brake_axis", 1),
                deadzone=wheel_cfg.get("deadzone", 0.02),
                hand_brake_button=wheel_cfg.get("hand_brake_button"),
            )
            if self.steering_wheel.is_connected:
                logger.info("Manual control — steering wheel + pedals. "
                            "Keyboard Space = hand-brake fallback.")
            else:
                logger.info("Manual control enabled — use W/A/S/D or arrow keys.")
        
    def run(self):
        image_queue = queue.Queue()
        depth_queue = queue.Queue()
        
        self.depth_camera.listen(depth_queue.put)
        self.camera.listen(image_queue.put)
        
        logger.info("Starting simulation loop. Press Ctrl+C or close window to stop.")

        crowd_mode = getattr(self.scenario, "crowd_mode", False)
        navigate_interval = max(1, self.config["simulation"].get("fps", 20) * 10)  # ~10s
        tick_count = 0

        try:
            while True:
                self.world.tick()
                tick_count += 1

                # Keep the crowd in perpetual motion by re-issuing destinations.
                if crowd_mode and self.walker_controllers and tick_count % navigate_interval == 0:
                    navigate_crowd(self.world, self.walker_controllers)
                
                try:
                    image = image_queue.get(timeout=2.0)
                    depth_image = depth_queue.get(timeout=2.0)
                except queue.Empty:
                    logger.warning("Timeout waiting for sensor data.")
                    continue

                self.state.depth_array = self.perception.process_depth(depth_image)

                current_vel = self.vehicle.get_velocity()
                self.state.ego_speed = np.sqrt(current_vel.x**2 + current_vel.y**2 + current_vel.z**2)

                # Forward (depth) component of the ego velocity in the camera
                # frame: project the world velocity onto the vehicle's forward
                # vector. Equals ego_speed when driving straight, drops when
                # braking/turning — which is what makes Kalman ego-motion
                # compensation correct at high speed and under braking.
                ego_tf = self.vehicle.get_transform()
                fwd = ego_tf.get_forward_vector()
                self.state.ego_vz = current_vel.x * fwd.x + current_vel.y * fwd.y + current_vel.z * fwd.z
                self.state.ego_vx = 0.0

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
                    if self.steering_wheel is not None and self.steering_wheel.is_connected:
                        # Wheel provides analog steer, throttle, brake.
                        # Keyboard Space still serves as hand-brake fallback.
                        ctrl = self.steering_wheel.parse(keys)
                        if keys[pygame.K_SPACE]:
                            ctrl["hand_brake"] = True
                    else:
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

        # Crowd mode tears down controllers and walkers together; in legacy
        # mode there are no controllers and walkers are destroyed below.
        if getattr(self, "walker_controllers", None):
            cleanup_crowd(self.world, self.walkers, self.walker_controllers)
            actors_to_destroy = []
        else:
            actors_to_destroy = list(self.walkers) if hasattr(self, "walkers") else []
        if self.vehicle is not None:
            actors_to_destroy.append(self.vehicle)
        safe_destroy(actors_to_destroy)
