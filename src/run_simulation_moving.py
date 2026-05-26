import yaml
import carla
import numpy as np
import typer
from typing import Optional

from utils.logger import get_logger
from utils.kalman import TrajectoryKalmanFilter
from utils.detector import ObjectDetector
from utils.visualizer import Visualizer
from sensors.sensor_manager import SensorManager
from scenarios import get_scenario, list_scenario_names
from utils.carla_utils import (
    image_to_bgr,
    move_spectator_to,
    safe_destroy,
    spawn_camera,
    spawn_depth_camera,
    spawn_vehicle,
)
from utils.collision_module import check_collision_risk

logger = get_logger(__name__)


def load_config(path: str) -> dict:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Parsed configuration as a dictionary.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


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

state = SimulationState()

def run_simulation(config_path: str, scenario_name: Optional[str]) -> None:
    """Run the moving-vehicle simulation with a selected scenario.

    Args:
        config_path: Path to the YAML configuration file.
        scenario_name: Registered scenario identifier (optional if default is in config).

    Returns:
        None.
    """
    config = load_config(config_path)
    scenario = get_scenario(config, scenario_name)
    logger.info("Loaded config: %s", config_path)
    logger.info("Running scenario: %s", scenario.name)

    client = carla.Client(config["carla"]["host"], config["carla"]["port"])
    client.set_timeout(config["carla"]["timeout"])

    vehicle = None
    walker = None
    
    try:
        world = client.get_world()
        spectator = world.get_spectator()
        
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / config["simulation"].get("fps", 20)
        world.apply_settings(settings)
        
        logger.info("Connected to CARLA and enabled sync mode: %s", world.get_map().name)
        
        detector = ObjectDetector()
        visualizer = Visualizer(
            width=config["sensors"]["camera"]["width"],
            height=config["sensors"]["camera"]["height"],
            title=f"SmartVehicular Detection - {scenario.name}"
        )
        
        blueprint_library = world.get_blueprint_library()
        try:
            vehicle, spawn_transform = spawn_vehicle(
                world,
                spawn_index=0,
                vehicle_filter=config["vehicle"]["blueprint"],
                autopilot=False,
            )
        except RuntimeError as exc:
            logger.error("Failed to spawn vehicle at any available spawn point: %s", exc)
            return

        logger.info("Vehicle spawned: %s at %s", vehicle.type_id, spawn_transform.location)
        
        distance_from_vehicle = scenario.walker_distance_from_vehicle
        walker_height = scenario.walker_height
        walker_horizontal_offset = scenario.walker_horizontal_offset

        p_bp = blueprint_library.filter("walker.pedestrian.*")[2]
        if p_bp.has_attribute('is_invincible'):
            p_bp.set_attribute('is_invincible', 'true')
        
        v_location = spawn_transform.location
        v_rotation = spawn_transform.rotation
        forward_vector = v_rotation.get_forward_vector()
        right_vector = v_rotation.get_right_vector()
        
        p_location = v_location + forward_vector * distance_from_vehicle + right_vector * walker_horizontal_offset
        p_rotation = v_rotation
        if(walker_horizontal_offset < 0):
            p_rotation.yaw += 180

        walker = None
        for offset in [0.0, 0.5, 1.0, 1.5]:
            p_location.z = v_location.z + walker_height + offset
            p_transform = carla.Transform(p_location, v_rotation)
            walker = world.try_spawn_actor(p_bp, p_transform)
            if walker is not None:
                break
                
        if walker is None:
            logger.warning("Can't spawn walker near the sidewalk.")
        else:
            logger.info("Walker spawned on the sidewalk at a distance of %s meters", distance_from_vehicle)
            
            if walker_horizontal_offset > 0:
                cross_direction = carla.Vector3D(-right_vector.x, -right_vector.y, -right_vector.z)
            else:
                cross_direction = right_vector
            cross_direction = cross_direction.make_unit_vector()
            
            walker.apply_control(
                carla.WalkerControl(direction=cross_direction, speed=scenario.walker_speed)
            )
        
        sensor_manager = SensorManager(world, vehicle)
        
        cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = spawn_camera(
            world,
            vehicle,
            cam_transform,
            width=config["sensors"]["camera"]["width"],
            height=config["sensors"]["camera"]["height"],
            fov=config["sensors"]["camera"]["fov"],
            tick=0.0,
        )
        sensor_manager.add_sensor(camera)
        
        depth_camera = spawn_depth_camera(
            world,
            vehicle,
            cam_transform,
            width=config["sensors"]["camera"]["width"],
            height=config["sensors"]["camera"]["height"],
            fov=config["sensors"]["camera"]["fov"],
            tick=0.0,
        )
        sensor_manager.add_sensor(depth_camera)
        
        def depth_callback(image):
            """Convert depth frames into a metric depth array.

            Args:
                image: CARLA depth camera image.

            Returns:
                None.
            """
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            # The depth map from CARLA is coded in RGB as: R + G*256 + B*256*256
            B = array[:, :, 0].astype(np.float32)
            G = array[:, :, 1].astype(np.float32)
            R = array[:, :, 2].astype(np.float32)
            normalized_depth = (R + G * 256.0 + B * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1.0)
            depth_in_meters = 1000.0 * normalized_depth
            state.depth_array = depth_in_meters

        depth_camera.listen(lambda image: depth_callback(image))
        
        def camera_callback(image):
            """Process camera frames and update detection state.

            Args:
                image: CARLA camera image.

            Returns:
                None.
            """
            bgr_array = image_to_bgr(image)
            rgb_array = bgr_array[:, :, ::-1]
            
            # Run detection
            results = detector.detect(rgb_array)
            
            # Kalman Filter Processing
            predictions = {}
            if results and results.boxes and results.boxes.id is not None:
                img_w = config["sensors"]["camera"]["width"]
                img_h = config["sensors"]["camera"]["height"]
                fov_rad = np.deg2rad(config["sensors"]["camera"]["fov"])
                f_length = (img_w / 2.0) / np.tan(fov_rad / 2.0)

                for i, box in enumerate(results.boxes):
                    obj_id = int(results.boxes.id[i])
                    coords = box.xyxy[0].tolist() # [x1, y1, x2, y2]
                    
                    x1, y1, x2, y2 = coords
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    w = x2 - x1
                    h = y2 - y1
                    
                    Z = 10.0 # fallback
                    if state.depth_array is not None:
                        cy_int = max(0, min(img_h - 1, int(cy)))
                        cx_int = max(0, min(img_w - 1, int(cx)))
                        Z = float(state.depth_array[cy_int, cx_int])
                        if Z <= 0.0: Z = 10.0
                        
                    X = (cx - img_w / 2.0) * Z / f_length
                    
                    if obj_id not in state.kalman_filters:
                        state.kalman_filters[obj_id] = TrajectoryKalmanFilter(dt=1.0/config["simulation"].get("fps", 20))
                    
                    est_x, est_z, vel_x, vel_z = state.kalman_filters[obj_id].predict_update(
                        X, Z, ego_vx=0.0, ego_vz=state.ego_speed
                    )
                    
                    real_v = vel_z - state.ego_speed
                    risk, msg, ttc = check_collision_risk(est_x, est_z, vel_x, real_v)
                    print(f"{risk}, {msg}, {ttc}")


                    future_frames = scenario.prediction_future_frames
                    dt = state.kalman_filters[obj_id].dt
                    pred_X = est_x + vel_x * dt * future_frames
                    pred_Z = est_z + vel_z * dt * future_frames
                    
                    if pred_Z <= 0.5:
                        continue
                        
                    pred_cx = (pred_X * f_length) / pred_Z + img_w / 2.0
                    
                    scale_factor = Z / pred_Z
                    pred_cy = img_h / 2.0 + (cy - img_h / 2.0) * scale_factor
                    pred_w = w * scale_factor
                    pred_h = h * scale_factor
                    
                    pred_box = [
                        pred_cx - pred_w / 2.0,
                        pred_cy - pred_h / 2.0,
                        pred_cx + pred_w / 2.0,
                        pred_cy + pred_h / 2.0
                    ]
                    
                    trajectory = []
                    for frm in range(2, future_frames + 1, 4):
                        pt_X = est_x + vel_x * dt * frm
                        pt_Z = est_z + vel_z * dt * frm
                        
                        if pt_Z > 0.5: 
                            pt_cx = (pt_X * f_length) / pt_Z + img_w / 2.0
                            pt_cy = img_h / 2.0 + (cy - img_h / 2.0) * (Z / pt_Z)
                            trajectory.append((pt_cx, pt_cy))
                    
                    predictions[obj_id] = {
                        "box": pred_box,
                        "trajectory": trajectory,
                        "time_horizon": future_frames * dt
                    }
            
            state.frame = rgb_array
            state.results = results
            state.kalman_predictions = predictions

        camera.listen(lambda image: camera_callback(image))
        
        logger.info("Starting simulation loop. Press Ctrl+C or close window to stop.")
        
        
        while True:
            world.tick()
            
            current_vel = vehicle.get_velocity()
            current_speed = np.sqrt(current_vel.x**2 + current_vel.y**2 + current_vel.z**2)
            state.ego_speed = current_speed
            target_speed = scenario.target_speed_mps
            
            error = target_speed - current_speed
            throttle = max(0.0, min(1.0, 0.4 * error + 0.1))
            vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=0.0))
            
            #move_spectator_to(vehicle.get_transform(), spectator, distance=9.0, z=3.5, pitch=-16.0)
            
            if state.frame is not None:
                if not visualizer.update(state.frame, state.results, state.kalman_predictions):
                    break
            
    except KeyboardInterrupt:
        logger.info("Simulation stopped by user.")
    finally:
        logger.info("Cleaning up actors...")
        
        if 'world' in locals():
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
            
        if 'visualizer' in locals():
            visualizer.close()
        if 'sensor_manager' in locals():
            sensor_manager.destroy()
        safe_destroy([walker, vehicle])


def main(
    config: str = typer.Option(
        "config/config.yaml",
        "--config",
        help="Path to the YAML configuration file.",
    ),
    scenario: Optional[str] = typer.Option(
        None,
        "--scenario",
        help="Scenario name to run. Defaults to simulation.default_scenario in config.",
    ),
    list_scenarios: bool = typer.Option(
        False,
        "--list-scenarios",
        help="List all available scenarios and exit.",
    ),
) -> None:
    """CLI entrypoint for running simulation scenarios."""
    if list_scenarios:
        config_data = load_config(config)
        for name in list_scenario_names(config_data):
            scenario_config = get_scenario(config_data, name)
            typer.echo(f"{name}: {scenario_config.description}")
        raise typer.Exit()

    try:
        run_simulation(config_path=config, scenario_name=scenario)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--scenario") from exc


if __name__ == "__main__":
    typer.run(main)
