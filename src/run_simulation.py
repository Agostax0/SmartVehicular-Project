import argparse
import yaml
import carla
import numpy as np
import time

from utils.logger import get_logger
from utils.kalman import TrajectoryKalmanFilter
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
        self.kalman_predictions = None
        self.kalman_filters = {}

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
        
        # Enable Synchronous Mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / config["simulation"].get("fps", 20)
        world.apply_settings(settings)
        
        logger.info("Connected to CARLA and enabled sync mode: %s", world.get_map().name)
        
        # 2. Setup Components
        detector = ObjectDetector()
        visualizer = Visualizer(
            width=config["sensors"]["camera"]["width"],
            height=config["sensors"]["camera"]["height"]
        )
        
        # 3. Spawn Vehicle
        blueprint_library = world.get_blueprint_library()
        v_bp = blueprint_library.find(config["vehicle"]["blueprint"])
        spawn_points = world.get_map().get_spawn_points()
        vehicle = None
        for sp in spawn_points:
            vehicle = world.try_spawn_actor(v_bp, sp)
            if vehicle is not None:
                spawn_point = sp
                break
        
        if vehicle is None:
            logger.error("Failed to spawn vehicle at any available spawn point")
            return
        logger.info("Vehicle spawned: %s at %s", vehicle.type_id, spawn_point.location)
        
        # --- PARAMETRI DI SPAWN PEDONE ---
        distanza_dal_veicolo = 8.0  # Molto vicino al muso per restare in strada
        altezza_iniziale = 1.0
        spawn_laterale = -2.0
        # ----------------------------------

        p_bp = blueprint_library.filter("walker.pedestrian.*")[0]
        if p_bp.has_attribute('is_invincible'):
            p_bp.set_attribute('is_invincible', 'true')
        
        # Usiamo direttamente il punto di spawn del veicolo per il calcolo
        v_location = spawn_point.location
        v_rotation = spawn_point.rotation
        forward_vector = v_rotation.get_forward_vector()
        right_vector = v_rotation.get_right_vector()
        
        # Calcolo posizione davanti all'auto
        p_location = v_location + forward_vector * (distanza_dal_veicolo - 2) + right_vector * spawn_laterale
        p_rotation = v_rotation
        p_rotation.yaw += 90
        
        walker = None
        walker1 = None
        for offset in [0.0, 0.5, 1.0, 1.5]:
            p_location.z = v_location.z + altezza_iniziale + offset
            p_transform = carla.Transform(p_location, p_rotation)
            walker = world.try_spawn_actor(p_bp, p_transform)
            p1_rotation = v_rotation
            p1_rotation.yaw += 180
            walker1 = world.try_spawn_actor(p_bp, carla.Transform(v_location + forward_vector * distanza_dal_veicolo, p1_rotation))
            if walker is not None:
                break
                
        if walker is None:
            logger.warning("Impossibile spawnare il pedone.")
        else:
            logger.info("Pedone spawnato a %s metri dal punto di spawn auto: %s", distanza_dal_veicolo, p_transform.location)
            direction = p_transform.get_forward_vector()
            d1 = carla.Vector3D(-direction.x, -direction.y, -direction.z)
            walker.apply_control(carla.WalkerControl(direction=direction, speed = 1.4))
            walker1.apply_control(carla.WalkerControl(direction=d1, speed = 1.4))
        
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
            
            # Kalman Filter Processing
            predictions = {}
            if results and results.boxes and results.boxes.id is not None:
                for i, box in enumerate(results.boxes):
                    obj_id = int(results.boxes.id[i])
                    coords = box.xyxy[0].tolist() # [x1, y1, x2, y2]
                    
                    x1, y1, x2, y2 = coords
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    w = x2 - x1
                    h = y2 - y1
                    
                    if obj_id not in state.kalman_filters:
                        # Usiamo il tuo filtro originale passandogli i pixel (cx, cy) al posto di (X, Z) in metri
                        state.kalman_filters[obj_id] = TrajectoryKalmanFilter(dt=1.0/config["simulation"].get("fps", 20))
                    
                    # Passiamo cx come X e cy come Z
                    est_x, est_z, vel_x, vel_z = state.kalman_filters[obj_id].predict_update(cx, cy)
                    
                    # Calcoliamo la box futura (estrapoliamo di 20 frame, circa 1.0s a 20 FPS, per vederla molto più avanti)
                    future_frames = 50
                    dt = state.kalman_filters[obj_id].dt
                    pred_cx = est_x + vel_x * dt * future_frames
                    pred_cy = est_z + vel_z * dt * future_frames
                    
                    pred_box = [
                        pred_cx - w / 2.0,
                        pred_cy - h / 2.0,
                        pred_cx + w / 2.0,
                        pred_cy + h / 2.0
                    ]
                    
                    # Generiamo una serie di punti intermedi per disegnare la traiettoria
                    trajectory = []
                    for f in range(2, future_frames + 1, 2):
                        pt_cx = est_x + vel_x * dt * f
                        pt_cy = est_z + vel_z * dt * f
                        trajectory.append((pt_cx, pt_cy))
                    
                    predictions[obj_id] = {
                        "box": pred_box,
                        "trajectory": trajectory
                    }
            
            # Update shared state
            state.frame = rgb_array
            state.results = results
            state.kalman_predictions = predictions

        camera.listen(lambda image: camera_callback(image))
        
        # 5. Simulation Loop
        logger.info("Starting simulation loop. Press Ctrl+C or close window to stop.")
        while True:
            world.tick()
            
            # Update visualizer from main thread
            if state.frame is not None:
                if not visualizer.update(state.frame, state.results, state.kalman_predictions):
                    break
            
    except KeyboardInterrupt:
        logger.info("Simulation stopped by user.")
    finally:
        logger.info("Cleaning up actors...")
        
        # Disable Sync Mode
        if 'world' in locals():
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
            
        if 'visualizer' in locals():
            visualizer.close()
        if 'sensor_manager' in locals():
            sensor_manager.destroy()
        if 'walker' in locals():
            walker.destroy()
            walker1.destroy()
        if 'vehicle' in locals():
            vehicle.destroy()


if __name__ == "__main__":
    main()
