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
from utils.carla_utils import move_spectator_to

logger = get_logger(__name__)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class SimulationState:
    """Helper to share data between camera callback and main loop."""
    def __init__(self):
        self.frame = None
        self.depth_array = None
        self.results = None
        self.kalman_predictions = None
        self.kalman_filters = {}
        self.ego_speed = 0.0

state = SimulationState()

def main():
    parser = argparse.ArgumentParser(description="SmartVehicular simulation runner with moving vehicle and crossing pedestrian")
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
            height=config["sensors"]["camera"]["height"],
            title="SmartVehicular Detection - Moving Vehicle & Crossing Walker"
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
        
        # --- PARAMETRI DI SPAWN PEDONE (LONTANO SUL MARCIAPIEDE CHE ATTRAVERSA) ---
        distanza_dal_veicolo = 35.0  # Spawna lontano
        altezza_iniziale = 1.0
        spawn_laterale = 5.0  # Lato marciapiede/banchina
        # --------------------------------------------------------------------------

        p_bp = blueprint_library.filter("walker.pedestrian.*")[0]
        if p_bp.has_attribute('is_invincible'):
            p_bp.set_attribute('is_invincible', 'true')
        
        # Usiamo il punto di spawn del veicolo per posizionare il pedone davanti
        v_location = spawn_point.location
        v_rotation = spawn_point.rotation
        forward_vector = v_rotation.get_forward_vector()
        right_vector = v_rotation.get_right_vector()
        
        # Calcolo posizione sul marciapiede davanti all'auto
        p_location = v_location + forward_vector * distanza_dal_veicolo + right_vector * spawn_laterale
        
        walker = None
        for offset in [0.0, 0.5, 1.0, 1.5]:
            p_location.z = v_location.z + altezza_iniziale + offset
            p_transform = carla.Transform(p_location, v_rotation)
            walker = world.try_spawn_actor(p_bp, p_transform)
            if walker is not None:
                break
                
        if walker is None:
            logger.warning("Impossibile spawnare il pedone sul marciapiede.")
        else:
            logger.info("Pedone spawnato sul marciapiede a %s metri di distanza", distanza_dal_veicolo)
            
            # Direzione per attraversare la strada (perpendicolare alla direzione dell'auto)
            # Se siamo a destra (laterale positivo), andiamo verso sinistra (negativo)
            if spawn_laterale > 0:
                cross_direction = carla.Vector3D(-right_vector.x, -right_vector.y, -right_vector.z)
            else:
                cross_direction = right_vector
            cross_direction = cross_direction.make_unit_vector()
            
            # Applica controllo al pedone per farlo camminare a 1.4 m/s attraverso la strada
            walker.apply_control(carla.WalkerControl(direction=cross_direction, speed=1.4))
        
        # 4. Setup Sensors
        sensor_manager = SensorManager(world, vehicle)
        
        cam_bp = blueprint_library.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(config["sensors"]["camera"]["width"]))
        cam_bp.set_attribute('image_size_y', str(config["sensors"]["camera"]["height"]))
        cam_bp.set_attribute('fov', str(config["sensors"]["camera"]["fov"]))
        
        cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
        sensor_manager.add_sensor(camera)
        
        # --- SENSOR DEPTH ---
        depth_bp = blueprint_library.find('sensor.camera.depth')
        depth_bp.set_attribute('image_size_x', str(config["sensors"]["camera"]["width"]))
        depth_bp.set_attribute('image_size_y', str(config["sensors"]["camera"]["height"]))
        depth_bp.set_attribute('fov', str(config["sensors"]["camera"]["fov"]))
        depth_camera = world.spawn_actor(depth_bp, cam_transform, attach_to=vehicle)
        sensor_manager.add_sensor(depth_camera)
        
        def depth_callback(image):
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
            # Convert raw data to numpy array
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            # CARLA camera raw data is in BGRA format. Slicing array[:, :, :3] gives BGR.
            # We convert BGR to RGB by reversing the channels along the third axis.
            rgb_array = array[:, :, :3][:, :, ::-1]
            
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
                    
                    # 1. Recuperiamo la profondità Z (in metri)
                    Z = 10.0 # fallback
                    if state.depth_array is not None:
                        cy_int = max(0, min(img_h - 1, int(cy)))
                        cx_int = max(0, min(img_w - 1, int(cx)))
                        Z = float(state.depth_array[cy_int, cx_int])
                        if Z <= 0.0: Z = 10.0
                        
                    # 2. Calcoliamo X in metri (distanza laterale)
                    X = (cx - img_w / 2.0) * Z / f_length
                    
                    if obj_id not in state.kalman_filters:
                        state.kalman_filters[obj_id] = TrajectoryKalmanFilter(dt=1.0/config["simulation"].get("fps", 20))
                    
                    # 3. Aggiorniamo Kalman con X e Z IN METRI (e forniamo la velocità dell'auto)
                    est_x, est_z, vel_x, vel_z = state.kalman_filters[obj_id].predict_update(
                        X, Z, ego_vx=0.0, ego_vz=state.ego_speed
                    )
                    
                    # 4. Prevediamo la posizione 3D futura (in coordinate telecamera)
                    future_frames = 100
                    dt = state.kalman_filters[obj_id].dt
                    pred_X = est_x + vel_x * dt * future_frames
                    pred_Z = est_z + vel_z * dt * future_frames
                    
                    # Se l'oggetto finisce dietro la telecamera (Z < 0), lo limitiamo
                    if pred_Z <= 0.5:
                        continue
                        
                    # 5. Riproiettiamo in coordinate Pixel 2D per il disegno
                    pred_cx = (pred_X * f_length) / pred_Z + img_w / 2.0
                    
                    # Scala della Y e dimensione box in base alla nuova profondità
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
                    
                    # Generiamo una serie di punti intermedi per disegnare la traiettoria
                    trajectory = []
                    for frm in range(2, future_frames + 1, 4):
                        pt_X = est_x + vel_x * dt * frm
                        pt_Z = est_z + vel_z * dt * frm
                        
                        if pt_Z > 0.5: # Disegna solo se davanti alla camera
                            pt_cx = (pt_X * f_length) / pt_Z + img_w / 2.0
                            pt_cy = img_h / 2.0 + (cy - img_h / 2.0) * (Z / pt_Z)
                            trajectory.append((pt_cx, pt_cy))
                    
                    predictions[obj_id] = {
                        "box": pred_box,
                        "trajectory": trajectory,
                        "time_horizon": future_frames * dt
                    }
            
            # Update shared state
            state.frame = rgb_array
            state.results = results
            state.kalman_predictions = predictions

        camera.listen(lambda image: camera_callback(image))
        
        # 5. Simulation Loop
        logger.info("Starting simulation loop. Press Ctrl+C or close window to stop.")
        
        spectator = world.get_spectator()
        
        while True:
            world.tick()
            
            # Regolatore di velocità per mantenere circa 20 km/h (5.56 m/s)
            current_vel = vehicle.get_velocity()
            current_speed = np.sqrt(current_vel.x**2 + current_vel.y**2 + current_vel.z**2)
            state.ego_speed = current_speed
            target_speed = 5.56 # 20 km/h in m/s
            
            error = target_speed - current_speed
            # Controllo P semplice per acceleratore
            throttle = max(0.0, min(1.0, 0.4 * error + 0.1))
            vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=0.0))
            
            # Sposta la telecamera dello spettatore dietro il veicolo in corsa
            move_spectator_to(vehicle.get_transform(), spectator, distance=9.0, z=3.5, pitch=-16.0)
            
            # Aggiorna il visualizzatore Pygame
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
        if 'vehicle' in locals():
            vehicle.destroy()


if __name__ == "__main__":
    main()
