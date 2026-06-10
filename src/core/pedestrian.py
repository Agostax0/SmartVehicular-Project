import carla
from utils.logger import get_logger

logger = get_logger(__name__)

def spawn_pedestrian(world, vehicle_transform, scenario):
    """Spawn a pedestrian (walker) or a cyclist near the vehicle based on scenario parameters."""
    blueprint_library = world.get_blueprint_library()
    w_type = getattr(scenario, "walker_type", "pedestrian")
    is_cyclist = w_type == "cyclist"
    
    if is_cyclist:
        bike_bps = blueprint_library.filter("vehicle.bh.crossbike")
        if not bike_bps:
            bike_bps = blueprint_library.filter("vehicle.diamondback.century")
        if not bike_bps:
            bike_bps = blueprint_library.filter("vehicle.gazelle.omafiets")
        p_bp = bike_bps[0]
    else:
        p_bp = blueprint_library.filter("walker.pedestrian.*")[2]
        if p_bp.has_attribute('is_invincible'):
            p_bp.set_attribute('is_invincible', 'true')
    
    distance_from_vehicle = scenario.walker_distance_from_vehicle
    walker_height = scenario.walker_height
    walker_horizontal_offset = scenario.walker_horizontal_offset
    walker_speed = scenario.walker_speed

    v_location = vehicle_transform.location
    v_rotation = vehicle_transform.rotation
    forward_vector = v_rotation.get_forward_vector()
    right_vector = v_rotation.get_right_vector()
    
    p_location = v_location + forward_vector * distance_from_vehicle + right_vector * walker_horizontal_offset
    
    if is_cyclist:
        if walker_horizontal_offset > 0:
            yaw = v_rotation.yaw - 90.0
        else:
            yaw = v_rotation.yaw + 90.0
        p_rotation = carla.Rotation(pitch=v_rotation.pitch, yaw=yaw, roll=v_rotation.roll)
    else:
        p_rotation = carla.Rotation(pitch=v_rotation.pitch, yaw=v_rotation.yaw, roll=v_rotation.roll)
        if walker_horizontal_offset < 0:
            p_rotation.yaw += 180

    walker = None
    spawn_z_offset = 0.2 if is_cyclist else walker_height
    for offset in [0.0, 0.5, 1.0, 1.5]:
        p_location.z = v_location.z + spawn_z_offset + offset
        p_transform = carla.Transform(p_location, p_rotation)
        walker = world.try_spawn_actor(p_bp, p_transform)
        if walker is not None:
            break
            
    if walker is None:
        logger.warning("Can't spawn actor near the sidewalk.")
        return None
    
    logger.info("%s spawned at a distance of %s meters", w_type.capitalize(), distance_from_vehicle)
    
    if walker_horizontal_offset > 0:
        cross_direction = carla.Vector3D(-right_vector.x, -right_vector.y, -right_vector.z)
    else:
        cross_direction = right_vector
    cross_direction = cross_direction.make_unit_vector()
    
    if is_cyclist:
        # Use physics-based control so the bike drops to the ground and wheels rotate
        throttle_val = min(1.0, walker_speed / 5.0)  # rough approximation
        walker.apply_control(carla.VehicleControl(throttle=max(0.3, throttle_val), steer=0.0))
    else:
        walker.apply_control(
            carla.WalkerControl(direction=cross_direction, speed=walker_speed)
        )
    return walker
