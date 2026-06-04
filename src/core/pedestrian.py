import carla
from utils.logger import get_logger

logger = get_logger(__name__)

def spawn_pedestrian(world, vehicle_transform, scenario):
    """Spawn a pedestrian (walker) near the vehicle based on scenario parameters."""
    blueprint_library = world.get_blueprint_library()
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
    p_rotation = carla.Rotation(pitch=v_rotation.pitch, yaw=v_rotation.yaw, roll=v_rotation.roll)
    if walker_horizontal_offset < 0:
        p_rotation.yaw += 180

    walker = None
    for offset in [0.0, 0.5, 1.0, 1.5]:
        p_location.z = v_location.z + walker_height + offset
        p_transform = carla.Transform(p_location, p_rotation)
        walker = world.try_spawn_actor(p_bp, p_transform)
        if walker is not None:
            break
            
    if walker is None:
        logger.warning("Can't spawn walker near the sidewalk.")
        return None
    
    logger.info("Walker spawned on the sidewalk at a distance of %s meters", distance_from_vehicle)
    
    if walker_horizontal_offset > 0:
        cross_direction = carla.Vector3D(-right_vector.x, -right_vector.y, -right_vector.z)
    else:
        cross_direction = right_vector
    cross_direction = cross_direction.make_unit_vector()
    
    walker.apply_control(
        carla.WalkerControl(direction=cross_direction, speed=walker_speed)
    )
    return walker
