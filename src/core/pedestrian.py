import random

import carla
from utils.carla_utils import safe_destroy
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


def spawn_crowd(world, client, num_walkers, max_speed, seed=None):
    """Spawn a crowd of AI-driven pedestrians that wander Town03 on the navmesh.

    Uses the official CARLA batch-spawn pattern: walkers are placed at random
    navigation locations (sidewalks/pedestrian areas), each paired with a
    ``controller.ai.walker`` that drives it toward random destinations and
    re-picks a new target on arrival.

    Args:
        world: CARLA world instance (must be in synchronous mode).
        client: CARLA client instance (needed for apply_batch_sync).
        num_walkers: How many pedestrians to spawn.
        max_speed: Max walking speed in m/s applied to every controller.
        seed: Optional RNG seed for reproducible blueprint/spawn selection.

    Returns:
        (walkers, controllers) — two parallel lists of successfully spawned
        actor objects (their lengths are equal and may be < num_walkers if
        some spawns collided on the navmesh).
    """
    rng = random.Random(seed)
    blueprint_library = world.get_blueprint_library()
    walker_blueprints = blueprint_library.filter("walker.pedestrian.*")

    # Pick one random navigation location per walker as the spawn point.
    spawn_locations = []
    for _ in range(num_walkers):
        loc = world.get_random_location_from_navigation()
        if loc is not None:
            spawn_locations.append(loc)

    if not spawn_locations:
        logger.warning("Crowd spawn failed: no navigation locations available "
                       "(navmesh not loaded for this map).")
        return [], []

    # Batch-spawn walkers at the chosen locations with randomized blueprints.
    walker_batch = []
    for loc in spawn_locations:
        bp = rng.choice(walker_blueprints)
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        if bp.has_attribute("speed"):
            bp.set_attribute("speed", "walk")
        transform = carla.Transform(loc)
        walker_batch.append(carla.command.SpawnActor(bp, transform))

    walker_results = client.apply_batch_sync(walker_batch, True)
    world.tick()

    walker_ids = [r.actor_id for r in walker_results if r.error is None]
    skipped = len(spawn_locations) - len(walker_ids)
    if skipped:
        logger.info("Crowd spawn: %d walker(s) skipped due to collisions.", skipped)

    # Batch-spawn one AI controller per successfully spawned walker, attached
    # to its walker as parent. Controller i pairs with walker i positionally.
    controller_bp = blueprint_library.find("controller.ai.walker")
    controller_batch = [
        carla.command.SpawnActor(controller_bp, carla.Transform(), wid)
        for wid in walker_ids
    ]

    controller_results = client.apply_batch_sync(controller_batch, True)
    world.tick()

    # Keep only walkers whose controller spawned successfully; the pairing is
    # positional, so walker_ids[i] matches controller_results[i].
    all_actor_ids = []
    pairs = []  # (walker_id, controller_id)
    for wid, res in zip(walker_ids, controller_results):
        if res.error is None:
            pairs.append((wid, res.actor_id))
            all_actor_ids.append(wid)
            all_actor_ids.append(res.actor_id)

    if not pairs:
        logger.warning("Crowd spawn failed: no AI controllers could be spawned.")
        return [], []

    all_actors = world.get_actors(all_actor_ids)
    paired_walkers, paired_controllers = [], []
    for wid, cid in pairs:
        w = all_actors.find(wid)
        c = all_actors.find(cid)
        if w is not None and c is not None:
            paired_walkers.append(w)
            paired_controllers.append(c)

    # Start AI: bind controller to walker, give a first random destination.
    for w, c in zip(paired_walkers, paired_controllers):
        c.start(w)
        c.go_to_location(world.get_random_location_from_navigation())
        c.set_max_speed(max_speed)

    logger.info("Crowd spawned: %d/%d AI pedestrians wandering at %.2f m/s.",
                len(paired_walkers), num_walkers, max_speed)
    return paired_walkers, paired_controllers


def navigate_crowd(world, controllers):
    """Re-issue a random navmesh destination to every crowd controller.

    Keeps the crowd in perpetual motion: pedestrians that reached their goal
    (and would otherwise idle) get a fresh target. Call periodically (~10 s).
    """
    for c in controllers:
        try:
            c.go_to_location(world.get_random_location_from_navigation())
        except RuntimeError:
            # Controller may have been destroyed mid-loop; skip it.
            continue


def cleanup_crowd(world, walkers, controllers):
    """Tear down a spawned crowd in the correct order to avoid errors.

    Controllers must be stopped and ticked before actors are destroyed, and
    controllers are destroyed before walkers.
    """
    if not walkers and not controllers:
        return

    for c in controllers:
        try:
            c.stop()
        except RuntimeError:
            continue

    try:
        world.tick()
    except RuntimeError:
        pass

    safe_destroy(controllers)
    safe_destroy(walkers)
    logger.info("Crowd cleaned up: %d walkers, %d controllers removed.",
                len(walkers), len(controllers))
