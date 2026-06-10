"""Reusable CARLA utility module.

This module exposes helper functions for actor spawning, sensor setup,
conversion utilities, and safe teardown.
"""

from __future__ import annotations

import random
import time
from typing import Iterable, Sequence

import carla
import numpy as np

def move_spectator_to(transform, spectator, distance=7.0, z=3.0, pitch=-15.0):
    """Move the spectator camera behind a given transform.

    Args:
        transform: Reference CARLA transform to follow.
        spectator: CARLA spectator actor to move.
        distance: Distance behind the reference transform in meters.
        z: Vertical offset in meters.
        pitch: Pitch angle in degrees.

    Returns:
        None.
    """
    back = transform.location - transform.get_forward_vector() * distance
    loc = carla.Location(back.x, back.y, back.z + z)
    rot = carla.Rotation(pitch=pitch, yaw=transform.rotation.yaw, roll=0.0)
    spectator.set_transform(carla.Transform(loc, rot))


def spawn_vehicle(
    world, spawn_index=0, vehicle_filter="vehicle.tesla.model3", autopilot=False
):
    """Spawn a vehicle in the world at the first available spawn point.

    Args:
        world: CARLA world instance.
        spawn_index: Preferred spawn point index.
        vehicle_filter: Blueprint filter for vehicle selection.
        autopilot: Whether to enable autopilot on the spawned actor.

    Returns:
        The spawned vehicle actor.
    """
    points = world.get_map().get_spawn_points()
    if not points:
        raise RuntimeError("No spawn points found")

    bps = world.get_blueprint_library().filter(vehicle_filter)
    if not bps:
        bps = world.get_blueprint_library().filter("vehicle.*")

    for k in range(len(points)):

        spawn_point = points[spawn_index + k % len(points)]
        actor = world.try_spawn_actor(
            random.choice(bps), spawn_point
        )
        if actor is not None:
            actor.set_autopilot(autopilot)
            actor.set_transform(spawn_point)
            return actor, spawn_point

    raise RuntimeError("Could not spawn vehicle")


def spawn_vehicle_ahead(
    world,
    ref_vehicle,
    distances: Sequence[float] = (20.0, 28.0, 36.0),
    same_lane_first=True,
    retries=6,
):
    """Spawn a vehicle ahead of a reference vehicle when possible.

    Args:
        world: CARLA world instance.
        ref_vehicle: The reference vehicle actor.
        distances: Candidate distances ahead in meters.
        same_lane_first: Prefer same-lane waypoints when available.
        retries: Number of retries when spawning at waypoints.

    Returns:
        A spawned vehicle actor or an existing one ahead.
    """
    ego_tf = ref_vehicle.get_transform()
    ego_loc = ego_tf.location
    fwd = ego_tf.get_forward_vector()
    right = ego_tf.get_right_vector()

    bps = world.get_blueprint_library().filter("vehicle.*")
    if not bps:
        raise RuntimeError("No vehicle blueprints found")

    # Phase 1: try exact transforms straight ahead of ego so the target is visually in front.
    direct = sorted(set(float(d) for d in distances))
    direct += [d + 4.0 for d in direct]
    for d in direct:
        loc = ego_loc + fwd * d + right * 0.0
        loc.z += 0.30
        tf = carla.Transform(loc, ego_tf.rotation)
        actor = world.try_spawn_actor(random.choice(bps), tf)
        if actor is not None:
            actor.set_autopilot(False)
            return actor

    # Phase 2: fallback to road waypoints in front.
    road_map = world.get_map()
    ego_wp = road_map.get_waypoint(
        ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving
    )

    wp_candidates = []
    d_pool = sorted(
        set(float(d) for d in distances) | set(float(d) + 8.0 for d in distances)
    )
    for d in d_pool:
        try:
            cands = ego_wp.next(float(d))
        except RuntimeError:
            cands = []
        if same_lane_first:
            cands = sorted(
                cands,
                key=lambda w: (
                    w.road_id != ego_wp.road_id,
                    w.lane_id != ego_wp.lane_id,
                    abs(w.lane_id - ego_wp.lane_id),
                ),
            )
        wp_candidates.extend(cands)

    for _ in range(max(1, int(retries))):
        for wp in wp_candidates:
            actor = world.try_spawn_actor(random.choice(bps), wp.transform)
            if actor is not None:
                actor.set_autopilot(False)
                return actor
        world.tick()
        time.sleep(0.02)

    # Phase 3: fallback to an already existing vehicle in front.
    best = None
    best_d = float("inf")
    for a in world.get_actors().filter("vehicle.*"):
        if a.id == ref_vehicle.id:
            continue
        loc = a.get_transform().location
        rel = loc - ego_loc
        d_fwd = rel.x * fwd.x + rel.y * fwd.y
        d_lat = abs(rel.x * (-fwd.y) + rel.y * fwd.x)
        if d_fwd > 6.0 and d_lat < 4.0 and d_fwd < best_d:
            best = a
            best_d = d_fwd

    if best is not None:
        try:
            best.set_autopilot(False)
        except RuntimeError:
            pass
        return best

    raise RuntimeError("Could not spawn or find a target vehicle ahead")


def spawn_camera(world, attach_to, transform, width=640, height=360, fov=95, tick=0.05):
    """Spawn an RGB camera sensor attached to an actor.

    Args:
        world: CARLA world instance.
        attach_to: Actor to attach the camera to.
        transform: Transform for the camera relative to the actor.
        width: Image width in pixels.
        height: Image height in pixels.
        fov: Field of view in degrees.
        tick: Sensor tick interval in seconds.

    Returns:
        The spawned camera actor.
    """
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(width))
    bp.set_attribute("image_size_y", str(height))
    bp.set_attribute("fov", str(fov))
    bp.set_attribute("sensor_tick", str(tick))
    return world.spawn_actor(bp, transform, attach_to=attach_to)


def spawn_depth_camera(
    world, attach_to, transform, width=640, height=360, fov=95, tick=0.05
):
    """Spawn a depth camera sensor attached to an actor.

    Args:
        world: CARLA world instance.
        attach_to: Actor to attach the camera to.
        transform: Transform for the camera relative to the actor.
        width: Image width in pixels.
        height: Image height in pixels.
        fov: Field of view in degrees.
        tick: Sensor tick interval in seconds.

    Returns:
        The spawned depth camera actor.
    """
    bp = world.get_blueprint_library().find('sensor.camera.depth')
    bp.set_attribute("image_size_x", str(width))
    bp.set_attribute("image_size_y", str(height))
    bp.set_attribute("fov", str(fov))
    bp.set_attribute("sensor_tick", str(tick))
    return world.spawn_actor(bp, transform, attach_to=attach_to)


def spawn_lidar(
    world,
    attach_to,
    transform,
    channels=32,
    points_per_second=56000,
    rotation_frequency=20,
    range_m=35,
):
    """Spawn a lidar sensor attached to an actor.

    Args:
        world: CARLA world instance.
        attach_to: Actor to attach the lidar to.
        transform: Transform for the lidar relative to the actor.
        channels: Number of lidar channels.
        points_per_second: Points generated per second.
        rotation_frequency: Rotation frequency in Hz.
        range_m: Lidar range in meters.

    Returns:
        The spawned lidar actor.
    """
    bp = world.get_blueprint_library().find("sensor.lidar.ray_cast")
    bp.set_attribute("channels", str(channels))
    bp.set_attribute("points_per_second", str(points_per_second))
    bp.set_attribute("rotation_frequency", str(rotation_frequency))
    bp.set_attribute("range", str(range_m))
    return world.spawn_actor(bp, transform, attach_to=attach_to)


def spawn_radar(
    world,
    attach_to,
    transform,
    horizontal_fov=30,
    vertical_fov=10,
    points_per_second=4000,
    range_m=100,
):
    """Spawn a radar sensor attached to an actor.

    Args:
        world: CARLA world instance.
        attach_to: Actor to attach the radar to.
        transform: Transform for the radar relative to the actor.
        horizontal_fov: Horizontal field of view in degrees.
        vertical_fov: Vertical field of view in degrees.
        points_per_second: Radar points generated per second.
        range_m: Radar range in meters.

    Returns:
        The spawned radar actor.
    """
    bp = world.get_blueprint_library().find("sensor.other.radar")
    bp.set_attribute("horizontal_fov", str(horizontal_fov))
    bp.set_attribute("vertical_fov", str(vertical_fov))
    bp.set_attribute("points_per_second", str(points_per_second))
    bp.set_attribute("range", str(range_m))
    return world.spawn_actor(bp, transform, attach_to=attach_to)


def image_to_bgr(image):
    """Convert a CARLA image to a BGR numpy array.

    Args:
        image: CARLA image object.

    Returns:
        Numpy array with shape (H, W, 3) in BGR order.
    """
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = np.reshape(arr, (image.height, image.width, 4))
    return arr[:, :, :3].copy()


def lidar_to_numpy(measurement):
    """Convert a CARLA lidar measurement to a numpy array.

    Args:
        measurement: CARLA lidar measurement.

    Returns:
        Numpy array of lidar points with shape (N, 4).
    """
    pts = np.frombuffer(measurement.raw_data, dtype=np.float32)
    return np.reshape(pts, (-1, 4))


def safe_destroy(actors: Iterable):
    """Safely destroy a collection of CARLA actors.

    Args:
        actors: Iterable of actors to destroy.

    Returns:
        None.
    """
    for a in actors:
        if a is not None:
            try:
                a.destroy()
            except RuntimeError:
                pass


__all__ = [
    "move_spectator_to",
    "spawn_vehicle",
    "spawn_vehicle_ahead",
    "spawn_camera",
    "spawn_depth_camera",
    "spawn_lidar",
    "spawn_radar",
    "image_to_bgr",
    "lidar_to_numpy",
    "safe_destroy",
]
