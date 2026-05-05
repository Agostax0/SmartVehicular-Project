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
    back = transform.location - transform.get_forward_vector() * distance
    loc = carla.Location(back.x, back.y, back.z + z)
    rot = carla.Rotation(pitch=pitch, yaw=transform.rotation.yaw, roll=0.0)
    spectator.set_transform(carla.Transform(loc, rot))


def spawn_vehicle(
    world, spawn_index=0, vehicle_filter="vehicle.tesla.model3", autopilot=False
):
    points = world.get_map().get_spawn_points()
    if not points:
        raise RuntimeError("No spawn points found")

    bps = world.get_blueprint_library().filter(vehicle_filter)
    if not bps:
        bps = world.get_blueprint_library().filter("vehicle.*")

    for k in range(len(points)):
        actor = world.try_spawn_actor(
            random.choice(bps), points[(spawn_index + k) % len(points)]
        )
        if actor is not None:
            actor.set_autopilot(autopilot)
            return actor

    raise RuntimeError("Could not spawn vehicle")


def spawn_vehicle_ahead(
    world,
    ref_vehicle,
    distances: Sequence[float] = (20.0, 28.0, 36.0),
    same_lane_first=True,
    retries=6,
):
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
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
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
    bp = world.get_blueprint_library().find("sensor.other.radar")
    bp.set_attribute("horizontal_fov", str(horizontal_fov))
    bp.set_attribute("vertical_fov", str(vertical_fov))
    bp.set_attribute("points_per_second", str(points_per_second))
    bp.set_attribute("range", str(range_m))
    return world.spawn_actor(bp, transform, attach_to=attach_to)


def image_to_bgr(image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = np.reshape(arr, (image.height, image.width, 4))
    return arr[:, :, :3].copy()


def lidar_to_numpy(measurement):
    pts = np.frombuffer(measurement.raw_data, dtype=np.float32)
    return np.reshape(pts, (-1, 4))


def safe_destroy(actors: Iterable):
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
    "spawn_lidar",
    "spawn_radar",
    "image_to_bgr",
    "lidar_to_numpy",
    "safe_destroy",
]
