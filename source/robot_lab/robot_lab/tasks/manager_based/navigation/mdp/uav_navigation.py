# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""MDP terms for planar navigation with the tilting UAV."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


PLANAR_LIDAR_RAY_COUNT = 36
"""Number of rays in the 360-degree, 10-degree planar scan."""


def planar_lidar_distances(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    max_distance: float = 4.0,
    expected_rays: int = PLANAR_LIDAR_RAY_COUNT,
) -> torch.Tensor:
    """Return one finite, clipped distance per horizontal ray."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    sensor_data = sensor.data
    ray_starts_w = getattr(sensor, "_ray_starts_w", None)
    if ray_starts_w is None or ray_starts_w.shape != sensor_data.ray_hits_w.shape:
        ray_starts_w = sensor_data.pos_w.unsqueeze(1)
    hit_vectors = sensor_data.ray_hits_w - ray_starts_w
    distances = torch.linalg.norm(hit_vectors, dim=-1)
    distances = torch.nan_to_num(
        distances,
        nan=max_distance,
        posinf=max_distance,
        neginf=max_distance,
    ).clamp(0.0, max_distance)
    if distances.shape[1] != expected_rays:
        raise RuntimeError(
            f"Expected {expected_rays} planar LiDAR rays, got {distances.shape[1]}."
        )
    return distances


def uav_planar_lidar_proximity(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    max_distance: float = 4.0,
) -> torch.Tensor:
    """Return 36 normalized proximity values with beam zero pointing forward.

    Isaac Lab generates a full scan from -180 degrees through +170 degrees. The
    circular roll changes that order to 0, +10, ..., +170, -180, ..., -10 degrees.
    A clear ray is zero and a close obstacle approaches one.
    """
    distances = planar_lidar_distances(env, sensor_cfg, max_distance)
    distances = torch.roll(distances, shifts=PLANAR_LIDAR_RAY_COUNT // 2, dims=1)
    return 1.0 - distances / max(max_distance, 1.0e-6)


def uav_planar_state(
    env: ManagerBasedRLEnv,
    command_name: str = "pose_command",
    goal_distance_scale: float = 50.0,
    velocity_scale: float = 1.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the six planar state features used beside the LiDAR scan.

    The features are relative goal XY, body-frame velocity XY, and sine/cosine
    of the live goal bearing. Position and velocity are normalized by fixed
    deployment-visible scales.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    goal_xy_b = command[:, :2] / max(goal_distance_scale, 1.0e-6)
    velocity_xy_b = asset.data.root_lin_vel_b[:, :2] / max(velocity_scale, 1.0e-6)
    goal_bearing = torch.atan2(command[:, 1], command[:, 0])
    bearing_features = torch.stack((torch.sin(goal_bearing), torch.cos(goal_bearing)), dim=-1)
    return torch.cat((goal_xy_b, velocity_xy_b, bearing_features), dim=-1)


def uav_planar_policy_observation(
    env: ManagerBasedRLEnv,
    command_name: str = "pose_command",
    goal_distance_scale: float = 50.0,
    velocity_scale: float = 1.5,
    max_distance: float = 4.0,
    debug_draw: bool = True,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
) -> torch.Tensor:
    """Concatenate six navigation features and the 36-ray scan for an MLP."""
    state = uav_planar_state(
        env,
        command_name=command_name,
        goal_distance_scale=goal_distance_scale,
        velocity_scale=velocity_scale,
        asset_cfg=asset_cfg,
    )
    lidar = uav_planar_lidar_proximity(env, sensor_cfg=sensor_cfg, max_distance=max_distance)
    if debug_draw:
        _draw_planar_lidar_debug_lines(
            env,
            command_name=command_name,
            sensor_cfg=sensor_cfg,
            max_distance=max_distance,
        )
    return torch.cat((state, lidar), dim=-1)


def _draw_planar_lidar_debug_lines(
    env: ManagerBasedRLEnv,
    command_name: str = "pose_command",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    max_distance: float = 4.0,
) -> None:
    """Draw the selected viewer environment's planar scan and navigation goal."""
    if not env.sim.has_gui():
        return

    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    ray_starts_w = getattr(sensor, "_ray_starts_w", None)
    ray_directions_w = getattr(sensor, "_ray_directions_w", None)
    if ray_starts_w is None or ray_directions_w is None:
        return

    env_index = int(env.cfg.viewer.env_index)
    if env_index < 0 or env_index >= sensor.data.ray_hits_w.shape[0]:
        return

    try:
        import isaacsim.util.debug_draw._debug_draw as omni_debug_draw

        if not hasattr(_draw_planar_lidar_debug_lines, "_draw_interface"):
            _draw_planar_lidar_debug_lines._draw_interface = (
                omni_debug_draw.acquire_debug_draw_interface()
            )
        draw_interface = _draw_planar_lidar_debug_lines._draw_interface

        starts = ray_starts_w[env_index]
        hits = sensor.data.ray_hits_w[env_index]
        directions = ray_directions_w[env_index]
        directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True).clamp_min(1.0e-6)
        hit_distances = torch.linalg.norm(hits - starts, dim=-1)
        valid_hits = (
            torch.isfinite(hits).all(dim=-1)
            & (hit_distances > 1.0e-6)
            & (hit_distances <= max_distance + 1.0e-4)
        )
        miss_ends = starts + directions * max_distance
        ends = torch.where(valid_hits.unsqueeze(-1), hits, miss_ends)

        proximity = (1.0 - hit_distances / max(max_distance, 1.0e-6)).clamp(0.0, 1.0)
        valid_hits_cpu = valid_hits.detach().cpu().tolist()
        proximity_cpu = proximity.detach().cpu().tolist()
        colors = []
        thicknesses = []
        for is_hit, proximity_value in zip(valid_hits_cpu, proximity_cpu):
            if is_hit:
                colors.append([proximity_value, 1.0 - proximity_value, 0.1, 1.0])
                thicknesses.append(1.8)
            else:
                colors.append([0.1, 0.55, 1.0, 0.35])
                thicknesses.append(0.8)

        command_term = env.command_manager.get_term(command_name)
        goal_w = command_term.pos_command_w[env_index].clone()

        # Keep the goal beacon and LiDAR scan in one draw call because
        # clear_lines() clears every line owned by this debug-draw interface.
        beacon_start = goal_w.clone()
        beacon_start[2] = env.scene.env_origins[env_index, 2] + 0.05
        beacon_end = goal_w.clone()
        beacon_end[2] += 1.5
        starts = torch.cat((starts, beacon_start[None]), dim=0)
        ends = torch.cat((ends, beacon_end[None]), dim=0)
        colors.append([0.1, 1.0, 0.2, 1.0])
        thicknesses.append(4.0)

        draw_interface.clear_lines()
        draw_interface.draw_lines(
            starts.detach().cpu().tolist(),
            ends.detach().cpu().tolist(),
            colors,
            thicknesses,
        )
        draw_interface.clear_points()
        draw_interface.draw_points(
            [goal_w.detach().cpu().tolist()],
            [[0.1, 1.0, 0.2, 1.0]],
            [25.0],
        )
    except Exception as error:
        if not hasattr(_draw_planar_lidar_debug_lines, "_warned"):
            print(f"[UAV LiDAR Debug Vis] Failed: {error}")
            _draw_planar_lidar_debug_lines._warned = True


def planar_lidar_collision(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    body_radius: float = 0.45,
    max_distance: float = 4.0,
    k_nearest: int = 2,
) -> torch.Tensor:
    """Detect a near collision from the mean of the closest horizontal rays."""
    distances = planar_lidar_distances(env, sensor_cfg, max_distance)
    nearest = torch.topk(
        distances,
        k=min(k_nearest, distances.shape[1]),
        dim=1,
        largest=False,
    ).values
    return nearest.mean(dim=1) < body_radius


def uav_contact_collision(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    threshold: float = 5.0,
) -> torch.Tensor:
    """Detect contact on any UAV or gripper rigid body."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_norm = torch.linalg.norm(sensor.data.net_forces_w_history, dim=-1)
    return force_norm.amax(dim=(1, 2)) > threshold


def altitude_out_of_bounds(
    env: ManagerBasedRLEnv,
    action_term_name: str = "uav_velocity",
    max_error: float = 0.5,
) -> torch.Tensor:
    """Terminate when the low-level altitude hold departs too far from its target."""
    action_term = env.action_manager.get_term(action_term_name)
    target_altitude = action_term.target_altitude
    robot: Articulation = env.scene["robot"]
    return torch.abs(robot.data.root_pos_w[:, 2] - target_altitude) > max_error


def excessive_tilt(
    env: ManagerBasedRLEnv,
    max_tilt: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when roll/pitch tilt exceeds the controller's normal envelope."""
    asset: Articulation = env.scene[asset_cfg.name]
    upright_alignment = -asset.data.projected_gravity_b[:, 2]
    return upright_alignment < math.cos(max_tilt)


def planar_out_of_bounds(
    env: ManagerBasedRLEnv,
    half_size: float = 25.0,
    distance_buffer: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Keep each UAV inside its assigned static-scene cell."""
    asset: Articulation = env.scene[asset_cfg.name]
    local_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return torch.any(torch.abs(local_xy) > half_size - distance_buffer, dim=1)


def setup_uav_static_usd_scene(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    scene_root: str = "/World/uav_navigation_scene",
    cell_size: float = 50.0,
    num_rows: int = 8,
    num_cols: int = 8,
) -> None:
    """Enable static-scene collisions and map environments onto its cell grid."""
    del env_ids
    if cell_size <= 0.0 or num_rows <= 0 or num_cols <= 0:
        raise ValueError("Static USD grid dimensions must be positive.")

    env_index = torch.arange(env.num_envs, device=env.device)
    cell_index = env_index % (num_rows * num_cols)
    row = torch.div(cell_index, num_cols, rounding_mode="floor")
    col = cell_index % num_cols
    origins = torch.zeros(env.num_envs, 3, device=env.device)
    origins[:, 0] = (col.to(torch.float32) + 0.5) * cell_size
    origins[:, 1] = (row.to(torch.float32) + 0.5) * cell_size
    env.scene._default_env_origins = origins
    env._uav_navigation_cell_index = cell_index

    import isaaclab.sim as sim_utils
    from isaaclab.sim import schemas
    from isaaclab.utils.mesh import PRIMITIVE_MESH_TYPES
    from pxr import UsdPhysics

    stage = sim_utils.get_current_stage()
    root_prim = stage.GetPrimAtPath(scene_root)
    if not root_prim.IsValid():
        raise RuntimeError(f"Static USD scene root not found: {scene_root}")

    geometry_types = set(PRIMITIVE_MESH_TYPES + ["Mesh"])
    geometry_prims = sim_utils.get_all_matching_child_prims(
        scene_root,
        predicate=lambda prim: prim.GetTypeName() in geometry_types,
        stage=stage,
    )
    if not geometry_prims:
        raise RuntimeError(f"No ray-castable geometry found under {scene_root}.")

    collision_cfg = schemas.CollisionPropertiesCfg(collision_enabled=True)
    mesh_collision_cfg = schemas.ConvexHullPropertiesCfg()
    mesh_count = 0
    for prim in geometry_prims:
        prim_path = prim.GetPath().pathString
        if UsdPhysics.CollisionAPI(prim):
            schemas.modify_collision_properties(prim_path, collision_cfg, stage=stage)
        else:
            schemas.define_collision_properties(prim_path, collision_cfg, stage=stage)
        if prim.GetTypeName() == "Mesh":
            schemas.define_mesh_collision_properties(prim_path, mesh_collision_cfg, stage=stage)
            mesh_count += 1

    print(
        "[INFO] UAV static USD scene: "
        f"{len(geometry_prims)} collision geometries ({mesh_count} meshes), "
        f"grid={num_rows}x{num_cols}, cell_size={cell_size:.1f} m, envs={env.num_envs}."
    )
