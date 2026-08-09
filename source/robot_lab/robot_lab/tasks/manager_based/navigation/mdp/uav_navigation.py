# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""MDP terms for planar navigation with the tilting UAV."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


PLANAR_LIDAR_RAY_COUNT = 72
"""Number of rays in the 360-degree, 5-degree planar scan."""

PLANAR_POLICY_STATE_DIM = 7
"""Deployment-visible state features concatenated before the planar LiDAR scan."""

UAV_PRIVILEGED_STATE_DIM = 95
"""Simulator-only expert features for the fixed 8-by-8 navigation scene."""

UAV_SINGLE_SCENE_PRIVILEGED_STATE_DIM = 29
"""Simulator-only expert features without the 8-by-8 cell encoding."""


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
    expected_rays: int = PLANAR_LIDAR_RAY_COUNT,
) -> torch.Tensor:
    """Return normalized proximity values with beam zero pointing forward.

    Isaac Lab generates a full scan from -180 degrees through the final sample
    before +180 degrees. The circular roll places the forward ray at index zero.
    A clear ray is zero and a close obstacle approaches one.
    """
    distances = planar_lidar_distances(
        env,
        sensor_cfg,
        max_distance,
        expected_rays=expected_rays,
    )
    distances = torch.roll(distances, shifts=expected_rays // 2, dims=1)
    return 1.0 - distances / max(max_distance, 1.0e-6)


def uav_planar_state(
    env: ManagerBasedRLEnv,
    command_name: str = "pose_command",
    goal_distance_scale: float = 50.0,
    velocity_scale: float = 1.5,
    yaw_rate_scale: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the seven deployment-visible state features beside the LiDAR scan.

    The features are relative goal XY, body-frame velocity XY, sine/cosine of the
    live goal bearing, and body yaw rate. All inputs are available from localization
    and the IMU.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    goal_xy_b = command[:, :2] / max(goal_distance_scale, 1.0e-6)
    velocity_xy_b = asset.data.root_lin_vel_b[:, :2] / max(velocity_scale, 1.0e-6)
    goal_bearing = torch.atan2(command[:, 1], command[:, 0])
    bearing_features = torch.stack((torch.sin(goal_bearing), torch.cos(goal_bearing)), dim=-1)
    yaw_rate = asset.data.root_ang_vel_b[:, 2:3] / max(yaw_rate_scale, 1.0e-6)
    return torch.cat((goal_xy_b, velocity_xy_b, bearing_features, yaw_rate), dim=-1)


def uav_planar_policy_observation(
    env: ManagerBasedRLEnv,
    command_name: str = "pose_command",
    goal_distance_scale: float = 50.0,
    velocity_scale: float = 1.5,
    yaw_rate_scale: float = 1.0,
    max_distance: float = 4.0,
    expected_rays: int = PLANAR_LIDAR_RAY_COUNT,
    debug_draw: bool = True,
    debug_draw_lidar: bool = True,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
) -> torch.Tensor:
    """Concatenate seven navigation features and the configured planar scan."""
    state = uav_planar_state(
        env,
        command_name=command_name,
        goal_distance_scale=goal_distance_scale,
        velocity_scale=velocity_scale,
        yaw_rate_scale=yaw_rate_scale,
        asset_cfg=asset_cfg,
    )
    lidar = uav_planar_lidar_proximity(
        env,
        sensor_cfg=sensor_cfg,
        max_distance=max_distance,
        expected_rays=expected_rays,
    )
    if debug_draw:
        _draw_planar_lidar_debug_lines(
            env,
            command_name=command_name,
            sensor_cfg=sensor_cfg,
            max_distance=max_distance,
            draw_lidar_rays=debug_draw_lidar,
        )
    return torch.cat((state, lidar), dim=-1)


def uav_navigation_privileged_state(
    env: ManagerBasedRLEnv,
    command_name: str = "pose_command",
    action_term_name: str = "uav_velocity",
    cell_size: float = 50.0,
    num_rows: int = 8,
    num_cols: int = 8,
    include_scene_identity: bool = True,
    velocity_scale: float = 1.5,
    altitude_error_scale: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return simulator-only state for the navigation data-collection expert.

    The actor and critic receive full rigid-body motion, attitude, controller tracking
    error, and actuator state. Grid tasks additionally receive exact map-cell identity
    and coordinates. This intentionally privileged policy is trained only to collect
    successful simulation trajectories.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    action_term = env.action_manager.get_term(action_term_name)

    half_cell = max(0.5 * cell_size, 1.0e-6)
    local_position_xy = (
        asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    ) / half_cell
    goal_local_xy = (
        command_term.pos_command_w[:, :2] - env.scene.env_origins[:, :2]
    ) / half_cell
    altitude_error = (
        (asset.data.root_pos_w[:, 2] - action_term.target_altitude)
        / max(altitude_error_scale, 1.0e-6)
    ).unsqueeze(-1)

    heading_w = asset.data.heading_w
    heading_features = torch.stack((torch.sin(heading_w), torch.cos(heading_w)), dim=-1)
    root_lin_vel_b = asset.data.root_lin_vel_b.clone()
    root_lin_vel_b[:, :2] /= max(velocity_scale, 1.0e-6)
    root_lin_vel_b[:, 2] /= max(action_term.cfg.altitude_hold_max_velocity, 1.0e-6)
    root_ang_vel_b = asset.data.root_ang_vel_b / max(action_term.cfg.yaw_rate_scale, 1.0e-6)
    projected_gravity_b = asset.data.projected_gravity_b

    controller_yaw_error = wrap_to_pi(action_term.target_yaw - heading_w)
    controller_yaw_features = torch.stack(
        (torch.sin(controller_yaw_error), torch.cos(controller_yaw_error)), dim=-1
    )

    velocity_integral = action_term.velocity_integral_error / max(
        action_term.cfg.vel_integral_limit, 1.0e-6
    )
    thrust_state = action_term.thrust_actual / max(action_term.cfg.thrust_max, 1.0e-6)
    servo_state = action_term.servo_angle_actual / max(
        action_term.cfg.servo_angle_limit, 1.0e-6
    )

    state_parts = [
        local_position_xy,
        goal_local_xy,
        altitude_error,
        heading_features,
        root_lin_vel_b,
        root_ang_vel_b,
        projected_gravity_b,
        controller_yaw_features,
    ]
    if include_scene_identity:
        cell_index = getattr(env, "_uav_navigation_cell_index", None)
        if cell_index is None:
            cell_index = torch.arange(env.num_envs, device=env.device) % (num_rows * num_cols)
        row = torch.div(cell_index, num_cols, rounding_mode="floor").to(torch.float32)
        col = (cell_index % num_cols).to(torch.float32)
        row = 2.0 * row / max(num_rows - 1, 1) - 1.0
        col = 2.0 * col / max(num_cols - 1, 1) - 1.0
        state_parts.extend(
            (
                torch.stack((row, col), dim=-1),
                F.one_hot(cell_index.to(torch.long), num_classes=num_rows * num_cols).to(
                    asset.data.root_pos_w.dtype
                ),
            )
        )
    state_parts.extend((velocity_integral, thrust_state, servo_state))
    return torch.cat(tuple(state_parts), dim=-1)


def _goal_pose_errors(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return planar distance and absolute terminal-heading error."""
    asset: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    goal_delta = command_term.pos_command_w[:, :2] - asset.data.root_pos_w[:, :2]
    distance = torch.linalg.norm(goal_delta, dim=1)
    live_goal_heading = torch.atan2(goal_delta[:, 1], goal_delta[:, 0])
    live_goal_heading = torch.where(
        distance > 1.0e-6,
        live_goal_heading,
        command_term.heading_command_w,
    )
    heading_error = torch.abs(wrap_to_pi(live_goal_heading - asset.data.heading_w))
    return distance, heading_error


def goal_reached_with_heading(
    env: ManagerBasedRLEnv,
    command_name: str = "pose_command",
    distance_threshold: float = 1.0,
    heading_threshold: float = math.radians(10.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Succeed only when both planar position and terminal yaw are within tolerance."""
    distance, heading_error = _goal_pose_errors(env, command_name, asset_cfg)
    return (distance < distance_threshold) & (heading_error < heading_threshold)


def goal_pose_bonus(
    env: ManagerBasedRLEnv,
    command_name: str = "pose_command",
    distance_threshold: float = 1.0,
    heading_threshold: float = math.radians(10.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return a sparse bonus using the same pose condition as success termination."""
    return goal_reached_with_heading(
        env,
        command_name=command_name,
        distance_threshold=distance_threshold,
        heading_threshold=heading_threshold,
        asset_cfg=asset_cfg,
    ).float()


def near_goal_heading_error(
    env: ManagerBasedRLEnv,
    command_name: str = "pose_command",
    distance_scale: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return normalized yaw error, smoothly gated to the terminal approach region."""
    distance, heading_error = _goal_pose_errors(env, command_name, asset_cfg)
    proximity = torch.exp(-torch.square(distance / max(distance_scale, 1.0e-6)))
    return proximity * heading_error / math.pi


def _draw_planar_lidar_debug_lines(
    env: ManagerBasedRLEnv,
    command_name: str = "pose_command",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    max_distance: float = 4.0,
    draw_lidar_rays: bool = True,
) -> None:
    """Draw the navigation goal and optionally the selected environment's planar scan."""
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

        colors = []
        thicknesses = []
        if draw_lidar_rays:
            starts = ray_starts_w[env_index]
            hits = sensor.data.ray_hits_w[env_index]
            directions = ray_directions_w[env_index]
            directions = directions / torch.linalg.norm(
                directions, dim=-1, keepdim=True
            ).clamp_min(1.0e-6)
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
            for is_hit, proximity_value in zip(valid_hits_cpu, proximity_cpu):
                if is_hit:
                    colors.append([proximity_value, 1.0 - proximity_value, 0.1, 1.0])
                    thicknesses.append(1.8)
                else:
                    colors.append([0.1, 0.55, 1.0, 0.35])
                    thicknesses.append(0.8)
        else:
            starts = ray_starts_w.new_empty((0, 3))
            ends = ray_starts_w.new_empty((0, 3))

        command_term = env.command_manager.get_term(command_name)
        goal_w = command_term.pos_command_w[env_index].clone()

        # Keep the goal beacon and optional LiDAR scan in one draw call because
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
    expected_rays: int = PLANAR_LIDAR_RAY_COUNT,
) -> torch.Tensor:
    """Detect a near collision from the mean of the closest horizontal rays."""
    distances = planar_lidar_distances(
        env,
        sensor_cfg,
        max_distance,
        expected_rays=expected_rays,
    )
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


def set_uav_navigation_cells(
    env: ManagerBasedEnv,
    cell_indices: int | Sequence[int] | torch.Tensor | None = None,
    cell_size: float = 50.0,
    num_rows: int = 8,
    num_cols: int = 8,
    grid_origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Map each environment origin to a cell in the shared static USD grid.

    Passing ``None`` distributes environments cyclically over the full grid. A
    scalar maps every environment to the same cell, which is used for policy
    evaluation on one selected obstacle layout.
    """
    if cell_size <= 0.0 or num_rows <= 0 or num_cols <= 0:
        raise ValueError("Static USD grid dimensions must be positive.")

    num_cells = num_rows * num_cols
    if cell_indices is None:
        indices = torch.arange(env.num_envs, device=env.device, dtype=torch.long) % num_cells
    else:
        indices = torch.as_tensor(cell_indices, device=env.device, dtype=torch.long)
        if indices.ndim == 0:
            indices = indices.expand(env.num_envs).clone()
        else:
            indices = indices.flatten()
            if indices.numel() != env.num_envs:
                raise ValueError(
                    f"Expected one cell index or {env.num_envs} indices, got {indices.numel()}."
                )

    if torch.any((indices < 0) | (indices >= num_cells)):
        minimum = int(indices.min().item())
        maximum = int(indices.max().item())
        raise ValueError(
            f"Cell indices must be in [0, {num_cells - 1}], got range [{minimum}, {maximum}]."
        )

    row = torch.div(indices, num_cols, rounding_mode="floor")
    col = indices % num_cols
    origin = torch.as_tensor(grid_origin, device=env.device, dtype=torch.float32).flatten()
    if origin.numel() != 3:
        raise ValueError(f"grid_origin must contain three values, got {origin.numel()}.")
    origins = origin.expand(env.num_envs, 3).clone()
    origins[:, 0] += (col.to(torch.float32) + 0.5) * cell_size
    origins[:, 1] += (row.to(torch.float32) + 0.5) * cell_size
    env.scene._default_env_origins = origins
    env._uav_navigation_cell_index = indices
    return indices


def set_uav_navigation_scene_origin(
    env: ManagerBasedEnv,
    scene_origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Place every parallel environment at one fixed shared-scene origin."""
    origin = torch.as_tensor(scene_origin, device=env.device, dtype=torch.float32).flatten()
    if origin.numel() != 3:
        raise ValueError(f"scene_origin must contain three values, got {origin.numel()}.")
    env.scene._default_env_origins = origin.expand(env.num_envs, 3).clone()
    indices = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    env._uav_navigation_cell_index = indices
    return indices


def _validate_uav_static_usd_collisions(
    env: ManagerBasedEnv,
    scene_root: str,
) -> tuple[int, int, int]:
    """Validate authored colliders without modifying the composed USD stage."""
    import isaaclab.sim as sim_utils
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

    authored_geometry_prims = [prim for prim in geometry_prims if not prim.IsInstanceProxy()]
    instance_proxy_count = len(geometry_prims) - len(authored_geometry_prims)
    if not authored_geometry_prims:
        raise RuntimeError(
            f"No authored geometry available for collision validation under {scene_root}; "
            f"found only {instance_proxy_count} instance-proxy geometries."
        )

    missing_collision_paths = []
    disabled_collision_paths = []
    for prim in authored_geometry_prims:
        prim_path = prim.GetPath().pathString
        collision_api = UsdPhysics.CollisionAPI(prim)
        if not collision_api:
            missing_collision_paths.append(prim_path)
        elif collision_api.GetCollisionEnabledAttr().Get() is False:
            disabled_collision_paths.append(prim_path)

    if missing_collision_paths or disabled_collision_paths:
        details = []
        if missing_collision_paths:
            details.append(
                f"missing CollisionAPI ({len(missing_collision_paths)}): "
                + ", ".join(missing_collision_paths[:10])
            )
        if disabled_collision_paths:
            details.append(
                f"collision disabled ({len(disabled_collision_paths)}): "
                + ", ".join(disabled_collision_paths[:10])
            )
        raise RuntimeError(
            f"Static USD scene collision validation failed under {scene_root}; "
            + "; ".join(details)
            + ". Author enabled colliders in the source USD before launching the environment."
        )

    mesh_count = sum(prim.GetTypeName() == "Mesh" for prim in authored_geometry_prims)
    return len(authored_geometry_prims), mesh_count, instance_proxy_count


def setup_uav_static_usd_scene(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    scene_root: str = "/World/uav_navigation_scene",
    cell_size: float = 50.0,
    num_rows: int = 8,
    num_cols: int = 8,
    grid_origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    cell_indices: int | Sequence[int] | torch.Tensor | None = None,
) -> None:
    """Validate authored static-scene collisions and map environments onto its cell grid."""
    del env_ids
    set_uav_navigation_cells(
        env,
        cell_indices=cell_indices,
        cell_size=cell_size,
        num_rows=num_rows,
        num_cols=num_cols,
        grid_origin=grid_origin,
    )
    geometry_count, mesh_count, instance_proxy_count = _validate_uav_static_usd_collisions(
        env, scene_root
    )

    print(
        "[INFO] UAV static USD scene: "
        f"validated {geometry_count} authored collision geometries ({mesh_count} meshes), "
        f"skipped {instance_proxy_count} read-only instance-proxy visual geometries, "
        f"grid={num_rows}x{num_cols}, cell_size={cell_size:.1f} m, envs={env.num_envs}."
    )


def setup_uav_single_static_usd_scene(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    scene_root: str = "/World/uav_navigation_scene",
    scene_origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    scene_size: float = 100.0,
) -> None:
    """Validate one shared static scene and give every environment the same origin."""
    del env_ids
    if scene_size <= 0.0:
        raise ValueError("Single static USD scene size must be positive.")
    set_uav_navigation_scene_origin(env, scene_origin=scene_origin)
    geometry_count, mesh_count, instance_proxy_count = _validate_uav_static_usd_collisions(
        env, scene_root
    )
    print(
        "[INFO] UAV single static USD scene: "
        f"validated {geometry_count} authored collision geometries ({mesh_count} meshes), "
        f"skipped {instance_proxy_count} read-only instance-proxy visual geometries, "
        f"origin={scene_origin}, size={scene_size:.1f} m, envs={env.num_envs}."
    )
