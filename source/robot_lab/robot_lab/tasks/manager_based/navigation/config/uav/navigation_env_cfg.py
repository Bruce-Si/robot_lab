# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Static-USD planar navigation environment for the tilting UAV and gripper."""

from __future__ import annotations

import math
from pathlib import Path

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import LidarPatternCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.navigation.mdp import UniformPose2dCommandCfg

from robot_lab.assets.tilting_uav import TILTING_UAV_CFG
from robot_lab.tasks.manager_based.navigation.mdp.commands import (
    TargetFacingNonStartEdgePose2dCommand,
)
from robot_lab.tasks.manager_based.navigation.mdp.events import reset_root_state_uniform_navigation
from robot_lab.tasks.manager_based.navigation.mdp import rewards as navigation_rewards
from robot_lab.tasks.manager_based.navigation.mdp.uav_navigation import (
    altitude_out_of_bounds,
    excessive_tilt,
    goal_pose_bonus,
    goal_reached_with_heading,
    near_goal_heading_error,
    planar_lidar_collision,
    planar_out_of_bounds,
    setup_uav_single_static_usd_scene,
    setup_uav_static_usd_scene,
    uav_contact_collision,
    uav_navigation_privileged_state,
    uav_planar_policy_observation,
)
from robot_lab.tasks.manager_based.navigation.mdp.uav_velocity_action import (
    TiltingUAVVelocityActionCfg,
)
from robot_lab.tasks.manager_based.navigation.config.uav.scene_profile import (
    GRID_8X8_PROFILE,
    WAREHOUSE_100M_PROFILE,
    UavNavigationSceneProfile,
    resolve_scene_path,
)


_SCENE_ROOT = "/World/uav_navigation_scene"
_CELL_SIZE = GRID_8X8_PROFILE.cell_size
_SCENE_ROWS = GRID_8X8_PROFILE.rows
_SCENE_COLS = GRID_8X8_PROFILE.columns
_LIDAR_RANGE = 4.0
_GOAL_DISTANCE_THRESHOLD = 1.0
_GOAL_HEADING_THRESHOLD = math.radians(10.0)


def _require_scene_path(scene_path: Path, profile_name: str) -> None:
    if not scene_path.is_file():
        raise FileNotFoundError(
            f"UAV navigation scene profile '{profile_name}' not found at {scene_path}. "
            "Set its profile-specific scene environment variable, set NAVRL_UAV_USD_SCENE, "
            "or pass --load_scene."
        )


def configure_tilting_uav_physics_hz(env_cfg, physics_hz: float) -> tuple[float, float, int]:
    """Set UAV physics frequency while preserving the configured policy cadence."""
    requested_hz = float(physics_hz)
    if not math.isfinite(requested_hz) or requested_hz <= 0.0:
        raise ValueError(f"physics_hz must be finite and positive, got {physics_hz!r}.")

    control_dt = float(env_cfg.sim.dt) * int(env_cfg.decimation)
    if control_dt <= 0.0:
        raise ValueError(f"Invalid existing control timestep: {control_dt!r}.")
    control_hz = 1.0 / control_dt
    decimation_float = requested_hz / control_hz
    decimation = int(round(decimation_float))
    if decimation < 1 or abs(decimation_float - decimation) > 1.0e-6:
        raise ValueError(
            f"physics_hz={requested_hz:g} must be an integer multiple of the "
            f"policy rate {control_hz:g} Hz."
        )

    env_cfg.sim.dt = 1.0 / requested_hz
    env_cfg.decimation = decimation
    env_cfg.sim.render_interval = decimation
    env_cfg.scene.lidar.update_period = decimation * env_cfg.sim.dt
    contact_forces = getattr(env_cfg.scene, "contact_forces", None)
    if contact_forces is not None:
        contact_forces.update_period = env_cfg.sim.dt
    if hasattr(env_cfg, "navigation_physics_hz"):
        env_cfg.navigation_physics_hz = requested_hz
    if hasattr(env_cfg, "navigation_control_hz"):
        env_cfg.navigation_control_hz = control_hz
    return requested_hz, control_hz, decimation


UAV_NAVIGATION_SCENE_PATH = resolve_scene_path(GRID_8X8_PROFILE)
UAV_WAREHOUSE_SCENE_PATH = resolve_scene_path(WAREHOUSE_100M_PROFILE)

_UAV_NAVIGATION_CFG = TILTING_UAV_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
_UAV_NAVIGATION_CFG.init_state.pos = (0.0, 0.0, 1.5)


@configclass
class TiltingUAVNavigationSceneCfg(InteractiveSceneCfg):
    """Shared static USD scene with one floating UAV articulation per environment."""

    usd_scene = AssetBaseCfg(
        prim_path=_SCENE_ROOT,
        spawn=sim_utils.UsdFileCfg(usd_path=str(UAV_NAVIGATION_SCENE_PATH)),
        collision_group=-1,
    )
    robot = _UAV_NAVIGATION_CFG
    lidar = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        # Preserve the RSD455's authored location while expressing the planar
        # scan in the UAV body frame instead of the optical camera frame.
        offset=RayCasterCfg.OffsetCfg(pos=(0.13, -0.0115, 0.03)),
        ray_alignment="base",
        pattern_cfg=LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=GRID_8X8_PROFILE.lidar_horizontal_res,
        ),
        max_distance=_LIDAR_RANGE,
        debug_vis=False,
        mesh_prim_paths=[
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr=_SCENE_ROOT,
                is_shared=True,
                merge_prim_meshes=True,
                track_mesh_transforms=False,
            )
        ],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=2,
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/domeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.82, 0.85, 0.9), intensity=300.0),
    )

    # sun_light = AssetBaseCfg(
    #     prim_path="/World/sunLight",
    #     spawn=sim_utils.DistantLightCfg(color=(1.0, 0.95, 0.88), intensity=1800.0),
    # )
    sun_light = AssetBaseCfg(
        prim_path="/World/sunLight",
        spawn=sim_utils.DistantLightCfg(
            color=(1.0, 0.95, 0.88),
            intensity=1800.0,
            angle=1.0,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            # XYZ 欧拉角约为 (0°, 45°, 45°)，四元数顺序是 wxyz
            rot=(0.853553, -0.146447, 0.353553, 0.353553),
        ),
    )


@configclass
class ActionsCfg:
    uav_velocity = TiltingUAVVelocityActionCfg(
        asset_name="robot",
        planar_mode=True,
        velocity_scale=(1.0, 1.0, 1.0),
        yaw_rate_scale=1.0,
        velocity_command_time_constant=0.15,
        altitude_hold_gain=1.5,
        altitude_hold_max_velocity=0.8,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        navigation = ObsTerm(
            func=uav_planar_policy_observation,
            params={
                "command_name": "pose_command",
                "goal_distance_scale": _CELL_SIZE,
                "velocity_scale": 1.0,
                "yaw_rate_scale": 1.0,
                "max_distance": _LIDAR_RANGE,
                "expected_rays": GRID_8X8_PROFILE.lidar_ray_count,
                "debug_draw": True,
            },
        )

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    @configclass
    class PrivilegedCfg(ObsGroup):
        state = ObsTerm(
            func=uav_navigation_privileged_state,
            params={
                "command_name": "pose_command",
                "action_term_name": "uav_velocity",
                "cell_size": _CELL_SIZE,
                "num_rows": _SCENE_ROWS,
                "num_cols": _SCENE_COLS,
                "velocity_scale": 1.0,
            },
        )

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class CommandsCfg:
    pose_command = UniformPose2dCommandCfg(
        class_type=TargetFacingNonStartEdgePose2dCommand,
        asset_name="robot",
        simple_heading=True,
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        ranges=UniformPose2dCommandCfg.Ranges(
            pos_x=(18.0, 20.0),
            pos_y=(-16.0, 16.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class RewardsCfg:
    termination_penalty = RewTerm(func=base_mdp.is_terminated, weight=-200.0)
    velocity_towards_goal = RewTerm(
        func=navigation_rewards.vel_towards_goal,
        weight=0.5,
        params={"command_name": "pose_command"},
    )
    progress_towards_goal = RewTerm(
        func=navigation_rewards.progress_towards_goal,
        weight=0.5,
        params={"command_name": "pose_command"},
    )
    lidar_obstacle = RewTerm(
        func=navigation_rewards.lidar_obstacle_penalty,
        weight=2.0,
        params={
            "max_distance": _LIDAR_RANGE,
            "k_nearest": 3,
            "soft_threshold": 1.8,
            "hard_threshold": 0.8,
        },
    )
    goal_bonus = RewTerm(
        func=goal_pose_bonus,
        weight=200.0,
        params={
            "command_name": "pose_command",
            "distance_threshold": _GOAL_DISTANCE_THRESHOLD,
            "heading_threshold": _GOAL_HEADING_THRESHOLD,
        },
    )
    terminal_heading = RewTerm(
        func=near_goal_heading_error,
        weight=-2.0,
        params={"command_name": "pose_command", "distance_scale": 2.0},
    )
    action_rate = RewTerm(func=navigation_rewards.action_rate_l2, weight=-0.01)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    contact_collision = DoneTerm(func=uav_contact_collision, params={"threshold": 5.0})
    lidar_collision = DoneTerm(
        func=planar_lidar_collision,
        params={
            "body_radius": 0.45,
            "max_distance": _LIDAR_RANGE,
            "k_nearest": 2,
            "expected_rays": GRID_8X8_PROFILE.lidar_ray_count,
        },
    )
    altitude = DoneTerm(func=altitude_out_of_bounds, params={"max_error": 0.5})
    tilt = DoneTerm(func=excessive_tilt, params={"max_tilt": 0.5})
    out_of_bounds = DoneTerm(
        func=planar_out_of_bounds,
        params={"half_size": _CELL_SIZE / 2.0, "distance_buffer": 0.5},
    )
    goal_reached = DoneTerm(
        func=goal_reached_with_heading,
        params={
            "command_name": "pose_command",
            "distance_threshold": _GOAL_DISTANCE_THRESHOLD,
            "heading_threshold": _GOAL_HEADING_THRESHOLD,
        },
        time_out=True,
    )


@configclass
class EventCfg:
    setup_static_scene = EventTerm(
        func=setup_uav_static_usd_scene,
        mode="prestartup",
        params={
            "scene_root": _SCENE_ROOT,
            "cell_size": _CELL_SIZE,
            "num_rows": _SCENE_ROWS,
            "num_cols": _SCENE_COLS,
        },
    )
    reset_joints = EventTerm(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_base = EventTerm(
        func=reset_root_state_uniform_navigation,
        mode="reset",
        params={
            "pose_range": {
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-0.5, 0.5),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )


@configclass
class TiltingUAVNavigationEnvCfg(ManagerBasedRLEnvCfg):
    scene: TiltingUAVNavigationSceneCfg = TiltingUAVNavigationSceneCfg(
        num_envs=_SCENE_ROWS * _SCENE_COLS,
        env_spacing=0.0,
        replicate_physics=False,
    )
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    navigation_scene_profile: str = GRID_8X8_PROFILE.name
    navigation_scene_layout: str = GRID_8X8_PROFILE.layout
    navigation_scene_rows: int = GRID_8X8_PROFILE.rows
    navigation_scene_columns: int = GRID_8X8_PROFILE.columns
    navigation_cell_size: float = GRID_8X8_PROFILE.cell_size
    navigation_scene_origin: tuple[float, float, float] = GRID_8X8_PROFILE.scene_origin
    navigation_bounds_half_size: float = GRID_8X8_PROFILE.bounds_half_size
    navigation_edge_offset: float = GRID_8X8_PROFILE.edge_offset
    navigation_lateral_range: tuple[float, float] = GRID_8X8_PROFILE.lateral_range
    navigation_evaluation_mode: str = GRID_8X8_PROFILE.evaluation_mode
    navigation_curriculum_enabled: bool = GRID_8X8_PROFILE.curriculum
    navigation_physics_hz: float = 400.0
    navigation_control_hz: float = 10.0

    def __post_init__(self):
        self._configure_common_runtime()
        self._apply_scene_profile(GRID_8X8_PROFILE, UAV_NAVIGATION_SCENE_PATH)

    def _configure_common_runtime(self) -> None:
        # Run flight physics at 400 Hz while preserving the 10 Hz policy and LiDAR period.
        self.sim.dt = 0.0025
        self.decimation = 40
        self.sim.render_interval = self.decimation
        self.scene.lidar.update_period = self.decimation * self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0
        self.viewer.eye = (-10.0, -10.0, 12.0)
        self.viewer.lookat = (0.0, 0.0, 1.5)

        # PhysX buffers for 1024 articulated UAVs sharing the obstacle-rich USD scene.
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**28
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**24
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**24
        self.sim.physx.gpu_max_rigid_contact_count = 2**24
        self.sim.physx.gpu_max_rigid_patch_count = 2**23
        self.sim.physx.gpu_collision_stack_size = 2**28
        self.sim.physx.gpu_heap_capacity = 2**28
        self.sim.physx.gpu_temp_buffer_capacity = 2**26

    def _apply_scene_profile(
        self,
        profile: UavNavigationSceneProfile,
        scene_path: Path,
    ) -> None:
        _require_scene_path(scene_path, profile.name)
        self.navigation_scene_profile = profile.name
        self.navigation_scene_layout = profile.layout
        self.navigation_scene_rows = profile.rows
        self.navigation_scene_columns = profile.columns
        self.navigation_cell_size = profile.cell_size
        self.navigation_scene_origin = profile.scene_origin
        self.navigation_bounds_half_size = profile.bounds_half_size
        self.navigation_edge_offset = profile.edge_offset
        self.navigation_lateral_range = profile.lateral_range
        self.navigation_evaluation_mode = profile.evaluation_mode
        self.navigation_curriculum_enabled = profile.curriculum

        self.scene.num_envs = profile.default_num_envs
        self.scene.usd_scene.spawn.usd_path = str(scene_path)
        self.scene.lidar.pattern_cfg.horizontal_res = profile.lidar_horizontal_res
        self.episode_length_s = profile.episode_length_s
        self.observations.policy.navigation.params["goal_distance_scale"] = (
            profile.goal_distance_scale
        )
        self.observations.policy.navigation.params["expected_rays"] = profile.lidar_ray_count
        self.observations.privileged.state.params.update(
            {
                "cell_size": profile.cell_size,
                "num_rows": profile.rows,
                "num_cols": profile.columns,
                "include_scene_identity": profile.use_scene_identity,
            }
        )
        self.terminations.out_of_bounds.params["half_size"] = profile.bounds_half_size
        self.terminations.lidar_collision.params["expected_rays"] = profile.lidar_ray_count

        if profile.layout == "grid":
            self.events.setup_static_scene.func = setup_uav_static_usd_scene
            self.events.setup_static_scene.params = {
                "scene_root": _SCENE_ROOT,
                "cell_size": profile.cell_size,
                "num_rows": profile.rows,
                "num_cols": profile.columns,
                "grid_origin": profile.scene_origin,
            }
        else:
            self.events.setup_static_scene.func = setup_uav_single_static_usd_scene
            self.events.setup_static_scene.params = {
                "scene_root": _SCENE_ROOT,
                "scene_origin": profile.scene_origin,
                "scene_size": profile.cell_size,
            }


@configclass
class TiltingUAVWarehouseNavigationEnvCfg(TiltingUAVNavigationEnvCfg):
    """Single 100 m warehouse scene with every UAV sharing the world origin."""

    def __post_init__(self):
        self._configure_common_runtime()
        self._apply_scene_profile(WAREHOUSE_100M_PROFILE, UAV_WAREHOUSE_SCENE_PATH)
        self.viewer.eye = (-35.0, -35.0, 30.0)
        self.viewer.lookat = (0.0, 0.0, 1.5)


@configclass
class TiltingUAVNavigationEnvCfg_PLAY(TiltingUAVNavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
        self.commands.pose_command.debug_vis = True
        self.scene.lidar.debug_vis = True


@configclass
class TiltingUAVWarehouseNavigationEnvCfg_PLAY(TiltingUAVWarehouseNavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
        self.commands.pose_command.debug_vis = True
        self.scene.lidar.debug_vis = True
