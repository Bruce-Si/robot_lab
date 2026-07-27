# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Static-USD planar navigation environment for the tilting UAV and gripper."""

from __future__ import annotations

import math
import os
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

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR
from robot_lab.assets.tilting_uav import TILTING_UAV_CFG
from robot_lab.tasks.manager_based.navigation.mdp.commands import NonStartEdgePose2dCommand
from robot_lab.tasks.manager_based.navigation.mdp.events import reset_root_state_uniform_navigation
from robot_lab.tasks.manager_based.navigation.mdp import rewards as navigation_rewards
from robot_lab.tasks.manager_based.navigation.mdp import terminations as navigation_terminations
from robot_lab.tasks.manager_based.navigation.mdp.uav_navigation import (
    altitude_out_of_bounds,
    excessive_tilt,
    planar_lidar_collision,
    planar_out_of_bounds,
    setup_uav_static_usd_scene,
    uav_contact_collision,
    uav_planar_policy_observation,
)
from robot_lab.tasks.manager_based.navigation.mdp.uav_velocity_action import (
    TiltingUAVVelocityActionCfg,
)


_SCENE_ROOT = "/World/uav_navigation_scene"
_CELL_SIZE = 50.0
_SCENE_ROWS = 8
_SCENE_COLS = 8
_LIDAR_RANGE = 4.0


def _resolve_scene_path() -> Path:
    default_path = (
        Path(ISAACLAB_ASSETS_DATA_DIR)
        / "environments"
        / "loco_navi_curriculum_flat_v1_baked_round_prims.usd"
    )
    configured_path = (
        os.environ.get("NAVRL_UAV_USD_SCENE")
        or os.environ.get("NAVRL_USD_SCENE")
        or str(default_path)
    )
    scene_path = Path(configured_path).expanduser().resolve()
    if not scene_path.is_file():
        raise FileNotFoundError(
            f"UAV navigation scene not found at {scene_path}. "
            "Set NAVRL_UAV_USD_SCENE or pass --load_scene."
        )
    return scene_path


UAV_NAVIGATION_SCENE_PATH = _resolve_scene_path()

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
            horizontal_res=10.0,
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
        spawn=sim_utils.DomeLightCfg(color=(0.82, 0.85, 0.9), intensity=900.0),
    )
    sun_light = AssetBaseCfg(
        prim_path="/World/sunLight",
        spawn=sim_utils.DistantLightCfg(color=(1.0, 0.95, 0.88), intensity=1800.0),
    )


@configclass
class ActionsCfg:
    uav_velocity = TiltingUAVVelocityActionCfg(
        asset_name="robot",
        planar_mode=True,
        velocity_scale=(1.0, 1.0, 1.0),
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
                "max_distance": _LIDAR_RANGE,
                "debug_draw": True,
            },
        )

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class CommandsCfg:
    pose_command = UniformPose2dCommandCfg(
        class_type=NonStartEdgePose2dCommand,
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
        func=navigation_rewards.goal_bonus,
        weight=200.0,
        params={"command_name": "pose_command", "distance_threshold": 1.0},
    )
    action_rate = RewTerm(func=navigation_rewards.action_rate_l2, weight=-0.01)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    contact_collision = DoneTerm(func=uav_contact_collision, params={"threshold": 5.0})
    lidar_collision = DoneTerm(
        func=planar_lidar_collision,
        params={"body_radius": 0.45, "max_distance": _LIDAR_RANGE, "k_nearest": 2},
    )
    altitude = DoneTerm(func=altitude_out_of_bounds, params={"max_error": 0.5})
    tilt = DoneTerm(func=excessive_tilt, params={"max_tilt": 0.5})
    out_of_bounds = DoneTerm(
        func=planar_out_of_bounds,
        params={"half_size": _CELL_SIZE / 2.0, "distance_buffer": 0.5},
    )
    goal_reached = DoneTerm(
        func=navigation_terminations.goal_reached,
        params={"command_name": "pose_command", "distance_threshold": 1.0},
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

    def __post_init__(self):
        self.sim.dt = 0.0025
        self.decimation = 40
        self.sim.render_interval = self.decimation
        self.episode_length_s = 100.0
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


@configclass
class TiltingUAVNavigationEnvCfg_PLAY(TiltingUAVNavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
        self.commands.pose_command.debug_vis = True
        self.scene.lidar.debug_vis = True
