# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Navigation environment configuration for Go2 with NavRL."""

import math

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import LidarPatternCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshRepeatedBoxesTerrainCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.markers import VisualizationMarkersCfg

import isaaclab.sim as sim_utils

from isaaclab_tasks.manager_based.navigation.mdp import UniformPose2dCommandCfg

# Large green sphere at goal position
_GOAL_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Command/goal_pose",
    markers={
        "goal_sphere": sim_utils.SphereCfg(
            radius=0.1,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
    },
)

from robot_lab.assets.unitree import UNITREE_GO2_CFG
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.unitree_go2.flat_env_cfg import (
    UnitreeGo2FlatEnvCfg,
)
import robot_lab.tasks.manager_based.navigation.mdp as mdp

# Low-level locomotion environment config for PreTrainedPolicyAction
LOW_LEVEL_ENV_CFG = UnitreeGo2FlatEnvCfg()

# Path to exported JIT locomotion policy
import os as _os
_POLICY_DIR = _os.path.dirname(_os.path.abspath(__file__))
# Walk up 8 levels from go2/ to reach project root, then into logs/
LOCOMOTION_POLICY_PATH = _os.path.join(
    _POLICY_DIR, *([".."] * 8),
    "logs", "rsl_rl", "unitree_go2_rough", "2026-05-29_11-11-29", "exported", "policy.pt",
)

# ---------------------------------------------------------------------
# Terrain: per-env obstacle cells with curriculum difficulty
#   - 64 x 32 = 2048 cells, each 35m x 35m (usable ~25m after borders)
#   - Easy cells: fewer, shorter, thinner obstacles
#   - Hard cells: more, taller, thicker obstacles
# ---------------------------------------------------------------------
# 8x8 = 64 unique obstacle layouts, envs cycle through them
_TERRAIN_COLS = 8
_TERRAIN_ROWS = 8
_CELL_SIZE = 50.0  # meters per cell (total terrain 400m x 400m)

OBSTACLE_BOXES_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(_CELL_SIZE, _CELL_SIZE),
    border_width=10.0,   # gap between cells
    border_height=0.5,    # raised ridge to visually separate cells
    num_rows=_TERRAIN_ROWS,
    num_cols=_TERRAIN_COLS,
    horizontal_scale=0.1,
    vertical_scale=0.1,
    slope_threshold=0.75,
    use_cache=False,
    color_scheme="height",
    sub_terrains={
        "mixed": MeshRepeatedBoxesTerrainCfg(
            proportion=1.0,
            function=mdp.mixed_objects_terrain,
            platform_width=2.0,
            object_params_start=MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=15,
                height=2.25,
                size=(0.1, 0.1),
            ),
            object_params_end=MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=90,
                height=2.25,
                size=(2.0, 2.0),
            ),
            abs_height_noise=(-1.75, 1.75),
        ),
    },
)


# ---------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------
@configclass
class NavigationSceneCfg(InteractiveSceneCfg):
    """Scene with flat terrain + box obstacles, Go2 robot, LiDAR."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=OBSTACLE_BOXES_CFG,
        max_init_terrain_level=3,  # start with easier obstacle layouts
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=1.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.5), metallic=0.1),
        debug_vis=False,
    )

    robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # LiDAR: 72 horizontal x 8 vertical beams, 4m range
    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.3)),
        ray_alignment="yaw",
        pattern_cfg=LidarPatternCfg(
            channels=8,
            vertical_fov_range=(-10.0, 20.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=5.0,
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # USD scene as overlay (set via _USD_SCENE_PATH)
    usd_scene: AssetBaseCfg | None = None


# ---------------------------------------------------------------------
# Actions: NavRL → frozen locomotion
# ---------------------------------------------------------------------
@configclass
class ActionsCfg:
    pre_trained_policy_action = mdp.HeadingLockedActionCfg(
        asset_name="robot",
        policy_path=LOCOMOTION_POLICY_PATH,
        low_level_decimation=4,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
    )


# ---------------------------------------------------------------------
# Observations: 1D state + 2D LiDAR
# ---------------------------------------------------------------------
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        state = ObsTerm(func=mdp.navrl_state, params={"command_name": "pose_command"})

        def __post_init__(self):
            self.concatenate_terms = True

    @configclass
    class LidarCfg(ObsGroup):
        lidar = ObsTerm(func=mdp.lidar_depth, params={"max_distance": 4.0})

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    lidar: LidarCfg = LidarCfg()


# ---------------------------------------------------------------------
# Commands: random 2D goal on a cell edge different from the start edge
# ---------------------------------------------------------------------


@configclass
class CommandsCfg:
    pose_command = UniformPose2dCommandCfg(
        class_type=mdp.NonStartEdgePose2dCommand,
        asset_name="robot",
        simple_heading=True,
        resampling_time_range=(1e9, 1e9),  # only resample at episode reset
        debug_vis=True,
        goal_pose_visualizer_cfg=_GOAL_MARKER_CFG,
        ranges=UniformPose2dCommandCfg.Ranges(
            pos_x=(18.0, 20.0),
            pos_y=(-16.0, 16.0),
            heading=(-math.pi, math.pi),
        ),
    )


# ---------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------
@configclass
class RewardsCfg:
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    vel_towards_goal = RewTerm(
        func=mdp.vel_towards_goal,
        weight=0.5,
        params={"command_name": "pose_command"},
    )
    progress_towards_goal = RewTerm(
        func=mdp.progress_towards_goal,
        weight=0.5,
        params={"command_name": "pose_command"},
    )
    lidar_obstacle = RewTerm(
        func=mdp.lidar_obstacle_penalty,
        weight=2.0,
        params={"max_distance": 4.0, "k_nearest": 5, "soft_threshold": 1.8, "hard_threshold": 0.8},
    )
    goal_bonus = RewTerm(
        func=mdp.goal_bonus,
        weight=200.0,
        params={"distance_threshold": 1.0},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)


# ---------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    lidar_collision = DoneTerm(
        func=mdp.lidar_collision,
        params={"body_radius": 0.4},
    )
    fallen_over = DoneTerm(
        func=mdp.fallen_over,
        params={"window_steps": 40, "displacement_threshold": 0.05, "grace_steps": 50},
    )
    terrain_out_of_bounds = None
    goal_reached = DoneTerm(
        func=mdp.goal_reached,
        params={"command_name": "pose_command", "distance_threshold": 1.0},
        time_out=True,  # reaching goal is success, no -200 penalty
    )
    # Let the dog explore freely; boundary walls constrain the cell physically.
    # Collision only penalizes via reward, does NOT end the episode
    # (dogs need time to explore without dying)


# ---------------------------------------------------------------------
# Events: random start on left side, facing toward goal (right side)
# ---------------------------------------------------------------------
@configclass
class EventCfg:
    setup_usd_scene = EventTerm(
        func=mdp.setup_usd_scene,
        mode="prestartup",
    )

    fix_env_origins = EventTerm(
        func=mdp.fix_origins_for_usd,
        mode="prestartup",
    )

    configure_viewer_visibility = EventTerm(
        func=mdp.configure_viewer_visibility,
        mode="prestartup",
    )

    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform_navigation,
        mode="reset",
        params={
            "pose_range": {
                "x": (-23.0, -21.0),
                "y": (-18.0, 18.0),
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

# ---------------------------------------------------------------------
# Curriculum: advance/reduce difficulty based on goal-reaching
# ---------------------------------------------------------------------
@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_by_goal_reached,
        params={"success_threshold": 8, "failure_threshold": 4},
    )


# ---------------------------------------------------------------------
# Main environment config
# ---------------------------------------------------------------------
_USD_SCENE_PATH = _os.environ.get("NAVRL_USD_SCENE", "")


@configclass
class NavigationEnvCfg(ManagerBasedRLEnvCfg):
    scene: NavigationSceneCfg = NavigationSceneCfg(
        num_envs=_TERRAIN_ROWS * _TERRAIN_COLS, env_spacing=0.0
    )
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum = CurriculumCfg()

    def __post_init__(self):
        self.sim.dt = 0.005
        self.decimation = 40  # NavRL @ 5Hz; locomotion internally @ 50Hz
        self.sim.render_interval = self.decimation
        self.episode_length_s = 100.0  # 500 steps, plenty for detours

        # Follow env 0 from high enough to see its whole 50m x 50m cell.
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0
        # PhysX buffers for 1024 quadrupeds moving through a shared obstacle-rich USD scene.
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**28
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**24
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**24
        self.sim.physx.gpu_max_rigid_contact_count = 2**24
        self.sim.physx.gpu_max_rigid_patch_count = 2**23
        self.sim.physx.gpu_collision_stack_size = 2**28
        self.sim.physx.gpu_heap_capacity = 2**28
        self.sim.physx.gpu_temp_buffer_capacity = 2**26
        # self.viewer.eye = (-35.0, -35.0, 40.0)
        # self.viewer.lookat = (0.0, 0.0, 0.5)
        self.viewer.eye = (-10.0, -10.0, 40.0)
        self.viewer.lookat = (25.0, 25.0, 0.5)

        # Sensor update periods
        if self.scene.lidar is not None:
            self.scene.lidar.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        if _USD_SCENE_PATH:
            self.scene.replicate_physics = False
            self.scene.terrain = None
            # USD provides the only terrain geometry. A prestartup event creates synthetic
            # terrain-level buffers for curriculum without spawning extra ground.
            self.scene.usd_scene = AssetBaseCfg(
                prim_path="/World/usd_scene",
                spawn=sim_utils.UsdFileCfg(usd_path=_os.path.abspath(_USD_SCENE_PATH)),
                collision_group=-1,
            )
            self.scene.lidar = MultiMeshRayCasterCfg(
                prim_path="{ENV_REGEX_NS}/Robot/base",
                offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.3)),
                ray_alignment="yaw",
                pattern_cfg=LidarPatternCfg(
                    channels=8, vertical_fov_range=(-10.0, 20.0),
                    horizontal_fov_range=(-180.0, 180.0), horizontal_res=5.0,
                ),
                debug_vis=False,
                mesh_prim_paths=[
                    MultiMeshRayCasterCfg.RaycastTargetCfg(
                        prim_expr="/World/usd_scene",
                        is_shared=True, merge_prim_meshes=True, track_mesh_transforms=False,
                    )
                ],
            )
            self.scene.lidar.update_period = self.decimation * self.sim.dt
            print(f"[INFO] USD scene: {_USD_SCENE_PATH}")
        if self.scene.terrain is not None and self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = True
