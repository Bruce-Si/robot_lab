# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Single-cell pillar-scene configuration for navigation-to-grasp handoff."""

from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.utils import configclass

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR
from robot_lab.tasks.manager_based.navigation.mdp.commands import (
    FixedLocalTargetFacingPose2dCommand,
)

from .navigation_env_cfg import (
    TiltingUAVNavigationEnvCfg,
    TiltingUAVNavigationSceneCfg,
)
from .scene_profile import GRID_8X8_PROFILE
from ...mdp.uav_navigation import setup_uav_static_usd_scene


PILLAR_GRASP_CELL_INDEX = 63
"""Use the hardest previously evaluated pillar cell (Cell_7_7)."""

PILLAR_GRASP_CELL_SCENE_PATH = (
    Path(ISAACLAB_ASSETS_DATA_DIR) / "environments" / "uav_eval_cell_7_7_baked_round_prims.usd"
).resolve()

PILLAR_GRASP_TABLE_CENTER_W = (397.0, 375.0, 0.2)
PILLAR_GRASP_TABLE_SIZE = (0.8, 0.6, 0.4)
PILLAR_GRASP_OBJECT_CENTER_W = (397.0, 375.0, 0.44)
PILLAR_GRASP_OBJECT_SIZE = (0.04, 0.04, 0.08)
# The original expert offset is expressed in its top-level UAV frame. RobotLab
# drives the /base_link articulation root while holding +30 deg pitch, so the
# equivalent calibrated root offset is different in z. This was the offset
# validated by the earlier RobotLab grasp smoke run.
PILLAR_GRASP_UAV_OFFSET_W = (-0.22, 0.0, 0.13)
PILLAR_GRASP_PRE_CLOSE_SECONDS = 1.0
PILLAR_GRASP_CLOSE_HOLD_SECONDS = 1.0
# Match the original pick-place navigation scene: carry the cube off the
# grasp table and release it on the ground-level goal patch in front of it.
PILLAR_PLACE_OBJECT_CENTER_W = (398.30, 375.0, 0.04)
PILLAR_PLACE_ZONE_SIZE = (0.12, 0.12, 0.01)
PILLAR_PLACE_ZONE_CENTER_W = (398.30, 375.0, 0.005)


@configclass
class TiltingUAVNavigationGraspSceneCfg(TiltingUAVNavigationSceneCfg):
    """One global table and one dynamic object over the selected pillar cell."""

    grasp_table = AssetBaseCfg(
        # Keep task-only grasp geometry outside the static navigation USD root.
        # The LiDAR ray target is /World/uav_navigation_scene, so the table is
        # physically collidable but intentionally invisible to navigation scans.
        prim_path="/World/GraspTable",
        spawn=sim_utils.CuboidCfg(
            size=PILLAR_GRASP_TABLE_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.32, 0.20, 0.10)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=PILLAR_GRASP_TABLE_CENTER_W),
    )
    place_zone = AssetBaseCfg(
        # Visual-only target marker. It is outside the LiDAR root and has no
        # collider, so it cannot interfere with navigation or object settling.
        prim_path="/World/PlaceZone",
        spawn=sim_utils.CuboidCfg(
            size=PILLAR_PLACE_ZONE_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.85, 0.2)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=PILLAR_PLACE_ZONE_CENTER_W),
    )
    grasp_object = RigidObjectCfg(
        prim_path="/World/GraspObject",
        collision_group=-1,
        spawn=sim_utils.CuboidCfg(
            size=PILLAR_GRASP_OBJECT_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=PILLAR_GRASP_OBJECT_CENTER_W),
    )


@configclass
class TiltingUAVNavigationGraspEnvCfg(TiltingUAVNavigationEnvCfg):
    """Navigation to the table followed by a scripted physical grasp phase."""

    scene: TiltingUAVNavigationGraspSceneCfg = TiltingUAVNavigationGraspSceneCfg(
        num_envs=1,
        env_spacing=0.0,
        replicate_physics=False,
    )

    navigation_grasp_cell_index: int = PILLAR_GRASP_CELL_INDEX
    navigation_fixed_target_local_xy: tuple[float, float] = (22.0, 0.0)
    navigation_table_goal_distance_threshold: float = 1.0
    navigation_table_goal_heading_threshold: float = math.radians(10.0)

    def __post_init__(self):
        self._configure_common_runtime()
        # Keep the pads on the two sides of this 0.04 m cube.  A zero target
        # for both prismatic joints drives the right finger through the cube
        # in this imported articulation and only produces a lateral push.
        self.actions.uav_velocity.gripper_closed_positions = (0.020, -0.020)
        if not PILLAR_GRASP_CELL_SCENE_PATH.is_file():
            raise FileNotFoundError(
                "Pillar grasp cell is missing: "
                f"{PILLAR_GRASP_CELL_SCENE_PATH}. Generate the derived Cell_7_7 USD first."
            )
        self._apply_scene_profile(
            # Keep the grid observation layout (including the cell-63 oracle one-hot)
            # while replacing the composed USD with the already extracted cell.
            GRID_8X8_PROFILE,
            PILLAR_GRASP_CELL_SCENE_PATH,
        )

        self.scene.num_envs = 1
        self.scene.usd_scene.spawn.usd_path = str(PILLAR_GRASP_CELL_SCENE_PATH)
        # This combined handoff uses LiDAR for navigation collision checks and
        # keeps physical contacts for the grasp itself. Do not allocate a GPU
        # ContactSensor stream that is not consumed by this scripted task.
        self.scene.contact_forces = None
        self.events.setup_static_scene.func = setup_uav_static_usd_scene
        self.events.setup_static_scene.params = {
            "scene_root": "/World/uav_navigation_scene",
            "cell_size": 50.0,
            "num_rows": 8,
            "num_cols": 8,
            "grid_origin": (0.0, 0.0, 0.0),
            "cell_indices": PILLAR_GRASP_CELL_INDEX,
        }
        self.events.reset_base.params["start_edge"] = 0
        self.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
        self.navigation_lateral_range = (0.0, 0.0)
        self.commands.pose_command.class_type = FixedLocalTargetFacingPose2dCommand

        # The handoff script owns success/collision decisions after navigation.
        # Keep altitude and attitude safety limits active.
        self.terminations.goal_reached = None
        self.terminations.contact_collision = None
        self.terminations.lidar_collision = None
        self.terminations.tilt.params["max_tilt"] = math.radians(45.0)
        self.episode_length_s = 180.0
