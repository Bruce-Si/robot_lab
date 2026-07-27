# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the tilting UAV with an attached gripper."""

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


TILTING_UAV_USD_ENV_VAR = "ROBOT_LAB_TILTING_UAV_USD"
"""Environment variable that overrides the derived tilting-UAV USD path."""

_DEFAULT_TILTING_UAV_USD_PATH = (
    Path(__file__).resolve().parents[5]
    / "TiltingUAV_Sim"
    / "TiltingUAV_Diffusion"
    / "TiltingUAV_Isaac"
    / "assets"
    / "uav_mesh"
    / "robotlab"
    / "uav_nav.usda"
)


def _resolve_tilting_uav_usd_path() -> Path:
    """Resolve the derived asset without coupling the config to one checkout path."""
    override = os.environ.get(TILTING_UAV_USD_ENV_VAR)
    path = Path(override).expanduser() if override else _DEFAULT_TILTING_UAV_USD_PATH
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Tilting-UAV USD not found at {path}. "
            f"Set {TILTING_UAV_USD_ENV_VAR} to the derived uav_nav.usda path."
        )
    return path


TILTING_UAV_USD_PATH = _resolve_tilting_uav_usd_path()
"""Resolved path to the derived RobotLab-compatible USD layer."""


TILTING_UAV_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(TILTING_UAV_USD_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=20.0,
            max_angular_velocity=50.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
            sleep_threshold=0.0,
            stabilization_threshold=0.0,
        ),
    ),
    articulation_root_prim_path="/base_link",
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={
            "joint_arm_.*": 0.0,
            "left_left_finger": 0.042,
            "left_right_finger": -0.042,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "tilt_servos": ImplicitActuatorCfg(
            joint_names_expr=["joint_arm_.*"],
            effort_limit_sim=5.0,
            velocity_limit_sim=15.0,
            stiffness=625.0,
            damping=25.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["left_(left|right)_finger"],
            effort_limit_sim=5.0,
            velocity_limit_sim=0.25,
            stiffness=400.0,
            damping=40.0,
        ),
    },
)
"""RobotLab asset configuration for the floating tilting UAV.

The navigation policy will not command the gripper. Its implicit actuator
holds the fingers at the open default positions until grasp integration.
"""
