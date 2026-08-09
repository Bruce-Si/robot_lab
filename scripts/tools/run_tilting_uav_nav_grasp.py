#!/usr/bin/env python3
"""Run navigation, pick the table cube, carry it, and place it on the ground goal."""

from __future__ import annotations

import argparse
import math
import sys
import traceback
from enum import Enum
from pathlib import Path

from isaaclab.app import AppLauncher


ROBOT_LAB_ROOT = Path(__file__).resolve().parents[2]
ROBOT_LAB_SOURCE_DIR = ROBOT_LAB_ROOT / "source" / "robot_lab"
if str(ROBOT_LAB_SOURCE_DIR) not in sys.path:
    # Keep this standalone handoff script usable after a checkout rename or
    # before the editable package has been installed in the active conda env.
    sys.path.insert(0, str(ROBOT_LAB_SOURCE_DIR))
DEFAULT_CHECKPOINT = (
    ROBOT_LAB_ROOT
    / "logs/rsl_rl/tilting_uav_navrl_planar_yaw_oracle"
    / "2026-07-27_21-27-40/model_2100.pt"
)
TASK_ID = "RobotLab-Navigation-Tilting-UAV-Grasp-v0"

parser = argparse.ArgumentParser(
    description="Navigate through Cell_7_7, pick the cube, carry it, and place it on the ground goal."
)
parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
parser.add_argument("--duration", type=float, default=180.0)
parser.add_argument("--status-hz", type=float, default=1.0)
parser.add_argument("--hold-seconds", type=float, default=5.0)
parser.add_argument("--pre-release-seconds", type=float, default=1.0)
parser.add_argument("--release-hold-seconds", type=float, default=1.0)
parser.add_argument(
    "--pre-close-seconds",
    type=float,
    default=1.0,
    help="Open-gripper settling time at the grasp pose; default matches the original expert.",
)
parser.add_argument(
    "--grasp-hold-seconds",
    type=float,
    default=1.0,
    help="Closed-gripper dwell before lifting; default matches the original expert.",
)
parser.add_argument(
    "--physics-hz",
    "--physics_hz",
    dest="physics_hz",
    type=float,
    default=400.0,
    help="UAV physics frequency; default 400. Use 60 for a lighter GUI test.",
)
parser.add_argument(
    "--navigation-only",
    action="store_true",
    help="Stop after the table-approach handoff condition is reached.",
)
parser.add_argument(
    "--show-lidar",
    action="store_true",
    help="Draw the planar LiDAR rays and goal beacon in GUI mode.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import importlib.metadata as metadata

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.math import quat_apply, quat_apply_inverse, wrap_to_pi, yaw_quat
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import robot_lab.tasks  # noqa: F401
from robot_lab.tasks.manager_based.navigation.config.uav.navigation_grasp_env_cfg import (
    PILLAR_GRASP_OBJECT_CENTER_W,
    PILLAR_PLACE_OBJECT_CENTER_W,
    PILLAR_GRASP_TABLE_CENTER_W,
    PILLAR_GRASP_TABLE_SIZE,
    PILLAR_GRASP_UAV_OFFSET_W,
)
from robot_lab.tasks.manager_based.navigation.mdp.uav_navigation import (
    planar_lidar_collision,
)
from robot_lab.tasks.manager_based.navigation.config.uav.navigation_env_cfg import (
    configure_tilting_uav_physics_hz,
)


class Phase(str, Enum):
    NAVIGATE = "NAVIGATE"
    APPROACH = "APPROACH"
    DESCEND = "DESCEND"
    PRE_CLOSE = "PRE_CLOSE"
    CLOSE = "CLOSE"
    LIFT = "LIFT"
    CARRY = "CARRY"
    PLACE_DESCEND = "PLACE_DESCEND"
    PRE_RELEASE = "PRE_RELEASE"
    RELEASE = "RELEASE"
    RECOVER = "RECOVER"
    DONE = "DONE"


FIRST_PERSON_CAMERA_PATH = (
    "/World/envs/env_0/Robot/sensors/RSD455/Camera_OmniVision_OV9782_Color"
)


def _scripted_velocity_action(
    env,
    target_w: torch.Tensor,
    *,
    target_yaw: float,
    max_speed: float,
) -> torch.Tensor:
    robot = env.scene["robot"]
    action_term = env.action_manager.get_term("uav_velocity")
    delta_xy_w = target_w[:2] - robot.data.root_pos_w[0, :2]
    distance = torch.linalg.norm(delta_xy_w)
    velocity_w = torch.zeros(3, device=env.device)
    if float(distance) > 1.0e-6:
        velocity_w[:2] = delta_xy_w * min(float(max_speed) / float(distance), 1.5)
    velocity_b = quat_apply_inverse(yaw_quat(robot.data.root_quat_w[0:1]), velocity_w[None])[0]

    yaw_error = wrap_to_pi(
        torch.tensor(target_yaw, device=env.device) - robot.data.heading_w[0]
    )
    action = torch.zeros((1, 3), device=env.device)
    action[0, :2] = velocity_b[:2] / action_term.velocity_scale[:2]
    action[0, 2] = torch.clamp(1.5 * yaw_error / action_term.cfg.yaw_rate_scale, -1.0, 1.0)
    return action.clamp(-1.0, 1.0)


def _pose_ready(env, target_w: torch.Tensor, *, xy_tol: float, z_tol: float, speed_tol: float) -> bool:
    robot = env.scene["robot"]
    pos = robot.data.root_pos_w[0]
    speed = torch.linalg.norm(robot.data.root_lin_vel_w[0])
    return bool(
        torch.linalg.norm(pos[:2] - target_w[:2]) < xy_tol
        and torch.abs(pos[2] - target_w[2]) < z_tol
        and speed < speed_tol
    )


def _object_placement_ready(
    env,
    target_w: torch.Tensor,
    *,
    xy_tol: float = 0.08,
    z_tol: float = 0.04,
    speed_tol: float = 0.12,
) -> bool:
    """Check that the released cube is resting inside the placement zone."""
    grasp_object = env.scene["grasp_object"]
    pos = grasp_object.data.root_pos_w[0]
    speed = torch.linalg.norm(grasp_object.data.root_lin_vel_w[0])
    return bool(
        torch.linalg.norm(pos[:2] - target_w[:2]) < xy_tol
        and torch.abs(pos[2] - target_w[2]) < z_tol
        and speed < speed_tol
    )


def _slew_target_altitude(env, target_z: float, *, max_rate: float) -> float:
    """Move the altitude setpoint without tripping the tracking-error safety limit."""
    action_term = env.action_manager.get_term("uav_velocity")
    current_target = float(action_term.target_altitude[0])
    max_step = max_rate * env.step_dt
    next_target = current_target + max(-max_step, min(max_step, target_z - current_target))
    action_term.set_target_altitude(next_target, env_ids=[0])
    return next_target


def _update_first_person_view(env) -> None:
    """Track the authored forward RGB camera pose with the interactive viewport."""
    controller = getattr(env, "viewport_camera_controller", None)
    if controller is None:
        return
    robot = env.scene["robot"]
    root_pos = robot.data.root_pos_w[0]
    root_quat = robot.data.root_quat_w[0:1]
    # Camera_OmniVision_OV9782_Color in the UAV USD. Its USD camera -Z axis is
    # aligned with the UAV +X forward axis.
    camera_offset_b = torch.tensor((0.13, -0.0115, 0.13), device=env.device)
    camera_forward_b = torch.tensor((1.0, 0.0, 0.0), device=env.device)
    camera_pos = root_pos + quat_apply(root_quat, camera_offset_b[None])[0]
    camera_forward = quat_apply(root_quat, camera_forward_b[None])[0]
    camera_target = camera_pos + 2.0 * camera_forward
    controller.set_view_env_index(env_index=0)
    controller.update_view_location(
        eye=camera_pos.detach().cpu().numpy(),
        lookat=camera_target.detach().cpu().numpy(),
    )


def _set_active_first_person_camera() -> bool:
    """Select the authored UAV color camera for the interactive viewport."""
    try:
        import omni.kit.viewport.utility as viewport_utility
        from isaacsim.core.utils.prims import is_prim_path_valid

        if not is_prim_path_valid(FIRST_PERSON_CAMERA_PATH):
            raise ValueError(f"camera prim does not exist: {FIRST_PERSON_CAMERA_PATH}")
        viewport = viewport_utility.get_active_viewport()
        if viewport is None:
            raise RuntimeError("no active viewport is available")
        viewport.camera_path = FIRST_PERSON_CAMERA_PATH
        if str(viewport.camera_path) != FIRST_PERSON_CAMERA_PATH:
            raise RuntimeError(f"viewport selected unexpected camera: {viewport.camera_path}")
    except (AttributeError, ImportError, RuntimeError, ValueError) as exc:
        print(
            f"[camera] authored color camera unavailable; using Perspective fallback: {exc}",
            flush=True,
        )
        return False
    print(f"[camera] active={FIRST_PERSON_CAMERA_PATH}", flush=True)
    return True


def main() -> None:
    if args_cli.duration <= 0.0:
        raise ValueError("--duration must be positive.")
    if args_cli.hold_seconds < 0.0:
        raise ValueError("--hold-seconds cannot be negative.")
    if args_cli.pre_release_seconds < 0.0:
        raise ValueError("--pre-release-seconds cannot be negative.")
    if args_cli.release_hold_seconds < 0.0:
        raise ValueError("--release-hold-seconds cannot be negative.")
    if args_cli.pre_close_seconds < 0.0:
        raise ValueError("--pre-close-seconds cannot be negative.")
    if args_cli.grasp_hold_seconds < 0.0:
        raise ValueError("--grasp-hold-seconds cannot be negative.")
    checkpoint_path = Path(args_cli.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Navigation checkpoint not found: {checkpoint_path}")

    env_cfg = parse_env_cfg(TASK_ID, device=args_cli.device, num_envs=1)
    env_cfg.seed = 42
    physics_hz, control_hz, decimation = configure_tilting_uav_physics_hz(
        env_cfg, args_cli.physics_hz
    )
    # The scan remains part of the policy observation; this only controls the
    # custom debug-draw overlay that is distracting from the first-person view.
    env_cfg.observations.policy.navigation.params["debug_draw"] = True
    env_cfg.observations.policy.navigation.params["debug_draw_lidar"] = args_cli.show_lidar
    agent_cfg = load_cfg_from_registry(TASK_ID, "rsl_rl_cfg_entry_point")
    installed_rsl_rl_version = metadata.version("rsl-rl-lib")
    if version.parse(installed_rsl_rl_version) < version.parse("3.0.1"):
        raise RuntimeError(
            f"RSL-RL >= 3.0.1 is required, found {installed_rsl_rl_version}."
        )
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_rsl_rl_version)
    agent_cfg.device = args_cli.device

    gym_env = gym.make(TASK_ID, cfg=env_cfg)
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint_path))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs, _ = env.reset()
    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    grasp_object = base_env.scene["grasp_object"]
    action_term = base_env.action_manager.get_term("uav_velocity")
    command_term = base_env.command_manager.get_term("pose_command")
    authored_camera_active = _set_active_first_person_camera()
    if not authored_camera_active:
        _update_first_person_view(base_env)

    table_top_z = PILLAR_GRASP_TABLE_CENTER_W[2] + 0.5 * PILLAR_GRASP_TABLE_SIZE[2]
    object_initial = grasp_object.data.root_pos_w[0].clone()
    grasp_offset = torch.tensor(PILLAR_GRASP_UAV_OFFSET_W, device=base_env.device)
    object_center = torch.tensor(PILLAR_GRASP_OBJECT_CENTER_W, device=base_env.device)
    object_xy = object_center[:2]
    approach = torch.tensor(
        (
            float(object_center[0] + grasp_offset[0]),
            float(object_center[1] + grasp_offset[1]),
            table_top_z + 0.35,
        ),
        device=base_env.device,
    )
    grasp = object_center + grasp_offset
    place_object = torch.tensor(PILLAR_PLACE_OBJECT_CENTER_W, device=base_env.device)
    place = place_object + grasp_offset
    carry = place.clone()
    carry[2] = table_top_z + 0.35
    # Match the original expert: lift at least 0.15 m, but never below the
    # configured carry height.
    lift = grasp.clone()
    lift[2] = max(float(grasp[2]) + 0.15, float(carry[2]))
    recover = carry.clone()

    phase = Phase.NAVIGATE
    phase_start_step = 0
    step = 0
    status_interval = max(1, int(round(1.0 / max(args_cli.status_hz, 1.0e-6) / base_env.step_dt)))
    max_steps = int(math.ceil(args_cli.duration / base_env.step_dt))
    success = False

    print("=" * 76, flush=True)
    print(" Tilting UAV Pillar Navigation -> Grasp ", flush=True)
    print("=" * 76, flush=True)
    print(f"[checkpoint] {checkpoint_path}", flush=True)
    print(
        f"[timing] physics={physics_hz:g}Hz policy={control_hz:g}Hz "
        f"decimation={decimation}",
        flush=True,
    )
    print(f"[lidar] debug_lines={'on' if args_cli.show_lidar else 'off'}", flush=True)
    print(
        f"[scene] cell=7_7 table={PILLAR_GRASP_TABLE_CENTER_W} "
        f"pick_object={tuple(round(float(v), 3) for v in object_initial.tolist())} "
        f"place_object={tuple(round(float(v), 3) for v in place_object.tolist())}",
        flush=True,
    )
    print(
        "[handoff] distance<1.0m and heading_error<10deg; "
        "phases=NAVIGATE->APPROACH->DESCEND->PRE_CLOSE->CLOSE->LIFT->CARRY->"
        "PLACE_DESCEND->PRE_RELEASE->RELEASE->RECOVER->DONE",
        flush=True,
    )
    print("=" * 76, flush=True)

    try:
        with torch.inference_mode():
            while simulation_app.is_running() and step < max_steps:
                current_pos = robot.data.root_pos_w[0]
                if not authored_camera_active:
                    _update_first_person_view(base_env)
                goal_delta = command_term.pos_command_w[0, :2] - current_pos[:2]
                goal_distance = torch.linalg.norm(goal_delta)
                goal_heading = torch.atan2(goal_delta[1], goal_delta[0])
                heading_error = torch.abs(wrap_to_pi(goal_heading - robot.data.heading_w[0]))

                if phase == Phase.NAVIGATE:
                    action_term.set_target_attitude(roll=0.0, pitch=0.0, env_ids=[0])
                    action_term.set_gripper_close_fraction(0.0, env_ids=[0])
                    actions = policy(obs)

                    reached = bool(
                        goal_distance
                        < base_env.cfg.navigation_table_goal_distance_threshold
                        and heading_error
                        < base_env.cfg.navigation_table_goal_heading_threshold
                    )
                    if reached:
                        if args_cli.navigation_only:
                            print("[result] navigation handoff condition reached: PASS", flush=True)
                            success = True
                            break
                        phase = Phase.APPROACH
                        phase_start_step = step
                        print(f"[phase] t={step * base_env.step_dt:.1f}s -> {phase.value}", flush=True)
                    else:
                        lidar_hit = bool(
                            planar_lidar_collision(
                                base_env,
                                expected_rays=36,
                            )[0]
                        )
                        if lidar_hit:
                            raise RuntimeError(
                                f"Navigation LiDAR collision before handoff: lidar={lidar_hit}."
                            )
                else:
                    target = approach
                    close_fraction = 0.0
                    pitch = math.radians(30.0)
                    max_speed = 0.35
                    altitude_rate = 0.25

                    if phase == Phase.APPROACH:
                        if _pose_ready(base_env, approach, xy_tol=0.06, z_tol=0.05, speed_tol=0.15):
                            phase = Phase.DESCEND
                            phase_start_step = step
                            print(f"[phase] t={step * base_env.step_dt:.1f}s -> {phase.value}", flush=True)
                    elif phase == Phase.DESCEND:
                        target = grasp
                        max_speed = 0.18
                        altitude_rate = 0.12
                        if _pose_ready(base_env, grasp, xy_tol=0.04, z_tol=0.03, speed_tol=0.12):
                            phase = Phase.PRE_CLOSE
                            phase_start_step = step
                            print(f"[phase] t={step * base_env.step_dt:.1f}s -> {phase.value}", flush=True)
                    elif phase == Phase.PRE_CLOSE:
                        target = grasp
                        close_fraction = 0.0
                        max_speed = 0.06
                        altitude_rate = 0.06
                        if (step - phase_start_step) * base_env.step_dt >= args_cli.pre_close_seconds:
                            phase = Phase.CLOSE
                            phase_start_step = step
                            print(f"[phase] t={step * base_env.step_dt:.1f}s -> {phase.value}", flush=True)
                    elif phase == Phase.CLOSE:
                        target = grasp
                        close_fraction = 1.0
                        max_speed = 0.08
                        altitude_rate = 0.08
                        if (step - phase_start_step) * base_env.step_dt >= args_cli.grasp_hold_seconds:
                            phase = Phase.LIFT
                            phase_start_step = step
                            print(f"[phase] t={step * base_env.step_dt:.1f}s -> {phase.value}", flush=True)
                    elif phase == Phase.LIFT:
                        target = lift
                        close_fraction = 1.0
                        max_speed = 0.20
                        altitude_rate = 0.15
                        object_lift = float(grasp_object.data.root_pos_w[0, 2] - object_initial[2])
                        if object_lift > 0.10 and _pose_ready(
                            base_env,
                            lift,
                            xy_tol=0.08,
                            z_tol=0.06,
                            speed_tol=0.18,
                        ):
                            phase = Phase.CARRY
                            phase_start_step = step
                            print(
                                f"[phase] t={step * base_env.step_dt:.1f}s -> {phase.value} "
                                f"object_lift={object_lift:.3f}m",
                                flush=True,
                            )
                        elif (step - phase_start_step) * base_env.step_dt > 8.0:
                            raise RuntimeError(
                                f"Object was not lifted after 8s (dz={object_lift:.3f}m)."
                            )
                    elif phase == Phase.CARRY:
                        target = carry
                        close_fraction = 1.0
                        max_speed = 0.12
                        altitude_rate = 0.12
                        object_lift = float(grasp_object.data.root_pos_w[0, 2] - object_initial[2])
                        if object_lift < 0.05 and (step - phase_start_step) * base_env.step_dt > 1.0:
                            raise RuntimeError(
                                f"Object was lost during carry (dz={object_lift:.3f}m)."
                            )
                        if _pose_ready(
                            base_env,
                            carry,
                            xy_tol=0.06,
                            z_tol=0.06,
                            speed_tol=0.12,
                        ):
                            phase = Phase.PLACE_DESCEND
                            phase_start_step = step
                            print(f"[phase] t={step * base_env.step_dt:.1f}s -> {phase.value}", flush=True)
                        elif (step - phase_start_step) * base_env.step_dt > 15.0:
                            raise RuntimeError("Timed out while carrying the object to the placement zone.")
                    elif phase == Phase.PLACE_DESCEND:
                        target = place
                        close_fraction = 1.0
                        max_speed = 0.12
                        altitude_rate = 0.10
                        if _pose_ready(
                            base_env,
                            place,
                            xy_tol=0.04,
                            z_tol=0.03,
                            speed_tol=0.10,
                        ):
                            phase = Phase.PRE_RELEASE
                            phase_start_step = step
                            print(f"[phase] t={step * base_env.step_dt:.1f}s -> {phase.value}", flush=True)
                    elif phase == Phase.PRE_RELEASE:
                        target = place
                        close_fraction = 1.0
                        max_speed = 0.06
                        altitude_rate = 0.06
                        if (step - phase_start_step) * base_env.step_dt >= args_cli.pre_release_seconds:
                            phase = Phase.RELEASE
                            phase_start_step = step
                            print(f"[phase] t={step * base_env.step_dt:.1f}s -> {phase.value}", flush=True)
                    elif phase == Phase.RELEASE:
                        target = place
                        close_fraction = 0.0
                        max_speed = 0.04
                        altitude_rate = 0.04
                        placement_ready = _object_placement_ready(base_env, place_object)
                        elapsed = (step - phase_start_step) * base_env.step_dt
                        if elapsed >= args_cli.release_hold_seconds and placement_ready:
                            phase = Phase.RECOVER
                            phase_start_step = step
                            print(
                                f"[phase] t={step * base_env.step_dt:.1f}s -> {phase.value} "
                                f"place_ok={placement_ready}",
                                flush=True,
                            )
                        elif elapsed > 5.0:
                            obj = grasp_object.data.root_pos_w[0]
                            raise RuntimeError(
                                "Released object did not settle in the placement zone "
                                f"(pos={[round(float(v), 3) for v in obj.tolist()]})."
                            )
                    elif phase == Phase.RECOVER:
                        target = recover
                        close_fraction = 0.0
                        pitch = 0.0
                        max_speed = 0.10
                        altitude_rate = 0.12
                        if _pose_ready(
                            base_env,
                            recover,
                            xy_tol=0.08,
                            z_tol=0.06,
                            speed_tol=0.15,
                        ) and _object_placement_ready(base_env, place_object):
                            phase = Phase.DONE
                            phase_start_step = step
                            success = True
                            print(f"[phase] t={step * base_env.step_dt:.1f}s -> {phase.value}", flush=True)
                    elif phase == Phase.DONE:
                        target = recover
                        close_fraction = 0.0
                        pitch = 0.0
                        max_speed = 0.05
                        altitude_rate = 0.05
                        if (step - phase_start_step) * base_env.step_dt >= args_cli.hold_seconds:
                            break

                    _slew_target_altitude(
                        base_env,
                        float(target[2]),
                        max_rate=altitude_rate,
                    )
                    action_term.set_target_attitude(roll=0.0, pitch=pitch, env_ids=[0])
                    action_term.set_gripper_close_fraction(close_fraction, env_ids=[0])
                    actions = _scripted_velocity_action(
                        base_env,
                        target,
                        target_yaw=0.0,
                        max_speed=max_speed,
                    )

                obs, _, dones, _ = env.step(actions)
                step += 1
                if bool(dones[0]):
                    termination_terms = {
                        name: bool(base_env.termination_manager.get_term(name)[0])
                        for name in base_env.termination_manager.active_terms
                    }
                    raise RuntimeError(
                        f"Environment terminated unexpectedly during {phase.value}: "
                        f"{termination_terms}."
                    )

                if step % status_interval == 0:
                    pos = robot.data.root_pos_w[0]
                    obj = grasp_object.data.root_pos_w[0]
                    place_xy_error = torch.linalg.norm(obj[:2] - place_object[:2])
                    print(
                        f"[t={step * base_env.step_dt:5.1f}s] phase={phase.value:8s} "
                        f"uav=[{pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f}] "
                        f"goal_d={float(goal_distance):.2f} yaw_err={math.degrees(float(heading_error)):.1f}deg "
                        f"z_ref={float(action_term.target_altitude[0]):.2f} "
                        f"obj=[{obj[0]:+.2f},{obj[1]:+.2f},{obj[2]:+.3f}] "
                        f"place_xy_err={float(place_xy_error):.3f}",
                        flush=True,
                    )
    except BaseException:
        print(f"[result] {phase.value} failed at t={step * base_env.step_dt:.1f}s", flush=True)
        traceback.print_exc()
        raise
    finally:
        env.close()

    if not success:
        message = f"Navigation-to-pick-and-place did not complete within {args_cli.duration:.1f}s."
        print(f"[result] FAIL: {message}", flush=True)
        raise RuntimeError(message)
    print("=" * 76, flush=True)
    print("Tilting UAV Pillar Navigation -> Pick -> Place: PASS", flush=True)
    print("=" * 76, flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
