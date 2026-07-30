# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Batched low-level velocity control for the tilting UAV articulation."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


class TiltingUAVVelocityAction(ActionTerm):
    """Convert normalized yaw-frame velocity commands into a tilting-UAV body wrench.

    In the default mode, the policy controls ``(vx, vy, vz)``. In planar mode, it
    controls ``(vx, vy, yaw_rate)`` while an outer loop supplies the vertical velocity
    needed to hold the reset altitude. Roll and pitch are held level, the commanded yaw
    rate is integrated into a yaw target, and the gripper remains at its open default
    joint positions. All controller, allocation, and actuator calculations are batched
    Torch operations on the simulation device.
    """

    _asset: Articulation

    def __init__(self, cfg: "TiltingUAVVelocityActionCfg", env):
        super().__init__(cfg, env)
        if not isinstance(self._asset, Articulation):
            raise TypeError(
                f"TiltingUAVVelocityAction requires an Articulation, got {type(self._asset).__name__}."
            )
        if len(cfg.velocity_scale) != 3:
            raise ValueError(f"velocity_scale must contain 3 values, got {cfg.velocity_scale}.")
        if cfg.planar_mode and cfg.altitude_hold_gain <= 0.0:
            raise ValueError("altitude_hold_gain must be positive in planar mode.")
        if cfg.planar_mode and cfg.altitude_hold_max_velocity <= 0.0:
            raise ValueError("altitude_hold_max_velocity must be positive in planar mode.")
        if cfg.velocity_command_time_constant < 0.0:
            raise ValueError("velocity_command_time_constant cannot be negative.")
        if cfg.planar_mode and cfg.yaw_rate_scale <= 0.0:
            raise ValueError("yaw_rate_scale must be positive in planar mode.")
        if len(cfg.rotor_positions) != 4 or len(cfg.rotor_alpha_deg) != 4:
            raise ValueError("The registered tilting UAV controller requires exactly four rotors.")

        base_body_ids, base_body_names = self._asset.find_bodies(cfg.base_body_name)
        if len(base_body_ids) != 1:
            raise ValueError(
                f"Expected one body matching '{cfg.base_body_name}', found {base_body_names}."
            )
        tilt_joint_ids, tilt_joint_names = self._asset.find_joints(cfg.tilt_joint_names, preserve_order=True)
        if tilt_joint_names != list(cfg.tilt_joint_names):
            raise ValueError(
                f"Expected tilt joints {list(cfg.tilt_joint_names)}, found {tilt_joint_names}."
            )
        gripper_joint_ids, gripper_joint_names = self._asset.find_joints(
            cfg.gripper_joint_names, preserve_order=True
        )
        if gripper_joint_names != list(cfg.gripper_joint_names):
            raise ValueError(
                f"Expected gripper joints {list(cfg.gripper_joint_names)}, found {gripper_joint_names}."
            )

        self._base_body_ids = torch.tensor(base_body_ids, dtype=torch.long, device=self.device)
        self._tilt_joint_ids = tilt_joint_ids
        self._gripper_joint_ids = gripper_joint_ids
        self._physics_dt = float(env.physics_dt)

        self._velocity_scale = self._tensor(cfg.velocity_scale)
        self._kp_vel = self._tensor(cfg.kp_vel)
        self._ki_vel = self._tensor(cfg.ki_vel)
        self._force_min = self._tensor(cfg.force_min_xyz)
        self._force_max = self._tensor(cfg.force_max_xyz)
        self._inertia = self._tensor(cfg.inertia_diag)
        self._kp_att = self._tensor(cfg.kp_att)
        self._ki_att = self._tensor(cfg.ki_att)
        self._kd_att = self._tensor(cfg.kd_att)
        self._kp_rate = self._tensor(cfg.kp_rate)
        self._ki_rate = self._tensor(cfg.ki_rate)
        self._kd_rate = self._tensor(cfg.kd_rate)
        self._torque_max = self._tensor(cfg.torque_max)

        self._velocity_dim = 2 if cfg.planar_mode else 3
        self._action_dim = self._velocity_dim + int(cfg.planar_mode)
        self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._velocity_command_target = torch.zeros(self.num_envs, 3, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 3, device=self.device)
        self._yaw_rate_command_target = torch.zeros(self.num_envs, device=self.device)
        self._processed_yaw_rate = torch.zeros(self.num_envs, device=self.device)
        self._desired_velocity_w = torch.zeros_like(self._processed_actions)
        self._desired_force_w = torch.zeros_like(self._processed_actions)
        self._desired_wrench_b = torch.zeros(self.num_envs, 6, device=self.device)
        self._achieved_wrench_b = torch.zeros_like(self._desired_wrench_b)

        self._velocity_integral_error = torch.zeros_like(self._processed_actions)
        self._attitude_integral_error = torch.zeros_like(self._processed_actions)
        self._rate_integral_error = torch.zeros_like(self._processed_actions)
        self._last_angular_velocity_b = torch.zeros_like(self._processed_actions)
        self._attitude_derivative_lpf = torch.zeros_like(self._processed_actions)
        self._rate_derivative_lpf = torch.zeros_like(self._processed_actions)
        self._target_yaw = torch.zeros(self.num_envs, device=self.device)
        self._target_altitude = torch.zeros(self.num_envs, device=self.device)

        self._allocation_matrix = self._build_allocation_matrix()
        if int(torch.linalg.matrix_rank(self._allocation_matrix).item()) != 6:
            raise ValueError("Tilting-UAV allocation matrix is not full wrench rank.")
        self._allocation_inverse = self._build_allocation_inverse()

        self._thrust_command = torch.zeros(self.num_envs, 4, device=self.device)
        self._thrust_actual = torch.zeros_like(self._thrust_command)
        self._motor_omega = torch.zeros_like(self._thrust_command)
        self._servo_angle_command = torch.zeros_like(self._thrust_command)
        self._servo_angle_actual = torch.zeros_like(self._thrust_command)
        self._gripper_position_target = self._asset.data.default_joint_pos[:, self._gripper_joint_ids].clone()

        self.reset()

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def velocity_command_target(self) -> torch.Tensor:
        return self._velocity_command_target

    @property
    def yaw_rate_command_target(self) -> torch.Tensor:
        return self._yaw_rate_command_target

    @property
    def processed_yaw_rate(self) -> torch.Tensor:
        return self._processed_yaw_rate

    @property
    def desired_velocity_w(self) -> torch.Tensor:
        """Current velocity target in the world frame."""
        return self._desired_velocity_w

    @property
    def desired_wrench_b(self) -> torch.Tensor:
        """Controller-requested body wrench before allocation and actuator dynamics."""
        return self._desired_wrench_b

    @property
    def achieved_wrench_b(self) -> torch.Tensor:
        """Body wrench produced after allocation limits and actuator dynamics."""
        return self._achieved_wrench_b

    @property
    def allocation_matrix(self) -> torch.Tensor:
        return self._allocation_matrix

    @property
    def thrust_command(self) -> torch.Tensor:
        return self._thrust_command

    @property
    def thrust_actual(self) -> torch.Tensor:
        return self._thrust_actual

    @property
    def servo_angle_command(self) -> torch.Tensor:
        return self._servo_angle_command

    @property
    def servo_angle_actual(self) -> torch.Tensor:
        return self._servo_angle_actual

    @property
    def target_yaw(self) -> torch.Tensor:
        return self._target_yaw

    @property
    def target_altitude(self) -> torch.Tensor:
        return self._target_altitude

    @property
    def velocity_integral_error(self) -> torch.Tensor:
        return self._velocity_integral_error

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape != self._raw_actions.shape:
            raise ValueError(f"Expected actions with shape {self._raw_actions.shape}, got {actions.shape}.")
        self._raw_actions[:] = actions
        bounded_actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        self._velocity_command_target[:, : self._velocity_dim] = (
            bounded_actions[:, : self._velocity_dim] * self._velocity_scale[: self._velocity_dim]
        )
        if self.cfg.planar_mode:
            self._velocity_command_target[:, 2] = 0.0
            self._yaw_rate_command_target[:] = bounded_actions[:, 2] * self.cfg.yaw_rate_scale
        if self.cfg.velocity_command_time_constant <= 0.0:
            self._processed_actions[:] = self._velocity_command_target
            self._processed_yaw_rate[:] = self._yaw_rate_command_target
        elif self.cfg.planar_mode:
            self._processed_actions[:, 2] = 0.0

    def apply_actions(self) -> None:
        root_quat_w = self._asset.data.root_quat_w
        root_lin_vel_w = self._asset.data.root_lin_vel_w
        root_ang_vel_b = self._asset.data.root_ang_vel_b

        if self.cfg.velocity_command_time_constant > 0.0:
            command_alpha = 1.0 - math.exp(
                -self._physics_dt / self.cfg.velocity_command_time_constant
            )
            self._processed_actions[:, : self._velocity_dim].add_(
                command_alpha
                * (
                    self._velocity_command_target[:, : self._velocity_dim]
                    - self._processed_actions[:, : self._velocity_dim]
                )
            )
            if self.cfg.planar_mode:
                self._processed_yaw_rate.add_(
                    command_alpha * (self._yaw_rate_command_target - self._processed_yaw_rate)
                )

        if self.cfg.planar_mode:
            altitude_error = self._target_altitude - self._asset.data.root_pos_w[:, 2]
            self._processed_actions[:, 2] = torch.clamp(
                self.cfg.altitude_hold_gain * altitude_error,
                -self.cfg.altitude_hold_max_velocity,
                self.cfg.altitude_hold_max_velocity,
            )
            self._target_yaw.add_(self._processed_yaw_rate * self._physics_dt)
            self._target_yaw[:] = math_utils.wrap_to_pi(self._target_yaw)

        self._desired_velocity_w[:] = math_utils.quat_apply_yaw(root_quat_w, self._processed_actions)
        velocity_error = self._desired_velocity_w - root_lin_vel_w
        self._velocity_integral_error.add_(velocity_error * self._physics_dt)
        self._velocity_integral_error.clamp_(-self.cfg.vel_integral_limit, self.cfg.vel_integral_limit)

        acceleration_command = self._kp_vel * velocity_error + self._ki_vel * self._velocity_integral_error
        self._desired_force_w[:] = self.cfg.mass * acceleration_command
        self._desired_force_w[:, 2].add_(self.cfg.mass * self.cfg.gravity)
        self._desired_force_w[:] = torch.maximum(
            torch.minimum(self._desired_force_w, self._force_max), self._force_min
        )
        force_norm = torch.linalg.norm(self._desired_force_w, dim=-1, keepdim=True)
        force_scale = torch.clamp(self.cfg.force_max / force_norm.clamp_min(1.0e-9), max=1.0)
        self._desired_force_w.mul_(force_scale)

        desired_force_b = math_utils.quat_apply_inverse(root_quat_w, self._desired_force_w)
        desired_torque_b = self._compute_attitude_torque(root_quat_w, root_ang_vel_b)
        self._desired_wrench_b[:, :3] = desired_force_b
        self._desired_wrench_b[:, 3:] = desired_torque_b

        allocation_coordinates = self._desired_wrench_b @ self._allocation_inverse.T
        p_coordinates = allocation_coordinates[:, 0::2]
        q_coordinates = allocation_coordinates[:, 1::2]
        self._thrust_command[:] = torch.hypot(p_coordinates, q_coordinates).clamp(
            self.cfg.thrust_min, self.cfg.thrust_max
        )
        self._servo_angle_command[:] = torch.atan2(p_coordinates, q_coordinates).clamp(
            -self.cfg.servo_angle_limit, self.cfg.servo_angle_limit
        )

        self._update_actuator_dynamics()
        actual_coordinates = torch.empty_like(allocation_coordinates)
        actual_coordinates[:, 0::2] = self._thrust_actual * torch.sin(self._servo_angle_actual)
        actual_coordinates[:, 1::2] = self._thrust_actual * torch.cos(self._servo_angle_actual)
        self._achieved_wrench_b[:] = actual_coordinates @ self._allocation_matrix.T

        self._asset.set_joint_position_target(self._servo_angle_actual, joint_ids=self._tilt_joint_ids)
        self._asset.set_joint_position_target(
            self._gripper_position_target, joint_ids=self._gripper_joint_ids
        )
        self._asset.instantaneous_wrench_composer.set_forces_and_torques(
            forces=self._achieved_wrench_b[:, None, :3],
            torques=self._achieved_wrench_b[:, None, 3:],
            body_ids=self._base_body_ids,
            is_global=False,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids_tensor = self._resolve_env_ids(env_ids)
        self._raw_actions[env_ids_tensor] = 0.0
        self._velocity_command_target[env_ids_tensor] = 0.0
        self._processed_actions[env_ids_tensor] = 0.0
        self._yaw_rate_command_target[env_ids_tensor] = 0.0
        self._processed_yaw_rate[env_ids_tensor] = 0.0
        self._desired_velocity_w[env_ids_tensor] = 0.0
        self._desired_force_w[env_ids_tensor] = 0.0
        self._desired_wrench_b[env_ids_tensor] = 0.0
        self._achieved_wrench_b[env_ids_tensor] = 0.0
        self._velocity_integral_error[env_ids_tensor] = 0.0
        self._attitude_integral_error[env_ids_tensor] = 0.0
        self._rate_integral_error[env_ids_tensor] = 0.0
        self._last_angular_velocity_b[env_ids_tensor] = 0.0
        self._attitude_derivative_lpf[env_ids_tensor] = 0.0
        self._rate_derivative_lpf[env_ids_tensor] = 0.0

        _, _, yaw = math_utils.euler_xyz_from_quat(self._asset.data.root_quat_w[env_ids_tensor])
        self._target_yaw[env_ids_tensor] = yaw
        self._target_altitude[env_ids_tensor] = self._asset.data.root_pos_w[env_ids_tensor, 2]

        hover_thrust = self.cfg.mass * self.cfg.gravity / 4.0
        hover_omega = math.sqrt(hover_thrust / self.cfg.rotor_kf)
        self._thrust_command[env_ids_tensor] = hover_thrust
        self._thrust_actual[env_ids_tensor] = hover_thrust
        self._motor_omega[env_ids_tensor] = hover_omega
        self._servo_angle_command[env_ids_tensor] = 0.0
        self._servo_angle_actual[env_ids_tensor] = 0.0
        self._gripper_position_target[env_ids_tensor] = self._asset.data.default_joint_pos[
            env_ids_tensor[:, None], self._gripper_joint_ids
        ]

        self._asset.set_joint_position_target(
            self._servo_angle_actual[env_ids_tensor],
            joint_ids=self._tilt_joint_ids,
            env_ids=env_ids_tensor,
        )
        self._asset.set_joint_position_target(
            self._gripper_position_target[env_ids_tensor],
            joint_ids=self._gripper_joint_ids,
            env_ids=env_ids_tensor,
        )

    def _compute_attitude_torque(
        self, root_quat_w: torch.Tensor, root_ang_vel_b: torch.Tensor
    ) -> torch.Tensor:
        zeros = torch.zeros_like(self._target_yaw)
        target_quat_w = math_utils.quat_from_euler_xyz(zeros, zeros, self._target_yaw)
        attitude_error_quat = math_utils.quat_mul(math_utils.quat_conjugate(root_quat_w), target_quat_w)
        attitude_error = math_utils.axis_angle_from_quat(attitude_error_quat)

        self._attitude_integral_error.add_(attitude_error * self._physics_dt)
        self._attitude_integral_error.clamp_(
            -self.cfg.att_integral_limit, self.cfg.att_integral_limit
        )
        self._update_lpf(
            self._attitude_derivative_lpf,
            root_ang_vel_b,
            self.cfg.att_derivative_lpf_cutoff_hz,
        )
        angular_velocity_command = (
            self._kp_att * attitude_error
            + self._ki_att * self._attitude_integral_error
            - self._kd_att * self._attitude_derivative_lpf
        )
        if self.cfg.planar_mode:
            angular_velocity_command[:, 2].add_(self._processed_yaw_rate)

        rate_error = angular_velocity_command - root_ang_vel_b
        angular_acceleration_feedback = (
            root_ang_vel_b - self._last_angular_velocity_b
        ) / max(self._physics_dt, 1.0e-6)
        self._last_angular_velocity_b[:] = root_ang_vel_b
        self._update_lpf(
            self._rate_derivative_lpf,
            angular_acceleration_feedback,
            self.cfg.rate_derivative_lpf_cutoff_hz,
        )
        self._rate_integral_error.add_(rate_error * self._physics_dt)
        self._rate_integral_error.clamp_(-self.cfg.rate_integral_limit, self.cfg.rate_integral_limit)

        angular_acceleration_command = (
            self._kp_rate * rate_error
            + self._ki_rate * self._rate_integral_error
            - self._kd_rate * self._rate_derivative_lpf
        )
        inertia_times_rate = self._inertia * root_ang_vel_b
        gyro_torque = torch.linalg.cross(root_ang_vel_b, inertia_times_rate, dim=-1)
        desired_torque = self._inertia * angular_acceleration_command + gyro_torque
        return torch.maximum(torch.minimum(desired_torque, self._torque_max), -self._torque_max)

    def _update_actuator_dynamics(self) -> None:
        servo_alpha = 1.0
        if self.cfg.servo_time_constant > 1.0e-6:
            servo_alpha = 1.0 - math.exp(-self._physics_dt / self.cfg.servo_time_constant)
        servo_next = self._servo_angle_actual + servo_alpha * (
            self._servo_angle_command - self._servo_angle_actual
        )
        if self.cfg.servo_rate_limit > 0.0:
            max_servo_step = self.cfg.servo_rate_limit * self._physics_dt
            servo_next = torch.maximum(
                torch.minimum(servo_next, self._servo_angle_actual + max_servo_step),
                self._servo_angle_actual - max_servo_step,
            )
        self._servo_angle_actual[:] = servo_next.clamp(
            -self.cfg.servo_angle_limit, self.cfg.servo_angle_limit
        )

        omega_command = torch.sqrt(
            self._thrust_command.clamp_min(0.0) / self.cfg.rotor_kf
        )
        omega_max = math.sqrt(self.cfg.thrust_max / self.cfg.rotor_kf)
        omega_command.clamp_(0.0, omega_max)
        if self.cfg.motor_time_constant > 0.0:
            motor_alpha = min(self._physics_dt / self.cfg.motor_time_constant, 1.0)
            self._motor_omega.add_(motor_alpha * (omega_command - self._motor_omega))
        else:
            self._motor_omega[:] = omega_command
        self._motor_omega.clamp_(0.0, omega_max)
        self._thrust_actual[:] = self.cfg.rotor_kf * self._motor_omega.square()

    def _build_allocation_matrix(self) -> torch.Tensor:
        rotor_positions = self._tensor(self.cfg.rotor_positions)
        rotor_alpha = torch.deg2rad(self._tensor(self.cfg.rotor_alpha_deg))
        servo_axes = torch.stack(
            (torch.cos(rotor_alpha), torch.sin(rotor_alpha), torch.zeros_like(rotor_alpha)), dim=-1
        )
        matrix = torch.zeros(6, 8, device=self.device)
        for rotor_index in range(4):
            force_from_p = torch.stack(
                (servo_axes[rotor_index, 1], -servo_axes[rotor_index, 0], torch.zeros((), device=self.device))
            )
            force_from_q = torch.tensor((0.0, 0.0, 1.0), device=self.device)
            p_index = 2 * rotor_index
            q_index = p_index + 1
            matrix[:3, p_index] = force_from_p
            matrix[:3, q_index] = force_from_q
            matrix[3:, p_index] = torch.linalg.cross(
                rotor_positions[rotor_index], force_from_p, dim=0
            )
            matrix[3:, q_index] = torch.linalg.cross(
                rotor_positions[rotor_index], force_from_q, dim=0
            )
        return matrix

    def _build_allocation_inverse(self) -> torch.Tensor:
        wrench_weights = torch.tensor(
            (self.cfg.allocation_weight_force,) * 3 + (self.cfg.allocation_weight_torque,) * 3,
            device=self.device,
        )
        weighted_transpose = self._allocation_matrix.T * wrench_weights.unsqueeze(0)
        normal_matrix = weighted_transpose @ self._allocation_matrix
        normal_matrix.add_(self.cfg.allocator_damping * torch.eye(8, device=self.device))
        return torch.linalg.solve(normal_matrix, weighted_transpose)

    def _update_lpf(self, state: torch.Tensor, measurement: torch.Tensor, cutoff_hz: float) -> None:
        if cutoff_hz <= 0.0:
            state[:] = measurement
            return
        time_constant = 1.0 / (2.0 * math.pi * cutoff_hz)
        alpha = self._physics_dt / (time_constant + self._physics_dt)
        state.add_(alpha * (measurement - state))

    def _resolve_env_ids(self, env_ids: Sequence[int] | None) -> torch.Tensor:
        if env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)):
            return torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.tensor(env_ids, dtype=torch.long, device=self.device)

    def _tensor(self, values) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.float32, device=self.device)


@configclass
class TiltingUAVVelocityActionCfg(ActionTermCfg):
    """Configuration matching the existing embedded tilting-UAV controller."""

    class_type: type[ActionTerm] = TiltingUAVVelocityAction

    planar_mode: bool = False
    altitude_hold_gain: float = 1.5
    altitude_hold_max_velocity: float = 0.8
    velocity_command_time_constant: float = 0.0
    velocity_scale: tuple[float, float, float] = (1.5, 1.5, 1.0)
    yaw_rate_scale: float = 1.0

    mass: float = 2.43
    gravity: float = 9.81
    inertia_diag: tuple[float, float, float] = (0.007648029, 0.014067495, 0.017081474)

    kp_vel: tuple[float, float, float] = (6.0, 6.0, 8.0)
    ki_vel: tuple[float, float, float] = (1.0, 1.0, 3.0)
    vel_integral_limit: float = 3.0

    kp_att: tuple[float, float, float] = (4.0, 5.0, 3.0)
    ki_att: tuple[float, float, float] = (0.0, 0.0, 0.0)
    kd_att: tuple[float, float, float] = (0.0, 0.0, 0.0)
    att_integral_limit: float = 0.5
    att_derivative_lpf_cutoff_hz: float = 0.0

    kp_rate: tuple[float, float, float] = (13.0, 18.0, 8.0)
    ki_rate: tuple[float, float, float] = (2.0, 4.0, 2.0)
    kd_rate: tuple[float, float, float] = (0.1, 0.1, 0.1)
    rate_integral_limit: float = 20.0
    rate_derivative_lpf_cutoff_hz: float = 30.0

    force_min_xyz: tuple[float, float, float] = (-20.0, -20.0, 5.0)
    force_max_xyz: tuple[float, float, float] = (20.0, 20.0, 40.0)
    force_max: float = 40.0
    torque_max: tuple[float, float, float] = (1.0, 1.0, 1.0)

    thrust_min: float = 0.01
    thrust_max: float = 18.0
    rotor_kf: float = 2.54e-6
    motor_time_constant: float = 0.05

    servo_angle_limit: float = math.pi
    servo_rate_limit: float = 15.0
    servo_time_constant: float = 0.087

    rotor_positions: tuple[tuple[float, float, float], ...] = (
        (0.09043, -0.12412, 0.0037),
        (-0.10293, 0.11085, 0.0037),
        (0.09043, 0.12412, 0.0037),
        (-0.10293, -0.11085, 0.0037),
    )
    rotor_alpha_deg: tuple[float, float, float, float] = (292.5, 135.0, 67.5, 225.0)
    allocation_weight_force: float = 10.0
    allocation_weight_torque: float = 1.0
    allocator_damping: float = 1.0e-3

    base_body_name: str = "base_link"
    tilt_joint_names: tuple[str, str, str, str] = (
        "joint_arm_1",
        "joint_arm_2",
        "joint_arm_3",
        "joint_arm_4",
    )
    gripper_joint_names: tuple[str, str] = (
        "left_left_finger",
        "left_right_finger",
    )
