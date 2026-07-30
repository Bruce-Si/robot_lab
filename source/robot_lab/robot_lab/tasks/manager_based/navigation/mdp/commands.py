# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Command generators for NavRL navigation."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.commands.pose_2d_command import UniformPose2dCommand
from isaaclab.utils.math import wrap_to_pi


class NonStartEdgePose2dCommand(UniformPose2dCommand):
    """Sample the goal on one of the three cell edges different from the reset start edge."""

    def _resample_command(self, env_ids: Sequence[int]):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)

        self.pos_command_w[env_ids] = self._env.scene.env_origins[env_ids]

        num_envs = len(env_ids)
        edge = getattr(self._env, "_nav_start_edge", None)
        if edge is None:
            edge = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        edge = edge[env_ids]

        edge_offset = 22.0
        lateral = torch.empty(num_envs, device=self.device).uniform_(-18.0, 18.0)
        local_xy = torch.zeros(num_envs, 2, device=self.device)

        # Edge convention from reset: 0 left, 1 right, 2 bottom, 3 top.
        # Pick uniformly from the three edges that are not the start edge.
        goal_edge = (edge + torch.randint(1, 4, (num_envs,), device=self.device)) % 4
        if not hasattr(self._env, "_nav_goal_edge"):
            self._env._nav_goal_edge = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._env._nav_goal_edge[env_ids] = goal_edge

        local_xy[:, 0] = torch.where(
            goal_edge == 0, -edge_offset, torch.where(goal_edge == 1, edge_offset, lateral)
        )
        local_xy[:, 1] = torch.where(
            goal_edge == 2, -edge_offset, torch.where(goal_edge == 3, edge_offset, lateral)
        )

        self.pos_command_w[env_ids, 0:2] += local_xy
        self.pos_command_w[env_ids, 2] += self.robot.data.default_root_state[env_ids, 2]

        if self.cfg.simple_heading:
            target_vec = self.pos_command_w[env_ids] - self.robot.data.root_pos_w[env_ids]
            target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])
            flipped_target_direction = wrap_to_pi(target_direction + torch.pi)

            curr_to_target = wrap_to_pi(target_direction - self.robot.data.heading_w[env_ids]).abs()
            curr_to_flipped_target = wrap_to_pi(flipped_target_direction - self.robot.data.heading_w[env_ids]).abs()

            self.heading_command_w[env_ids] = torch.where(
                curr_to_target < curr_to_flipped_target,
                target_direction,
                flipped_target_direction,
            )
        else:
            self.heading_command_w[env_ids] = torch.empty(num_envs, device=self.device).uniform_(
                *self.cfg.ranges.heading
            )


class OppositeEdgePose2dCommand(NonStartEdgePose2dCommand):
    """Backward-compatible alias for configs that still reference the old class name."""


class TargetFacingNonStartEdgePose2dCommand(NonStartEdgePose2dCommand):
    """Sample edge goals while continuously pointing the heading at the target.

    The base command may flip the heading by 180 degrees so a legged robot can walk
    backward instead of turning around. The UAV carries a forward-facing gripper, so
    its desired heading follows the live target bearing throughout the approach.
    """

    def _update_command(self):
        target_vec = self.pos_command_w - self.robot.data.root_pos_w
        target_distance = torch.linalg.norm(target_vec[:, :2], dim=1)
        target_heading = torch.atan2(target_vec[:, 1], target_vec[:, 0])
        self.heading_command_w[:] = torch.where(
            target_distance > 1.0e-6,
            target_heading,
            self.heading_command_w,
        )
        super()._update_command()
