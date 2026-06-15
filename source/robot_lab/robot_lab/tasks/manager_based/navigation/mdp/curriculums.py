# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Curriculum: advance difficulty on consecutive goal reaches."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .debug_vis import _ensure_usd_terrain_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_by_goal_reached(
    env: "ManagerBasedRLEnv", env_ids: torch.Tensor, success_threshold: int = 3, failure_threshold: int = 3
) -> torch.Tensor:
    if env.scene.terrain is None or getattr(env.scene.terrain, "is_usd_curriculum", False):
        terrain = _ensure_usd_terrain_state(env)
    else:
        terrain = env.scene.terrain

    # Init tracking state
    if not hasattr(env, "_nav_success"):
        n = env.num_envs
        env._nav_success = torch.zeros(n, device=env.device)
        env._nav_failure = torch.zeros(n, device=env.device)

    # Which envs just finished?
    t = getattr(env, "reset_terminated", None)
    to = getattr(env, "reset_time_outs", None)
    if t is None or to is None:
        return torch.tensor(0.0, device=env.device)
    done = t.bool() | to.bool()
    if not done.any():
        return terrain.terrain_levels.float().mean()

    # Which ended due to reaching goal?
    cmd = env.command_manager.get_term("pose_command")
    rpos = env.scene["robot"].data.root_pos_w
    dist = torch.norm(cmd.pos_command_w[:, :2] - rpos[:, :2], dim=1)
    reached = (dist < 1.0) & done

    # Update counters
    env._nav_success[reached] += 1
    env._nav_success[~reached & done] = 0
    env._nav_failure[done & ~reached] += 1
    env._nav_failure[reached] = 0

    cur = terrain.terrain_levels.clone()
    max_lv = terrain.max_terrain_level - 1

    # Level up
    up = env._nav_success >= success_threshold
    cur[up] = torch.clamp(cur[up] + 1, max=max_lv)
    env._nav_success[up] = 0

    # Level down
    dn = env._nav_failure >= failure_threshold
    cur[dn] = torch.clamp(cur[dn] - 1, min=0)
    env._nav_failure[dn] = 0

    # Apply changes
    changed = cur != terrain.terrain_levels
    if changed.any():
        terrain.terrain_levels[:] = cur
        c = changed.nonzero(as_tuple=False).squeeze(-1)
        terrain.env_origins[c] = terrain.terrain_origins[cur[c].long(), terrain.terrain_types[c].long()]

    return cur.float().mean()
