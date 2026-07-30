# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a tilting-UAV RSL-RL checkpoint in one static USD scene cell."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


ROBOT_LAB_ROOT = Path(__file__).resolve().parents[3]
ENVIRONMENT_ASSET_DIR = ROBOT_LAB_ROOT / "source/robot_lab/data/environments"

parser = argparse.ArgumentParser(description="Evaluate a tilting-UAV navigation checkpoint.")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of parallel evaluation UAVs.")
parser.add_argument(
    "--task",
    type=str,
    default="RobotLab-Navigation-Tilting-UAV-v0",
    help="Registered tilting-UAV navigation task.",
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="RSL-RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=42, help="Seed for repeatable start and goal samples.")
parser.add_argument("--load_scene", type=str, default=None, help="Path to an explicit navigation USD scene.")
parser.add_argument(
    "--full_grid_scene",
    action="store_true",
    help="Load the original 8-by-8 scene instead of the derived single-cell scene.",
)
parser.add_argument("--terrain_level", type=int, default=7, help="USD grid row to evaluate.")
parser.add_argument("--terrain_variant", type=int, default=7, help="USD grid column to evaluate.")
parser.add_argument("--episodes_per_env", type=int, default=1, help="Completed episodes to collect per UAV.")
parser.add_argument("--output", type=str, default=None, help="Optional evaluation JSON path.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.checkpoint is None:
    parser.error("--checkpoint is required.")
if args_cli.num_envs <= 0:
    parser.error("--num_envs must be positive.")
if not 0 <= args_cli.terrain_level < 8:
    parser.error("--terrain_level must be in [0, 7].")
if not 0 <= args_cli.terrain_variant < 8:
    parser.error("--terrain_variant must be in [0, 7].")
if args_cli.episodes_per_env <= 0:
    parser.error("--episodes_per_env must be positive.")
if args_cli.full_grid_scene and args_cli.load_scene:
    parser.error("--full_grid_scene and --load_scene cannot be used together.")

if not args_cli.full_grid_scene and args_cli.load_scene is None:
    single_cell_scene = (
        ENVIRONMENT_ASSET_DIR
        / f"uav_eval_cell_{args_cli.terrain_level}_{args_cli.terrain_variant}_baked_round_prims.usd"
    )
    if not single_cell_scene.is_file():
        parser.error(
            f"Derived single-cell scene not found: {single_cell_scene}. "
            "Generate it with scripts/tools/extract_navrl_usd_cell.py or pass --full_grid_scene."
        )
    args_cli.load_scene = str(single_cell_scene)

if args_cli.load_scene is not None:
    scene_path = str(Path(args_cli.load_scene).expanduser().resolve())
    os.environ["NAVRL_UAV_USD_SCENE"] = scene_path
    os.environ["NAVRL_USD_SCENE"] = scene_path
    print(f"[INFO] Using selected navigation scene: {scene_path}")
else:
    print("[INFO] Using the configured full-grid navigation scene.")

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs after Isaac Sim starts."""

import importlib.metadata as metadata

import gymnasium as gym
from packaging import version
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401  # isort: skip

from uav_navigation_eval import (  # isort: skip
    evaluate_uav_navigation,
    print_evaluation_summary,
    write_evaluation_json,
)


installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse("3.0.1"):
    raise RuntimeError(f"RSL-RL >= 3.0.1 is required, found {installed_version}.")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: RslRlBaseRunnerCfg):
    """Load one checkpoint and evaluate deterministic actions."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    checkpoint_path = Path(retrieve_file_path(args_cli.checkpoint)).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    env_cfg.log_dir = str(checkpoint_path.parent)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        print(f"[INFO] Loading model checkpoint: {checkpoint_path}")
        runner.load(str(checkpoint_path))
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        cell_index = args_cli.terrain_level * 8 + args_cli.terrain_variant
        result = evaluate_uav_navigation(
            env,
            policy,
            cell_index=cell_index,
            episodes_per_env=args_cli.episodes_per_env,
            seed=args_cli.seed,
        )
        result["task"] = args_cli.task
        result["checkpoint"] = str(checkpoint_path)
        result["checkpoint_iteration"] = runner.current_learning_iteration
        result["scene_mode"] = "full_grid" if args_cli.full_grid_scene else "single_cell"
        result["scene_usd"] = str(env.unwrapped.cfg.scene.usd_scene.spawn.usd_path)

        if args_cli.output:
            output_path = Path(args_cli.output)
        else:
            scene_tag = "full_grid" if args_cli.full_grid_scene else "single"
            output_path = (
                checkpoint_path.parent
                / "evaluations"
                / (
                    f"{checkpoint_path.stem}_{scene_tag}_cell_{cell_index:02d}_seed_{args_cli.seed}"
                    f"_episodes_{args_cli.episodes_per_env}.json"
                )
            )
        output_path = write_evaluation_json(result, output_path)
        print_evaluation_summary(result, output_path)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
