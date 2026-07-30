# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument("--load_scene", type=str, default=None, help="Path to a USD scene file to load.")
parser.add_argument(
    "--eval_interval",
    type=int,
    default=0,
    help="Run deterministic UAV evaluation every N learning iterations; zero disables it.",
)
parser.add_argument("--eval_level", type=int, default=7, help="USD grid row used for periodic UAV evaluation.")
parser.add_argument("--eval_variant", type=int, default=7, help="USD grid column used for periodic UAV evaluation.")
parser.add_argument("--eval_seed", type=int, default=42, help="Fixed seed used for comparable periodic evaluations.")
parser.add_argument(
    "--eval_episodes_per_env", type=int, default=1, help="Completed evaluation episodes per environment."
)
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.task is None:
    parser.error(
        "--task is required. Example: --task=RobotLab-Navigation-Go2-v0 "
        "--num_envs 1024 --load_scene source/robot_lab/data/environments/loco_navi_v1.usd"
    )
if args_cli.eval_interval < 0:
    parser.error("--eval_interval cannot be negative.")
if not 0 <= args_cli.eval_level < 8:
    parser.error("--eval_level must be in [0, 7].")
if not 0 <= args_cli.eval_variant < 8:
    parser.error("--eval_variant must be in [0, 7].")
if args_cli.eval_episodes_per_env <= 0:
    parser.error("--eval_episodes_per_env must be positive.")

# Pass USD scene path via env var before hydra loads config
if args_cli.load_scene:
    import os as _os_train
    _os_train.environ["NAVRL_USD_SCENE"] = args_cli.load_scene
    # Also set obstacle JSON path (derived from USD path)
    json_path = args_cli.load_scene.replace(".usda", "_obstacles.json").replace(".usd", "_obstacles.json")
    _os_train.environ["NAVRL_OBSTACLE_JSON"] = json_path

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import logging
import os
import time
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401  # isort: skip

from uav_navigation_eval import (  # isort: skip
    PeriodicEvaluationOnPolicyRunner,
    evaluate_uav_navigation,
    log_evaluation_scalars,
    print_evaluation_summary,
    write_evaluation_json,
)

# import logger
logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    if args_cli.eval_interval > 0:
        if args_cli.distributed:
            raise ValueError("Periodic UAV evaluation currently supports single-process training only.")
        if args_cli.task.split(":")[-1] != "RobotLab-Navigation-Tilting-UAV-v0":
            raise ValueError("--eval_interval is currently implemented only for the tilting-UAV navigation task.")
        if agent_cfg.class_name != "OnPolicyRunner":
            raise ValueError("Periodic UAV evaluation requires an OnPolicyRunner agent.")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not
    # change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    eval_cell_index = args_cli.eval_level * 8 + args_cli.eval_variant

    def run_periodic_evaluation(eval_runner: OnPolicyRunner, iteration: int) -> None:
        checkpoint_path = os.path.join(log_dir, f"model_{iteration}.pt")
        if iteration % eval_runner.cfg["save_interval"] != 0 or not os.path.isfile(checkpoint_path):
            eval_runner.save(checkpoint_path)

        print(
            f"[INFO] Starting deterministic UAV evaluation at iteration {iteration}: "
            f"cell={eval_cell_index} (level={args_cli.eval_level}, variant={args_cli.eval_variant}), "
            f"envs={env.num_envs}, episodes_per_env={args_cli.eval_episodes_per_env}."
        )
        policy = eval_runner.get_inference_policy(device=env.unwrapped.device)
        result = evaluate_uav_navigation(
            env,
            policy,
            cell_index=eval_cell_index,
            episodes_per_env=args_cli.eval_episodes_per_env,
            seed=args_cli.eval_seed,
            restore_training_cells=True,
        )
        result["task"] = args_cli.task
        result["checkpoint"] = os.path.abspath(checkpoint_path)
        result["checkpoint_iteration"] = iteration
        output_path = write_evaluation_json(
            result,
            os.path.join(
                log_dir,
                "evaluations",
                f"iteration_{iteration:06d}_cell_{eval_cell_index:02d}.json",
            ),
        )
        log_evaluation_scalars(eval_runner.logger.writer, result, iteration)
        print_evaluation_summary(result, output_path)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        if args_cli.eval_interval > 0:
            runner = PeriodicEvaluationOnPolicyRunner(
                env,
                agent_cfg.to_dict(),
                log_dir=log_dir,
                device=agent_cfg.device,
                evaluation_interval=args_cli.eval_interval,
                evaluation_callback=run_periodic_evaluation,
            )
        else:
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    if args_cli.eval_interval > 0:
        print(
            f"[INFO] Periodic UAV evaluation enabled: interval={args_cli.eval_interval}, "
            f"cell={eval_cell_index}, seed={args_cli.eval_seed}, "
            f"episodes_per_env={args_cli.eval_episodes_per_env}."
        )
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
