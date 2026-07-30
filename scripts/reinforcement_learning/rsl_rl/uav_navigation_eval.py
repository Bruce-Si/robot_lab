# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Deterministic checkpoint evaluation for tilting-UAV navigation."""

from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import check_nan

from robot_lab.tasks.manager_based.navigation.mdp.uav_navigation import set_uav_navigation_cells


EDGE_NAMES = ("left", "right", "bottom", "top")
TERMINATION_TERMS = (
    "time_out",
    "contact_collision",
    "lidar_collision",
    "altitude",
    "tilt",
    "out_of_bounds",
    "goal_reached",
)
FAILURE_TERMS = (
    "contact_collision",
    "lidar_collision",
    "altitude",
    "tilt",
    "out_of_bounds",
)
PRIMARY_OUTCOMES = (
    "success",
    "contact_collision",
    "lidar_collision",
    "altitude",
    "tilt",
    "out_of_bounds",
    "time_out",
    "unknown",
)


def _capture_rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    torch.backends.cudnn.benchmark = state["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = state["cudnn_deterministic"]


def _seed_evaluation(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _count_and_rate(count: int, total: int) -> dict[str, int | float]:
    return {"count": count, "rate": count / total if total > 0 else 0.0}


def evaluate_uav_navigation(
    env,
    policy,
    *,
    cell_index: int,
    episodes_per_env: int = 1,
    seed: int = 42,
    cell_size: float = 50.0,
    num_rows: int = 8,
    num_cols: int = 8,
    restore_training_cells: bool = False,
    progress_interval_steps: int = 100,
) -> dict[str, Any]:
    """Run deterministic episodes with every environment in one USD scene cell.

    Isaac Lab resets completed environments inside ``step``. Termination signals,
    start edges, and goal edges are therefore captured for the just-completed
    episode before accepting the freshly reset state.
    """
    if episodes_per_env <= 0:
        raise ValueError("episodes_per_env must be positive.")
    num_cells = num_rows * num_cols
    if cell_index < 0 or cell_index >= num_cells:
        raise ValueError(f"cell_index must be in [0, {num_cells - 1}], got {cell_index}.")

    raw_env = env.unwrapped
    if not hasattr(raw_env, "termination_manager"):
        raise TypeError("UAV evaluation requires a ManagerBasedRLEnv termination manager.")
    missing_terms = set(TERMINATION_TERMS) - set(raw_env.termination_manager.active_terms)
    if missing_terms:
        raise ValueError(f"UAV evaluation termination terms are missing: {sorted(missing_terms)}")

    original_cells = getattr(raw_env, "_uav_navigation_cell_index", None)
    if restore_training_cells:
        if original_cells is None:
            raise RuntimeError("Training cell mapping is unavailable and cannot be restored after evaluation.")
        original_cells = original_cells.clone()
        rng_state = _capture_rng_state()
    else:
        rng_state = None

    num_envs = env.num_envs
    total_trials = num_envs * episodes_per_env
    device = torch.device(raw_env.device)
    episode_counts = torch.zeros(num_envs, device=device, dtype=torch.long)
    current_returns = torch.zeros(num_envs, device=device)
    current_lengths = torch.zeros(num_envs, device=device, dtype=torch.long)
    episode_returns = torch.zeros(num_envs, episodes_per_env, device=device)
    episode_lengths = torch.zeros(num_envs, episodes_per_env, device=device, dtype=torch.long)
    signal_counts = {name: torch.zeros((), device=device, dtype=torch.long) for name in TERMINATION_TERMS}
    primary_counts = {name: torch.zeros((), device=device, dtype=torch.long) for name in PRIMARY_OUTCOMES}
    collision_count = torch.zeros((), device=device, dtype=torch.long)
    edge_trials = torch.zeros(16, device=device, dtype=torch.long)
    edge_successes = torch.zeros_like(edge_trials)
    edge_collisions = torch.zeros_like(edge_trials)

    completed_trials = 0
    rollout_steps = 0
    started_at = datetime.now(timezone.utc)
    wall_start = time.time()
    try:
        _seed_evaluation(seed)
        set_uav_navigation_cells(
            raw_env,
            cell_indices=cell_index,
            cell_size=cell_size,
            num_rows=num_rows,
            num_cols=num_cols,
        )
        with torch.inference_mode():
            obs, _ = env.reset()
            policy.reset(torch.ones(num_envs, device=device, dtype=torch.bool))

        if not hasattr(raw_env, "_nav_start_edge") or not hasattr(raw_env, "_nav_goal_edge"):
            raise RuntimeError("Navigation start/goal edge buffers were not created during reset.")

        max_rollout_steps = int(env.max_episode_length) * episodes_per_env
        while completed_trials < total_trials and rollout_steps < max_rollout_steps:
            active = episode_counts < episodes_per_env
            start_edges = raw_env._nav_start_edge.clone()
            goal_edges = raw_env._nav_goal_edge.clone()

            with torch.inference_mode():
                actions = policy(obs)
                actions[~active] = 0.0
                obs, rewards, dones, _ = env.step(actions)
                policy.reset(dones)

            current_returns[active] += rewards[active]
            current_lengths[active] += 1
            done_active = dones.bool() & active
            done_ids = done_active.nonzero(as_tuple=False).squeeze(-1)
            rollout_steps += 1

            if done_ids.numel() > 0:
                slots = episode_counts[done_ids]
                episode_returns[done_ids, slots] = current_returns[done_ids]
                episode_lengths[done_ids, slots] = current_lengths[done_ids]

                terms = {
                    name: raw_env.termination_manager.get_term(name)[done_ids]
                    for name in TERMINATION_TERMS
                }
                for name, values in terms.items():
                    signal_counts[name] += values.sum()

                any_failure = torch.zeros_like(terms[FAILURE_TERMS[0]])
                for name in FAILURE_TERMS:
                    any_failure |= terms[name]
                strict_success = terms["goal_reached"] & ~any_failure
                any_collision = terms["contact_collision"] | terms["lidar_collision"]
                collision_count += any_collision.sum()

                remaining = torch.ones_like(strict_success)
                outcome_masks = {
                    "success": strict_success,
                    "contact_collision": terms["contact_collision"],
                    "lidar_collision": terms["lidar_collision"],
                    "altitude": terms["altitude"],
                    "tilt": terms["tilt"],
                    "out_of_bounds": terms["out_of_bounds"],
                    "time_out": terms["time_out"],
                }
                for name, mask in outcome_masks.items():
                    selected = remaining & mask
                    primary_counts[name] += selected.sum()
                    remaining &= ~selected
                primary_counts["unknown"] += remaining.sum()

                pair_indices = start_edges[done_ids] * len(EDGE_NAMES) + goal_edges[done_ids]
                edge_trials += torch.bincount(pair_indices, minlength=16)
                edge_successes += torch.bincount(pair_indices[strict_success], minlength=16)
                edge_collisions += torch.bincount(pair_indices[any_collision], minlength=16)

                episode_counts[done_ids] += 1
                current_returns[done_ids] = 0.0
                current_lengths[done_ids] = 0
                completed_trials += done_ids.numel()

            if progress_interval_steps > 0 and (
                rollout_steps % progress_interval_steps == 0 or completed_trials == total_trials
            ):
                print(
                    f"[eval] cell={cell_index} step={rollout_steps}/{max_rollout_steps} "
                    f"completed={completed_trials}/{total_trials}"
                )

        if completed_trials != total_trials:
            raise RuntimeError(
                f"Evaluation completed {completed_trials}/{total_trials} trials after "
                f"{rollout_steps} environment steps."
            )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        wall_time = time.time() - wall_start
        return_values = episode_returns.flatten()
        length_values = episode_lengths.flatten().to(torch.float32)
        signal_values = {name: int(value.item()) for name, value in signal_counts.items()}
        primary_values = {name: int(value.item()) for name, value in primary_counts.items()}
        collision_value = int(collision_count.item())
        edge_pair_results = []
        edge_trials_cpu = edge_trials.cpu().tolist()
        edge_successes_cpu = edge_successes.cpu().tolist()
        edge_collisions_cpu = edge_collisions.cpu().tolist()
        for start_index, start_name in enumerate(EDGE_NAMES):
            for goal_index, goal_name in enumerate(EDGE_NAMES):
                pair_index = start_index * len(EDGE_NAMES) + goal_index
                trials = edge_trials_cpu[pair_index]
                if trials == 0:
                    continue
                successes = edge_successes_cpu[pair_index]
                collisions = edge_collisions_cpu[pair_index]
                edge_pair_results.append(
                    {
                        "start_edge": start_name,
                        "goal_edge": goal_name,
                        "trials": trials,
                        "successes": successes,
                        "success_rate": successes / trials,
                        "collisions": collisions,
                        "collision_rate": collisions / trials,
                    }
                )

        result = {
            "schema_version": 1,
            "evaluation_started_at": started_at.isoformat(),
            "evaluation_finished_at": datetime.now(timezone.utc).isoformat(),
            "policy_mode": "deterministic_mean",
            "seed": seed,
            "scene_cell": {
                "index": cell_index,
                "level": cell_index // num_cols,
                "variant": cell_index % num_cols,
                "rows": num_rows,
                "columns": num_cols,
                "cell_size_m": cell_size,
            },
            "num_envs": num_envs,
            "episodes_per_env": episodes_per_env,
            "trials": total_trials,
            "rollout_steps": rollout_steps,
            "step_dt_s": float(raw_env.step_dt),
            "wall_time_s": wall_time,
            "throughput_env_steps_per_second": rollout_steps * num_envs / max(wall_time, 1.0e-9),
            "outcomes": {
                name: _count_and_rate(primary_values[name], total_trials) for name in PRIMARY_OUTCOMES
            },
            "termination_signals": {
                name: _count_and_rate(signal_values[name], total_trials) for name in TERMINATION_TERMS
            },
            "any_collision": _count_and_rate(collision_value, total_trials),
            "episode": {
                "mean_return": float(return_values.mean().item()),
                "std_return": float(return_values.std(unbiased=False).item()),
                "mean_length_steps": float(length_values.mean().item()),
                "std_length_steps": float(length_values.std(unbiased=False).item()),
                "mean_length_seconds": float(length_values.mean().item() * raw_env.step_dt),
            },
            "edge_pairs": edge_pair_results,
        }
        return result
    finally:
        if restore_training_cells:
            _restore_rng_state(rng_state)
            set_uav_navigation_cells(
                raw_env,
                cell_indices=original_cells,
                cell_size=cell_size,
                num_rows=num_rows,
                num_cols=num_cols,
            )
            with torch.inference_mode():
                env.reset()
            raw_env.extras["log"] = {}


def write_evaluation_json(result: dict[str, Any], output_path: str | os.PathLike[str]) -> Path:
    """Write an evaluation result as stable, human-readable JSON."""
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return path


def print_evaluation_summary(result: dict[str, Any], output_path: Path | None = None) -> None:
    """Print the high-signal metrics from an evaluation result."""
    cell = result["scene_cell"]
    outcomes = result["outcomes"]
    collision = result["any_collision"]
    episode = result["episode"]
    print("=" * 72)
    print("Tilting UAV Navigation Evaluation")
    print("=" * 72)
    print(
        f"cell={cell['index']} (level={cell['level']}, variant={cell['variant']}) "
        f"envs={result['num_envs']} trials={result['trials']} seed={result['seed']}"
    )
    print(
        f"success={outcomes['success']['count']}/{result['trials']} "
        f"({100.0 * outcomes['success']['rate']:.2f}%) "
        f"collision={collision['count']}/{result['trials']} "
        f"({100.0 * collision['rate']:.2f}%)"
    )
    print(
        f"timeout={outcomes['time_out']['count']} altitude={outcomes['altitude']['count']} "
        f"tilt={outcomes['tilt']['count']} out_of_bounds={outcomes['out_of_bounds']['count']}"
    )
    print(
        f"mean_return={episode['mean_return']:.3f} "
        f"mean_episode={episode['mean_length_steps']:.1f} steps "
        f"({episode['mean_length_seconds']:.2f}s)"
    )
    print(
        f"wall_time={result['wall_time_s']:.2f}s "
        f"throughput={result['throughput_env_steps_per_second']:.1f} env-steps/s"
    )
    if output_path is not None:
        print(f"json={output_path}")
    print("=" * 72)


def log_evaluation_scalars(writer, result: dict[str, Any], iteration: int) -> None:
    """Append evaluation metrics to the active RSL-RL logger."""
    if writer is None:
        return
    writer.add_scalar("Eval/success_rate", result["outcomes"]["success"]["rate"], iteration)
    writer.add_scalar("Eval/collision_rate", result["any_collision"]["rate"], iteration)
    writer.add_scalar("Eval/timeout_rate", result["outcomes"]["time_out"]["rate"], iteration)
    writer.add_scalar("Eval/mean_return", result["episode"]["mean_return"], iteration)
    writer.add_scalar("Eval/mean_episode_length", result["episode"]["mean_length_steps"], iteration)
    writer.flush()


class PeriodicEvaluationOnPolicyRunner(OnPolicyRunner):
    """RSL-RL runner that refreshes observations after in-process evaluation."""

    def __init__(
        self,
        *args,
        evaluation_interval: int,
        evaluation_callback: Callable[[OnPolicyRunner, int], None],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if evaluation_interval <= 0:
            raise ValueError("evaluation_interval must be positive.")
        self.evaluation_interval = evaluation_interval
        self.evaluation_callback = evaluation_callback

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run PPO and evaluate after each configured nonzero iteration."""
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                stop = time.time()
                collect_time = stop - start
                start = stop
                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
            )

            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

            if it > 0 and it % self.evaluation_interval == 0:
                self.evaluation_callback(self, it)
                self.alg.train_mode()
                reset_mask = torch.ones(self.env.num_envs, device=self.device, dtype=torch.bool)
                self.alg.actor.reset(reset_mask)
                self.alg.critic.reset(reset_mask)
                self.logger.cur_reward_sum.zero_()
                self.logger.cur_episode_length.zero_()
                if self.cfg["algorithm"]["rnd_cfg"]:
                    self.logger.cur_ereward_sum.zero_()
                    self.logger.cur_ireward_sum.zero_()
                with torch.inference_mode():
                    obs = self.env.get_observations().to(self.device)

        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
            self.logger.stop_logging_writer()
