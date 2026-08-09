# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""RSL-RL PPO configuration for the privileged UAV data-collection expert."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class TiltingUAVNavPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 5000
    save_interval = 100
    experiment_name = "tilting_uav_navrl_planar_yaw_oracle"
    clip_actions = 1.0
    obs_groups = {
        "actor": ["policy", "privileged"],
        "critic": ["policy", "privileged"],
    }

    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.3,
            std_type="scalar",
        ),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class TiltingUAVWarehouseNavPPORunnerCfg(TiltingUAVNavPPORunnerCfg):
    """PPO run namespace for the single-scene warehouse data-collection expert."""

    experiment_name = "tilting_uav_warehouse_navrl_planar_yaw_oracle"
