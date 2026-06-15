# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""RSL-RL PPO runner configuration for NavRL navigation policy with CNNModel."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlCNNModelCfg,
)


@configclass
class NavRLPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "unitree_go2_navrl"
    clip_actions = 1.0

    # Observation groups: 1D "policy" (state) + 2D "lidar"
    obs_groups = {
        "actor": ["policy", "lidar"],
        "critic": ["policy", "lidar"],
    }

    actor = RslRlCNNModelCfg(
        class_name="CNNModel",
        hidden_dims=[256, 256],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(
            init_std=1.0,
            std_type="scalar",
        ),
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[4, 16, 32, 32],
            kernel_size=[5, 5, 3, 3],
            stride=[1, 2, 2, 2],
            padding="zeros",
            activation="elu",
            flatten=True,
        ),
    )

    critic = RslRlCNNModelCfg(
        class_name="CNNModel",
        hidden_dims=[256, 256],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=None,
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[4, 16, 32, 32],
            kernel_size=[5, 5, 3, 3],
            stride=[1, 2, 2, 2],
            padding="zeros",
            activation="elu",
            flatten=True,
        ),
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
        share_cnn_encoders=True,
    )
