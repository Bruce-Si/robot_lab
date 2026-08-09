# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Register tilting-UAV navigation environments."""

import gymnasium as gym

from . import agents


gym.register(
    id="RobotLab-Navigation-Tilting-UAV-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:TiltingUAVNavigationEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:TiltingUAVNavPPORunnerCfg"
        ),
    },
)


gym.register(
    id="RobotLab-Navigation-Tilting-UAV-Warehouse-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.navigation_env_cfg:TiltingUAVWarehouseNavigationEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:TiltingUAVWarehouseNavPPORunnerCfg"
        ),
    },
)


gym.register(
    id="RobotLab-Navigation-Tilting-UAV-Grasp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.navigation_grasp_env_cfg:TiltingUAVNavigationGraspEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:TiltingUAVNavPPORunnerCfg"
        ),
    },
)
