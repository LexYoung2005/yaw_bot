# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-Yaw-Bot-Predictive-Gated-Direct-v0",
    entry_point=f"{__name__}.yaw_bot_env:YawBotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.yaw_bot_env_cfg:YawBotPredictiveGatedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PredictiveGatedPPORunnerCfg",
    },
)


gym.register(
    id="Template-Yaw-Bot-Outer-Only-PPO-Direct-v0",
    entry_point=f"{__name__}.yaw_bot_env:YawBotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.yaw_bot_env_cfg:YawBotOuterOnlyPPOEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PredictiveGatedPPORunnerCfg",
    },
)


gym.register(
    id="Template-Yaw-Bot-Uniform-Reward-PPO-Direct-v0",
    entry_point=f"{__name__}.yaw_bot_env:YawBotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.yaw_bot_env_cfg:YawBotUniformRewardPPOEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PredictiveGatedPPORunnerCfg",
    },
)


gym.register(
    id="Template-Yaw-Bot-Static-Reward-PPO-Direct-v0",
    entry_point=f"{__name__}.yaw_bot_env:YawBotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.yaw_bot_env_cfg:YawBotStaticRewardPPOEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PredictiveGatedPPORunnerCfg",
    },
)


gym.register(
    id="Template-Yaw-Bot-LIRPG-PPO-Direct-v0",
    entry_point=f"{__name__}.yaw_bot_env:YawBotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.yaw_bot_env_cfg:YawBotLIRPGPPOEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LIRPGPPORunnerCfg",
    },
)


gym.register(
    id="Template-Yaw-Bot-ReLara-PPO-Direct-v0",
    entry_point=f"{__name__}.yaw_bot_env:YawBotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.yaw_bot_env_cfg:YawBotReLaraPPOEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PredictiveGatedPPORunnerCfg",
    },
)
