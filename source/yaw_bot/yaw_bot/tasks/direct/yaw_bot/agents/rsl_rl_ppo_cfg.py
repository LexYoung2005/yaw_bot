# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PredictiveGatedActorCriticCfg(RslRlPpoActorCriticCfg):
    """Legacy v2-v4 multi-critic configuration retained for checkpoint tests."""

    class_name: str = "MultiCriticActorCritic"
    num_value_heads: int = 4


@configclass
class PredictiveGatedPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Legacy v2-v4 routed-PPO configuration retained for compatibility."""

    class_name: str = "PredictiveGatedPPO"
    num_tiers: int = 4


@configclass
class BoundedStdActorCriticCfg(RslRlPpoActorCriticCfg):
    """Single-critic actor with a trust region on exploration scale."""

    class_name: str = "BoundedStdActorCritic"
    noise_std_type: str = "log"
    minimum_noise_std: float = 0.05
    maximum_noise_std: float = 1.05
    # float32 tanh actions become non-invertible when an unbounded actor mean
    # escapes far into saturation. Four keeps the mean comfortably below the
    # atanh(1 - 1e-6) replay limit while preserving nearly the full action span.
    maximum_latent_mean: float = 4.0
    action_squash: bool = True


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "yaw_bot_direct"
    # The installed RSL-RL ActorCritic only supports 1D observations. The raw
    # depth image is consumed by the task-owned pose predictor, whose forecast is
    # appended to the vector "policy" observation.
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.1,
        entropy_coef=0.01,
        num_learning_epochs=3,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class PredictiveGatedPPORunnerCfg(PPORunnerCfg):
    """Standard PPO fed by predictor-weighted atomic rewards.

    The historical class name remains as a configuration entry-point alias, but
    there is no reward tier, curriculum stage, or multi-critic routing.
    """

    class_name = "BoundedStdOnPolicyRunner"
    experiment_name = "yaw_bot_predictive_gated"
    # RSL-RL's minibatch schedule collapses to 1e-5 in one update on this task.
    # First reproduce the baseline's fast acquisition to 1e-2, then regulate
    # complete-rollout KL so the late update cannot stay at 1e-2 while KL grows.
    rollout_adaptive_schedule = True
    # A complete-rollout trust region may consolidate to the same floor as
    # upstream adaptive PPO. The previous 1e-3 floor made a late high-KL update
    # irreversible and produced saturated spinning in composer and outer-only.
    rollout_adaptive_min_learning_rate = 1.0e-5
    rollout_adaptive_max_learning_rate = 1.0e-3
    rollout_adaptive_factor = 1.15
    rollout_adaptive_acquisition_rollouts = 0
    # Stock desired_kl=0.01 is evaluated per minibatch. A complete three-epoch
    # update naturally lands higher, so regulate around 0.03 after acquisition.
    rollout_adaptive_desired_kl = 0.03
    rollout_trust_region_maximum_kl = 0.06
    rollout_trust_region_backtrack_factor = 2.0
    rollout_trust_region_maximum_backtracks = 4
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    policy = BoundedStdActorCriticCfg(
        # Restore the exploration regime that actually learned the 7.17 task.
        # The previous low-noise/fixed-rate combination stayed conservative but
        # never acquired stable standing or command following.
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        # Match the 7.17 prediction-on/gate-off control.  Keeping a different
        # inner PPO here made the allocator comparison confounded: it received
        # only 60% as many optimization epochs and half the clipping range.
        clip_param=0.1,
        # The fixed outer objective already supplies stochastic exploration
        # through the Gaussian policy. A persistent entropy bonus held std at
        # 1.0 while the successful PPO control naturally reduced it below 0.7.
        entropy_coef=0.0,
        num_learning_epochs=3,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="fixed",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class LIRPGPPORunnerCfg(PredictiveGatedPPORunnerCfg):
    """Official PPO-LIRPG optimizer schedule on the unchanged yaw_bot runner."""

    rollout_adaptive_schedule = False
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=10,
        num_mini_batches=32,
        learning_rate=3.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=None,
        max_grad_norm=0.5,
    )
