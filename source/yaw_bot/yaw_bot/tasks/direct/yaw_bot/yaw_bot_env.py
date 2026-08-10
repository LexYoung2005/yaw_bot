# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch
import torch.distributed as dist

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, Imu, RayCasterCamera
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.math import quat_apply

from .agents.relara import ReLaraConfig, ReLaraRewardAgent, relara_policy_reward
from .outer_advantage_composer import (
    REWARD_GROUP_NAMES,
    CenteredTanhComposer,
    LIRPGIntrinsicReward,
    LIRPGOuterCritic,
    OuterCritic,
    RunningGroupRMS,
    beta_schedule,
    composer_meta_gradient_loss,
    effective_composer_weights,
    generalized_advantage,
    group_atomic_rewards,
    lirpg_actor_reward,
    lirpg_meta_gradient_loss,
    outer_reward,
    reward_group_index_tensor,
    select_actor_reward,
    static_group_weight_tensor,
    validate_outer_checkpoint_state,
    validate_static_group_weights,
)
from .pose_predictor import DepthPosePredictor, future_state_prediction_loss
from .predictive_feasibility import (
    DepthFeatureEncoder,
    PredictiveFeasibilityModel,
    censored_diagonal_gaussian_component_log_prob,
    componentwise_rollout_causal_progress_credit,
    statewise_rollout_causal_progress_credit,
    straight_through_clamp,
)
from .predictive_labels import (
    DIRECT_REWARD_NAMES,
    aggregate_prerequisite_targets,
    command_aligned_velocities,
    command_trackable_label,
    differential_drive_yaw_proxy,
    linear_warmup_blend,
    ordered_partial_horizon_indices,
    stable_event_label,
)
from .yaw_bot_env_cfg import YawBotEnvCfg


class YawBotEnv(DirectRLEnv):
    cfg: YawBotEnvCfg

    @staticmethod
    def _distributed_training_active() -> bool:
        return bool(
            dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        )

    @staticmethod
    @torch.no_grad()
    def _distributed_all_reduce_in_place(
        tensor: torch.Tensor,
        *,
        op: dist.ReduceOp,
    ) -> None:
        if dist.get_backend() == "gloo" and tensor.device.type != "cpu":
            host_tensor = tensor.detach().cpu()
            dist.all_reduce(host_tensor, op=op)
            tensor.copy_(host_tensor.to(tensor.device))
        else:
            dist.all_reduce(tensor, op=op)

    @staticmethod
    @torch.no_grad()
    def _distributed_broadcast_in_place(
        tensor: torch.Tensor,
        *,
        source: int = 0,
    ) -> None:
        if dist.get_backend() == "gloo" and tensor.device.type != "cpu":
            host_tensor = tensor.detach().cpu()
            dist.broadcast(host_tensor, src=source)
            tensor.copy_(host_tensor.to(tensor.device))
        else:
            dist.broadcast(tensor, src=source)

    @staticmethod
    @torch.no_grad()
    def _synchronize_module(
        module: torch.nn.Module,
        *,
        average: bool,
    ) -> None:
        """Broadcast or average one task-owned module across all ranks."""
        if not YawBotEnv._distributed_training_active():
            return
        world_size = dist.get_world_size()
        for tensor in module.state_dict().values():
            if tensor.is_floating_point() or tensor.is_complex():
                if average:
                    YawBotEnv._distributed_all_reduce_in_place(
                        tensor, op=dist.ReduceOp.SUM
                    )
                    tensor.div_(world_size)
                else:
                    YawBotEnv._distributed_broadcast_in_place(tensor, source=0)
            else:
                # Integer counters/buffers do not admit a meaningful average.
                YawBotEnv._distributed_broadcast_in_place(tensor, source=0)

    @staticmethod
    @torch.no_grad()
    def _average_adam_optimizer(optimizer: torch.optim.Optimizer) -> None:
        """Average Adam/AdamW moments, tolerating unequal local update counts."""
        if not YawBotEnv._distributed_training_active():
            return
        world_size = dist.get_world_size()
        for parameter_group in optimizer.param_groups:
            amsgrad = bool(parameter_group.get("amsgrad", False))
            for parameter in parameter_group["params"]:
                state = optimizer.state[parameter]
                has_state = torch.tensor(
                    1 if "exp_avg" in state else 0,
                    device=(
                        "cpu"
                        if dist.get_backend() == "gloo"
                        else parameter.device
                    ),
                    dtype=torch.long,
                )
                YawBotEnv._distributed_all_reduce_in_place(
                    has_state, op=dist.ReduceOp.MAX
                )
                if int(has_state.item()) == 0:
                    continue

                if "exp_avg" not in state:
                    state["step"] = torch.tensor(0.0)
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(parameter)

                for state_name in ("exp_avg", "exp_avg_sq"):
                    YawBotEnv._distributed_all_reduce_in_place(
                        state[state_name], op=dist.ReduceOp.SUM
                    )
                    state[state_name].div_(world_size)
                if amsgrad:
                    YawBotEnv._distributed_all_reduce_in_place(
                        state["max_exp_avg_sq"], op=dist.ReduceOp.SUM
                    )
                    state["max_exp_avg_sq"].div_(world_size)

                local_step = state.get("step", torch.tensor(0.0))
                step_value = torch.tensor(
                    float(local_step.item()),
                    device=(
                        "cpu"
                        if dist.get_backend() == "gloo"
                        else parameter.device
                    ),
                    dtype=torch.float32,
                )
                YawBotEnv._distributed_all_reduce_in_place(
                    step_value, op=dist.ReduceOp.MAX
                )
                if isinstance(local_step, torch.Tensor):
                    local_step.fill_(float(step_value.item()))
                else:
                    state["step"] = float(step_value.item())

    @staticmethod
    def _average_gradients(parameters: Sequence[torch.nn.Parameter]) -> None:
        """Average a same-shape task-owned gradient set across ranks."""
        if not YawBotEnv._distributed_training_active():
            return
        world_size = dist.get_world_size()
        for parameter in parameters:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            YawBotEnv._distributed_all_reduce_in_place(
                parameter.grad, op=dist.ReduceOp.SUM
            )
            parameter.grad.div_(world_size)

    @staticmethod
    def _distributed_mean_metrics(metrics: dict[str, float]) -> dict[str, float]:
        if not YawBotEnv._distributed_training_active() or not metrics:
            return metrics
        names = tuple(metrics)
        # NCCL requires a CUDA tensor. Every metric dictionary here belongs to
        # the environment, whose values originated on its current CUDA device.
        rank_device = (
            torch.device("cpu")
            if dist.get_backend() == "gloo"
            else torch.device("cuda", torch.cuda.current_device())
        )
        values = torch.tensor(
            [metrics[name] for name in names],
            device=rank_device,
            dtype=torch.float64,
        )
        YawBotEnv._distributed_all_reduce_in_place(
            values, op=dist.ReduceOp.SUM
        )
        values.div_(dist.get_world_size())
        return {
            name: float(value)
            for name, value in zip(names, values.cpu().tolist(), strict=True)
        }

    @torch.no_grad()
    def synchronize_distributed_training_state(self, *, initial: bool) -> None:
        """Synchronize task-owned state outside the PPO/DDP policy.

        RSL-RL synchronizes actor/critic gradients itself. Predictors learn from
        rank-local online trajectories, so they use rollout-local SGD followed
        by parameter and Adam-moment averaging at each rollout boundary.
        """
        if not self._distributed_training_active():
            return

        if initial:
            self._synchronize_module(self.pose_predictor, average=False)
            if self._predictive_training_enabled:
                self._synchronize_module(
                    self.predictive_feasibility_model, average=False
                )
            if self.outer_advantage_composer_enabled():
                self._synchronize_module(self.outer_reward_composer, average=False)
                self._synchronize_module(self._outer_rollout_composer, average=False)
                if self._outer_reward_composition_mode == "lirpg":
                    self._synchronize_module(
                        self.lirpg_intrinsic_reward, average=False
                    )
                    self._synchronize_module(
                        self._lirpg_rollout_reward, average=False
                    )
                self._synchronize_module(self.outer_critic, average=False)
                self._synchronize_module(self.outer_group_rms, average=False)
            dist.barrier()
            return

        self._synchronize_module(self.pose_predictor, average=True)
        self._average_adam_optimizer(self._pose_predictor_optimizer)
        if self._predictive_training_enabled:
            self._synchronize_module(
                self.predictive_feasibility_model, average=True
            )
            self._average_adam_optimizer(self._predictive_optimizer)
        dist.barrier()

    def _configure_gym_env_spaces(self) -> None:
        """Expose vector and image observations as separate top-level groups."""
        super()._configure_gym_env_spaces()
        grouped_space = self.single_observation_space["policy"]
        if not isinstance(grouped_space, gym.spaces.Dict):
            raise TypeError("YawBotEnvCfg.observation_space must define grouped policy and depth observations.")
        self.single_observation_space = grouped_space
        self.observation_space = gym.vector.utils.batch_space(grouped_space, self.num_envs)

    def __init__(self, cfg: YawBotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._predictive_gating_enabled = bool(getattr(self.cfg, "predictive_gating_enable", False))

        # joint indices
        self._left_hip_dof_idx, _ = self.robot.find_joints(self.cfg.left_hip_joint_name)
        self._left_knee_dof_idx, _ = self.robot.find_joints(self.cfg.left_knee_joint_name)
        self._left_wheel_dof_idx, _ = self.robot.find_joints(self.cfg.left_wheel_joint_name)
        self._right_hip_dof_idx, _ = self.robot.find_joints(self.cfg.right_hip_joint_name)
        self._right_knee_dof_idx, _ = self.robot.find_joints(self.cfg.right_knee_joint_name)
        self._right_wheel_dof_idx, _ = self.robot.find_joints(self.cfg.right_wheel_joint_name)
        self._left_wheel_body_ids, _ = self.robot.find_bodies(self.cfg.left_wheel_body_name)
        self._right_wheel_body_ids, _ = self.robot.find_bodies(self.cfg.right_wheel_body_name)

        self._servo_joint_ids = torch.tensor(
            [
                self._left_hip_dof_idx[0],
                self._left_knee_dof_idx[0],
                self._right_hip_dof_idx[0],
                self._right_knee_dof_idx[0],
            ],
            device=self.device,
            dtype=torch.long,
        )

        self._wheel_joint_ids = torch.tensor(
            [
                self._left_wheel_dof_idx[0],
                self._right_wheel_dof_idx[0],
            ],
            device=self.device,
            dtype=torch.long,
        )

        self._all_joint_ids = torch.tensor(
            [
                self._left_hip_dof_idx[0],
                self._left_knee_dof_idx[0],
                self._left_wheel_dof_idx[0],
                self._right_hip_dof_idx[0],
                self._right_knee_dof_idx[0],
                self._right_wheel_dof_idx[0],
            ],
            device=self.device,
            dtype=torch.long,
        )

        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.last_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._mapped_parallel_hip_targets = torch.zeros((self.num_envs, 2), device=self.device)
        self._wheel_contact_armed = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self._commands = torch.zeros((self.num_envs, 2), device=self.device)
        self._command_time_left = torch.zeros((self.num_envs,), device=self.device)
        self._command_dt = self.cfg.sim.dt * self.cfg.decimation
        self._curriculum_stage = 1
        self._curriculum_active_stage = 1
        self._curriculum_episode_count = 0
        self._curriculum_ema = torch.zeros(3, device=self.device)
        self._curriculum_history = [[], [], []]
        self._curriculum_last_unlock = 0
        self._curriculum_last_unlock_episode = 0
        self._curriculum_last_means = torch.zeros(3, device=self.device)
        self._ep_len = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self._ep_gate_sum = torch.zeros((self.num_envs, 3), device=self.device)
        self._ep_gate_denom = torch.zeros((self.num_envs, 3), device=self.device)
        self._prev_root_pos_w = self.robot.data.root_pos_w.clone()
        self._commanded_position_w = self.robot.data.root_pos_w[:, :2].clone()
        body_forward_axis = torch.zeros((self.num_envs, 3), device=self.device)
        body_forward_axis[:, 1] = 1.0
        commanded_heading_w = quat_apply(self.robot.data.root_quat_w, body_forward_axis)[:, :2]
        self._commanded_heading_w = commanded_heading_w / torch.linalg.vector_norm(
            commanded_heading_w, dim=1, keepdim=True
        ).clamp(min=1.0e-6)
        self._wheel_body_ids = torch.tensor(
            [self._left_wheel_body_ids[0], self._right_wheel_body_ids[0]],
            device=self.device,
            dtype=torch.long,
        )
        self._prev_wheel_body_pos_w = self.robot.data.body_pos_w[:, self._wheel_body_ids].clone()

        if self._predictive_gating_enabled:
            self._initialize_pose_predictor()
            # Environment-owned Predictor initialization used to consume the
            # global Torch stream before RSL-RL constructed its Actor. Thus a
            # same-seed baseline and Predictor run silently started from
            # different policies. Fork and restore every CUDA stream here, and
            # use a private generator for all later Predictor sampling.
            cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=cuda_devices, enabled=True):
                torch.manual_seed(int(self.cfg.predictive_random_seed))
                self._initialize_predictive_gating()
            self._predictive_generator = torch.Generator(device=self.device)
            self._predictive_generator.manual_seed(int(self.cfg.predictive_random_seed) + 1)
        else:
            self._initialize_pose_predictor()

        # default servo pose is defined in (a, b) space and mapped into simulation joints.
        default_joint_pos = self.robot.data.default_joint_pos
        self._default_branch_hip_joint_pos = torch.full(
            (self.num_envs, 2),
            self.cfg.default_branch_hip_angle,
            device=self.device,
            dtype=default_joint_pos.dtype,
        )
        self._default_mapped_parallel_hip_pos = torch.full(
            (self.num_envs, 2),
            self.cfg.default_mapped_hip_angle,
            device=self.device,
            dtype=default_joint_pos.dtype,
        )
        self._default_servo_joint_pos = self._map_branch_and_parallel_hips_to_sim_servo_targets(
            self._default_branch_hip_joint_pos,
            self._default_mapped_parallel_hip_pos,
        )

        # per-step commanded servo targets
        self._servo_position_targets = self._default_servo_joint_pos.clone()
        self._wheel_velocity_targets = torch.zeros((self.num_envs, 2), device=self.device)

        # sign mapping for servo semantic consistency
        # left hip a, left mapped hip b, right hip a, right mapped hip b
        self._servo_action_sign = torch.tensor(
            [
                1.0,   # left hip
                1.0,   # left mapped hip
                1.0,   # right hip
                1.0,   # right mapped hip
            ],
            device=self.device,
            dtype=torch.float,
        ).unsqueeze(0)

        # sign mapping for wheels so same positive action means forward motion
        # left wheel, right wheel
        self._wheel_action_sign = torch.tensor(
            [
                1.0,   # left wheel
                -1.0,  # right wheel: mirrored axis, so flip sign
            ],
            device=self.device,
            dtype=torch.float,
        ).unsqueeze(0)
        # joint limits for clamping servo targets
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits
        self._servo_lower_limits = soft_joint_pos_limits[:, self._servo_joint_ids, 0]
        self._servo_upper_limits = soft_joint_pos_limits[:, self._servo_joint_ids, 1]
        self._branch_hip_lower_limits = self._servo_lower_limits[:, [0, 2]]
        self._branch_hip_upper_limits = self._servo_upper_limits[:, [0, 2]]
        self._mapped_parallel_hip_lower_limits = torch.full(
            (self.num_envs, 2),
            self.cfg.mapped_hip_lower_limit,
            device=self.device,
        )
        self._mapped_parallel_hip_upper_limits = torch.full(
            (self.num_envs, 2),
            self.cfg.mapped_hip_upper_limit,
            device=self.device,
        )

    def _initialize_pose_predictor(self) -> None:
        """Initialize the original future-pose baseline without changing its behavior."""
        depth_observation_dim = self.cfg.depth_observation_height * self.cfg.depth_observation_width
        self.pose_predictor = DepthPosePredictor(
            depth_observation_height=self.cfg.depth_observation_height,
            depth_observation_width=self.cfg.depth_observation_width,
            state_observation_dim=self.cfg.pose_predictor_state_dim,
            history_steps=self.cfg.pose_predictor_history_steps,
            prediction_steps=self.cfg.pose_predictor_future_steps,
            hidden_dims=self.cfg.pose_predictor_hidden_dims,
        ).to(self.device)
        self._pose_predictor_optimizer = torch.optim.Adam(
            self.pose_predictor.parameters(), lr=self.cfg.pose_predictor_learning_rate
        )
        self._depth_observation_history = torch.zeros(
            (self.num_envs, self.cfg.pose_predictor_history_steps, depth_observation_dim), device=self.device
        )
        self._state_observation_history = torch.zeros(
            (
                self.num_envs,
                self.cfg.pose_predictor_history_steps,
                self.cfg.pose_predictor_state_dim,
            ),
            device=self.device,
        )
        supervision_queue_length = self.cfg.pose_predictor_future_steps + 1
        predictor_depth_input_dim = self.cfg.pose_predictor_history_steps * depth_observation_dim
        predictor_state_input_dim = self.cfg.pose_predictor_history_steps * self.cfg.pose_predictor_state_dim
        self._predictor_depth_input_queue = torch.zeros(
            (self.num_envs, supervision_queue_length, predictor_depth_input_dim), device=self.device
        )
        self._predictor_state_input_queue = torch.zeros(
            (self.num_envs, supervision_queue_length, predictor_state_input_dim), device=self.device
        )
        self._future_state_target_queue = torch.zeros(
            (
                self.num_envs,
                supervision_queue_length,
                self.cfg.pose_predictor_output_dim,
            ),
            device=self.device,
        )
        self._pose_sequence_age = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._cached_pose_prediction = torch.zeros(
            (
                self.num_envs,
                self.cfg.pose_predictor_future_steps,
                self.cfg.pose_predictor_output_dim,
            ),
            device=self.device,
        )
        self._cached_pose_prediction[:, :, 0] = 1.0
        self._pose_predictor_last_step = -1
        self._pose_predictor_last_loss = 0.0
        self._pose_predictor_last_quaternion_loss = 0.0
        self._pose_predictor_last_linear_velocity_loss = 0.0
        self._pose_predictor_last_angular_velocity_loss = 0.0
        self._pose_predictor_last_batch_size = 0

    def _initialize_predictive_gating(self) -> None:
        """Initialize the training-only predictor or the deployable encoder."""
        self._outer_reward_composition_mode = str(
            getattr(self.cfg, "outer_reward_composition_mode", "composer")
        ).lower()
        if self._outer_reward_composition_mode not in {
            "composer",
            "uniform",
            "static",
            "lirpg",
            "relara",
        }:
            raise ValueError(
                "outer_reward_composition_mode must be composer, uniform, static, lirpg, or relara."
            )
        self._outer_static_group_weights = validate_static_group_weights(
            getattr(self.cfg, "outer_static_group_weights", (1.0,) * len(REWARD_GROUP_NAMES))
        )
        if self._outer_reward_composition_mode == "uniform" and self._outer_static_group_weights != (
            1.0,
        ) * len(REWARD_GROUP_NAMES):
            raise ValueError("The A1 uniform baseline requires every group weight to equal one.")
        history_steps = self.cfg.predictive_history_steps
        height = self.cfg.depth_observation_height
        width = self.cfg.depth_observation_width
        self._predictive_training_enabled = bool(self.cfg.predictive_feasibility_train)
        # A full model may be present for inference while all of its updates are
        # frozen (the final-policy phase loaded with --pretrained_predictor).
        self._predictive_updates_enabled = self._predictive_training_enabled

        self._predictive_depth_history = torch.zeros(
            (self.num_envs, history_steps, height, width), device=self.device
        )
        self._predictive_state_history = torch.zeros(
            (self.num_envs, history_steps, self.cfg.predictive_state_dim), device=self.device
        )
        self._predictive_history_age = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._predictive_last_observation_step = -1
        self._predictive_last_gate_step = -1
        self._predictive_optimizer_steps = 0
        self._predictive_valid_labeled_sequences = 0
        self._predictive_gate_control_steps = 0
        self._predictive_gate_blend = 0.0
        self._predictive_cached_policy_latent = torch.zeros(
            (self.num_envs, self.cfg.predictive_depth_latent_dim), device=self.device
        )
        self._predictive_cached_policy_observation = torch.zeros(
            (self.num_envs, self.cfg.predictive_policy_observation_dim), device=self.device
        )
        self._predictive_cached_critic_observation = torch.zeros(
            (
                self.num_envs,
                self.cfg.predictive_policy_observation_dim + self.cfg.predictive_critic_privileged_dim,
            ),
            device=self.device,
        )
        self._predictive_cached_probabilities = torch.ones(
            (self.num_envs, self.cfg.predictive_event_dim), device=self.device
        )
        self._predictive_cached_uncertainty = torch.zeros_like(self._predictive_cached_probabilities)
        self._predictive_cached_gates = torch.ones(
            (self.num_envs, self.cfg.predictive_reward_dim), device=self.device
        )
        self._predictive_cached_fused_latent = torch.zeros(
            (self.num_envs, 128), device=self.device
        )
        self._outer_cached_group_weights = torch.ones(
            (self.num_envs, len(REWARD_GROUP_NAMES)), device=self.device
        )
        self._lirpg_cached_intrinsic_reward = torch.zeros(
            self.num_envs, device=self.device
        )
        self._relara_cached_proposed_reward = torch.zeros(
            self.num_envs, device=self.device
        )

        if self._predictive_training_enabled:
            self.predictive_feasibility_model = PredictiveFeasibilityModel(
                history_steps=history_steps,
                depth_height=height,
                depth_width=width,
                state_dim=self.cfg.predictive_state_dim,
                action_dim=self.cfg.predictive_action_dim,
                event_dim=self.cfg.predictive_event_dim,
                latent_dim=self.cfg.predictive_depth_latent_dim,
                future_steps=self.cfg.predictive_future_steps,
                future_state_dim=self.cfg.predictive_future_state_dim,
                reward_dim=self.cfg.predictive_reward_dim,
                ensemble_size=self.cfg.predictive_ensemble_size,
                ema_decay=self.cfg.predictive_ema_decay,
                uncertainty_scale=self.cfg.predictive_uncertainty_beta,
                ensemble_bootstrap_probability=self.cfg.predictive_ensemble_bootstrap_probability,
            ).to(self.device)
            initial_reward_weights = torch.as_tensor(
                self.cfg.predictive_reward_initial_weights,
                device=self.device,
                dtype=torch.float32,
            )
            if initial_reward_weights.shape != (self.cfg.predictive_reward_dim,):
                raise ValueError(
                    "predictive_reward_initial_weights must contain exactly "
                    f"{self.cfg.predictive_reward_dim} values."
                )
            lower = float(self.cfg.predictive_reward_weight_min)
            upper = float(self.cfg.predictive_reward_weight_max)
            if torch.any((initial_reward_weights < lower) | (initial_reward_weights > upper)):
                raise ValueError("Initial predictive reward weights must lie within configured bounds.")
            reward_output = self.predictive_feasibility_model.reward_head[-1]
            if not isinstance(reward_output, torch.nn.Linear):
                raise TypeError("Predictive reward head must end in a linear layer.")
            with torch.no_grad():
                # A bias-only prior is not a prior if the randomly initialized
                # output matrix already adds state-dependent offsets. Start at
                # the exact configured allocation; state dependence is learned
                # continuously as soon as outer gradients arrive.
                reward_output.weight.zero_()
                reward_output.bias.copy_(
                    initial_reward_weights.to(dtype=reward_output.bias.dtype)
                )
            self._predictive_reward_prior = initial_reward_weights.detach().clone()
            reward_parameter_ids = {
                id(parameter) for parameter in self.predictive_feasibility_model.reward_head.parameters()
            }
            trainable_parameters = [
                parameter
                for parameter in self.predictive_feasibility_model.parameters()
                if parameter.requires_grad and id(parameter) not in reward_parameter_ids
            ]
            # Keep the exact parameter set owned by the auxiliary optimizer.
            # The allocator leaves reward-head gradients populated after its
            # rollout update; including those stale gradients in the auxiliary
            # clip norm would silently shrink the event/future update.
            self._predictive_auxiliary_parameters = tuple(trainable_parameters)
            self._predictive_optimizer = torch.optim.AdamW(
                self._predictive_auxiliary_parameters,
                lr=self.cfg.predictive_learning_rate,
                weight_decay=self.cfg.predictive_weight_decay,
            )
            if bool(getattr(self.cfg, "outer_advantage_composer_enable", False)):
                self._predictive_allocator_optimizer = None
            else:
                self._predictive_allocator_optimizer = torch.optim.AdamW(
                    self.predictive_feasibility_model.reward_head.parameters(),
                    lr=self.cfg.predictive_allocator_learning_rate,
                    weight_decay=0.0,
                )
            allocator_context_dim = self.predictive_feasibility_model.reward_context_dim
            self._predictive_cached_allocator_contexts = torch.zeros(
                (self.num_envs, allocator_context_dim), device=self.device
            )
            self._predictive_cached_allocator_sample = torch.ones(
                (self.num_envs, self.cfg.predictive_reward_dim), device=self.device
            )
            self._predictive_cached_allocator_mean = torch.ones(
                (self.num_envs, self.cfg.predictive_reward_dim), device=self.device
            )
            self._predictive_cached_allocator_log_prob = torch.zeros(
                (self.num_envs, self.cfg.predictive_reward_dim), device=self.device
            )
            self._predictive_allocator_context_ema = torch.zeros(
                allocator_context_dim, device=self.device
            )
            self._predictive_allocator_context_initialized = False
            self._predictive_rollout_allocator_context = torch.zeros(
                allocator_context_dim, device=self.device
            )
            self._predictive_rollout_allocator_mean = torch.ones(
                self.cfg.predictive_reward_dim, device=self.device
            )
            self._predictive_rollout_allocator_sample = torch.ones(
                self.cfg.predictive_reward_dim, device=self.device
            )
            self._predictive_rollout_allocator_residual = torch.zeros(
                self.cfg.predictive_reward_dim, device=self.device
            )
            self._predictive_rollout_allocator_log_prob = torch.zeros(
                self.cfg.predictive_reward_dim, device=self.device
            )
            self._predictive_rollout_allocator_coordinate = -1
            self._predictive_allocator_candidate_signs = torch.ones(
                (self.num_envs, 1), device=self.device
            )
            self._predictive_rollout_gate_blend = 0.0
            self._predictive_reference_ema: float | None = None
            self._predictive_allocator_steps = 0
            self._predictive_last_allocator_loss = 0.0
            self._predictive_last_reference_score = 0.0
            self._predictive_last_reference_progress = 0.0
            self._predictive_last_ppo_credit = 0.0
            self._predictive_last_component_alignment = 0.0

            if bool(getattr(self.cfg, "outer_advantage_composer_enable", False)):
                fused_dim = int(self.predictive_feasibility_model.fusion_hidden_dims[-1])
                self._predictive_cached_fused_latent = torch.zeros(
                    (self.num_envs, fused_dim), device=self.device
                )
                self.outer_reward_composer = CenteredTanhComposer(
                    fused_dim,
                    num_groups=len(REWARD_GROUP_NAMES),
                    half_range=float(self.cfg.outer_composer_weight_half_range),
                ).to(self.device)
                # Collection uses an explicit frozen snapshot. The trainable
                # phi is copied here only at rollout boundaries.
                self._outer_rollout_composer = copy.deepcopy(self.outer_reward_composer)
                self._outer_rollout_composer.requires_grad_(False).eval()
                if self._outer_reward_composition_mode == "lirpg":
                    self.outer_critic = LIRPGOuterCritic(
                        int(self.cfg.predictive_policy_observation_dim)
                    ).to(self.device)
                else:
                    outer_critic_observation_dim = (
                        int(self.cfg.predictive_policy_observation_dim)
                        + int(self.cfg.predictive_critic_privileged_dim)
                    )
                    self.outer_critic = OuterCritic(
                        outer_critic_observation_dim
                    ).to(self.device)
                self.outer_group_rms = RunningGroupRMS(
                    len(REWARD_GROUP_NAMES), decay=float(self.cfg.outer_group_rms_decay)
                ).to(self.device)
                self._outer_composer_optimizer = torch.optim.AdamW(
                    self.outer_reward_composer.parameters(),
                    lr=float(self.cfg.outer_composer_learning_rate),
                    weight_decay=0.0,
                )
                if self._outer_reward_composition_mode == "lirpg":
                    observation_dim = int(self.cfg.predictive_policy_observation_dim)
                    self.lirpg_intrinsic_reward = LIRPGIntrinsicReward(
                        observation_dim,
                        int(self.cfg.predictive_action_dim),
                    ).to(self.device)
                    self._lirpg_rollout_reward = copy.deepcopy(
                        self.lirpg_intrinsic_reward
                    )
                    self._lirpg_rollout_reward.requires_grad_(False).eval()
                    self._lirpg_optimizer = torch.optim.Adam(
                        self.lirpg_intrinsic_reward.parameters(),
                        lr=float(self.cfg.lirpg_learning_rate),
                        eps=1.0e-5,
                    )
                    self._lirpg_updates = 0
                elif self._outer_reward_composition_mode == "relara":
                    relara_config = ReLaraConfig(
                        gamma=float(self.cfg.relara_gamma),
                        reward_scale=float(self.cfg.relara_reward_scale),
                        beta=float(self.cfg.relara_beta),
                        replay_capacity=int(self.cfg.relara_replay_capacity),
                        batch_size=int(self.cfg.relara_batch_size),
                        learning_starts=int(self.cfg.relara_learning_starts),
                        actor_lr=float(self.cfg.relara_actor_learning_rate),
                        critic_lr=float(self.cfg.relara_critic_learning_rate),
                        alpha_lr=float(self.cfg.relara_alpha_learning_rate),
                        policy_frequency=int(self.cfg.relara_policy_frequency),
                        target_frequency=int(self.cfg.relara_target_frequency),
                        tau=float(self.cfg.relara_tau),
                        initial_alpha=float(self.cfg.relara_initial_alpha),
                        alpha_autotune=bool(self.cfg.relara_alpha_autotune),
                    )
                    self.relara_reward_agent = ReLaraRewardAgent(
                        int(self.cfg.predictive_policy_observation_dim)
                        + int(self.cfg.predictive_action_dim),
                        relara_config,
                        device=self.device,
                        seed=int(self.cfg.predictive_random_seed) + 200,
                    )
                    self._relara_updates = 0
                if self._outer_reward_composition_mode == "lirpg":
                    # Match the reference PPO-LIRPG intrinsic/V_ex Adam.
                    self._outer_critic_optimizer = torch.optim.Adam(
                        self.outer_critic.parameters(),
                        lr=float(self.cfg.lirpg_learning_rate),
                        eps=1.0e-5,
                    )
                else:
                    self._outer_critic_optimizer = torch.optim.AdamW(
                        self.outer_critic.parameters(),
                        lr=float(self.cfg.outer_critic_learning_rate),
                        weight_decay=0.0,
                    )
                self._outer_beta_iteration = 0
                self._outer_beta_total_iterations = 1
                self._outer_beta = 0.0
                self._outer_composer_updates = 0
                self._outer_critic_updates = 0
                self._outer_last_metrics: dict[str, float] = {}
                self._outer_learning_auc = 0.0
                self._outer_previous_score: float | None = None

            horizon = self.cfg.predictive_future_steps
            self._predictive_depth_input_queue = torch.zeros(
                (self.num_envs, horizon, history_steps, height, width), device=self.device
            )
            self._predictive_state_input_queue = torch.zeros(
                (
                    self.num_envs,
                    horizon,
                    history_steps,
                    self.cfg.predictive_state_dim,
                ),
                device=self.device,
            )
            self._predictive_action_input_queue = torch.zeros(
                (self.num_envs, horizon, self.cfg.predictive_action_dim), device=self.device
            )
            self._predictive_event_target_queue = torch.zeros(
                (self.num_envs, horizon, self.cfg.predictive_event_dim), device=self.device
            )
            self._predictive_future_target_queue = torch.zeros(
                (self.num_envs, horizon, self.cfg.predictive_future_state_dim), device=self.device
            )
            self._predictive_sequence_age = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            self._predictive_queue_cursor = 0
        else:
            # Playback instantiates only the component that is actually deployed.
            self.deployable_depth_encoder = DepthFeatureEncoder(
                depth_history_steps=history_steps,
                depth_height=height,
                depth_width=width,
                latent_dim=self.cfg.predictive_depth_latent_dim,
            ).to(self.device)
            self.deployable_depth_encoder.eval()

        self._predictive_last_loss = 0.0
        self._predictive_last_event_loss = 0.0
        self._predictive_last_future_loss = 0.0
        self._predictive_last_reward_loss = 0.0
        self._predictive_last_brier = 0.0
        self._predictive_last_accuracy = 0.0
        self._predictive_last_batch_size = 0
        self._predictive_last_horizon_event_rates = [0.0] * self.cfg.predictive_event_dim

    def _map_normalized_actions_to_range(
        self, actions: torch.Tensor, lower_limits: torch.Tensor, upper_limits: torch.Tensor
    ) -> torch.Tensor:
        """Map normalized actions in [-1, 1] to the full closed interval [lower, upper]."""
        midpoint = 0.5 * (upper_limits + lower_limits)
        half_range = 0.5 * (upper_limits - lower_limits)
        return midpoint + actions * half_range

    def _predictive_reward_bounds(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the common bounds used by all direct reward weights."""
        lower = torch.full_like(reference, float(self.cfg.predictive_reward_weight_min))
        upper = torch.full_like(reference, float(self.cfg.predictive_reward_weight_max))
        if torch.any(lower >= upper):
            raise ValueError("Every predictive reward lower bound must be below its upper bound.")
        return lower, upper

    def _bounded_predictive_reward_mean(self, raw_mean: torch.Tensor) -> torch.Tensor:
        """Map allocator logits to the complete configured reward-weight range."""
        # The straight-through map keeps deployment weights in bounds without
        # creating dead reward-head coordinates when an optimizer step moves a
        # zero-initialized raw mean infinitesimally below the lower bound.
        lower, upper = self._predictive_reward_bounds(raw_mean)
        return straight_through_clamp(raw_mean, lower, upper)

    def _get_depth_observation(self) -> torch.Tensor:
        """Return a normalized, pooled depth image for each environment."""
        depth = self.depth_camera.data.output["distance_to_image_plane"]
        depth = torch.nan_to_num(
            depth,
            nan=self.cfg.depth_max_distance,
            posinf=self.cfg.depth_max_distance,
            neginf=0.0,
        )
        depth = torch.clamp(depth, min=0.0, max=self.cfg.depth_max_distance)
        depth = depth.permute(0, 3, 1, 2)
        target_size = (self.cfg.depth_observation_height, self.cfg.depth_observation_width)
        if tuple(depth.shape[-2:]) != target_size:
            depth = torch.nn.functional.adaptive_avg_pool2d(depth, target_size)
        return depth / self.cfg.depth_max_distance

    def _get_predictive_observations(self) -> dict[str, torch.Tensor]:
        """Build deployable actor observations and privileged critic observations."""
        current_step = int(self.common_step_counter)
        if self._predictive_last_observation_step == current_step:
            return {
                "policy": self._predictive_cached_policy_observation,
                "critic": self._predictive_cached_critic_observation,
            }
        if self._predictive_last_observation_step != current_step:
            # Resample between transitions so the actor, predictor, physics, and
            # reward all see the same command for an action.
            if self.cfg.use_velocity_commands and self.cfg.resample_commands:
                self._command_time_left -= self._command_dt
                resample_env_ids = torch.nonzero(
                    self._command_time_left <= 0.0, as_tuple=False
                ).squeeze(-1)
                if resample_env_ids.numel() > 0:
                    self._resample_commands(resample_env_ids)

        servo_pos = self.robot.data.joint_pos[:, self._servo_joint_ids]
        servo_vel = self.robot.data.joint_vel[:, self._servo_joint_ids]
        wheel_vel = self.robot.data.joint_vel[:, self._wheel_joint_ids]
        imu_data = self.body_imu.data

        if self.cfg.enable_imu_noise:
            imu_quat = imu_data.quat_w + torch.randn_like(imu_data.quat_w) * self.cfg.imu_quat_noise_std
            imu_quat = torch.nn.functional.normalize(imu_quat, dim=-1, eps=1.0e-6)
            imu_ang_vel = imu_data.ang_vel_b + torch.randn_like(imu_data.ang_vel_b) * self.cfg.imu_ang_vel_noise_std
            projected_gravity = (
                imu_data.projected_gravity_b
                + torch.randn_like(imu_data.projected_gravity_b) * self.cfg.imu_projected_gravity_noise_std
            )
        else:
            imu_quat = imu_data.quat_w
            imu_ang_vel = imu_data.ang_vel_b
            projected_gravity = imu_data.projected_gravity_b

        state_observation = torch.cat(
            [
                imu_quat,            # 4: IMU orientation
                imu_ang_vel,         # 3: IMU angular velocity
                projected_gravity,   # 3: gravity in the body frame
                self._commands,      # 2: desired linear/yaw velocity
                servo_pos,           # 4: measured servo-equivalent positions
                servo_vel,           # 4: measured servo-equivalent velocities
                wheel_vel,           # 2: wheel encoders
                self.actions,        # 6: action applied in the previous transition
            ],
            dim=-1,
        )
        if state_observation.shape[-1] != self.cfg.predictive_state_dim:
            raise RuntimeError(
                f"Predictive actor state dimension is {state_observation.shape[-1]}, "
                f"expected {self.cfg.predictive_state_dim}."
            )

        # Do not feed the Predictor's richer, continuously changing training
        # state directly to PPO.  The 7.17 no-gate control learned reliably
        # from this exact 25-D ordering; keeping it here makes reward allocation
        # the only experimental variable seen by the inner learner.
        wheel_pos = self.robot.data.joint_pos[:, self._wheel_joint_ids]
        actor_state_observation = torch.cat(
            [
                imu_quat,                         # 4
                imu_ang_vel,                      # 3
                self.robot.data.root_lin_vel_b,   # 3
                projected_gravity,                # 3
                self._commands,                   # 2
                wheel_pos,                        # 2
                wheel_vel,                        # 2
                self.last_actions,                # 6
            ],
            dim=-1,
        )
        if actor_state_observation.shape[-1] != self.cfg.predictive_actor_state_dim:
            raise RuntimeError(
                f"Predictive actor state dimension is {actor_state_observation.shape[-1]}, "
                f"expected {self.cfg.predictive_actor_state_dim}."
            )

        if self._predictive_last_observation_step != current_step:
            raw_depth_observation = self._get_depth_observation()
            depth_observation = raw_depth_observation.squeeze(1)
            fresh_mask = self._predictive_history_age == 0
            active_mask = ~fresh_mask
            if torch.any(active_mask):
                self._predictive_depth_history[active_mask, :-1] = self._predictive_depth_history[
                    active_mask, 1:
                ].clone()
                self._predictive_depth_history[active_mask, -1] = depth_observation[active_mask]
                self._predictive_state_history[active_mask, :-1] = self._predictive_state_history[
                    active_mask, 1:
                ].clone()
                self._predictive_state_history[active_mask, -1] = state_observation[active_mask]
            if torch.any(fresh_mask):
                self._predictive_depth_history[fresh_mask] = depth_observation[fresh_mask].unsqueeze(1)
                self._predictive_state_history[fresh_mask] = state_observation[fresh_mask].unsqueeze(1)
            self._predictive_history_age += 1

            with torch.no_grad():
                if self._predictive_training_enabled:
                    self._predictive_cached_policy_latent = (
                        self.predictive_feasibility_model.encode_for_policy(self._predictive_depth_history).detach()
                    )
                else:
                    self._predictive_cached_policy_latent = self.deployable_depth_encoder(
                        self._predictive_depth_history
                    ).detach()
        pose_prediction = self._update_pose_prediction(
            raw_depth_observation, actor_state_observation
        )
        # The reward allocator is an outer learner and never enters the policy.
        # Preserve the separately supervised actor-side future-pose signal that
        # made the 7.17 prediction-on/no-gate control learn reliably.
        policy_observation = torch.cat(
            [actor_state_observation, pose_prediction], dim=-1
        )
        wheel_contact_normal_force = torch.abs(self.wheel_contact_sensor.data.net_forces_w[:, :, 2])
        semantic_wheel_vel = wheel_vel * self._wheel_action_sign
        wheel_surface_speed = self.cfg.wheel_radius * torch.mean(semantic_wheel_vel, dim=1)
        slip = wheel_surface_speed - self.robot.data.root_lin_vel_b[:, 1]
        privileged_observation = torch.cat(
            [
                self.robot.data.root_lin_vel_b,
                torch.clamp(
                    wheel_contact_normal_force / self.cfg.predictive_contact_force_scale,
                    min=0.0,
                    max=5.0,
                ),
                torch.clamp(
                    slip / self.cfg.predictive_slip_scale,
                    min=-5.0,
                    max=5.0,
                ).unsqueeze(-1),
            ],
            dim=-1,
        )
        if self.cfg.predictive_critic_privileged_dim == 0:
            critic_observation = policy_observation
        elif self.cfg.predictive_critic_privileged_dim == privileged_observation.shape[-1]:
            critic_observation = torch.cat([policy_observation, privileged_observation], dim=-1)
        else:
            raise ValueError(
                "predictive_critic_privileged_dim must be zero or match the complete "
                f"privileged observation ({privileged_observation.shape[-1]})."
            )
        self._predictive_cached_policy_observation.copy_(policy_observation)
        self._predictive_cached_critic_observation.copy_(critic_observation)
        self._predictive_last_observation_step = current_step

        log = self.extras.setdefault("log", {})
        log["PredictiveGating/prob_stable"] = torch.mean(self._predictive_cached_probabilities[:, 0]).item()
        log["PredictiveGating/prob_grounded"] = torch.mean(self._predictive_cached_probabilities[:, 1]).item()
        log["PredictiveGating/prob_trackable"] = torch.mean(self._predictive_cached_probabilities[:, 2]).item()
        log["PredictiveGating/prob_low_slip"] = torch.mean(self._predictive_cached_probabilities[:, 3]).item()
        log["PredictiveGating/uncertainty"] = torch.mean(self._predictive_cached_uncertainty).item()
        log["PredictiveGating/predictor_loss"] = self._predictive_last_loss
        log["PredictiveGating/event_loss"] = self._predictive_last_event_loss
        log["PredictiveGating/future_loss"] = self._predictive_last_future_loss
        log["PredictiveGating/reward_aux_loss"] = self._predictive_last_reward_loss
        if self._predictive_training_enabled and not bool(
            getattr(self.cfg, "outer_advantage_composer_enable", False)
        ):
            log["RewardAllocator/allocator_loss"] = self._predictive_last_allocator_loss
            log["RewardAllocator/reference_score"] = self._predictive_last_reference_score
            log["RewardAllocator/reference_progress"] = self._predictive_last_reference_progress
            log["RewardAllocator/component_credit_abs"] = self._predictive_last_ppo_credit
            log["RewardAllocator/component_alignment_abs"] = (
                self._predictive_last_component_alignment
            )
            log["RewardAllocator/allocator_steps"] = float(self._predictive_allocator_steps)
            log["RewardAllocator/selected_coordinate"] = float(
                self._predictive_rollout_allocator_coordinate
            )
            log["RewardAllocator/mean_min"] = torch.amin(
                self._predictive_cached_allocator_mean, dim=-1
            ).mean().item()
            log["RewardAllocator/mean_max"] = torch.amax(
                self._predictive_cached_allocator_mean, dim=-1
            ).mean().item()
            log["RewardAllocator/state_weight_std"] = self._predictive_cached_allocator_mean.std(
                dim=0, unbiased=False
            ).mean().item()
            allocator_lower, allocator_upper = self._predictive_reward_bounds(
                self._predictive_cached_allocator_sample
            )
            allocator_sample_at_bound = (
                (self._predictive_cached_allocator_sample <= allocator_lower)
                | (self._predictive_cached_allocator_sample >= allocator_upper)
            )
            log["RewardAllocator/sample_bound_rate"] = allocator_sample_at_bound.float().mean().item()
            contextual_exploration = bool(
                getattr(self.cfg, "predictive_allocator_statewise_exploration", False)
                and self._predictive_allocator_active()
            )
            log["RewardAllocator/contextual_action_count"] = float(
                self.num_envs if contextual_exploration else 1
            )
            if contextual_exploration:
                coordinate = self._predictive_rollout_allocator_coordinate
                state_residual = (
                    self._predictive_cached_allocator_sample
                    - self._predictive_cached_allocator_mean
                )
                if coordinate >= 0:
                    state_residual = state_residual[:, coordinate]
                log["RewardAllocator/state_residual_std"] = state_residual.std(
                    unbiased=False
                ).item()
                log["RewardAllocator/state_residual_positive_rate"] = (
                    state_residual > 0.0
                ).float().mean().item()
                log["RewardAllocator/explored_dimensions"] = float(
                    1 if coordinate >= 0 else self.cfg.predictive_reward_dim
                )
            elif self._predictive_allocator_active() and torch.any(
                self._predictive_rollout_allocator_residual != 0.0
            ):
                shared_residual = self._predictive_rollout_allocator_residual
                log["RewardAllocator/state_residual_std"] = shared_residual.std(
                    unbiased=False
                ).item()
                log["RewardAllocator/state_residual_positive_rate"] = (
                    shared_residual > 0.0
                ).float().mean().item()
                log["RewardAllocator/explored_dimensions"] = float(
                    self.cfg.predictive_reward_dim
                )
            else:
                log["RewardAllocator/state_residual_std"] = 0.0
                log["RewardAllocator/state_residual_positive_rate"] = 0.0
                log["RewardAllocator/explored_dimensions"] = 0.0
            log["RewardAllocator/rollout_gate_blend"] = float(
                self._predictive_rollout_gate_blend
            )
            log["RewardAllocator/context_norm"] = torch.linalg.vector_norm(
                self._predictive_rollout_allocator_context
            ).item()
        log["PredictiveGating/brier"] = self._predictive_last_brier
        log["PredictiveGating/accuracy"] = self._predictive_last_accuracy
        log["PredictiveGating/batch_size"] = float(self._predictive_last_batch_size)
        log["PredictiveGating/optimizer_steps"] = float(self._predictive_optimizer_steps)
        log["PredictiveGating/valid_labeled_sequences"] = float(self._predictive_valid_labeled_sequences)
        log["PredictiveGating/gate_control_steps"] = float(self._predictive_gate_control_steps)
        log["PredictiveGating/gate_blend"] = float(self._predictive_gate_blend)
        for reward_index, reward_name in enumerate(DIRECT_REWARD_NAMES):
            log[f"PredictiveWeights/{reward_name}"] = torch.mean(
                self._predictive_cached_gates[:, reward_index]
            ).item()
        log["PredictiveGating/horizon_target_stable"] = self._predictive_last_horizon_event_rates[0]
        log["PredictiveGating/horizon_target_grounded"] = self._predictive_last_horizon_event_rates[1]
        log["PredictiveGating/horizon_target_trackable"] = self._predictive_last_horizon_event_rates[2]
        log["PredictiveGating/horizon_target_low_slip"] = self._predictive_last_horizon_event_rates[3]
        return {
            "policy": self._predictive_cached_policy_observation,
            "critic": self._predictive_cached_critic_observation,
        }

    def _predictive_allocator_active(self) -> bool:
        """Return whether the direct reward allocator is trainable."""
        return bool(
            self._predictive_updates_enabled
            and self.cfg.predictive_allocator_train
        )

    def set_predictive_allocator_candidate_signs(self, signs: torch.Tensor) -> None:
        """Assign each environment to the positive or negative outer candidate."""
        if not self._predictive_gating_enabled or not self._predictive_training_enabled:
            raise RuntimeError("Candidate signs require a training-time predictive allocator.")
        signs = signs.to(device=self.device, dtype=torch.float32).reshape(-1, 1)
        if signs.shape != self._predictive_allocator_candidate_signs.shape:
            raise ValueError(
                f"Candidate signs have shape {tuple(signs.shape)}; expected "
                f"{tuple(self._predictive_allocator_candidate_signs.shape)}."
            )
        if not torch.all((signs == 1.0) | (signs == -1.0)):
            raise ValueError("Every allocator candidate sign must be +1 or -1.")
        self._predictive_allocator_candidate_signs.copy_(signs)

    def begin_predictive_allocator_rollout(
        self,
        aggregate_context: torch.Tensor | None = None,
        *,
        explore: bool | None = None,
        resample_exploration: bool = True,
    ) -> None:
        """Prepare the contextual reward policy for the next PPO rollout."""
        if not self._predictive_gating_enabled or not self._predictive_training_enabled:
            return
        if aggregate_context is not None:
            aggregate_context = aggregate_context.to(
                device=self.device,
                dtype=self._predictive_allocator_context_ema.dtype,
            ).reshape(-1)
            if aggregate_context.shape != self._predictive_allocator_context_ema.shape:
                raise ValueError(
                    "Allocator aggregate context has shape "
                    f"{tuple(aggregate_context.shape)}; expected "
                    f"{tuple(self._predictive_allocator_context_ema.shape)}."
                )
            decay = float(self.cfg.predictive_allocator_context_ema_decay)
            if not 0.0 <= decay < 1.0:
                raise ValueError("predictive_allocator_context_ema_decay must be in [0, 1).")
            if self._predictive_allocator_context_initialized:
                self._predictive_allocator_context_ema.mul_(decay).add_(
                    aggregate_context, alpha=1.0 - decay
                )
            else:
                self._predictive_allocator_context_ema.copy_(aggregate_context)
                self._predictive_allocator_context_initialized = True

        context = self._predictive_allocator_context_ema.detach().clone()
        allocator_active = self._predictive_allocator_active()
        allocator_applied = bool(
            self.cfg.predictive_allocator_train
            or getattr(self.cfg, "predictive_fixed_initial_weights_control", False)
        )
        should_explore = allocator_active if explore is None else bool(explore) and allocator_active
        with torch.no_grad():
            reward_mean = self._bounded_predictive_reward_mean(
                self.predictive_feasibility_model.reward_head(context.unsqueeze(0))
            ).squeeze(0)
            if (
                should_explore
                and allocator_active
                and self.cfg.predictive_allocator_exploration_std > 0.0
            ):
                if getattr(self.cfg, "predictive_allocator_coordinate_exploration", False):
                    sweep, position = divmod(
                        int(self._predictive_allocator_steps),
                        int(self.cfg.predictive_reward_dim),
                    )
                    coordinate_generator = torch.Generator(device="cpu")
                    coordinate_generator.manual_seed(
                        int(self.cfg.predictive_random_seed) + 104729 * sweep
                    )
                    coordinate_order = torch.randperm(
                        self.cfg.predictive_reward_dim,
                        generator=coordinate_generator,
                    )
                    coordinate = int(coordinate_order[position].item())
                    self._predictive_rollout_allocator_coordinate = coordinate
                    if getattr(
                        self.cfg, "predictive_allocator_statewise_exploration", False
                    ):
                        # The actual stochastic reward actions are sampled from
                        # each state-conditioned mean in _cache_predictive_gates.
                        allocator_sample = reward_mean
                    else:
                        residual = torch.zeros_like(reward_mean)
                        residual[coordinate] = float(
                            self.cfg.predictive_allocator_exploration_std
                        ) * torch.randn(
                            (),
                            device=reward_mean.device,
                            dtype=reward_mean.dtype,
                            generator=self._predictive_generator,
                        )
                        allocator_sample = reward_mean + residual
                else:
                    if getattr(
                        self.cfg, "predictive_allocator_statewise_exploration", False
                    ):
                        # Every state will sample the complete 22-D action in
                        # _cache_predictive_gates. The aggregate action is only
                        # a rollout diagnostic in this mode.
                        allocator_sample = reward_mean
                    else:
                        if resample_exploration:
                            self._predictive_rollout_allocator_residual.copy_(
                                float(self.cfg.predictive_allocator_exploration_std)
                                * torch.randn(
                                    reward_mean.shape,
                                    device=reward_mean.device,
                                    dtype=reward_mean.dtype,
                                    generator=self._predictive_generator,
                                )
                            )
                        allocator_sample = (
                            reward_mean + self._predictive_rollout_allocator_residual
                        )
                    self._predictive_rollout_allocator_coordinate = -1
            else:
                allocator_sample = reward_mean
                self._predictive_rollout_allocator_residual.zero_()
                self._predictive_rollout_allocator_coordinate = -1
            lower, upper = self._predictive_reward_bounds(allocator_sample)
            allocator_sample = torch.maximum(torch.minimum(allocator_sample, upper), lower)
            log_probability = censored_diagonal_gaussian_component_log_prob(
                allocator_sample,
                reward_mean,
                float(self.cfg.predictive_allocator_exploration_std),
                lower,
                upper,
            )

        self._predictive_rollout_allocator_context.copy_(context)
        self._predictive_rollout_allocator_mean.copy_(reward_mean)
        self._predictive_rollout_allocator_sample.copy_(allocator_sample)
        self._predictive_rollout_allocator_log_prob.copy_(log_probability)
        if allocator_applied:
            if (
                self.cfg.predictive_gate_warmup_control_steps == 0
                and self.cfg.predictive_gate_ramp_control_steps == 0
            ):
                self._predictive_rollout_gate_blend = 1.0
            else:
                self._predictive_rollout_gate_blend = linear_warmup_blend(
                    self._predictive_gate_control_steps,
                    self.cfg.predictive_gate_warmup_control_steps,
                    self.cfg.predictive_gate_ramp_control_steps,
                )
        else:
            self._predictive_rollout_gate_blend = 0.0
        self._predictive_gate_blend = self._predictive_rollout_gate_blend
        self._predictive_cached_allocator_mean.copy_(
            reward_mean.unsqueeze(0).expand(self.num_envs, -1)
        )
        self._predictive_cached_allocator_sample.copy_(
            allocator_sample.unsqueeze(0).expand(self.num_envs, -1)
        )
        self._predictive_cached_allocator_log_prob.copy_(
            log_probability.unsqueeze(0).expand(self.num_envs, -1)
        )

    def _cache_predictive_gates(self) -> None:
        """Apply a state-conditioned reward allocation before executing the action."""
        if not self._predictive_gating_enabled:
            return
        current_step = int(self.common_step_counter)
        if self._predictive_last_gate_step == current_step:
            return

        if not self._predictive_training_enabled:
            self._predictive_cached_probabilities.fill_(1.0)
            self._predictive_cached_uncertainty.zero_()
            self._predictive_cached_gates.fill_(1.0)
            self._predictive_last_gate_step = current_step
            return

        with torch.no_grad():
            # Heads are optimized on online-encoder latents, so gate inference
            # uses that same latent distribution. The actor alone uses EMA.
            prediction = self.predictive_feasibility_model.predict_for_gate(
                self._predictive_depth_history,
                self._predictive_state_history,
                self.actions,
                conservative_beta=self.cfg.predictive_uncertainty_beta,
            )
            probabilities = prediction["conservative_probability"]
            self._predictive_cached_probabilities.copy_(probabilities)
            self._predictive_cached_uncertainty.copy_(prediction["std_probability"])
            self._predictive_cached_fused_latent.copy_(prediction["fused_features"])
            self._predictive_cached_allocator_contexts.copy_(
                prediction["allocator_context"]
            )
            if bool(getattr(self.cfg, "outer_advantage_composer_enable", False)):
                if self._outer_reward_composition_mode == "composer":
                    group_weights = self._outer_rollout_composer(
                        prediction["fused_features"].detach()
                    )
                else:
                    group_weights = static_group_weight_tensor(
                        self._outer_static_group_weights,
                        self._outer_cached_group_weights,
                    )
                self._outer_cached_group_weights.copy_(group_weights)
                owners = reward_group_index_tensor(self.device)
                self._predictive_cached_gates.copy_(group_weights[:, owners])
                if self._outer_reward_composition_mode == "lirpg":
                    self._lirpg_cached_intrinsic_reward.copy_(
                        self._lirpg_rollout_reward(
                            self._predictive_cached_policy_observation,
                            self.actions,
                        )
                    )
                    self._relara_cached_proposed_reward.zero_()
                elif self._outer_reward_composition_mode == "relara":
                    relara_context = torch.cat(
                        (self._predictive_cached_policy_observation, self.actions),
                        dim=-1,
                    )
                    self._relara_cached_proposed_reward.copy_(
                        self.relara_reward_agent.propose(relara_context).squeeze(-1)
                    )
                    self._lirpg_cached_intrinsic_reward.zero_()
                else:
                    self._lirpg_cached_intrinsic_reward.zero_()
                    self._relara_cached_proposed_reward.zero_()
                self._predictive_last_gate_step = current_step
                cursor = self._predictive_queue_cursor
                self._predictive_depth_input_queue[:, cursor].copy_(self._predictive_depth_history)
                self._predictive_state_input_queue[:, cursor].copy_(self._predictive_state_history)
                self._predictive_action_input_queue[:, cursor].copy_(self.actions)
                self._predictive_sequence_age.add_(1).clamp_(max=self.cfg.predictive_future_steps)
                return
            state_reward_mean = self._bounded_predictive_reward_mean(
                self.predictive_feasibility_model.reward_head(
                    prediction["allocator_context"]
                )
            )
            # The deterministic head performs the state-by-state allocation.
            # Contextual exploration must also own one independent stochastic
            # action per state; a single shared residual cannot identify how a
            # state-conditioned reward head should vary its output.
            if (
                getattr(self.cfg, "predictive_allocator_statewise_exploration", False)
                and self._predictive_allocator_active()
            ):
                coordinate = self._predictive_rollout_allocator_coordinate
                rollout_residual = float(
                    self.cfg.predictive_allocator_exploration_std
                ) * torch.randn(
                    state_reward_mean.shape,
                    device=state_reward_mean.device,
                    dtype=state_reward_mean.dtype,
                    generator=self._predictive_generator,
                )
                if coordinate >= 0:
                    coordinate_mask = torch.zeros_like(rollout_residual)
                    coordinate_mask[:, coordinate] = 1.0
                    rollout_residual.mul_(coordinate_mask)
            else:
                rollout_residual = (
                    self._predictive_rollout_allocator_sample
                    - self._predictive_rollout_allocator_mean
                ).unsqueeze(0)
            state_reward_sample = state_reward_mean + (
                self._predictive_allocator_candidate_signs * rollout_residual
            )
            lower, upper = self._predictive_reward_bounds(state_reward_sample)
            state_reward_sample = torch.maximum(
                torch.minimum(state_reward_sample, upper), lower
            )
            self._predictive_cached_allocator_mean.copy_(state_reward_mean)
            self._predictive_cached_allocator_sample.copy_(state_reward_sample)
            self._predictive_cached_allocator_log_prob.copy_(
                censored_diagonal_gaussian_component_log_prob(
                    state_reward_sample,
                    state_reward_mean,
                    float(self.cfg.predictive_allocator_exploration_std),
                    lower,
                    upper,
                )
            )
            self._predictive_cached_gates.copy_(
                1.0 + self._predictive_rollout_gate_blend * (state_reward_sample - 1.0)
            )

        cursor = self._predictive_queue_cursor
        self._predictive_depth_input_queue[:, cursor].copy_(self._predictive_depth_history)
        self._predictive_state_input_queue[:, cursor].copy_(self._predictive_state_history)
        self._predictive_action_input_queue[:, cursor].copy_(self.actions)
        self._predictive_sequence_age.add_(1).clamp_(max=self.cfg.predictive_future_steps)
        if self._predictive_allocator_active():
            self._predictive_gate_control_steps += 1
        self._predictive_last_gate_step = current_step

    def _train_predictive_feasibility(
        self,
        env_ids: torch.Tensor | None = None,
        *,
        allow_partial: bool = False,
    ) -> None:
        """Train on complete windows or consume partial windows at failures."""
        if not self._predictive_updates_enabled:
            return
        horizon = self.cfg.predictive_future_steps
        minimum_age = 1 if allow_partial else horizon
        if env_ids is None:
            valid_ids = torch.nonzero(self._predictive_sequence_age >= minimum_age, as_tuple=False).squeeze(-1)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long).reshape(-1)
            valid_ids = env_ids[self._predictive_sequence_age[env_ids] >= minimum_age]
        if valid_ids.numel() == 0:
            self._predictive_last_batch_size = 0
            return

        if env_ids is None:
            batch_size = min(self.cfg.predictive_batch_size, valid_ids.numel())
            sampled_ids = valid_ids[
                torch.randperm(
                    valid_ids.numel(),
                    device=self.device,
                    generator=self._predictive_generator,
                )[:batch_size]
            ]
        else:
            # Every terminated environment must contribute its final failure
            # outcome before reset clears the per-episode circular queue.
            sampled_ids = valid_ids
            batch_size = sampled_ids.numel()

        sequence_lengths = self._predictive_sequence_age[sampled_ids].clamp(max=horizon)
        oldest_indices, ordered_indices, valid_mask = ordered_partial_horizon_indices(
            self._predictive_queue_cursor,
            sequence_lengths,
            horizon,
        )

        def gather_sequence(queue: torch.Tensor) -> torch.Tensor:
            selected = queue[sampled_ids]
            gather_shape = (batch_size, horizon) + (1,) * (selected.ndim - 2)
            gather_index = ordered_indices.reshape(gather_shape).expand_as(selected)
            return torch.gather(selected, dim=1, index=gather_index)

        with torch.inference_mode(False), torch.enable_grad():
            depth_input = self._predictive_depth_input_queue[sampled_ids, oldest_indices].detach().clone()
            state_input = self._predictive_state_input_queue[sampled_ids, oldest_indices].detach().clone()
            action_input = self._predictive_action_input_queue[sampled_ids, oldest_indices].detach().clone()
            event_sequence = gather_sequence(self._predictive_event_target_queue).detach().clone()
            future_targets = gather_sequence(self._predictive_future_target_queue).detach().clone()

            # All events use occupancy: brief wheel-contact interruptions are
            # normal for this wheel-legged platform and must not erase an entire
            # otherwise useful supervision window.
            event_targets = aggregate_prerequisite_targets(event_sequence, valid_mask)
            horizon_event_rates = torch.mean(event_targets, dim=0)
            future_targets[..., :4] = torch.nn.functional.normalize(
                future_targets[..., :4], dim=-1, eps=1.0e-6
            )
            prediction = self.predictive_feasibility_model.predict(
                depth_input,
                state_input,
                action_input,
                conservative_beta=self.cfg.predictive_uncertainty_beta,
            )
            future_weight = self.cfg.predictive_future_loss_weight
            losses = self.predictive_feasibility_model.compute_loss(
                prediction,
                event_targets,
                future_targets,
                reward_targets=None,
                event_weight=self.cfg.predictive_event_loss_weight,
                quaternion_weight=future_weight,
                linear_velocity_weight=0.25 * future_weight,
                angular_velocity_weight=0.10 * future_weight,
                future_mask=valid_mask,
                reward_weight=0.0,
                generator=self._predictive_generator,
            )
            self._predictive_optimizer.zero_grad(set_to_none=True)
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                self._predictive_auxiliary_parameters, self.cfg.predictive_gradient_clip
            )
            self._predictive_optimizer.step()
            self.predictive_feasibility_model.update_ema()

        self._predictive_optimizer_steps += 1
        self._predictive_valid_labeled_sequences += int(batch_size)

        self._predictive_last_loss = float(losses["total_loss"].detach().item())
        self._predictive_last_event_loss = float(losses["event_loss"].detach().item())
        self._predictive_last_future_loss = float(
            (
                losses["quaternion_loss"]
                + 0.25 * losses["linear_velocity_loss"]
                + 0.10 * losses["angular_velocity_loss"]
            ).detach().item()
        )
        self._predictive_last_reward_loss = float(losses["reward_loss"].detach().item())
        self._predictive_last_brier = float(losses["event_brier"].detach().item())
        self._predictive_last_accuracy = float(losses["event_accuracy"].detach().item())
        self._predictive_last_batch_size = batch_size
        self._predictive_last_horizon_event_rates = [
            float(value) for value in horizon_event_rates.detach().cpu().tolist()
        ]

    def _record_predictive_targets(
        self,
        event_targets: torch.Tensor,
    ) -> None:
        """Record event/future supervision and run the delayed auxiliary update."""
        if not self._predictive_gating_enabled or not self._predictive_updates_enabled:
            return
        cursor = self._predictive_queue_cursor
        self._predictive_event_target_queue[:, cursor].copy_(event_targets)
        current_quaternion = torch.nn.functional.normalize(
            self.body_imu.data.quat_w, dim=-1, eps=1.0e-6
        )
        current_future_target = torch.cat(
            [current_quaternion, self.robot.data.root_lin_vel_b, self.body_imu.data.ang_vel_b], dim=-1
        )
        self._predictive_future_target_queue[:, cursor].copy_(current_future_target)

        terminated_ids = torch.nonzero(self.reset_terminated, as_tuple=False).squeeze(-1)
        if terminated_ids.numel() > 0:
            self._train_predictive_feasibility(
                terminated_ids,
                allow_partial=not bool(
                    getattr(self.cfg, "outer_advantage_composer_enable", False)
                ),
            )

        current_step = int(self.common_step_counter)
        if current_step > 0 and current_step % self.cfg.predictive_train_interval == 0:
            self._train_predictive_feasibility()
        self._predictive_queue_cursor = (cursor + 1) % self.cfg.predictive_future_steps

    def update_predictive_reward_allocator(
        self,
        allocator_context: torch.Tensor,
        allocator_sample: torch.Tensor,
        old_log_probability: torch.Tensor,
        behavior_reference_rewards: torch.Tensor,
        behavior_dones: torch.Tensor,
        behavior_reward_components: torch.Tensor,
        ppo_advantages: torch.Tensor,
        reference_improvement: float,
        normalized_reference_improvement: float,
        allocation_blend: float,
        allocator_coordinate: int | None = None,
    ) -> dict[str, float]:
        """Update the 22 reward weights as an outer stochastic policy.

        Every stored context owns its actual stochastic reward action. The
        caller supplies the fixed-objective change observed on the rollout
        after those actions trained the inner PPO. That delayed real outcome
        determines the outer sign; PPO advantage only attributes it locally.
        """
        if (
            not self._predictive_gating_enabled
            or not self._predictive_updates_enabled
            or not self.cfg.predictive_allocator_train
            or not self._predictive_allocator_active()
        ):
            return {}
        # Rollout metadata is created under RSL-RL's inference_mode. Copy the
        # outer action tensors into ordinary tensors before reward-head
        # autograd; inference tensors cannot be saved for backward.
        with torch.inference_mode(False):
            allocator_context = allocator_context.detach().clone().to(self.device)
            allocator_sample = allocator_sample.detach().clone().to(self.device)
            old_log_probability = old_log_probability.detach().clone().to(self.device)
        expected_context_dim = self.predictive_feasibility_model.reward_context_dim
        if allocator_context.ndim < 1 or allocator_context.shape[-1] != expected_context_dim:
            raise ValueError(
                "Allocator context must end in the Predictor fusion dimension "
                f"{expected_context_dim}; got {tuple(allocator_context.shape)}."
            )
        allocator_prefix = allocator_context.shape[:-1]
        expected_allocator_shape = allocator_prefix + (self.cfg.predictive_reward_dim,)
        if allocator_sample.shape != expected_allocator_shape:
            raise ValueError(
                f"Allocator sample has shape {tuple(allocator_sample.shape)}; "
                f"expected {expected_allocator_shape}."
            )
        if old_log_probability.shape != allocator_sample.shape:
            raise ValueError("Allocator component log probabilities must match the reward action shape.")
        if behavior_reference_rewards.ndim != 2:
            raise ValueError("Reference rollout rewards must have shape [time, env].")
        expected_prefix = behavior_reference_rewards.shape
        if allocator_prefix not in (torch.Size(), expected_prefix):
            raise ValueError(
                "State-level allocator context must match reference rollout prefix "
                f"{tuple(expected_prefix)}; got {tuple(allocator_prefix)}."
            )
        behavior_reference_rewards = behavior_reference_rewards.to(self.device)
        behavior_dones = behavior_dones.reshape(expected_prefix).to(device=self.device, dtype=torch.bool)
        behavior_reward_components = behavior_reward_components.to(self.device)
        if behavior_reward_components.shape != expected_prefix + (self.cfg.predictive_reward_dim,):
            raise ValueError(
                "Behavior reward components must have shape [time, env, predictive_reward_dim]."
            )
        ppo_advantages = ppo_advantages.reshape(expected_prefix).to(self.device)
        if not 0.0 <= float(allocation_blend) <= 1.0:
            raise ValueError("allocation_blend must lie in [0, 1].")

        with torch.no_grad():
            def discounted_returns(rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
                returns = torch.zeros_like(rewards)
                running_return = torch.zeros(rewards.shape[1], device=self.device)
                gamma = float(self.cfg.predictive_allocator_gamma)
                for step in range(rewards.shape[0] - 1, -1, -1):
                    running_return = rewards[step] + gamma * running_return * (~dones[step]).float()
                    returns[step] = running_return
                return returns

            behavior_returns = discounted_returns(
                behavior_reference_rewards, behavior_dones
            )
            component_returns = torch.zeros_like(behavior_reward_components)
            running_component_return = torch.zeros(
                (behavior_reward_components.shape[1], self.cfg.predictive_reward_dim),
                device=self.device,
            )
            gamma = float(self.cfg.predictive_allocator_gamma)
            for step in range(behavior_reward_components.shape[0] - 1, -1, -1):
                running_component_return = behavior_reward_components[step] + (
                    gamma
                    * running_component_return
                    * (~behavior_dones[step]).float().unsqueeze(-1)
                )
                component_returns[step] = running_component_return

            reference_score = float(behavior_returns.mean().item())
            previous_reference_score = self._predictive_reference_ema
            reference_score_delta = (
                0.0
                if previous_reference_score is None
                else reference_score - float(previous_reference_score)
            )
            self._predictive_reference_ema = reference_score
            policy_improvement = float(reference_improvement)
            progress_scale = float(self.cfg.predictive_reference_progress_scale)
            expected_normalized_improvement = float(
                torch.tanh(torch.tensor(policy_improvement / progress_scale)).item()
            )
            normalized_progress = float(normalized_reference_improvement)
            same_direction = (
                abs(normalized_progress) <= 1.0e-8
                or normalized_progress * expected_normalized_improvement >= 0.0
            )
            confidence_only_shrinks = (
                abs(normalized_progress) <= abs(expected_normalized_improvement) + 1.0e-5
            )
            if not same_direction or not confidence_only_shrinks:
                raise ValueError(
                    "Normalized fixed-reference improvement must preserve the raw slope sign "
                    "and may only be shrunk by estimator confidence."
                )
            if allocator_prefix == expected_prefix:
                component_credit, component_alignment = statewise_rollout_causal_progress_credit(
                    policy_improvement,
                    progress_scale,
                    ppo_advantages,
                    component_returns,
                    advantage_weight=float(self.cfg.predictive_allocator_ppo_credit),
                    progress_weight=float(self.cfg.predictive_allocator_progress_credit),
                )
            else:
                # Compatibility for the older same-storage counterfactual path,
                # which owns one aggregate action rather than state actions.
                component_credit, component_alignment = componentwise_rollout_causal_progress_credit(
                    policy_improvement,
                    progress_scale,
                    ppo_advantages,
                    component_returns,
                    advantage_weight=float(self.cfg.predictive_allocator_ppo_credit),
                    progress_weight=float(self.cfg.predictive_allocator_progress_credit),
                )
            # Keep causal accounting correct if a future experiment explicitly
            # opts into numerical blending. The direct configuration uses 1.0.
            component_credit = component_credit * float(allocation_blend)
            component_credit = component_credit.detach().clone()
            component_alignment = component_alignment.detach().clone()

        with torch.inference_mode(False), torch.enable_grad():
            flat_context = allocator_context.reshape(-1, expected_context_dim)
            reward_mean = self._bounded_predictive_reward_mean(
                self.predictive_feasibility_model.reward_head(flat_context)
            ).reshape(expected_allocator_shape)
            exploration_std = float(self.cfg.predictive_allocator_exploration_std)
            if exploration_std <= 0.0:
                raise ValueError("predictive_allocator_exploration_std must be positive during training.")
            # Each reward multiplier receives its own advantage-attributed
            # credit. A joint scalar log-probability would push all twenty-two
            # dimensions with the same noisy rollout signal.
            lower, upper = self._predictive_reward_bounds(allocator_sample)
            log_probability = censored_diagonal_gaussian_component_log_prob(
                allocator_sample,
                reward_mean,
                exploration_std,
                lower,
                upper,
            )
            importance_ratio = torch.exp(
                (log_probability - old_log_probability).clamp(-20.0, 20.0)
            )
            clip = float(self.cfg.predictive_allocator_importance_clip)
            clipped_ratio = importance_ratio.clamp(1.0 - clip, 1.0 + clip)
            reinforce_loss = -torch.minimum(
                importance_ratio * component_credit,
                clipped_ratio * component_credit,
            ).mean()
            # No-evidence updates must preserve the explicit stability-first
            # prior. Pulling every multiplier toward one silently acted like a
            # motion curriculum and caused the observed v32 drift.
            regularization_loss = torch.square(
                reward_mean - self._predictive_reward_prior
            ).mean()
            allocator_loss = (
                reinforce_loss
                + float(self.cfg.predictive_allocator_regularization) * regularization_loss
            )
            self._predictive_allocator_optimizer.zero_grad(set_to_none=True)
            allocator_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.predictive_feasibility_model.reward_head.parameters(),
                self.cfg.predictive_gradient_clip,
            )
            self._predictive_allocator_optimizer.step()

        self._predictive_allocator_steps += 1
        self._predictive_last_allocator_loss = float(allocator_loss.detach().item())
        self._predictive_last_reference_score = reference_score
        self._predictive_last_reference_progress = reference_score_delta
        self._predictive_last_ppo_credit = float(component_credit.abs().mean().item())
        self._predictive_last_component_alignment = float(
            component_alignment.abs().mean().item()
        )
        metrics = {
            "allocator_loss": self._predictive_last_allocator_loss,
            "reference_score": reference_score,
            "reference_score_delta": reference_score_delta,
            "policy_improvement": policy_improvement,
            "policy_improvement_baseline": 0.0,
            "normalized_reference_progress": normalized_progress,
            "reference_credit": float(self.cfg.predictive_allocator_progress_credit)
            * normalized_progress
            * float(allocation_blend),
            "allocation_blend": float(allocation_blend),
            "advantage_attribution_abs": self._predictive_last_component_alignment,
            "advantage_attribution_mean": float(component_alignment.mean().item()),
            "advantage_attribution_min": float(component_alignment.min().item()),
            "advantage_attribution_max": float(component_alignment.max().item()),
            "importance_ratio": float(importance_ratio.detach().mean().item()),
            "component_alignment_abs": self._predictive_last_component_alignment,
            "component_credit_abs": self._predictive_last_ppo_credit,
            "allocator_steps": float(self._predictive_allocator_steps),
            "selected_coordinate": float(
                self._predictive_rollout_allocator_coordinate
                if allocator_coordinate is None
                else allocator_coordinate
            ),
        }
        log = self.extras.setdefault("log", {})
        for name, value in metrics.items():
            log[f"RewardAllocator/{name}"] = value
        return metrics

    def reset_predictive_allocator_learning_context(self) -> None:
        """Reset policy-specific outer-loop baselines for a fresh inner PPO."""
        if not self._predictive_gating_enabled or not self._predictive_training_enabled:
            return
        self._predictive_reference_ema = None
        self._predictive_allocator_steps = 0
        self._predictive_last_allocator_loss = 0.0
        self._predictive_last_reference_score = 0.0
        self._predictive_last_reference_progress = 0.0
        self._predictive_last_ppo_credit = 0.0
        self._predictive_last_component_alignment = 0.0
        self._predictive_allocator_context_ema.zero_()
        self._predictive_allocator_context_initialized = False
        self._predictive_rollout_allocator_residual.zero_()

    def freeze_predictive_updates(self) -> None:
        """Freeze a loaded full predictor for the fresh final-policy phase."""
        if not self._predictive_gating_enabled or not self._predictive_training_enabled:
            raise RuntimeError("A full predictive model is required before it can be frozen.")
        self._predictive_updates_enabled = False
        self.predictive_feasibility_model.eval()
        self.predictive_feasibility_model.requires_grad_(False)
        # A frozen allocator is deterministic.  Clear any exploratory sample
        # cached before the checkpoint was loaded.
        self._predictive_cached_allocator_sample.copy_(self._predictive_cached_allocator_mean)
        self._predictive_cached_allocator_log_prob.zero_()
        self._predictive_rollout_allocator_residual.zero_()

    def predictive_allocator_updates_enabled(self) -> bool:
        """Return whether rollout metadata and outer updates are required."""
        return bool(
            self._predictive_gating_enabled
            and self._predictive_updates_enabled
            and self.cfg.predictive_allocator_train
        )

    def predictive_allocator_rollout_blend(self) -> float:
        """Return the causal strength of the allocation used by this rollout."""
        return float(self._predictive_rollout_gate_blend)

    def outer_advantage_composer_enabled(self) -> bool:
        """Return whether the new five-group outer-guided path is active."""
        return bool(
            self._predictive_training_enabled
            and getattr(self.cfg, "outer_advantage_composer_enable", False)
        )

    def outer_beta_total_iterations(self) -> int:
        """Return the persisted absolute horizon of the beta schedule."""
        if not self.outer_advantage_composer_enabled():
            return 0
        return int(self._outer_beta_total_iterations)

    def outer_beta_iteration(self) -> int:
        """Return the number of completed composer rollouts."""
        if not self.outer_advantage_composer_enabled():
            return 0
        return int(self._outer_beta_iteration)

    @torch.no_grad()
    def begin_outer_advantage_rollout(self, total_iterations: int) -> float:
        """Freeze phi_k and beta_k for one complete PPO rollout."""
        if not self.outer_advantage_composer_enabled():
            return 0.0
        if total_iterations <= 0:
            raise ValueError("total_iterations must be positive.")
        self._outer_beta_total_iterations = int(total_iterations)
        if self._outer_reward_composition_mode in {"uniform", "static", "lirpg"}:
            self._outer_beta = 1.0
        elif self._outer_reward_composition_mode == "relara":
            self._outer_beta = float(self.cfg.relara_beta)
        else:
            self._outer_beta = beta_schedule(
                self._outer_beta_iteration,
                self._outer_beta_total_iterations,
                maximum=float(self.cfg.outer_beta_maximum),
                warmup_fraction=float(self.cfg.outer_beta_warmup_fraction),
            )
        if self._outer_reward_composition_mode == "composer":
            self._outer_rollout_composer.load_state_dict(
                self.outer_reward_composer.state_dict()
            )
            self._outer_rollout_composer.requires_grad_(False).eval()
        elif self._outer_reward_composition_mode == "lirpg":
            self._lirpg_rollout_reward.load_state_dict(
                self.lirpg_intrinsic_reward.state_dict()
            )
            self._lirpg_rollout_reward.requires_grad_(False).eval()
        elif self._outer_reward_composition_mode == "relara":
            self.relara_reward_agent.freeze_for_rollout()
        return float(self._outer_beta)

    def update_outer_advantage_composer(
        self,
        *,
        fused_latents: torch.Tensor,
        fixed_internal_rewards: torch.Tensor,
        normalized_group_rewards: torch.Tensor,
        fixed_outer_rewards: torch.Tensor,
        dones: torch.Tensor,
        time_outs: torch.Tensor,
        critic_observations: torch.Tensor,
        final_critic_observation: torch.Tensor,
        actor: torch.nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        actor_observations: torch.Tensor,
        actor_actions: torch.Tensor,
        actor_old_log_probabilities: torch.Tensor,
        actor_action_standard_deviations: torch.Tensor,
        actor_maximum_latent_mean: float,
        ppo_clip_param: float,
        gamma: float,
        lam: float,
        defer_lirpg_reward_update: bool = False,
    ) -> dict[str, float | torch.Tensor]:
        """Train the outer critic and phi through a virtual PPO step."""
        if not self.outer_advantage_composer_enabled() or not bool(
            getattr(self.cfg, "outer_composer_train", True)
        ):
            return {}
        if (
            self._outer_reward_composition_mode == "lirpg"
            and not defer_lirpg_reward_update
        ):
            raise RuntimeError(
                "A3 must use the official minibatch PPO-LIRPG adapter; "
                "the retired rollout-level approximation is disabled."
            )
        with torch.inference_mode(False), torch.enable_grad():
            fused_latents = fused_latents.detach().clone().to(self.device)
            fixed_internal_rewards = (
                fixed_internal_rewards.detach().clone().to(self.device)
            )
            normalized_group_rewards = normalized_group_rewards.detach().clone().to(self.device)
            fixed_outer_rewards = fixed_outer_rewards.detach().clone().to(self.device)
            dones = dones.detach().clone().to(self.device).float()
            time_outs = time_outs.detach().clone().to(self.device).float()
            critic_observations = critic_observations.detach().clone().to(self.device)
            final_critic_observation = final_critic_observation.detach().clone().to(self.device)

            with torch.no_grad():
                outer_values = self.outer_critic(critic_observations)
                final_outer_value = self.outer_critic(final_critic_observation)
                # Match standard PPO timeout semantics: bootstrap a truncation
                # before treating its rollout-storage transition as done.
                adjusted_outer_rewards = (
                    fixed_outer_rewards + float(gamma) * outer_values * time_outs
                )
                outer_advantages, outer_returns = generalized_advantage(
                    adjusted_outer_rewards,
                    outer_values,
                    final_outer_value,
                    dones,
                    gamma=float(gamma),
                    lam=float(lam),
                )

            if (
                self._outer_reward_composition_mode == "lirpg"
                and defer_lirpg_reward_update
            ):
                outer_score_tensor = fixed_outer_rewards.mean().detach()
                if self._distributed_training_active():
                    self._distributed_all_reduce_in_place(
                        outer_score_tensor, op=dist.ReduceOp.SUM
                    )
                    outer_score_tensor.div_(dist.get_world_size())
                outer_score = float(outer_score_tensor.item())
                if self._outer_previous_score is not None:
                    self._outer_learning_auc += 0.5 * (
                        self._outer_previous_score + outer_score
                    )
                self._outer_previous_score = outer_score
                metrics: dict[str, float | torch.Tensor] = {
                    "outer_advantage_mean": float(outer_advantages.mean().item()),
                    "outer_advantage_std": float(
                        outer_advantages.std(unbiased=False).item()
                    ),
                    "outer_reward": outer_score,
                    "learning_auc": float(self._outer_learning_auc),
                    "_outer_advantages_tensor": outer_advantages.detach(),
                    "_outer_returns_tensor": outer_returns.detach(),
                    "_outer_values_tensor": outer_values.detach(),
                }
                for group_name in REWARD_GROUP_NAMES:
                    metrics[f"weight_{group_name}"] = 1.0
                return metrics

            # Fit the training-only critic before asking it to attribute outer
            # credit.  Previously composer used the stale critic and this
            # network received only one update versus PPO's twelve.
            critic_loss_values: list[float] = []
            critic_grad_norm_values: list[float] = []
            critic_epochs = int(self.cfg.outer_critic_learning_epochs)
            for _ in range(critic_epochs):
                self._outer_critic_optimizer.zero_grad(set_to_none=True)
                predicted_values = self.outer_critic(critic_observations)
                critic_loss = torch.nn.functional.smooth_l1_loss(
                    predicted_values, outer_returns.detach()
                )
                critic_loss.backward()
                self._average_gradients(tuple(self.outer_critic.parameters()))
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.outer_critic.parameters(), float(self.cfg.outer_critic_gradient_clip)
                )
                self._outer_critic_optimizer.step()
                critic_loss_values.append(float(critic_loss.detach().item()))
                critic_grad_norm_values.append(float(critic_grad_norm))

            with torch.no_grad():
                # Recompute GAE from the newly fitted value baseline.  The
                # composer therefore sees current-rollout residual outer credit
                # rather than a one-rollout-stale estimate.
                outer_values = self.outer_critic(critic_observations)
                final_outer_value = self.outer_critic(final_critic_observation)
                adjusted_outer_rewards = (
                    fixed_outer_rewards + float(gamma) * outer_values * time_outs
                )
                outer_advantages, _ = generalized_advantage(
                    adjusted_outer_rewards,
                    outer_values,
                    final_outer_value,
                    dones,
                    gamma=float(gamma),
                    lam=float(lam),
                )

            if self._outer_reward_composition_mode == "lirpg":
                reward_optimizer = self._lirpg_optimizer
                reward_module = self.lirpg_intrinsic_reward
                reward_optimizer.zero_grad(set_to_none=True)
                reward_loss, reward_metrics = lirpg_meta_gradient_loss(
                    reward_module,
                    actor,
                    actor_optimizer,
                    fixed_internal_rewards,
                    outer_advantages.detach(),
                    actor_observations.detach(),
                    actor_actions.detach(),
                    actor_old_log_probabilities.detach(),
                    actor_action_standard_deviations.detach(),
                    dones,
                    gamma=float(gamma),
                    lam=float(lam),
                    clip_param=float(ppo_clip_param),
                    maximum_latent_mean=float(actor_maximum_latent_mean),
                    intrinsic_coefficient=float(self.cfg.lirpg_intrinsic_coefficient),
                    reward_l2=float(self.cfg.lirpg_reward_l2),
                    temporal_smoothness=float(self.cfg.lirpg_temporal_smoothness),
                )
            else:
                reward_optimizer = self._outer_composer_optimizer
                reward_module = self.outer_reward_composer
                reward_optimizer.zero_grad(set_to_none=True)
                reward_loss, reward_metrics = composer_meta_gradient_loss(
                    reward_module,
                    actor,
                    actor_optimizer,
                    fused_latents,
                    fixed_internal_rewards,
                    normalized_group_rewards,
                    outer_advantages.detach(),
                    actor_observations.detach(),
                    actor_actions.detach(),
                    actor_old_log_probabilities.detach(),
                    actor_action_standard_deviations.detach(),
                    dones,
                    gamma=float(gamma),
                    lam=float(lam),
                    clip_param=float(ppo_clip_param),
                    maximum_latent_mean=float(actor_maximum_latent_mean),
                    weight_to_one=float(self.cfg.outer_composer_weight_to_one),
                    temporal_smoothness=float(self.cfg.outer_composer_temporal_smoothness),
                )
            reward_parameters = tuple(reward_module.parameters())
            reward_gradients = torch.autograd.grad(reward_loss, reward_parameters)
            # Request gradients only for phi. The virtual PPO graph may contain
            # the real actor, but neither its parameters nor Adam state are
            # mutated by this outer update.
            for parameter, gradient in zip(
                reward_parameters, reward_gradients, strict=True
            ):
                parameter.grad = gradient.detach()
            self._average_gradients(reward_parameters)
            reward_grad_norm = torch.nn.utils.clip_grad_norm_(
                reward_parameters,
                float(self.cfg.outer_composer_gradient_clip),
            )
            reward_optimizer.step()

        self._outer_composer_updates += 1
        if self._outer_reward_composition_mode == "lirpg":
            self._lirpg_updates += 1
        self._outer_critic_updates += critic_epochs
        outer_score_tensor = fixed_outer_rewards.mean().detach()
        if self._distributed_training_active():
            self._distributed_all_reduce_in_place(
                outer_score_tensor, op=dist.ReduceOp.SUM
            )
            outer_score_tensor.div_(dist.get_world_size())
        outer_score = float(outer_score_tensor.item())
        if self._outer_previous_score is not None:
            self._outer_learning_auc += 0.5 * (self._outer_previous_score + outer_score)
        self._outer_previous_score = outer_score
        metrics = {
            "composer_loss": float(reward_loss.detach().item()),
            "outer_critic_loss": float(sum(critic_loss_values) / len(critic_loss_values)),
            "meta_policy_loss": float(reward_metrics["meta_policy_loss"].item()),
            "meta_outer_loss_before": float(
                reward_metrics["meta_outer_loss_before"].item()
            ),
            "meta_outer_loss_after": float(
                reward_metrics["meta_outer_loss_after"].item()
            ),
            "predicted_outer_improvement": float(
                reward_metrics["predicted_outer_improvement"].item()
            ),
            "meta_inner_policy_loss": float(
                reward_metrics["inner_policy_loss"].item()
            ),
            # Kept as a diagnostic only. It no longer supplies composer credit.
            "advantage_cosine": float(
                reward_metrics["advantage_cosine"].item()
            ),
            "weight_to_one": float(
                reward_metrics.get("weight_to_one_loss", torch.zeros(())).item()
            ),
            "temporal_smoothness": float(
                reward_metrics["temporal_smoothness_loss"].item()
            ),
            "composer_gradient_norm": float(reward_grad_norm),
            "outer_critic_gradient_norm": float(
                sum(critic_grad_norm_values) / len(critic_grad_norm_values)
            ),
            "outer_advantage_mean": float(outer_advantages.mean().item()),
            "outer_advantage_std": float(outer_advantages.std(unbiased=False).item()),
            "shaping_advantage_mean": float(
                reward_metrics.get(
                    "shaping_advantage",
                    reward_metrics.get("intrinsic_advantage_mean", torch.zeros(())),
                ).mean().item()
            ),
            "outer_reward": outer_score,
            "learning_auc": float(self._outer_learning_auc),
        }
        if self._outer_reward_composition_mode == "lirpg":
            for name in (
                "intrinsic_reward_mean",
                "intrinsic_reward_std",
                "intrinsic_reward_abs",
                "reward_l2_loss",
            ):
                metrics[name] = float(reward_metrics[name].item())
            rollout_weights = self._outer_cached_group_weights
        else:
            rollout_weights = reward_metrics["weights"]
        for group_index, group_name in enumerate(REWARD_GROUP_NAMES):
            metrics[f"weight_{group_name}"] = float(
                rollout_weights[..., group_index].mean().item()
            )
        metrics = self._distributed_mean_metrics(metrics)
        self._outer_last_metrics = metrics
        self._outer_beta_iteration += 1
        # This is the only point at which phi_{k+1} becomes eligible for
        # collection. Rewards already copied into PPO storage remain immutable.
        self.begin_outer_advantage_rollout(self._outer_beta_total_iterations)
        return metrics

    def update_relara_reward_agent(
        self,
        *,
        actor_observations: torch.Tensor,
        actor_actions: torch.Tensor,
        final_actor_observation: torch.Tensor,
        final_actor_action: torch.Tensor,
        proposed_rewards: torch.Tensor,
        fixed_outer_rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict[str, float]:
        """Store one vector rollout and apply the official Reward-Agent SAC update.

        One batch update per vector control step preserves ReLara's sample
        update ratio: a 24x512 rollout yields 24 updates of batch size 512.
        The updated actor is frozen only after all rollout-k rewards are stored.
        """
        if self._outer_reward_composition_mode != "relara":
            raise RuntimeError("ReLara update called for another reward mode.")
        time_steps, environments, _ = actor_observations.shape
        if actor_actions.shape[:2] != (time_steps, environments):
            raise ValueError("ReLara observations and actions must share [T,N].")
        next_observations = torch.cat(
            (actor_observations[1:], final_actor_observation.unsqueeze(0)), dim=0
        )
        next_actions = torch.cat(
            (actor_actions[1:], final_actor_action.unsqueeze(0)), dim=0
        )
        contexts = torch.cat((actor_observations, actor_actions), dim=-1)
        next_contexts = torch.cat((next_observations, next_actions), dim=-1)
        self.relara_reward_agent.add_rollout(
            contexts,
            next_contexts,
            proposed_rewards,
            fixed_outer_rewards,
            dones,
        )
        metrics = self.relara_reward_agent.optimize(time_steps)
        outer_score = float(fixed_outer_rewards.detach().mean().item())
        if self._outer_previous_score is not None:
            self._outer_learning_auc += 0.5 * (
                self._outer_previous_score + outer_score
            )
        self._outer_previous_score = outer_score
        self._relara_updates = int(self.relara_reward_agent.gradient_steps)
        metrics.update(
            outer_reward=outer_score,
            learning_auc=float(self._outer_learning_auc),
            proposed_reward=float(proposed_rewards.detach().mean().item()),
            proposed_reward_std=float(
                proposed_rewards.detach().std(unbiased=False).item()
            ),
            beta=float(self.cfg.relara_beta),
        )
        self._outer_last_metrics = metrics
        self._outer_composer_updates += 1
        self._outer_beta_iteration += 1
        self.begin_outer_advantage_rollout(self._outer_beta_total_iterations)
        return metrics

    def complete_official_lirpg_rollout(
        self,
        metrics: dict[str, float],
        *,
        meta_updates: int,
    ) -> None:
        """Activate the official LIRPG reward parameters for the next rollout."""

        if self._outer_reward_composition_mode != "lirpg":
            raise RuntimeError("Official LIRPG completion called for another mode.")
        if meta_updates <= 0:
            raise ValueError("Official LIRPG must perform at least one meta update.")
        self._outer_composer_updates += 1
        self._lirpg_updates += int(meta_updates)
        self._outer_critic_updates += int(meta_updates)
        self._outer_last_metrics = self._distributed_mean_metrics(metrics)
        self._outer_beta_iteration += 1
        self.begin_outer_advantage_rollout(self._outer_beta_total_iterations)

    def save_predictive_feasibility(
        self,
        full_checkpoint_path: str,
        encoder_checkpoint_path: str,
        checkpoint_group_id: str | None = None,
    ) -> None:
        """Save the training model and the encoder-only deployment artifact."""
        self.save_predictive_feasibility_full(full_checkpoint_path, checkpoint_group_id)
        self.save_deployable_depth_encoder(encoder_checkpoint_path, checkpoint_group_id)

    def _predictive_checkpoint_metadata(
        self, checkpoint_group_id: str | None, *, include_training_state: bool = False
    ) -> dict[str, Any]:
        metadata = {
            # Version 34 holds the reward policy fixed for a complete block of
            # inner updates and measures its real fixed-objective learning speed
            # on a PPO-free evaluation rollout before changing the allocator.
            "predictive_allocator_credit_version": 37,
            "common_step_counter": int(self.common_step_counter),
            "predictive_optimizer_steps": int(self._predictive_optimizer_steps),
            "predictive_valid_labeled_sequences": int(self._predictive_valid_labeled_sequences),
            "predictive_gate_control_steps": int(self._predictive_gate_control_steps),
            "predictive_allocator_steps": int(getattr(self, "_predictive_allocator_steps", 0)),
            "predictive_reference_ema": getattr(self, "_predictive_reference_ema", None),
            "predictive_allocator_context_ema": self._predictive_allocator_context_ema.detach()
            .cpu()
            .tolist(),
            "predictive_allocator_context_initialized": bool(
                self._predictive_allocator_context_initialized
            ),
            "predictive_updates_enabled": bool(self._predictive_updates_enabled),
            "checkpoint_group_id": checkpoint_group_id,
        }
        if self.outer_advantage_composer_enabled():
            metadata["outer_advantage_composer_version"] = 4
            metadata["outer_reward_composition_mode"] = (
                self._outer_reward_composition_mode
            )
            metadata["outer_static_group_weights"] = list(
                self._outer_static_group_weights
            )
            metadata["outer_beta_iteration"] = int(self._outer_beta_iteration)
            metadata["outer_beta_total_iterations"] = int(self._outer_beta_total_iterations)
            metadata["outer_beta"] = float(self._outer_beta)
            metadata["outer_composer_updates"] = int(self._outer_composer_updates)
            metadata["outer_critic_updates"] = int(self._outer_critic_updates)
            metadata["outer_learning_auc"] = float(self._outer_learning_auc)
            metadata["outer_previous_score"] = self._outer_previous_score
            if include_training_state:
                metadata["outer_composer_state_dict"] = self.outer_reward_composer.state_dict()
                metadata["outer_rollout_composer_state_dict"] = (
                    self._outer_rollout_composer.state_dict()
                )
                metadata["outer_critic_state_dict"] = self.outer_critic.state_dict()
                metadata["outer_group_rms_state_dict"] = self.outer_group_rms.state_dict()
                metadata["outer_composer_optimizer_state_dict"] = (
                    self._outer_composer_optimizer.state_dict()
                )
                metadata["outer_critic_optimizer_state_dict"] = (
                    self._outer_critic_optimizer.state_dict()
                )
                if self._outer_reward_composition_mode == "lirpg":
                    metadata["lirpg_implementation_version"] = int(
                        self.cfg.lirpg_implementation_version
                    )
                    metadata["lirpg_extrinsic_coefficient"] = float(
                        self.cfg.lirpg_extrinsic_coefficient
                    )
                    metadata["lirpg_intrinsic_coefficient"] = float(
                        self.cfg.lirpg_intrinsic_coefficient
                    )
                    metadata["lirpg_intrinsic_reward_state_dict"] = (
                        self.lirpg_intrinsic_reward.state_dict()
                    )
                    metadata["lirpg_rollout_reward_state_dict"] = (
                        self._lirpg_rollout_reward.state_dict()
                    )
                    metadata["lirpg_optimizer_state_dict"] = (
                        self._lirpg_optimizer.state_dict()
                    )
                    metadata["lirpg_updates"] = int(self._lirpg_updates)
                elif self._outer_reward_composition_mode == "relara":
                    metadata["relara_implementation_version"] = int(
                        self.cfg.relara_implementation_version
                    )
                    metadata["relara_beta"] = float(self.cfg.relara_beta)
                    metadata["relara_reward_agent_state"] = (
                        self.relara_reward_agent.training_state_dict()
                    )
                    metadata["relara_updates"] = int(self._relara_updates)
        return metadata

    def save_predictive_feasibility_full(
        self, checkpoint_path: str, checkpoint_group_id: str | None = None
    ) -> None:
        """Save the full predictor/optimizer training sidecar."""
        if not self._predictive_gating_enabled or not self._predictive_training_enabled:
            return
        full_path = Path(checkpoint_path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        self.predictive_feasibility_model.save_full(
            full_path,
            optimizer=self._predictive_optimizer,
            allocator_optimizer=self._predictive_allocator_optimizer,
            extra=self._predictive_checkpoint_metadata(
                checkpoint_group_id, include_training_state=True
            ),
        )

    def save_deployable_depth_encoder(
        self, checkpoint_path: str, checkpoint_group_id: str | None = None
    ) -> None:
        """Save the EMA encoder-only deployment sidecar."""
        if not self._predictive_gating_enabled or not self._predictive_training_enabled:
            return
        encoder_path = Path(checkpoint_path)
        encoder_path.parent.mkdir(parents=True, exist_ok=True)
        self.predictive_feasibility_model.save_encoder(
            encoder_path,
            extra=self._predictive_checkpoint_metadata(checkpoint_group_id),
        )

    def load_predictive_feasibility(self, checkpoint_path: str, load_optimizer: bool = True) -> bool:
        """Restore the full training-only predictor when resuming PPO."""
        path = Path(checkpoint_path)
        if not path.is_file():
            print(f"[YawBot] Predictive feasibility checkpoint not found: {path}")
            return False
        if not self._predictive_training_enabled:
            raise RuntimeError("Full predictive model loading is only valid in training mode.")
        optimizer = self._predictive_optimizer if load_optimizer else None
        required_extra = (
            {"outer_advantage_composer_version": 4}
            if self.outer_advantage_composer_enabled()
            else {"predictive_allocator_credit_version": 37}
        )
        metadata = self.predictive_feasibility_model.load_full(
            path,
            optimizer=optimizer,
            allocator_optimizer=self._predictive_allocator_optimizer if load_optimizer else None,
            map_location=self.device,
            required_extra=required_extra,
        )
        if self.outer_advantage_composer_enabled():
            validate_outer_checkpoint_state(metadata)
            saved_mode = str(
                metadata.get("outer_reward_composition_mode", "composer")
            )
            if saved_mode != self._outer_reward_composition_mode:
                raise ValueError(
                    "Reward-composition checkpoint mode mismatch: "
                    f"saved={saved_mode}, requested={self._outer_reward_composition_mode}."
                )
            saved_static_weights = validate_static_group_weights(
                metadata.get("outer_static_group_weights", (1.0,) * len(REWARD_GROUP_NAMES))
            )
            if saved_static_weights != self._outer_static_group_weights:
                raise ValueError(
                    "Static reward weights differ from the resumed checkpoint."
                )
            self.outer_reward_composer.load_state_dict(metadata["outer_composer_state_dict"])
            self._outer_rollout_composer.load_state_dict(
                metadata["outer_rollout_composer_state_dict"]
            )
            self.outer_critic.load_state_dict(metadata["outer_critic_state_dict"])
            self.outer_group_rms.load_state_dict(metadata["outer_group_rms_state_dict"])
            if load_optimizer:
                self._outer_composer_optimizer.load_state_dict(
                    metadata["outer_composer_optimizer_state_dict"]
                )
                self._outer_critic_optimizer.load_state_dict(
                    metadata["outer_critic_optimizer_state_dict"]
                )
            if self._outer_reward_composition_mode == "lirpg":
                lirpg_keys = {
                    "lirpg_implementation_version",
                    "lirpg_extrinsic_coefficient",
                    "lirpg_intrinsic_coefficient",
                    "lirpg_intrinsic_reward_state_dict",
                    "lirpg_rollout_reward_state_dict",
                    "lirpg_optimizer_state_dict",
                    "lirpg_updates",
                }
                missing_lirpg = sorted(lirpg_keys.difference(metadata))
                if missing_lirpg:
                    raise ValueError(
                        f"LIRPG checkpoint is incomplete: {missing_lirpg}."
                    )
                saved_lirpg_version = int(
                    metadata["lirpg_implementation_version"]
                )
                requested_lirpg_version = int(
                    self.cfg.lirpg_implementation_version
                )
                if saved_lirpg_version != requested_lirpg_version:
                    raise ValueError(
                        "LIRPG checkpoint implementation mismatch: "
                        f"saved={saved_lirpg_version}, "
                        f"requested={requested_lirpg_version}."
                    )
                for coefficient_name in (
                    "lirpg_extrinsic_coefficient",
                    "lirpg_intrinsic_coefficient",
                ):
                    saved_coefficient = float(metadata[coefficient_name])
                    requested_coefficient = float(
                        getattr(self.cfg, coefficient_name)
                    )
                    if saved_coefficient != requested_coefficient:
                        raise ValueError(
                            f"{coefficient_name} differs from the resumed "
                            "LIRPG checkpoint."
                        )
                self.lirpg_intrinsic_reward.load_state_dict(
                    metadata["lirpg_intrinsic_reward_state_dict"]
                )
                self._lirpg_rollout_reward.load_state_dict(
                    metadata["lirpg_rollout_reward_state_dict"]
                )
                if load_optimizer:
                    self._lirpg_optimizer.load_state_dict(
                        metadata["lirpg_optimizer_state_dict"]
                    )
                self._lirpg_updates = int(metadata["lirpg_updates"])
            elif self._outer_reward_composition_mode == "relara":
                relara_keys = {
                    "relara_implementation_version",
                    "relara_beta",
                    "relara_reward_agent_state",
                    "relara_updates",
                }
                missing_relara = sorted(relara_keys.difference(metadata))
                if missing_relara:
                    raise ValueError(
                        f"ReLara checkpoint is incomplete: {missing_relara}."
                    )
                if int(metadata["relara_implementation_version"]) != int(
                    self.cfg.relara_implementation_version
                ):
                    raise ValueError("ReLara checkpoint implementation mismatch.")
                if float(metadata["relara_beta"]) != float(self.cfg.relara_beta):
                    raise ValueError("ReLara beta differs from the resumed checkpoint.")
                self.relara_reward_agent.load_training_state_dict(
                    metadata["relara_reward_agent_state"]
                )
                self._relara_updates = int(metadata["relara_updates"])
            self._outer_beta_iteration = int(metadata["outer_beta_iteration"])
            self._outer_beta_total_iterations = int(metadata["outer_beta_total_iterations"])
            self._outer_beta = float(metadata["outer_beta"])
            self._outer_composer_updates = int(metadata["outer_composer_updates"])
            self._outer_critic_updates = int(metadata["outer_critic_updates"])
            self._outer_learning_auc = float(metadata["outer_learning_auc"])
            previous_score = metadata.get("outer_previous_score")
            self._outer_previous_score = (
                None if previous_score is None else float(previous_score)
            )
        self._predictive_optimizer_steps = int(metadata.get("predictive_optimizer_steps", 0))
        self._predictive_valid_labeled_sequences = int(metadata.get("predictive_valid_labeled_sequences", 0))
        # Pre-fix checkpoints have no dedicated gate clock.  Restart their gate
        # bootstrap instead of mapping the old simulator counter to an already
        # saturated schedule.
        self._predictive_gate_control_steps = int(metadata.get("predictive_gate_control_steps", 0))
        self._predictive_allocator_steps = int(metadata.get("predictive_allocator_steps", 0))
        reference_ema = metadata.get("predictive_reference_ema")
        self._predictive_reference_ema = None if reference_ema is None else float(reference_ema)
        allocator_context = metadata.get("predictive_allocator_context_ema")
        if allocator_context is not None:
            allocator_context_tensor = torch.as_tensor(
                allocator_context,
                device=self.device,
                dtype=self._predictive_allocator_context_ema.dtype,
            )
            if allocator_context_tensor.shape != self._predictive_allocator_context_ema.shape:
                raise ValueError(
                    "Predictive checkpoint allocator context has shape "
                    f"{tuple(allocator_context_tensor.shape)}; expected "
                    f"{tuple(self._predictive_allocator_context_ema.shape)}."
                )
            self._predictive_allocator_context_ema.copy_(allocator_context_tensor)
            self._predictive_allocator_context_initialized = bool(
                metadata.get("predictive_allocator_context_initialized", True)
            )
        else:
            self._predictive_allocator_context_ema.zero_()
            self._predictive_allocator_context_initialized = False
        self._predictive_updates_enabled = bool(
            metadata.get("predictive_updates_enabled", self._predictive_updates_enabled)
        )
        if not self._predictive_updates_enabled:
            self.predictive_feasibility_model.eval()
            self.predictive_feasibility_model.requires_grad_(False)
        if "predictive_gate_control_steps" not in metadata:
            print("[YawBot] Migrating a legacy predictor checkpoint: predictive gate warm-up restarts at zero.")
        self._predictive_gate_blend = linear_warmup_blend(
            self._predictive_gate_control_steps,
            self.cfg.predictive_gate_warmup_control_steps,
            self.cfg.predictive_gate_ramp_control_steps,
        )
        self._predictive_last_observation_step = -1
        self._predictive_last_gate_step = -1
        print(f"[YawBot] Loaded predictive feasibility model: {path}")
        return True

    def load_deployable_depth_encoder(self, checkpoint_path: str) -> bool:
        """Load only the depth encoder used by the actor during playback/deployment."""
        path = Path(checkpoint_path)
        if not path.is_file():
            print(f"[YawBot] Deployable depth encoder checkpoint not found: {path}")
            return False
        encoder, _ = DepthFeatureEncoder.from_checkpoint(path, map_location=self.device)
        expected = {
            "depth_history_steps": self.cfg.predictive_history_steps,
            "depth_height": self.cfg.depth_observation_height,
            "depth_width": self.cfg.depth_observation_width,
            "latent_dim": self.cfg.predictive_depth_latent_dim,
        }
        if encoder.get_config() != expected:
            raise ValueError(
                f"Depth encoder checkpoint config {encoder.get_config()} does not match task config {expected}."
            )
        if self._predictive_training_enabled:
            self.predictive_feasibility_model.policy_depth_encoder.load_state_dict(encoder.state_dict())
        else:
            self.deployable_depth_encoder = encoder.to(self.device)
            self.deployable_depth_encoder.eval()
        self._predictive_last_observation_step = -1
        print(f"[YawBot] Loaded deployable depth encoder: {path}")
        return True

    def _train_pose_predictor(self) -> None:
        valid_ids = torch.nonzero(
            self._pose_sequence_age
            >= self.cfg.pose_predictor_history_steps + self.cfg.pose_predictor_future_steps,
            as_tuple=False,
        ).squeeze(-1)
        if valid_ids.numel() == 0:
            self._pose_predictor_last_batch_size = 0
            return

        batch_size = min(self.cfg.pose_predictor_batch_size, valid_ids.numel())
        sampled_ids = valid_ids[
            torch.randperm(valid_ids.numel(), device=self.device)[:batch_size]
        ]
        predictor_depth_input = self._predictor_depth_input_queue[sampled_ids, 0]
        predictor_state_input = self._predictor_state_input_queue[sampled_ids, 0]
        future_state_target = self._future_state_target_queue[sampled_ids, 1:].clone()
        target_quaternion = torch.nn.functional.normalize(
            future_state_target[..., :4], dim=-1, eps=1.0e-6
        )
        future_state_target = torch.cat([target_quaternion, future_state_target[..., 4:]], dim=-1)

        # RSL-RL collects rollouts under inference_mode, so explicitly create
        # ordinary autograd tensors for this auxiliary update.
        with torch.inference_mode(False), torch.enable_grad():
            predictor_depth_input = predictor_depth_input.detach().clone()
            predictor_state_input = predictor_state_input.detach().clone()
            future_state_target = future_state_target.detach().clone()
            prediction = self.pose_predictor(predictor_depth_input, predictor_state_input)
            loss, quaternion_loss, linear_velocity_loss, angular_velocity_loss = future_state_prediction_loss(
                prediction,
                future_state_target,
                linear_velocity_weight=self.cfg.pose_predictor_linear_velocity_loss_weight,
                angular_velocity_weight=self.cfg.pose_predictor_angular_velocity_loss_weight,
            )
            self._pose_predictor_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.pose_predictor.parameters(), self.cfg.pose_predictor_gradient_clip
            )
            self._pose_predictor_optimizer.step()

        self._pose_predictor_last_loss = float(loss.detach().item())
        self._pose_predictor_last_quaternion_loss = float(quaternion_loss.detach().item())
        self._pose_predictor_last_linear_velocity_loss = float(linear_velocity_loss.detach().item())
        self._pose_predictor_last_angular_velocity_loss = float(angular_velocity_loss.detach().item())
        self._pose_predictor_last_batch_size = batch_size

    def _update_pose_prediction(
        self, depth_observation: torch.Tensor, state_observation: torch.Tensor
    ) -> torch.Tensor:
        """Update temporal buffers and return five future poses and body velocities."""
        current_step = int(self.common_step_counter)
        if self._pose_predictor_last_step == current_step:
            return self._cached_pose_prediction.flatten(start_dim=1)
        if not self.cfg.pose_predictor_enabled:
            self._cached_pose_prediction = torch.zeros(
                self.num_envs,
                self.cfg.pose_predictor_future_steps,
                self.cfg.pose_predictor_output_dim,
                device=self.device,
            )
            self._pose_predictor_last_loss = 0.0
            self._pose_predictor_last_quaternion_loss = 0.0
            self._pose_predictor_last_linear_velocity_loss = 0.0
            self._pose_predictor_last_angular_velocity_loss = 0.0
            self._pose_predictor_last_batch_size = 0
            self._pose_predictor_last_step = current_step
            return self._cached_pose_prediction.flatten(start_dim=1)

        fresh_mask = self._pose_sequence_age == 0
        active_mask = ~fresh_mask
        if torch.any(active_mask):
            self._depth_observation_history[active_mask, :-1] = self._depth_observation_history[
                active_mask, 1:
            ].clone()
            self._depth_observation_history[active_mask, -1] = depth_observation[active_mask].flatten(start_dim=1)
            self._state_observation_history[active_mask, :-1] = self._state_observation_history[
                active_mask, 1:
            ].clone()
            self._state_observation_history[active_mask, -1] = state_observation[active_mask]
        if torch.any(fresh_mask):
            self._depth_observation_history[fresh_mask] = depth_observation[fresh_mask].flatten(start_dim=1).unsqueeze(1)
            self._state_observation_history[fresh_mask] = state_observation[fresh_mask].unsqueeze(1)

        predictor_depth_input = self._depth_observation_history.flatten(start_dim=1)
        predictor_state_input = self._state_observation_history.flatten(start_dim=1)
        self._predictor_depth_input_queue[:, :-1] = self._predictor_depth_input_queue[:, 1:].clone()
        self._predictor_depth_input_queue[:, -1] = predictor_depth_input
        self._predictor_state_input_queue[:, :-1] = self._predictor_state_input_queue[:, 1:].clone()
        self._predictor_state_input_queue[:, -1] = predictor_state_input
        self._future_state_target_queue[:, :-1] = self._future_state_target_queue[:, 1:].clone()
        current_pose = torch.nn.functional.normalize(self.body_imu.data.quat_w, dim=-1, eps=1.0e-6)
        current_target = torch.cat(
            [current_pose, self.robot.data.root_lin_vel_b, self.body_imu.data.ang_vel_b],
            dim=-1,
        )
        self._future_state_target_queue[:, -1] = current_target
        self._pose_sequence_age += 1

        with torch.no_grad():
            self._cached_pose_prediction = self.pose_predictor(
                predictor_depth_input, predictor_state_input
            ).detach()

        if (
            self.cfg.pose_predictor_train
            and current_step > 0
            and current_step % self.cfg.pose_predictor_train_interval == 0
        ):
            self._train_pose_predictor()

        self._pose_predictor_last_step = current_step
        return self._cached_pose_prediction.flatten(start_dim=1)

    def save_pose_predictor(
        self, checkpoint_path: str, checkpoint_group_id: str | None = None
    ) -> None:
        """Save the auxiliary predictor beside an RSL-RL checkpoint."""
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.pose_predictor.state_dict(),
                "optimizer_state_dict": self._pose_predictor_optimizer.state_dict(),
                "checkpoint_group_id": checkpoint_group_id,
            },
            path,
        )

    def load_pose_predictor(self, checkpoint_path: str, load_optimizer: bool = True) -> bool:
        """Load the auxiliary predictor if its checkpoint exists."""
        path = Path(checkpoint_path)
        if not path.is_file():
            print(f"[YawBot] Pose predictor checkpoint not found: {path}")
            return False
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.pose_predictor.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self._pose_predictor_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._pose_predictor_last_step = -1
        print(f"[YawBot] Loaded pose predictor: {path}")
        return True

    def get_curriculum_state(self) -> dict:
        """Return the global curriculum state stored with a training checkpoint."""
        return {
            "version": 1,
            "stage": int(self._curriculum_stage),
            "active_stage": int(self._curriculum_active_stage),
            "episode_count": int(self._curriculum_episode_count),
            "ema": self._curriculum_ema.detach().cpu(),
            "history": [list(values) for values in self._curriculum_history],
            "last_unlock": int(self._curriculum_last_unlock),
            "last_unlock_episode": int(self._curriculum_last_unlock_episode),
            "last_means": self._curriculum_last_means.detach().cpu(),
        }

    def load_curriculum_state(self, state: dict | None) -> bool:
        """Restore global curriculum progress from a training checkpoint."""
        if not state:
            return False

        stage = max(1, min(4, int(state.get("stage", 1))))
        active_stage = max(1, min(stage, int(state.get("active_stage", stage))))
        self._curriculum_stage = stage
        self._curriculum_active_stage = active_stage
        self._curriculum_episode_count = max(0, int(state.get("episode_count", 0)))
        self._curriculum_ema.copy_(
            torch.as_tensor(state.get("ema", self._curriculum_ema), device=self.device, dtype=self._curriculum_ema.dtype)
        )

        history = state.get("history", [])
        self._curriculum_history = []
        for index in range(3):
            values = list(history[index]) if index < len(history) else []
            self._curriculum_history.append(values[-self.cfg.curriculum_window_episodes :])

        self._curriculum_last_unlock = int(state.get("last_unlock", stage))
        self._curriculum_last_unlock_episode = max(0, int(state.get("last_unlock_episode", 0)))
        self._curriculum_last_means.copy_(
            torch.as_tensor(
                state.get("last_means", self._curriculum_last_means),
                device=self.device,
                dtype=self._curriculum_last_means.dtype,
            )
        )
        print(
            f"[YawBot] Restored curriculum: stage={self._curriculum_stage}, "
            f"active_stage={self._curriculum_active_stage}, episodes={self._curriculum_episode_count}"
        )
        return True

    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        """Resample linear and yaw velocity commands."""
        if env_ids.numel() == 0:
            return

        if not self._velocity_commands_enabled():
            self._commands[env_ids] = 0.0
            self._command_time_left[env_ids] = 0.0
            return

        if self.cfg.use_fixed_velocity_command:
            self._commands[env_ids, 0] = self.cfg.fixed_command_lin_vel_x
            self._commands[env_ids, 1] = self.cfg.fixed_command_yaw_vel
            self._command_time_left[env_ids] = float("inf")
            return

        lin_low, lin_high = self.cfg.command_lin_vel_x_range
        yaw_low, yaw_high = self.cfg.command_yaw_vel_range
        time_low, time_high = self.cfg.command_resample_time_range

        lin_cmd = self._sample_command_with_deadzone(
            env_ids.numel(),
            lin_low,
            lin_high,
            self.cfg.command_lin_vel_x_min_abs,
        )
        yaw_cmd = torch.zeros(env_ids.numel(), device=self.device)
        yaw_active = torch.rand(env_ids.numel(), device=self.device) < self.cfg.command_yaw_probability
        if torch.any(yaw_active):
            yaw_cmd[yaw_active] = self._sample_command_with_deadzone(
                int(torch.sum(yaw_active).item()),
                yaw_low,
                yaw_high,
                self.cfg.command_yaw_vel_min_abs,
            )
        stop_command = torch.rand(env_ids.numel(), device=self.device) < self.cfg.command_stop_probability
        lin_cmd[stop_command] = 0.0
        yaw_cmd[stop_command] = 0.0
        self._commands[env_ids, 0] = lin_cmd
        self._commands[env_ids, 1] = yaw_cmd
        self._command_time_left[env_ids] = time_low + (time_high - time_low) * torch.rand(
            env_ids.numel(), device=self.device
        )

    def _velocity_commands_enabled(self) -> bool:
        """Return whether command rewards are active enough to expose nonzero commands."""
        if not self.cfg.curriculum_enable:
            return True
        return self._curriculum_active_stage >= self.cfg.velocity_command_curriculum_start_stage

    def _advance_commanded_planar_pose(self) -> torch.Tensor:
        """Integrate velocity commands and return planar distance from the ideal trajectory."""
        yaw_delta = self._commands[:, 1] * self._command_dt
        half_yaw_delta = 0.5 * yaw_delta
        half_cos = torch.cos(half_yaw_delta)
        half_sin = torch.sin(half_yaw_delta)
        heading_x = self._commanded_heading_w[:, 0]
        heading_y = self._commanded_heading_w[:, 1]
        midpoint_heading = torch.stack(
            [
                half_cos * heading_x - half_sin * heading_y,
                half_sin * heading_x + half_cos * heading_y,
            ],
            dim=1,
        )
        self._commanded_position_w += self._commands[:, :1] * self._command_dt * midpoint_heading

        yaw_cos = torch.cos(yaw_delta)
        yaw_sin = torch.sin(yaw_delta)
        self._commanded_heading_w = torch.stack(
            [
                yaw_cos * heading_x - yaw_sin * heading_y,
                yaw_sin * heading_x + yaw_cos * heading_y,
            ],
            dim=1,
        )
        return torch.linalg.vector_norm(
            self.robot.data.root_pos_w[:, :2] - self._commanded_position_w,
            dim=1,
        )

    def _sample_command_with_deadzone(
        self, num_samples: int, low: float, high: float, min_abs: float
    ) -> torch.Tensor:
        """Sample commands while avoiding a dead-zone around zero when requested."""
        if num_samples == 0:
            return torch.zeros(0, device=self.device)
        if min_abs <= 0.0 or not (low < 0.0 < high):
            return low + (high - low) * torch.rand(num_samples, device=self.device)

        samples = torch.empty(num_samples, device=self.device)
        choose_positive = torch.rand(num_samples, device=self.device) < 0.5

        pos_low = max(min_abs, 0.0)
        pos_high = high
        neg_low = low
        neg_high = min(-min_abs, 0.0)

        if pos_low < pos_high:
            pos_count = int(torch.sum(choose_positive).item())
            if pos_count > 0:
                samples[choose_positive] = pos_low + (pos_high - pos_low) * torch.rand(pos_count, device=self.device)
        else:
            choose_positive[:] = False

        choose_negative = ~choose_positive
        if neg_low < neg_high:
            neg_count = int(torch.sum(choose_negative).item())
            if neg_count > 0:
                samples[choose_negative] = neg_low + (neg_high - neg_low) * torch.rand(neg_count, device=self.device)
        else:
            samples[choose_negative] = pos_low + (pos_high - pos_low) * torch.rand(
                int(torch.sum(choose_negative).item()), device=self.device
            )

        return samples

    def _compute_equivalent_knee_angle_from_branch_hips(
        self, branch_hip_angles_deg: torch.Tensor, mapped_hip_angles_deg: torch.Tensor
    ) -> torch.Tensor:
        """Compute equivalent knee angle t from branch hip angle a and mapped hip angle b.

        All angles are in degrees. This implements the provided closed-form relationship but
        uses atan2 for stable quadrant handling:

            t = -180 + a + atan2(y, x) + acos(-sqrt(x^2 + y^2) / 200)

        where:
            x = 60 * (cos(b) - cos(a))
            y = 60 * (sin(b) + sin(a)) + 45
        """
        a_rad = torch.deg2rad(branch_hip_angles_deg)
        b_rad = torch.deg2rad(mapped_hip_angles_deg)
        x = 60.0 * (torch.cos(b_rad) - torch.cos(a_rad))
        y = 60.0 * (torch.sin(b_rad) + torch.sin(a_rad)) + 45.0
        chord = torch.sqrt(torch.square(x) + torch.square(y))
        inner_angle_deg = torch.rad2deg(torch.atan2(y, x))
        knee_triangle_angle_deg = torch.rad2deg(torch.arccos(torch.clamp(-chord / 200.0, -1.0, 1.0)))
        return -180.0 + branch_hip_angles_deg + inner_angle_deg + knee_triangle_angle_deg

    def _compute_equivalent_knee_angle_from_branch_hips_rad(
        self, branch_hip_angles_rad: torch.Tensor, mapped_hip_angles_rad: torch.Tensor
    ) -> torch.Tensor:
        """Same as `_compute_equivalent_knee_angle_from_branch_hips` but with radian inputs/outputs."""
        knee_deg = self._compute_equivalent_knee_angle_from_branch_hips(
            torch.rad2deg(branch_hip_angles_rad),
            torch.rad2deg(mapped_hip_angles_rad),
        )
        return torch.deg2rad(knee_deg)

    def _map_branch_and_parallel_hips_to_sim_servo_targets(
        self, branch_hip_targets: torch.Tensor, mapped_parallel_hip_targets: torch.Tensor
    ) -> torch.Tensor:
        """Map branch hip angle a and parallel hip angle b to simulation servo targets [hip, knee, hip, knee]."""
        semantic_knee_targets = self._compute_equivalent_knee_angle_from_branch_hips_rad(
            branch_hip_targets,
            mapped_parallel_hip_targets,
        )
        servo_targets = torch.zeros(
            (branch_hip_targets.shape[0], 4),
            device=branch_hip_targets.device,
            dtype=branch_hip_targets.dtype,
        )
        servo_targets[:, [0, 2]] = branch_hip_targets
        servo_targets[:, 1] = semantic_knee_targets[:, 0]
        servo_targets[:, 3] = -semantic_knee_targets[:, 1]
        return servo_targets

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.termination_contact_sensor = ContactSensor(self.cfg.termination_contact_sensor)
        self.wheel_contact_sensor = ContactSensor(self.cfg.wheel_contact_sensor)
        self.body_imu = Imu(self.cfg.body_imu)
        self.depth_camera = RayCasterCamera(self.cfg.depth_camera)
        terrain_cfg = self.cfg.terrain
        terrain_cfg.num_envs = self.scene.cfg.num_envs
        terrain_cfg.env_spacing = self.scene.cfg.env_spacing
        self._terrain = TerrainImporter(terrain_cfg)

        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[terrain_cfg.prim_path])

        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["termination_contact"] = self.termination_contact_sensor
        self.scene.sensors["wheel_contact"] = self.wheel_contact_sensor
        self.scene.sensors["body_imu"] = self.body_imu
        self.scene.sensors["depth_camera"] = self.depth_camera

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)

        if (
            not self._predictive_gating_enabled
            and self.cfg.use_velocity_commands
            and self.cfg.resample_commands
        ):
            if self._velocity_commands_enabled():
                self._command_time_left -= self._command_dt
                resample_env_ids = torch.nonzero(self._command_time_left <= 0.0, as_tuple=False).squeeze(-1)
                if resample_env_ids.numel() > 0:
                    self._resample_commands(resample_env_ids)
            else:
                self._commands[:] = 0.0
                self._command_time_left[:] = 0.0

        if self._predictive_gating_enabled:
            self._cache_predictive_gates()

    def _apply_action(self) -> None:
        # -----------------------------
        # 1) servo joints: position control
        # actions[0:4] -> position offsets around the default [left hip a, left mapped hip b, right hip a, right mapped hip b]
        # -----------------------------
        servo_actions = self.actions[:, 0:4] * self._servo_action_sign
        branch_hip_targets = torch.clamp(
            self._default_branch_hip_joint_pos + servo_actions[:, [0, 2]] * self.cfg.branch_hip_action_scale,
            min=self._branch_hip_lower_limits,
            max=self._branch_hip_upper_limits,
        )
        self._mapped_parallel_hip_targets = torch.clamp(
            self._default_mapped_parallel_hip_pos + servo_actions[:, [1, 3]] * self.cfg.mapped_hip_action_scale,
            min=self._mapped_parallel_hip_lower_limits,
            max=self._mapped_parallel_hip_upper_limits,
        )
        servo_targets = self._map_branch_and_parallel_hips_to_sim_servo_targets(
            branch_hip_targets,
            self._mapped_parallel_hip_targets,
        )
        self._servo_position_targets = torch.clamp(
            servo_targets,
            min=self._servo_lower_limits,
            max=self._servo_upper_limits,
        )

        self.robot.set_joint_position_target(
            self._servo_position_targets,
            joint_ids=self._servo_joint_ids,
        )

        # -----------------------------
        # 2) wheel joints: velocity control
        # actions[4:6] -> wheel velocity targets
        # same positive action = robot forward
        # -----------------------------
        wheel_actions = self.actions[:, 4:6] * self._wheel_action_sign
        self._wheel_velocity_targets = wheel_actions * self.cfg.wheel_action_scale

        self.robot.set_joint_velocity_target(
            self._wheel_velocity_targets,
            joint_ids=self._wheel_joint_ids,
        )

    def _get_observations(self) -> dict:
        if self._predictive_gating_enabled:
            return self._get_predictive_observations()

        wheel_pos = self.robot.data.joint_pos[:, self._wheel_joint_ids]
        wheel_vel = self.robot.data.joint_vel[:, self._wheel_joint_ids]
        root_lin_vel = self.robot.data.root_lin_vel_b

        imu_data = self.body_imu.data
        imu_quat_src = imu_data.quat_w
        imu_ang_vel_src = imu_data.ang_vel_b
        imu_projected_gravity_src = imu_data.projected_gravity_b

        if self.cfg.enable_imu_noise:
            imu_quat = imu_quat_src + torch.randn_like(imu_quat_src) * self.cfg.imu_quat_noise_std
            imu_quat = torch.nn.functional.normalize(imu_quat, dim=-1)
            imu_ang_vel = imu_ang_vel_src + torch.randn_like(imu_ang_vel_src) * self.cfg.imu_ang_vel_noise_std
            imu_projected_gravity = (
                imu_projected_gravity_src
                + torch.randn_like(imu_projected_gravity_src) * self.cfg.imu_projected_gravity_noise_std
            )
        else:
            imu_quat = imu_quat_src
            imu_ang_vel = imu_ang_vel_src
            imu_projected_gravity = imu_projected_gravity_src

        state_parts = [
            imu_quat,               # 4
            imu_ang_vel,            # 3
            root_lin_vel,           # 3
            imu_projected_gravity,  # 3
        ]
        if self.cfg.use_velocity_commands:
            state_parts.append(self._commands)  # 2
        state_parts.extend(
            [
                wheel_pos,          # 2
                wheel_vel,          # 2
                self.last_actions,  # 6
            ]
        )
        state_observation = torch.cat(state_parts, dim=-1)
        if state_observation.shape[-1] != self.cfg.pose_predictor_state_dim:
            raise RuntimeError(
                f"Pose predictor state dimension is {state_observation.shape[-1]}, "
                f"expected {self.cfg.pose_predictor_state_dim}."
            )

        depth_obs = self._get_depth_observation()
        pose_prediction = self._update_pose_prediction(depth_obs, state_observation)
        obs = torch.cat([state_observation, pose_prediction], dim=-1)

        log = self.extras.setdefault("log", {})
        log["PosePredictor/loss"] = self._pose_predictor_last_loss
        log["PosePredictor/quaternion_loss"] = self._pose_predictor_last_quaternion_loss
        log["PosePredictor/linear_velocity_loss"] = self._pose_predictor_last_linear_velocity_loss
        log["PosePredictor/angular_velocity_loss"] = self._pose_predictor_last_angular_velocity_loss
        log["PosePredictor/batch_size"] = float(self._pose_predictor_last_batch_size)

        return {"policy": obs, "depth": depth_obs}

    def _get_rewards(self) -> torch.Tensor:
        self.extras["log"] = {}
        servo_joint_vel = self.robot.data.joint_vel[:, self._servo_joint_ids]
        projected_gravity = self.robot.data.projected_gravity_b
        root_lin_vel = self.robot.data.root_lin_vel_b
        root_ang_vel = self.robot.data.root_ang_vel_w
        wheel_vel = self.robot.data.joint_vel[:, self._wheel_joint_ids]
        # `wheel_contact_sensor` tracks only the two wheel links, so its force
        # tensor is already ordered over [left_wheel, right_wheel].
        wheel_contact_normal_force = torch.abs(self.wheel_contact_sensor.data.net_forces_w[:, :, 2])

        gravity_xy_error = torch.sum(torch.square(projected_gravity[:, :2]), dim=1)
        projected_gravity_reward = torch.exp(
            -gravity_xy_error / self.cfg.command_tracking_gravity_sigma
        )
        joint_action_rate = torch.sum(torch.square(self.actions[:, 0:4] - self.last_actions[:, 0:4]), dim=1)

        rew_alive = self.cfg.rew_scale_alive * (1.0 - self.reset_terminated.float())
        rew_termination = self.cfg.rew_scale_terminated * self.reset_terminated.float()
        rew_angle = self.cfg.rew_scale_angle * gravity_xy_error
        rew_ang_vel = self.cfg.rew_scale_ang_vel * torch.sum(torch.square(root_ang_vel[:, :2]), dim=1)
        rew_projected_gravity = self.cfg.rew_scale_projected_gravity * projected_gravity_reward
        rew_vertical_vel = self.cfg.rew_scale_vertical_vel * torch.square(root_lin_vel[:, 2])
        rew_action_magnitude = self.cfg.rew_scale_action_magnitude * torch.sum(
            torch.square(self.actions), dim=1
        )
        if self.cfg.use_velocity_commands:
            planar_position_error = self._advance_commanded_planar_pose()
            rew_planar_position_error = self.cfg.rew_scale_planar_position_error * planar_position_error
        else:
            planar_position_error = torch.zeros(self.num_envs, device=self.device)
            rew_planar_position_error = torch.zeros(self.num_envs, device=self.device)
        standstill_weight = torch.ones(self.num_envs, device=self.device)
        upright_motion_weight = torch.ones(self.num_envs, device=self.device)
        if self.cfg.use_velocity_commands:
            lin_vel_error = torch.square(root_lin_vel[:, 1] - self._commands[:, 0])
            yaw_vel_error = torch.square(root_ang_vel[:, 2] - self._commands[:, 1])
            command_motion_mag = torch.abs(self._commands[:, 0]) + 0.5 * torch.abs(self._commands[:, 1])
            standstill_weight = torch.exp(-command_motion_mag / 0.25)
            turn_stability_mask = (torch.abs(self._commands[:, 1]) <= 0.05).float()
            rew_joint_action_rate = self.cfg.rew_scale_joint_action_rate * standstill_weight * joint_action_rate
            rew_yaw_ang_vel = self.cfg.rew_scale_yaw_ang_vel * turn_stability_mask * torch.square(root_ang_vel[:, 2])
            lin_track_score = torch.exp(-lin_vel_error / self.cfg.command_tracking_sigma_lin)
            yaw_track_score = torch.exp(-yaw_vel_error / self.cfg.command_tracking_sigma_yaw)
            lin_standstill_baseline = torch.exp(
                -torch.square(self._commands[:, 0]) / self.cfg.command_tracking_sigma_lin
            )
            yaw_standstill_baseline = torch.exp(
                -torch.square(self._commands[:, 1]) / self.cfg.command_tracking_sigma_yaw
            )
            upright_motion_weight = torch.exp(
                -gravity_xy_error / self.cfg.command_tracking_upright_sigma
            ) * torch.exp(
                -torch.sum(torch.square(root_ang_vel[:, :2]), dim=1) / self.cfg.command_tracking_stability_sigma
            )
            rew_track_lin_vel = self.cfg.rew_scale_track_lin_vel * (
                lin_track_score - lin_standstill_baseline
            ) * upright_motion_weight
            rew_track_yaw_vel = self.cfg.rew_scale_track_yaw_vel * (
                yaw_track_score - yaw_standstill_baseline
            ) * upright_motion_weight
        else:
            rew_joint_action_rate = self.cfg.rew_scale_joint_action_rate * joint_action_rate
            rew_yaw_ang_vel = self.cfg.rew_scale_yaw_ang_vel * torch.square(root_ang_vel[:, 2])
            rew_track_lin_vel = torch.zeros(self.num_envs, device=self.device)
            rew_track_yaw_vel = torch.zeros(self.num_envs, device=self.device)
            lin_track_score = torch.zeros(self.num_envs, device=self.device)
            yaw_track_score = torch.zeros(self.num_envs, device=self.device)

        semantic_wheel_vel = wheel_vel * self._wheel_action_sign

        if self.cfg.use_velocity_commands:
            semantic_wheel_forward_vel = torch.mean(semantic_wheel_vel, dim=1)
            semantic_wheel_yaw_vel = differential_drive_yaw_proxy(semantic_wheel_vel)
            rew_track_wheel_lin = self.cfg.rew_scale_track_wheel_lin * torch.tanh(
                3.0 * self._commands[:, 0] * semantic_wheel_forward_vel
            ) * upright_motion_weight
            rew_track_wheel_yaw = self.cfg.rew_scale_track_wheel_yaw * torch.tanh(
                2.0 * self._commands[:, 1] * semantic_wheel_yaw_vel
            ) * upright_motion_weight
        else:
            rew_track_wheel_lin = torch.zeros(self.num_envs, device=self.device)
            rew_track_wheel_yaw = torch.zeros(self.num_envs, device=self.device)
        wheel_in_contact = wheel_contact_normal_force > self.cfg.wheel_contact_normal_force_threshold
        self._wheel_contact_armed |= torch.any(wheel_in_contact, dim=1)
        wheel_air_count = torch.sum((~wheel_in_contact).float(), dim=1)
        wheel_contact_count = torch.sum(wheel_in_contact.float(), dim=1)
        wheel_air_penalty_count = torch.where(
            self._wheel_contact_armed,
            (wheel_air_count > 0.0).float(),
            torch.zeros_like(wheel_air_count),
        )
        rew_wheel_air = self.cfg.rew_scale_wheel_air * wheel_air_penalty_count

        # per-step diagnostics for wheel behavior
        log = self.extras["log"]
        if hasattr(self, "_last_non_wheel_contact_force_norm"):
            termination_forces = self._last_non_wheel_contact_force_norm
            log["Diagnostics/termination_rate"] = torch.mean(self.reset_terminated.float()).item()
            log["Diagnostics/timeout_rate"] = torch.mean(self.reset_time_outs.float()).item()
            log["Diagnostics/termination_force_max"] = torch.mean(
                torch.amax(termination_forces, dim=1)
            ).item()
            for body_index, body_name in enumerate(self.termination_contact_sensor.body_names):
                body_force = termination_forces[:, body_index]
                metric_name = body_name.lower()
                log[f"Diagnostics/termination_force_{metric_name}"] = torch.mean(body_force).item()
                log[f"Diagnostics/termination_trigger_{metric_name}"] = torch.mean(
                    (body_force > self.cfg.termination_contact_force_threshold).float()
                ).item()
            log["Diagnostics/termination_trigger_excessive_tilt"] = torch.mean(
                self._last_excessive_tilt.float()
            ).item()
        log["Diagnostics/wheel_vel_left"] = torch.mean(wheel_vel[:, 0]).item()
        log["Diagnostics/wheel_vel_right"] = torch.mean(wheel_vel[:, 1]).item()
        log["Diagnostics/wheel_air_count"] = torch.mean(wheel_air_count).item()
        log["Diagnostics/wheel_contact_count"] = torch.mean(wheel_contact_count).item()
        log["Diagnostics/wheel_air_penalty_count"] = torch.mean(wheel_air_penalty_count).item()
        log["Diagnostics/wheel_contact_armed_rate"] = torch.mean(self._wheel_contact_armed.float()).item()
        log["Diagnostics/wheel_semantic_vel_mean"] = torch.mean(semantic_wheel_vel).item()
        log["Diagnostics/wheel_semantic_forward_vel"] = torch.mean(torch.mean(semantic_wheel_vel, dim=1)).item()
        log["Diagnostics/root_lin_vel_y"] = torch.mean(root_lin_vel[:, 1]).item()
        log["Diagnostics/root_ang_vel_z"] = torch.mean(root_ang_vel[:, 2]).item()
        log["Diagnostics/wheel_action_abs"] = torch.mean(torch.abs(self.actions[:, 4:6])).item()
        log["Diagnostics/action_saturation_rate"] = torch.mean(
            (torch.abs(self.actions) >= 0.999).float()
        ).item()
        log["Diagnostics/servo_action_saturation_rate"] = torch.mean(
            (torch.abs(self.actions[:, 0:4]) >= 0.999).float()
        ).item()
        log["Diagnostics/wheel_action_saturation_rate"] = torch.mean(
            (torch.abs(self.actions[:, 4:6]) >= 0.999).float()
        ).item()
        log["Diagnostics/wheel_velocity_cmd_abs"] = torch.mean(torch.abs(self._wheel_velocity_targets)).item()
        log["Diagnostics/servo_joint_vel_sq"] = torch.mean(torch.sum(torch.square(servo_joint_vel), dim=1)).item()
        log["Diagnostics/gravity_xy_error"] = torch.mean(gravity_xy_error).item()
        log["Diagnostics/root_vertical_vel_abs"] = torch.mean(torch.abs(root_lin_vel[:, 2])).item()
        log["Diagnostics/projected_gravity_reward"] = torch.mean(projected_gravity_reward).item()
        log["Diagnostics/joint_action_rate"] = torch.mean(joint_action_rate).item()
        log["Diagnostics/rew_action_magnitude"] = torch.mean(rew_action_magnitude).item()

        wheel_radius = self.cfg.wheel_radius
        wheel_forward_surface_speed = wheel_radius * torch.mean(semantic_wheel_vel, dim=1)
        slip_error = wheel_forward_surface_speed - root_lin_vel[:, 1]
        forward_vel = root_lin_vel[:, 1]

        if self.cfg.use_velocity_commands:
            straight_weight = torch.exp(-torch.square(self._commands[:, 1]) / self.cfg.command_tracking_sigma_yaw)
            moving_command_mask = (
                (torch.abs(self._commands[:, 0]) > self.cfg.command_stop_threshold)
                | (torch.abs(self._commands[:, 1]) > self.cfg.command_stop_threshold)
            ).float()
            stop_command_mask = 1.0 - moving_command_mask
        else:
            straight_weight = torch.ones(self.num_envs, device=self.device)
            moving_command_mask = torch.zeros(self.num_envs, device=self.device)
            stop_command_mask = torch.ones(self.num_envs, device=self.device)

        active_linear_mask, command_aligned_forward_vel, command_aligned_wheel_speed = command_aligned_velocities(
            self._commands[:, 0],
            forward_vel,
            wheel_forward_surface_speed,
            stop_threshold=self.cfg.command_stop_threshold,
        )
        linear_motion_mask = active_linear_mask.float()
        rew_forward_vel = self.cfg.rew_scale_forward_vel * linear_motion_mask * straight_weight * torch.clamp(
            command_aligned_forward_vel,
            min=0.0,
            max=self.cfg.forward_velocity_cap,
        )
        rew_backward_vel = (
            self.cfg.rew_scale_backward_vel
            * linear_motion_mask
            * torch.clamp(command_aligned_forward_vel, max=0.0).abs()
        )
        rew_forward_progress = self.cfg.rew_scale_forward_progress * linear_motion_mask * straight_weight * torch.clamp(
            command_aligned_wheel_speed,
            min=0.0,
            max=self.cfg.forward_progress_cap,
        )
        rew_command_stop_motion = self.cfg.rew_scale_command_stop_motion * stop_command_mask * (
            torch.abs(forward_vel) + 0.25 * torch.abs(root_ang_vel[:, 2])
        )
        rew_direction = self.cfg.rew_scale_direction * (
            active_linear_mask & (command_aligned_forward_vel > 0.0)
        ).float()
        active_yaw_mask = torch.abs(self._commands[:, 1]) > self.cfg.command_stop_threshold
        rew_yaw_direction = self.cfg.rew_scale_yaw_direction * (
            active_yaw_mask & (self._commands[:, 1] * root_ang_vel[:, 2] > 0.0)
        ).float()
        # Keep a proven no-gate reward basis for the Predictor task.  The newer
        # command-aligned formulas make forward and reverse locomotion equally
        # active from the first random rollout; the 7.17 control instead learned
        # balance with a single forward convention and used tracking terms for
        # command response.
        legacy_rew_forward_vel = (
            self.cfg.rew_scale_forward_vel
            * moving_command_mask
            * straight_weight
            * torch.clamp(forward_vel, min=0.0, max=self.cfg.forward_velocity_cap)
        )
        legacy_rew_backward_vel = (
            self.cfg.rew_scale_backward_vel
            * moving_command_mask
            * torch.clamp(forward_vel, max=0.0).abs()
        )
        legacy_rew_direction = self.cfg.rew_scale_direction * (
            self._commands[:, 0] * forward_vel > 0.0
        ).float()
        rew_pre_stage3_still = self.cfg.rew_scale_pre_stage3_still * torch.abs(forward_vel)
        rew_pre_stage3_servo_motion = self.cfg.rew_scale_pre_stage3_servo_motion * torch.sum(
            torch.square(servo_joint_vel), dim=1
        )
        stable_stand = (gravity_xy_error <= self.cfg.stability_gravity_error_threshold) & (
            torch.norm(root_ang_vel[:, :2], dim=1) <= self.cfg.stability_ang_vel_threshold
        )
        rew_wheel_contact = self.cfg.rew_scale_wheel_contact * stable_stand.float() * (wheel_contact_count / 2.0)

        stage2_stable = stable_stand
        # Grounded is a real-time gate: higher-tier rewards are enabled only while
        # both wheels are currently in contact with the ground.
        stage3_grounded = stage2_stable & (wheel_air_count == 0.0)
        trackable_event = command_trackable_label(
            self._commands,
            forward_vel,
            root_ang_vel[:, 2],
            stop_threshold=self.cfg.command_stop_threshold,
            linear_error_threshold=self.cfg.command_tracking_lin_error_threshold,
            yaw_error_threshold=self.cfg.command_tracking_yaw_error_threshold,
            stop_linear_threshold=self.cfg.body_speed_gate_threshold,
            stop_yaw_threshold=self.cfg.command_stop_yaw_rate_threshold,
        )
        stage4_track_ready = stage3_grounded & trackable_event

        stage1_mask = torch.ones(self.num_envs, device=self.device)
        stage2_mask = stage2_stable.float()
        stage3_mask = stage3_grounded.float()
        stage4_mask = stage4_track_ready.float()
        stage3_denom_mask = self._wheel_contact_armed.float()

        # accumulate per-episode gate success ratios
        self._ep_len += 1
        self._ep_gate_sum += torch.stack([stage2_mask, stage3_mask, stage4_mask], dim=1)
        self._ep_gate_denom += torch.stack(
            [
                torch.ones_like(stage2_mask),
                stage3_denom_mask,
                torch.ones_like(stage4_mask),
            ],
            dim=1,
        )

        if self.cfg.curriculum_enable:
            batch_rates = torch.stack(
                [
                    torch.mean(stage2_mask),
                    torch.mean(stage3_mask),
                    torch.mean(stage4_mask),
                ]
            )
            alpha = self.cfg.curriculum_ema_alpha
            self._curriculum_ema = (1.0 - alpha) * self._curriculum_ema + alpha * batch_rates
        else:
            self._curriculum_stage = 4
            self._curriculum_active_stage = 4

        if self.cfg.reward_gate_enable:
            pre_stage3_servo_motion_gate = 1.0 if self._curriculum_active_stage < 3 else 0.0
            stage2_gate = stage2_mask * (1.0 if self._curriculum_active_stage >= 2 else 0.0)
            stage3_gate = stage3_mask * (1.0 if self._curriculum_active_stage >= 3 else 0.0)
            stage4_gate = stage4_mask * (1.0 if self._curriculum_active_stage >= 4 else 0.0)
        else:
            pre_stage3_servo_motion_gate = 0.0
            stage2_gate = torch.ones_like(stage2_mask)
            stage3_gate = torch.ones_like(stage3_mask)
            stage4_gate = torch.ones_like(stage4_mask)

        tier1_reward = (
            rew_angle
            + rew_ang_vel
            + rew_projected_gravity
            + rew_yaw_ang_vel
            + rew_joint_action_rate
            + rew_vertical_vel
        )
        tier2_reward = rew_wheel_contact + rew_wheel_air + rew_track_wheel_yaw
        tier3_reward = (
            rew_track_wheel_lin
            + rew_forward_vel
            + rew_backward_vel
            + 0.5 * rew_direction
            + rew_track_yaw_vel
            + rew_command_stop_motion
        )
        tier4_reward = 0.5 * rew_direction + rew_track_lin_vel + rew_planar_position_error
        legacy_tier3_reward = (
            rew_track_wheel_lin
            + legacy_rew_forward_vel
            + legacy_rew_backward_vel
            + 0.5 * legacy_rew_direction
            + rew_track_yaw_vel
            + rew_command_stop_motion
        )
        legacy_tier4_reward = (
            0.5 * legacy_rew_direction + rew_track_lin_vel + rew_planar_position_error
        )

        predictive_total_reward = None
        if self._predictive_gating_enabled:
            grounded_event = wheel_air_count == 0.0
            stable_event = stable_event_label(stage2_stable, self.reset_terminated)
            # Slip is only defined while rolling on the ground.  Treat airborne
            # samples as vacuously low-slip; the independent grounded head is
            # responsible for suppressing those transitions.
            low_slip = (~grounded_event) | (
                torch.abs(slip_error) <= self.cfg.predictive_slip_threshold
            )
            event_targets = torch.stack(
                [
                    stable_event.float(),
                    grounded_event.float(),
                    trackable_event.float(),
                    low_slip.float(),
                ],
                dim=-1,
            )
            reward_terms = torch.stack(
                [
                    rew_angle,
                    rew_ang_vel,
                    rew_projected_gravity,
                    rew_yaw_ang_vel,
                    rew_joint_action_rate,
                    rew_action_magnitude,
                    rew_pre_stage3_still,
                    rew_pre_stage3_servo_motion,
                    rew_vertical_vel,
                    rew_wheel_contact,
                    rew_wheel_air,
                    rew_track_wheel_yaw,
                    rew_track_wheel_lin,
                    legacy_rew_forward_vel,
                    legacy_rew_backward_vel,
                    rew_forward_progress,
                    legacy_rew_direction,
                    rew_yaw_direction,
                    rew_track_yaw_vel,
                    rew_command_stop_motion,
                    rew_track_lin_vel,
                    rew_planar_position_error,
                ],
                dim=-1,
            )
            self._record_predictive_targets(event_targets)
            reward_weights = self._predictive_cached_gates
            # The direct allocator has sole control over all twenty-two atomic
            # terms. Hand-written curriculum gates remain a baseline-only
            # mechanism and are deliberately absent from this reward path.
            applied_reward_components = reward_terms
            if bool(
                getattr(self.cfg, "outer_advantage_composer_enable", False)
                and self._predictive_training_enabled
            ):
                raw_group_rewards, normalized_group_rewards, normalized_atomic_rewards = (
                    group_atomic_rewards(
                        applied_reward_components,
                        self.outer_group_rms,
                        update_rms=self._predictive_updates_enabled,
                    )
                )
                fixed_outer_reward = outer_reward(
                    lin_track_score,
                    yaw_track_score,
                    self.reset_terminated,
                    self.actions,
                    termination_penalty=float(self.cfg.outer_reward_termination_penalty),
                    action_cost=float(self.cfg.outer_reward_action_cost),
                )
                # These are the original PPO's real safety terms, not the outer
                # objective. Without them, a mostly-negative shaping basis
                # rewards early termination because falling stops future costs.
                fixed_internal_reward = rew_alive + rew_termination
                if self._outer_reward_composition_mode == "lirpg":
                    predictive_total_reward = lirpg_actor_reward(
                        fixed_outer_reward,
                        self._lirpg_cached_intrinsic_reward,
                        extrinsic_coefficient=float(
                            self.cfg.lirpg_extrinsic_coefficient
                        ),
                        intrinsic_coefficient=float(
                            self.cfg.lirpg_intrinsic_coefficient
                        ),
                    )
                elif self._outer_reward_composition_mode == "relara":
                    predictive_total_reward = relara_policy_reward(
                        fixed_outer_reward,
                        self._relara_cached_proposed_reward,
                        beta=float(self.cfg.relara_beta),
                    )
                else:
                    predictive_total_reward = select_actor_reward(
                        fixed_outer_reward,
                        fixed_internal_reward,
                        normalized_group_rewards,
                        self._outer_cached_group_weights,
                        self._outer_beta,
                        outer_only=bool(
                            getattr(self.cfg, "outer_only_actor_reward", False)
                        ),
                    )
                if self._predictive_training_enabled:
                    self.extras["outer_composer"] = {
                        "fixed_outer_reward": fixed_outer_reward.detach().clone(),
                        "fixed_internal_reward": fixed_internal_reward.detach().clone(),
                        "normalized_group_rewards": normalized_group_rewards.detach().clone(),
                        "raw_group_rewards": raw_group_rewards.detach().clone(),
                        "fused_latent": self._predictive_cached_fused_latent.detach().clone(),
                        "group_weights": self._outer_cached_group_weights.detach().clone(),
                        "lirpg_intrinsic_reward": self._lirpg_cached_intrinsic_reward.detach().clone(),
                        "relara_proposed_reward": self._relara_cached_proposed_reward.detach().clone(),
                        "cached_actor_reward": predictive_total_reward.detach().clone(),
                    }
                log["OuterComposer/outer_reward"] = torch.mean(fixed_outer_reward).item()
                log["OuterComposer/fixed_internal_reward"] = torch.mean(
                    fixed_internal_reward
                ).item()
                log["OuterComposer/beta"] = float(self._outer_beta)
                log["RewardBaseline/mode_composer"] = float(
                    self._outer_reward_composition_mode == "composer"
                )
                log["RewardBaseline/mode_uniform"] = float(
                    self._outer_reward_composition_mode == "uniform"
                )
                log["RewardBaseline/mode_static"] = float(
                    self._outer_reward_composition_mode == "static"
                )
                log["RewardBaseline/mode_lirpg"] = float(
                    self._outer_reward_composition_mode == "lirpg"
                )
                log["RewardBaseline/mode_relara"] = float(
                    self._outer_reward_composition_mode == "relara"
                )
                log["LIRPG/intrinsic_reward"] = torch.mean(
                    self._lirpg_cached_intrinsic_reward
                ).item()
                log["ReLara/proposed_reward"] = torch.mean(
                    self._relara_cached_proposed_reward
                ).item()
                log["ReLara/proposed_reward_std"] = torch.std(
                    self._relara_cached_proposed_reward, unbiased=False
                ).item()
                log["OuterComposer/termination"] = torch.mean(
                    self.reset_terminated.float()
                ).item()
                log["OuterComposer/command_success"] = torch.mean(
                    trackable_event.float()
                ).item()
                log["OuterComposer/learning_auc"] = float(self._outer_learning_auc)
                effective_weights = effective_composer_weights(
                    self._outer_cached_group_weights,
                    self._outer_beta,
                )
                for group_index, group_name in enumerate(REWARD_GROUP_NAMES):
                    # Raw weight is what phi learned; effective weight is what
                    # this rollout's PPO reward actually used after beta.
                    log[f"OuterComposer/weight_{group_name}"] = torch.mean(
                        self._outer_cached_group_weights[:, group_index]
                    ).item()
                    log[f"OuterComposer/effective_weight_{group_name}"] = torch.mean(
                        effective_weights[:, group_index]
                    ).item()
                    log[f"OuterComposer/rms_{group_name}"] = float(
                        self.outer_group_rms.rms[group_index].item()
                    )
                for reward_index, reward_name in enumerate(DIRECT_REWARD_NAMES):
                    log[f"AtomicRewardRaw/{reward_name}"] = torch.mean(
                        torch.abs(applied_reward_components[:, reward_index])
                    ).item()
                    log[f"AtomicRewardNormalized/{reward_name}"] = torch.mean(
                        torch.abs(normalized_atomic_rewards[:, reward_index])
                    ).item()
                for metric_name, metric_value in self._outer_last_metrics.items():
                    log[f"OuterComposer/{metric_name}"] = float(metric_value)
            else:
                predictive_total_reward = (
                    rew_alive
                    + rew_termination
                    + pre_stage3_servo_motion_gate * rew_pre_stage3_still
                    + pre_stage3_servo_motion_gate * rew_pre_stage3_servo_motion
                    + torch.sum(applied_reward_components * reward_weights, dim=-1)
                )
            if (
                self.cfg.predictive_fixed_weight_control_use_baseline_reward
                and not getattr(self.cfg, "outer_advantage_composer_enable", False)
            ):
                predictive_total_reward = (
                    rew_alive
                    + rew_termination
                    + stage1_mask * tier1_reward
                    + pre_stage3_servo_motion_gate * rew_pre_stage3_still
                    + pre_stage3_servo_motion_gate * rew_pre_stage3_servo_motion
                    + stage2_gate * tier2_reward
                    + stage3_gate * legacy_tier3_reward
                    + stage4_gate * legacy_tier4_reward
                )

            # Immutable outer objective.  The predictor cannot change these
            # coefficients, the alive term, or the terminal penalty.  Its only
            # control is the twenty-two training-reward multipliers above.
            reference_stability = torch.exp(
                -gravity_xy_error / self.cfg.command_tracking_gravity_sigma
            )
            reference_grounded = wheel_contact_count / 2.0
            reference_tracking = 0.5 * (lin_track_score + yaw_track_score)
            reference_low_slip = 1.0 - torch.tanh(
                torch.abs(slip_error) / max(self.cfg.predictive_slip_threshold, 1.0e-6)
            )
            reference_action_cost = torch.tanh(torch.mean(torch.square(self.actions), dim=-1))
            reference_reward = (
                self.cfg.reference_alive_weight * (1.0 - self.reset_terminated.float())
                + self.cfg.reference_terminal_penalty * self.reset_terminated.float()
                + self.cfg.reference_stability_weight * reference_stability
                + self.cfg.reference_grounded_weight * reference_grounded
                + self.cfg.reference_tracking_weight * reference_tracking
                + self.cfg.reference_low_slip_weight * reference_low_slip
                - self.cfg.reference_action_penalty * reference_action_cost
            )
            if self._predictive_training_enabled and self.cfg.predictive_allocator_train:
                self.extras["predictive_meta"] = {
                    # DirectRLEnv resets completed environments before returning
                    # from step(), so freeze terminal-transition metadata here.
                    "context_contribution": self._predictive_cached_allocator_contexts.detach()
                    .mean(dim=0)
                    .clone(),
                    "allocator_context": self._predictive_rollout_allocator_context.detach().clone(),
                    "allocator_mean": self._predictive_rollout_allocator_mean.detach().clone(),
                    "allocator_sample": self._predictive_rollout_allocator_sample.detach().clone(),
                    "allocator_log_probability": self._predictive_rollout_allocator_log_prob.detach().clone(),
                    "allocator_residual": self._predictive_rollout_allocator_residual.detach().clone(),
                    # State-level allocator actions are required to train the
                    # reward head on the fused feature that actually produced
                    # each transition.  The aggregate tensors above remain the
                    # generation-level exploration identity/check.
                    "state_allocator_context": self._predictive_cached_allocator_contexts.detach().clone(),
                    "state_allocator_mean": self._predictive_cached_allocator_mean.detach().clone(),
                    "state_allocator_sample": self._predictive_cached_allocator_sample.detach().clone(),
                    "state_allocator_log_probability": self._predictive_cached_allocator_log_prob.detach().clone(),
                    "allocator_coordinate": int(
                        self._predictive_rollout_allocator_coordinate
                    ),
                    "state_reward_mean": self._predictive_cached_allocator_mean.detach().clone(),
                    "unallocated_rewards": (
                        rew_alive
                        + rew_termination
                        + pre_stage3_servo_motion_gate * rew_pre_stage3_still
                        + pre_stage3_servo_motion_gate * rew_pre_stage3_servo_motion
                    ).detach().clone(),
                    "reference_rewards": reference_reward.detach().clone(),
                    "reward_components": applied_reward_components.detach().clone(),
                }
            if not bool(getattr(self.cfg, "outer_advantage_composer_enable", False)):
                log["ReferenceObjective/reward"] = torch.mean(reference_reward).item()
                log["ReferenceObjective/stability"] = torch.mean(reference_stability).item()
                log["ReferenceObjective/grounded"] = torch.mean(reference_grounded).item()
                log["ReferenceObjective/tracking"] = torch.mean(reference_tracking).item()
                log["ReferenceObjective/low_slip"] = torch.mean(reference_low_slip).item()
                log["ReferenceObjective/terminal_cost"] = torch.mean(
                    self.cfg.reference_terminal_penalty * self.reset_terminated.float()
                ).item()
            log["PredictiveWeights/mean"] = torch.mean(reward_weights).item()
            log["PredictiveWeights/min"] = torch.mean(torch.amin(reward_weights, dim=-1)).item()
            log["PredictiveWeights/max"] = torch.mean(torch.amax(reward_weights, dim=-1)).item()
            log["PredictiveGating/label_stable"] = torch.mean(event_targets[:, 0]).item()
            log["PredictiveGating/label_grounded"] = torch.mean(event_targets[:, 1]).item()
            log["PredictiveGating/label_trackable"] = torch.mean(event_targets[:, 2]).item()
            log["PredictiveGating/label_low_slip"] = torch.mean(event_targets[:, 3]).item()

        log["Diagnostics/wheel_surface_speed"] = torch.mean(wheel_forward_surface_speed).item()
        log["Diagnostics/wheel_surface_speed_abs"] = torch.mean(torch.abs(wheel_forward_surface_speed)).item()
        log["Diagnostics/wheel_body_speed_slip"] = torch.mean(slip_error).item()
        log["Diagnostics/wheel_body_speed_slip_abs"] = torch.mean(torch.abs(slip_error)).item()
        log["Diagnostics/rew_forward_vel"] = torch.mean(rew_forward_vel).item()
        log["Diagnostics/rew_backward_vel"] = torch.mean(rew_backward_vel).item()
        log["Diagnostics/rew_forward_progress"] = torch.mean(rew_forward_progress).item()
        log["Diagnostics/rew_direction"] = torch.mean(rew_direction).item()
        log["Diagnostics/rew_yaw_direction"] = torch.mean(rew_yaw_direction).item()
        log["Diagnostics/command_stop_rate"] = torch.mean(stop_command_mask).item()
        log["Diagnostics/rew_command_stop_motion"] = torch.mean(rew_command_stop_motion).item()
        log["Diagnostics/planar_position_error"] = torch.mean(planar_position_error).item()
        # Deployment-only play evaluation must use the same immutable objective
        # without constructing Composer, Reward Agent, outer critic, or event
        # heads. Keep these task metrics available in every environment mode.
        evaluation_outer_reward = outer_reward(
            lin_track_score,
            yaw_track_score,
            self.reset_terminated,
            self.actions,
            termination_penalty=float(
                getattr(self.cfg, "outer_reward_termination_penalty", 5.0)
            ),
            action_cost=float(getattr(self.cfg, "outer_reward_action_cost", 0.01)),
        )
        log["Evaluation/fixed_outer_reward"] = torch.mean(
            evaluation_outer_reward
        ).item()
        log["Evaluation/command_success"] = torch.mean(
            trackable_event.float()
        ).item()
        log["Evaluation/termination_rate"] = torch.mean(
            self.reset_terminated.float()
        ).item()
        log["Evaluation/action_saturation_rate"] = torch.mean(
            (torch.abs(self.actions) >= 0.98).float()
        ).item()
        log["Diagnostics/rew_planar_position_error"] = torch.mean(rew_planar_position_error).item()
        log["Diagnostics/stage1_stable_rate"] = torch.mean(stage1_mask).item()
        log["Diagnostics/stage2_stable_stand_rate"] = torch.mean(stage2_mask).item()
        if torch.any(self._wheel_contact_armed):
            log["Diagnostics/stage3_grounded_rate"] = torch.mean(stage3_mask[self._wheel_contact_armed]).item()
        else:
            log["Diagnostics/stage3_grounded_rate"] = 0.0
        log["Diagnostics/stage4_track_ready_rate"] = torch.mean(stage4_mask).item()
        log["Diagnostics/reward_gate_enable"] = float(self.cfg.reward_gate_enable)
        log["Diagnostics/pose_predictor_enabled"] = float(self.cfg.pose_predictor_enabled)
        log["Diagnostics/curriculum_stage"] = float(self._curriculum_stage)
        log["Diagnostics/curriculum_active_stage"] = float(self._curriculum_active_stage)
        log["Diagnostics/curriculum_stage_unlocked"] = float(self._curriculum_last_unlock)
        log["Diagnostics/curriculum_unlocked_at_episode"] = float(self._curriculum_last_unlock_episode)
        log["Diagnostics/curriculum_mean_stage2"] = float(self._curriculum_last_means[0].item())
        log["Diagnostics/curriculum_mean_stage3"] = float(self._curriculum_last_means[1].item())
        log["Diagnostics/curriculum_mean_stage4"] = float(self._curriculum_last_means[2].item())
        root_quat_w = self.robot.data.root_quat_w
        body_forward_axis_w = quat_apply(
            root_quat_w,
            torch.tensor([0.0, 1.0, 0.0], device=self.device, dtype=root_quat_w.dtype).unsqueeze(0).repeat(self.num_envs, 1),
        )
        root_delta_pos_w = self.robot.data.root_pos_w - self._prev_root_pos_w
        root_forward_displacement = torch.sum(root_delta_pos_w * body_forward_axis_w, dim=-1)
        wheel_body_delta_pos_w = self.robot.data.body_pos_w[:, self._wheel_body_ids] - self._prev_wheel_body_pos_w
        wheel_body_forward_displacement = torch.sum(
            wheel_body_delta_pos_w * body_forward_axis_w.unsqueeze(1), dim=-1
        )
        wheel_grounded_motion_mask = wheel_in_contact & (
            torch.maximum(
                torch.abs(wheel_body_forward_displacement),
                torch.abs(root_forward_displacement).unsqueeze(1),
            )
            > 1.0e-4
        )
        if torch.any(wheel_grounded_motion_mask):
            root_forward_displacement_expanded = root_forward_displacement.unsqueeze(1).expand_as(
                wheel_body_forward_displacement
            )
            grounded_root_displacement = root_forward_displacement_expanded[wheel_grounded_motion_mask]
            grounded_wheel_displacement = wheel_body_forward_displacement[wheel_grounded_motion_mask]
            displacement_scale = torch.maximum(
                torch.abs(grounded_root_displacement),
                torch.abs(grounded_wheel_displacement),
            ).clamp(min=1.0e-4)
            grounded_displacement_match = 1.0 - (
                torch.abs(grounded_root_displacement - grounded_wheel_displacement) / displacement_scale
            )
            grounded_displacement_match = torch.clamp(grounded_displacement_match, min=0.0, max=1.0)
            log["Diagnostics/grounded_wheel_body_sign_match_rate"] = torch.mean(
                grounded_displacement_match
            ).item()
        else:
            log["Diagnostics/grounded_wheel_body_sign_match_rate"] = 0.0

        if self.cfg.use_velocity_commands:
            lin_pos_mask = self._commands[:, 0] > 0.05
            cmd_match = torch.sign(self._commands[:, 0]) * torch.sign(root_lin_vel[:, 1])
            active_lin_mask = torch.abs(self._commands[:, 0]) > 0.05
            active_yaw_mask = torch.abs(self._commands[:, 1]) > 0.05
            stop_cmd_mask_bool = stop_command_mask > 0.0
            moving_cmd_mask_bool = moving_command_mask > 0.0
            lin_cmd_success = (~active_lin_mask) | (
                torch.abs(root_lin_vel[:, 1] - self._commands[:, 0])
                <= self.cfg.command_tracking_lin_error_threshold
            )
            yaw_cmd_success = (~active_yaw_mask) | (
                torch.abs(root_ang_vel[:, 2] - self._commands[:, 1])
                <= self.cfg.command_tracking_yaw_error_threshold
            )
            moving_cmd_success = moving_cmd_mask_bool & lin_cmd_success & yaw_cmd_success
            stop_cmd_success = stop_cmd_mask_bool & (
                torch.abs(forward_vel) <= self.cfg.body_speed_gate_threshold
            ) & (torch.abs(root_ang_vel[:, 2]) <= self.cfg.command_stop_yaw_rate_threshold)
            command_success = trackable_event
            if torch.any(moving_cmd_mask_bool):
                log["Diagnostics/moving_cmd_success_rate"] = torch.mean(
                    moving_cmd_success[moving_cmd_mask_bool].float()
                ).item()
            else:
                log["Diagnostics/moving_cmd_success_rate"] = 0.0
            if torch.any(stop_cmd_mask_bool):
                log["Diagnostics/stop_cmd_success_rate"] = torch.mean(
                    stop_cmd_success[stop_cmd_mask_bool].float()
                ).item()
            else:
                log["Diagnostics/stop_cmd_success_rate"] = 0.0
            log["Diagnostics/velocity_command_success_rate"] = torch.mean(command_success.float()).item()
            if torch.any(active_lin_mask):
                log["Diagnostics/lin_cmd_sign_match_rate"] = torch.mean(
                    (cmd_match[active_lin_mask] > 0).float()
                ).item()
                log["Diagnostics/lin_cmd_response_mag"] = torch.mean(
                    torch.abs(root_lin_vel[active_lin_mask, 1])
                ).item()
                log["Diagnostics/lin_cmd_success_rate"] = torch.mean(
                    lin_cmd_success[active_lin_mask].float()
                ).item()
            else:
                log["Diagnostics/lin_cmd_sign_match_rate"] = 0.0
                log["Diagnostics/lin_cmd_response_mag"] = 0.0
                log["Diagnostics/lin_cmd_success_rate"] = 0.0
            if torch.any(active_yaw_mask):
                yaw_cmd_match = torch.sign(self._commands[:, 1]) * torch.sign(root_ang_vel[:, 2])
                log["Diagnostics/yaw_cmd_sign_match_rate"] = torch.mean(
                    (yaw_cmd_match[active_yaw_mask] > 0).float()
                ).item()
                log["Diagnostics/yaw_cmd_response_mag"] = torch.mean(
                    torch.abs(root_ang_vel[active_yaw_mask, 2])
                ).item()
                log["Diagnostics/yaw_cmd_signed_success_rate"] = torch.mean(
                    yaw_cmd_success[active_yaw_mask].float()
                ).item()
            else:
                log["Diagnostics/yaw_cmd_sign_match_rate"] = 0.0
                log["Diagnostics/yaw_cmd_response_mag"] = 0.0
                log["Diagnostics/yaw_cmd_signed_success_rate"] = 0.0
            if torch.any(lin_pos_mask):
                log["Diagnostics/forward_cmd_root_lin_vel_y"] = torch.mean(root_lin_vel[lin_pos_mask, 1]).item()
                log["Diagnostics/forward_cmd_wheel_semantic_vel"] = torch.mean(
                    torch.mean(semantic_wheel_vel[lin_pos_mask], dim=1)
                ).item()
                log["Diagnostics/forward_cmd_success_rate"] = torch.mean(
                    (root_lin_vel[lin_pos_mask, 1] > 0.02).float()
                ).item()
                log["Diagnostics/forward_cmd_slip_abs"] = torch.mean(torch.abs(slip_error[lin_pos_mask])).item()
            yaw_pos_mask = self._commands[:, 1] > 0.05
            if torch.any(yaw_pos_mask):
                log["Diagnostics/yaw_cmd_root_ang_vel_z"] = torch.mean(root_ang_vel[yaw_pos_mask, 2]).item()
                log["Diagnostics/yaw_cmd_success_rate"] = torch.mean(
                    (root_ang_vel[yaw_pos_mask, 2] > 0.05).float()
                ).item()
            else:
                log["Diagnostics/yaw_cmd_root_ang_vel_z"] = 0.0
                log["Diagnostics/yaw_cmd_success_rate"] = 0.0

        if predictive_total_reward is not None:
            return predictive_total_reward

        total_reward = (
            rew_alive
            + rew_termination
            + stage1_mask * tier1_reward
            + pre_stage3_servo_motion_gate * rew_pre_stage3_still
            + pre_stage3_servo_motion_gate * rew_pre_stage3_servo_motion
            + stage2_gate * tier2_reward
            + stage3_gate * tier3_reward
            + stage4_gate * tier4_reward
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.disable_termination:
            false_dones = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            return false_dones, false_dones

        contact_force_norm = torch.norm(self.termination_contact_sensor.data.net_forces_w, dim=-1)
        contact_trigger = contact_force_norm > self.cfg.termination_contact_force_threshold
        if self.cfg.termination_body_only:
            try:
                body_index = self.termination_contact_sensor.body_names.index(self.cfg.body_link_name)
            except ValueError as error:
                raise RuntimeError(
                    f"Termination sensor does not contain required body {self.cfg.body_link_name!r}."
                ) from error
            non_wheel_body_contact = contact_trigger[:, body_index]
        else:
            non_wheel_body_contact = torch.any(contact_trigger, dim=1)
        self._last_non_wheel_contact_force_norm = contact_force_norm

        max_gravity_xy_error = self.cfg.termination_max_gravity_xy_error
        if max_gravity_xy_error is None:
            excessive_tilt = torch.zeros_like(non_wheel_body_contact)
        else:
            gravity_xy_error = torch.sum(
                torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=1
            )
            excessive_tilt = gravity_xy_error > float(max_gravity_xy_error)
        self._last_excessive_tilt = excessive_tilt

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return non_wheel_body_contact | excessive_tilt, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # curriculum update uses episode-averaged gate rates over the last N episodes
        if self.cfg.curriculum_enable and env_ids.numel() > 0:
            ep_len = self._ep_len[env_ids].clamp(min=1).float()
            ep_gate_denom = self._ep_gate_denom[env_ids].clamp(min=1.0)
            ep_gate_mean = self._ep_gate_sum[env_ids] / ep_gate_denom
            for i in range(ep_gate_mean.shape[0]):
                for k in range(3):
                    self._curriculum_history[k].append(float(ep_gate_mean[i, k].item()))
                    if len(self._curriculum_history[k]) > self.cfg.curriculum_window_episodes:
                        self._curriculum_history[k].pop(0)
                self._curriculum_episode_count += 1

            if (
                self._curriculum_episode_count >= self.cfg.curriculum_warmup_episodes
                and self._curriculum_episode_count % self.cfg.curriculum_check_interval_episodes == 0
                and len(self._curriculum_history[0]) >= self.cfg.curriculum_window_episodes
            ):
                means = torch.tensor(
                    [
                        sum(self._curriculum_history[0]) / len(self._curriculum_history[0]),
                        sum(self._curriculum_history[1]) / len(self._curriculum_history[1]),
                        sum(self._curriculum_history[2]) / len(self._curriculum_history[2]),
                    ],
                    device=self.device,
                )
                episode_length_ratio = torch.mean(ep_len / float(self.max_episode_length))
                self._curriculum_last_means = means
                unlocked = False
                if (
                    self._curriculum_stage == 1
                    and means[0] >= self.cfg.curriculum_unlock_rate
                    and episode_length_ratio >= self.cfg.curriculum_stage2_min_episode_ratio
                ):
                    self._curriculum_stage = 2
                    unlocked = True
                elif self._curriculum_stage == 2 and means[1] >= self.cfg.curriculum_unlock_rate:
                    self._curriculum_stage = 3
                    unlocked = True
                elif self._curriculum_stage == 3 and means[2] >= self.cfg.curriculum_unlock_rate:
                    self._curriculum_stage = 4
                    unlocked = True

                if unlocked:
                    self._curriculum_last_unlock = self._curriculum_stage
                    self._curriculum_last_unlock_episode = self._curriculum_episode_count
                    print(
                        f"[YawBot] Curriculum unlocked stage {self._curriculum_stage} at episode {self._curriculum_episode_count} "
                        f"(means: {means.tolist()})"
                    )

                # soft rollback: do not downgrade max stage, only pause higher-tier rewards
                active = 1
                if self._curriculum_stage >= 2 and means[0] >= self.cfg.curriculum_unlock_rate:
                    active = 2
                if self._curriculum_stage >= 3 and means[1] >= self.cfg.curriculum_unlock_rate:
                    active = 3
                if self._curriculum_stage >= 4 and means[2] >= self.cfg.curriculum_unlock_rate:
                    active = 4
                self._curriculum_active_stage = active

        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        joint_pos[:, self._servo_joint_ids] = self._default_servo_joint_pos[env_ids]

        joint_vel[:, self._all_joint_ids] = 0.0

        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        default_root_state[:, 2] += self.cfg.reset_height_offset
        default_root_state[:, 7:13] = 0.0

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self._servo_position_targets[env_ids] = self._default_servo_joint_pos[env_ids]
        self._wheel_velocity_targets[env_ids] = 0.0
        self._mapped_parallel_hip_targets[env_ids] = 0.0
        self._wheel_contact_armed[env_ids] = False
        self._ep_len[env_ids] = 0
        self._ep_gate_sum[env_ids] = 0.0
        self._ep_gate_denom[env_ids] = 0.0
        self._prev_root_pos_w[env_ids] = default_root_state[:, :3]
        self._commanded_position_w[env_ids] = default_root_state[:, :2]
        body_forward_axis = torch.zeros((env_ids.numel(), 3), device=self.device)
        body_forward_axis[:, 1] = 1.0
        commanded_heading_w = quat_apply(default_root_state[:, 3:7], body_forward_axis)[:, :2]
        self._commanded_heading_w[env_ids] = commanded_heading_w / torch.linalg.vector_norm(
            commanded_heading_w, dim=1, keepdim=True
        ).clamp(min=1.0e-6)
        self._prev_wheel_body_pos_w[env_ids] = self.robot.data.body_pos_w[env_ids][:, self._wheel_body_ids]
        if self._predictive_gating_enabled:
            self._predictive_depth_history[env_ids] = 0.0
            self._predictive_state_history[env_ids] = 0.0
            self._predictive_history_age[env_ids] = 0
            self._predictive_cached_policy_latent[env_ids] = 0.0
            self._predictive_cached_probabilities[env_ids] = 1.0
            self._predictive_cached_uncertainty[env_ids] = 0.0
            if self._predictive_training_enabled:
                self._predictive_depth_input_queue[env_ids] = 0.0
                self._predictive_state_input_queue[env_ids] = 0.0
                self._predictive_action_input_queue[env_ids] = 0.0
                self._predictive_event_target_queue[env_ids] = 0.0
                self._predictive_future_target_queue[env_ids] = 0.0
                self._predictive_cached_allocator_contexts[env_ids] = 0.0
                self._predictive_sequence_age[env_ids] = 0
            self._predictive_last_observation_step = -1
            self._predictive_last_gate_step = -1
        if hasattr(self, "_depth_observation_history"):
            self._depth_observation_history[env_ids] = 0.0
            self._state_observation_history[env_ids] = 0.0
            self._predictor_depth_input_queue[env_ids] = 0.0
            self._predictor_state_input_queue[env_ids] = 0.0
            self._future_state_target_queue[env_ids] = 0.0
            self._pose_sequence_age[env_ids] = 0
            self._cached_pose_prediction[env_ids] = 0.0
            self._cached_pose_prediction[env_ids, :, 0] = 1.0
            self._pose_predictor_last_step = -1
        if self.cfg.use_velocity_commands and (self.cfg.resample_commands or self.cfg.use_fixed_velocity_command):
            self._resample_commands(env_ids)
        else:
            self._commands[env_ids] = 0.0
            self._command_time_left[env_ids] = 0.0

    def _post_physics_step(self):
        self.last_actions.copy_(self.actions)
        self._prev_root_pos_w.copy_(self.robot.data.root_pos_w)
        self._prev_wheel_body_pos_w.copy_(self.robot.data.body_pos_w[:, self._wheel_body_ids])
