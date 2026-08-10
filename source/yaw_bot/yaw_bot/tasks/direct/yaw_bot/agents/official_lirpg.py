"""Isaac/RSL tensor adapter for the official PPO-LIRPG update.

Algorithm source:
https://github.com/Hwhitetooth/lirpg/blob/ca85c523beae4feb0fa128c90f145961a9d0de36/baselines/ppo2/ppo2.py

The source implementation is TensorFlow 1.x and cannot share a process with
Isaac Sim 5.1's Python/PyTorch runtime.  This module preserves its update order
and equations while replacing only TensorFlow placeholders, the environment
runner, and policy/value calls with the existing RSL-RL tensors and modules.
"""

from __future__ import annotations

import torch
from torch import nn

try:
    from ..outer_advantage_composer import official_lirpg_meta_gradient_loss
except ImportError:  # Standalone CPU formula tests without launching Isaac Sim.
    from outer_advantage_composer import official_lirpg_meta_gradient_loss


def official_lirpg_ppo_update(
    algorithm,
    *,
    reward_model: nn.Module,
    reward_optimizer: torch.optim.Optimizer,
    outer_critic: nn.Module,
    outer_critic_optimizer: torch.optim.Optimizer,
    extrinsic_rewards: torch.Tensor,
    outer_advantages: torch.Tensor,
    outer_returns: torch.Tensor,
    outer_values: torch.Tensor,
    critic_observations: torch.Tensor,
    actor_observations: torch.Tensor,
    actions: torch.Tensor,
    old_action_log_probabilities: torch.Tensor,
    dones: torch.Tensor,
    final_mixed_value: torch.Tensor,
    actor_maximum_latent_mean: float,
    extrinsic_coefficient: float = 0.01,
    intrinsic_coefficient: float = 1.0,
    reward_gradient_clip: float = 0.5,
    outer_critic_gradient_clip: float = 0.5,
) -> tuple[dict[str, float], dict[str, float], int]:
    """Run the official alternating intrinsic/PPO minibatch updates."""

    if algorithm.policy.is_recurrent:
        raise NotImplementedError("Official yaw_bot LIRPG supports non-recurrent PPO only.")
    if algorithm.rnd or algorithm.symmetry:
        raise NotImplementedError("Official LIRPG adapter excludes RND and symmetry losses.")

    storage = algorithm.storage
    rollout_shape = extrinsic_rewards.shape
    expected_shape = (storage.num_transitions_per_env, storage.num_envs)
    if rollout_shape != expected_shape:
        raise ValueError(
            f"Expected LIRPG rollout {expected_shape}, received {rollout_shape}."
        )
    for name, tensor in (
        ("outer advantages", outer_advantages),
        ("outer returns", outer_returns),
        ("outer values", outer_values),
    ):
        if tensor.shape != expected_shape:
            raise ValueError(f"Expected {name} {expected_shape}, received {tensor.shape}.")
    if critic_observations.shape[:2] != expected_shape:
        raise ValueError(
            "Outer critic observations must start with the rollout dimensions "
            f"{expected_shape}, received {tuple(critic_observations.shape)}."
        )
    mixed_values = storage.values.squeeze(-1).detach()
    final_mixed_value = final_mixed_value.squeeze(-1).detach()
    if final_mixed_value.shape != (storage.num_envs,):
        raise ValueError("Final mixed critic value must have shape [num_envs].")

    observations = storage.observations.flatten(0, 1)
    flat_actions = storage.actions.flatten(0, 1)
    target_values = storage.values.flatten(0, 1)
    old_log_probabilities = storage.actions_log_prob.flatten(0, 1)
    old_mu = storage.mu.flatten(0, 1)
    old_sigma = storage.sigma.flatten(0, 1)
    flat_outer_critic_observations = critic_observations.flatten(0, 1)
    flat_outer_returns = outer_returns.flatten().detach()
    flat_outer_values = outer_values.flatten().detach()
    batch_size = storage.num_envs * storage.num_transitions_per_env
    mini_batch_size = batch_size // algorithm.num_mini_batches
    if mini_batch_size <= 0:
        raise ValueError("LIRPG minibatch size must be positive.")

    ppo_totals = {"value_function": 0.0, "surrogate": 0.0, "entropy": 0.0}
    meta_names = (
        "meta_policy_loss",
        "meta_outer_loss_before",
        "meta_outer_loss_after",
        "predicted_outer_improvement",
        "inner_policy_loss",
        "intrinsic_reward_mean",
        "intrinsic_reward_std",
        "intrinsic_reward_abs",
        "intrinsic_advantage_mean",
        "advantage_cosine",
    )
    meta_totals = {name: 0.0 for name in meta_names}
    meta_gradient_norm_total = 0.0
    outer_critic_loss_total = 0.0
    outer_critic_gradient_norm_total = 0.0
    update_count = 0

    for _ in range(algorithm.num_learning_epochs):
        # The official implementation reshuffles at every PPO epoch.
        permutation = torch.randperm(batch_size, device=algorithm.device)
        for mini_batch in range(algorithm.num_mini_batches):
            start = mini_batch * mini_batch_size
            stop = (
                batch_size
                if mini_batch == algorithm.num_mini_batches - 1
                else (mini_batch + 1) * mini_batch_size
            )
            batch_indices = permutation[start:stop]

            reward_loss, reward_metrics = official_lirpg_meta_gradient_loss(
                reward_model,
                algorithm.policy.actor,
                algorithm.policy.log_std,
                algorithm.optimizer,
                extrinsic_rewards,
                mixed_values,
                final_mixed_value,
                outer_advantages,
                actor_observations,
                actions,
                old_action_log_probabilities,
                dones,
                batch_indices,
                gamma=float(algorithm.gamma),
                lam=float(algorithm.lam),
                clip_param=float(algorithm.clip_param),
                maximum_latent_mean=float(actor_maximum_latent_mean),
                extrinsic_coefficient=float(extrinsic_coefficient),
                intrinsic_coefficient=float(intrinsic_coefficient),
            )
            reward_parameters = tuple(reward_model.parameters())
            reward_gradients = torch.autograd.grad(reward_loss, reward_parameters)

            # TensorFlow evaluates policy_train and intrinsic_train from the
            # same pre-update graph. Keep that mixed advantage for this real
            # PPO minibatch even though eta has just advanced.
            advantages_batch = reward_metrics["mixed_advantages_batch"].detach()
            obs_batch = observations[batch_indices]
            actions_batch = flat_actions[batch_indices]
            target_values_batch = target_values[batch_indices]
            returns_batch = reward_metrics["mixed_returns_batch"].detach().unsqueeze(-1)
            old_log_probability_batch = old_log_probabilities[batch_indices]
            old_mu_batch = old_mu[batch_indices]
            old_sigma_batch = old_sigma[batch_indices]

            # The reference implementation places eta and V_ex in one Adam
            # optimizer and executes both updates for every minibatch. They
            # have disjoint parameters, so two Adam instances with identical
            # hyperparameters preserve the same parameter updates.
            predicted_outer_values = outer_critic(
                flat_outer_critic_observations[batch_indices]
            ).flatten()
            old_outer_values_batch = flat_outer_values[batch_indices]
            outer_returns_batch = flat_outer_returns[batch_indices]
            clipped_outer_values = old_outer_values_batch + (
                predicted_outer_values - old_outer_values_batch
            ).clamp(-algorithm.clip_param, algorithm.clip_param)
            outer_critic_loss = 0.5 * torch.maximum(
                torch.square(predicted_outer_values - outer_returns_batch),
                torch.square(clipped_outer_values - outer_returns_batch),
            ).mean()
            outer_critic_parameters = tuple(outer_critic.parameters())
            outer_critic_gradients = torch.autograd.grad(
                algorithm.value_loss_coef * outer_critic_loss,
                outer_critic_parameters,
            )
            reward_optimizer.zero_grad(set_to_none=True)
            outer_critic_optimizer.zero_grad(set_to_none=True)
            for parameter, gradient in zip(
                reward_parameters, reward_gradients, strict=True
            ):
                parameter.grad = gradient.detach()
            for parameter, gradient in zip(
                outer_critic_parameters, outer_critic_gradients, strict=True
            ):
                parameter.grad = gradient.detach()
            intrinsic_parameters = reward_parameters + outer_critic_parameters
            intrinsic_gradient_norm = nn.utils.clip_grad_norm_(
                intrinsic_parameters,
                min(
                    float(reward_gradient_clip),
                    float(outer_critic_gradient_clip),
                ),
            )
            reward_optimizer.step()
            outer_critic_optimizer.step()

            algorithm.policy.act(obs_batch)
            actions_log_probability_batch = algorithm.policy.get_actions_log_prob(
                actions_batch
            )
            value_batch = algorithm.policy.evaluate(obs_batch)
            mu_batch = algorithm.policy.action_mean
            sigma_batch = algorithm.policy.action_std
            entropy_batch = algorithm.policy.entropy

            if algorithm.desired_kl is not None and algorithm.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (
                            torch.square(old_sigma_batch)
                            + torch.square(old_mu_batch - mu_batch)
                        )
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        dim=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if kl_mean > algorithm.desired_kl * 2.0:
                        algorithm.learning_rate = max(
                            1.0e-5, algorithm.learning_rate / 1.5
                        )
                    elif 0.0 < kl_mean < algorithm.desired_kl / 2.0:
                        algorithm.learning_rate = min(
                            1.0e-2, algorithm.learning_rate * 1.5
                        )
                    for group in algorithm.optimizer.param_groups:
                        group["lr"] = algorithm.learning_rate

            ratio = torch.exp(
                actions_log_probability_batch
                - old_log_probability_batch.squeeze(-1)
            )
            surrogate = -advantages_batch * ratio
            surrogate_clipped = -advantages_batch * torch.clamp(
                ratio,
                1.0 - algorithm.clip_param,
                1.0 + algorithm.clip_param,
            )
            surrogate_loss = torch.maximum(surrogate, surrogate_clipped).mean()
            if algorithm.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-algorithm.clip_param, algorithm.clip_param)
                value_loss = torch.maximum(
                    torch.square(value_batch - returns_batch),
                    torch.square(value_clipped - returns_batch),
                ).mean()
            else:
                value_loss = torch.square(returns_batch - value_batch).mean()
            entropy = entropy_batch.mean()
            policy_loss = (
                surrogate_loss
                + algorithm.value_loss_coef * value_loss
                - algorithm.entropy_coef * entropy
            )
            algorithm.optimizer.zero_grad()
            policy_loss.backward()
            if algorithm.is_multi_gpu:
                algorithm.reduce_parameters()
            nn.utils.clip_grad_norm_(
                algorithm.policy.parameters(), algorithm.max_grad_norm
            )
            algorithm.optimizer.step()

            ppo_totals["value_function"] += float(value_loss.detach())
            ppo_totals["surrogate"] += float(surrogate_loss.detach())
            ppo_totals["entropy"] += float(entropy.detach())
            for name in meta_names:
                meta_totals[name] += float(reward_metrics[name])
            meta_gradient_norm_total += float(intrinsic_gradient_norm)
            outer_critic_loss_total += float(outer_critic_loss.detach())
            outer_critic_gradient_norm_total += float(intrinsic_gradient_norm)
            update_count += 1

    storage.clear()
    if update_count <= 0:
        raise RuntimeError("Official LIRPG performed no minibatch updates.")
    ppo_metrics = {
        name: value / update_count for name, value in ppo_totals.items()
    }
    meta_metrics = {
        name: value / update_count for name, value in meta_totals.items()
    }
    meta_metrics["composer_loss"] = meta_metrics["meta_policy_loss"]
    meta_metrics["composer_gradient_norm"] = (
        meta_gradient_norm_total / update_count
    )
    meta_metrics["shaping_advantage_mean"] = meta_metrics[
        "intrinsic_advantage_mean"
    ]
    meta_metrics["outer_critic_loss"] = outer_critic_loss_total / update_count
    meta_metrics["outer_critic_gradient_norm"] = (
        outer_critic_gradient_norm_total / update_count
    )
    return ppo_metrics, meta_metrics, update_count


__all__ = ["official_lirpg_ppo_update"]
