from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


EVENT_NAMES = ("stable", "grounded", "trackable", "low_slip")


def diagonal_gaussian_log_prob(
    value: torch.Tensor,
    mean: torch.Tensor,
    standard_deviation: float,
) -> torch.Tensor:
    """Return the joint log-probability of a diagonal Gaussian action."""
    if value.shape != mean.shape:
        raise ValueError("Gaussian value and mean tensors must have identical shapes.")
    if value.ndim < 1:
        raise ValueError("Gaussian action tensors must have at least one dimension.")
    if standard_deviation <= 0.0:
        raise ValueError("Gaussian standard deviation must be positive.")
    return torch.distributions.Normal(mean, float(standard_deviation)).log_prob(value).sum(dim=-1)


def diagonal_gaussian_component_log_prob(
    value: torch.Tensor,
    mean: torch.Tensor,
    standard_deviation: float,
) -> torch.Tensor:
    """Return one log-probability per diagonal-Gaussian action component."""
    if value.shape != mean.shape:
        raise ValueError("Gaussian value and mean tensors must have identical shapes.")
    if value.ndim < 1:
        raise ValueError("Gaussian action tensors must have at least one dimension.")
    if standard_deviation <= 0.0:
        raise ValueError("Gaussian standard deviation must be positive.")
    return torch.distributions.Normal(mean, float(standard_deviation)).log_prob(value)


def censored_diagonal_gaussian_component_log_prob(
    value: torch.Tensor,
    mean: torch.Tensor,
    standard_deviation: float,
    lower_bound: float | torch.Tensor,
    upper_bound: float | torch.Tensor,
) -> torch.Tensor:
    """Return the likelihood of a Gaussian action clipped to closed bounds.

    Clipping creates probability mass at both bounds. Treating a clipped value
    as an ordinary Gaussian sample gives a zero score-function gradient when
    both sample and mean are zero, making zero-initialized rewards one-sided.
    This is the exact component likelihood of the censored action actually
    applied by the environment.
    """
    if value.shape != mean.shape:
        raise ValueError("Gaussian value and mean tensors must have identical shapes.")
    if standard_deviation <= 0.0:
        raise ValueError("Gaussian standard deviation must be positive.")
    lower = torch.as_tensor(lower_bound, device=mean.device, dtype=mean.dtype)
    upper = torch.as_tensor(upper_bound, device=mean.device, dtype=mean.dtype)
    try:
        lower, upper = torch.broadcast_tensors(lower, upper)
        torch.broadcast_shapes(mean.shape, lower.shape)
    except RuntimeError as error:
        raise ValueError("Gaussian bounds must broadcast to the action shape.") from error
    if torch.any(lower >= upper):
        raise ValueError("Every Gaussian lower bound must be below its upper bound.")

    scale = float(standard_deviation)
    interior_log_probability = torch.distributions.Normal(mean, scale).log_prob(value)
    lower_log_mass = torch.special.log_ndtr((lower - mean) / scale)
    upper_log_mass = torch.special.log_ndtr((mean - upper) / scale)
    return torch.where(
        value <= lower,
        lower_log_mass,
        torch.where(value >= upper, upper_log_mass, interior_log_probability),
    )


def straight_through_clamp(
    value: torch.Tensor,
    lower_bound: float | torch.Tensor,
    upper_bound: float | torch.Tensor,
) -> torch.Tensor:
    """Clamp in the forward pass while retaining an identity inward gradient."""
    lower = torch.as_tensor(lower_bound, device=value.device, dtype=value.dtype)
    upper = torch.as_tensor(upper_bound, device=value.device, dtype=value.dtype)
    if torch.any(lower >= upper):
        raise ValueError("Every clamp lower bound must be below its upper bound.")
    bounded = torch.maximum(torch.minimum(value, upper), lower)
    return value + (bounded - value).detach()


def fixed_reference_policy_improvement(
    old_log_probability: torch.Tensor,
    new_log_probability: torch.Tensor,
    reference_rewards: torch.Tensor,
    dones: torch.Tensor,
    *,
    gamma: float,
    importance_clip: float,
    normalization_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure whether one PPO update favors fixed-reference high-return actions.

    All quantities come from the same rollout, so simulator/reset noise cannot
    masquerade as learning progress. The clipped importance surrogate is zero
    when the policy did not change, positive when probability moves toward
    above-average fixed-reference returns, and negative in the opposite case.
    """
    expected_shape = reference_rewards.shape
    if reference_rewards.ndim != 2:
        raise ValueError("Reference rewards must have shape [time, env].")
    if old_log_probability.shape != expected_shape or new_log_probability.shape != expected_shape:
        raise ValueError("Policy log probabilities must match reference reward shape [time, env].")
    if dones.shape != expected_shape:
        raise ValueError("Done flags must match reference reward shape [time, env].")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0, 1].")
    if not 0.0 <= importance_clip < 1.0:
        raise ValueError("importance_clip must lie in [0, 1).")
    if normalization_scale <= 0.0:
        raise ValueError("normalization_scale must be positive.")

    returns = torch.zeros_like(reference_rewards)
    running_return = torch.zeros(reference_rewards.shape[1], device=reference_rewards.device)
    dones = dones.to(device=reference_rewards.device, dtype=torch.bool)
    for step in range(reference_rewards.shape[0] - 1, -1, -1):
        running_return = reference_rewards[step] + float(gamma) * running_return * (~dones[step]).float()
        returns[step] = running_return
    reference_advantage = (
        (returns - returns.mean())
        / returns.std(unbiased=False).clamp_min(1.0e-6)
    )
    ratio = torch.exp((new_log_probability - old_log_probability).clamp(-20.0, 20.0))
    clipped_ratio = ratio.clamp(1.0 - float(importance_clip), 1.0 + float(importance_clip))
    raw_improvement = torch.minimum(
        ratio * reference_advantage,
        clipped_ratio * reference_advantage,
    ).mean()
    normalized_improvement = torch.tanh(raw_improvement / float(normalization_scale))
    return raw_improvement, normalized_improvement


def antithetic_reward_candidates(
    unallocated_rewards: torch.Tensor,
    reward_components: torch.Tensor,
    state_reward_means: torch.Tensor,
    residual: torch.Tensor,
    *,
    lower_bound: float,
    upper_bound: float,
    allocation_blend: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build paired training rewards that differ only by allocator residual sign.

    The two candidates share every transition, atomic reward, and deterministic
    state-conditioned head output. This makes their subsequent PPO-update
    difference attributable to the explored reward allocation.
    """
    if reward_components.ndim < 2:
        raise ValueError("Reward components must include sample and reward dimensions.")
    if state_reward_means.shape != reward_components.shape:
        raise ValueError("State reward means must match reward component shape.")
    if unallocated_rewards.shape != reward_components.shape[:-1]:
        raise ValueError("Unallocated rewards must match reward component sample dimensions.")
    if residual.shape == reward_components.shape:
        # Contextual exploration: every transition owns an independent reward
        # action. This is the identifiable form used by the direct allocator.
        contextual_residual = residual
    elif residual.ndim == 1 and residual.shape[0] == reward_components.shape[-1]:
        # Legacy/shared exploration remains useful for numerical regression and
        # old aggregate experiments.
        expand_shape = (1,) * (state_reward_means.ndim - 1) + (residual.numel(),)
        contextual_residual = residual.reshape(expand_shape)
    else:
        raise ValueError(
            "Residual must contain one shared value per reward component or "
            "match the complete state reward tensor."
        )
    if lower_bound >= upper_bound:
        raise ValueError("Reward lower bound must be smaller than upper bound.")
    if not 0.0 <= allocation_blend <= 1.0:
        raise ValueError("allocation_blend must lie in [0, 1].")

    contextual_residual = contextual_residual.to(
        device=state_reward_means.device, dtype=state_reward_means.dtype
    )
    positive_weights = (state_reward_means + contextual_residual).clamp(
        lower_bound, upper_bound
    )
    negative_weights = (state_reward_means - contextual_residual).clamp(
        lower_bound, upper_bound
    )
    positive_weights = 1.0 + float(allocation_blend) * (positive_weights - 1.0)
    negative_weights = 1.0 + float(allocation_blend) * (negative_weights - 1.0)
    positive_rewards = unallocated_rewards + torch.sum(
        reward_components * positive_weights, dim=-1
    )
    negative_rewards = unallocated_rewards + torch.sum(
        reward_components * negative_weights, dim=-1
    )
    return positive_rewards, negative_rewards


def linear_trend(values: torch.Tensor) -> torch.Tensor:
    """Return the least-squares slope of a one-dimensional metric sequence."""
    values = values.reshape(-1)
    if values.numel() < 2:
        raise ValueError("A trend requires at least two values.")
    steps = torch.arange(values.numel(), device=values.device, dtype=values.dtype)
    centered_steps = steps - steps.mean()
    denominator = torch.square(centered_steps).sum().clamp_min(1.0e-6)
    return torch.sum(centered_steps * (values - values.mean())) / denominator


def winner_conditioned_pair_progress(pair_progress: float, winner_sign: float) -> float:
    """Express an antithetic slope gap in the selected candidate's action frame.

    ``pair_progress`` is defined in the positive candidate's frame as
    ``0.5 * (positive_slope - negative_slope)``. If the negative candidate is
    selected, its Gaussian score function already reverses the action direction;
    its credit must therefore reverse too. Reversing the sample but retaining
    the signed gap would train the allocator back toward the losing candidate.
    """
    if winner_sign not in (-1.0, 1.0):
        raise ValueError("winner_sign must be exactly -1 or +1.")
    winner_progress = float(pair_progress) * float(winner_sign)
    if winner_progress < -1.0e-8:
        raise ValueError("winner_sign is inconsistent with the antithetic slope gap.")
    return max(winner_progress, 0.0)


def discounted_rollout_score(
    reference_rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Summarize one rollout with the immutable discounted reference return."""
    if reference_rewards.ndim != 2:
        raise ValueError("Reference rewards must have shape [time, env].")
    if dones.shape != reference_rewards.shape:
        raise ValueError("Done flags must match reference rewards.")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0, 1].")
    running_return = torch.zeros(reference_rewards.shape[1], device=reference_rewards.device)
    returns = torch.zeros_like(reference_rewards)
    done_mask = dones.to(device=reference_rewards.device, dtype=torch.bool)
    for step in range(reference_rewards.shape[0] - 1, -1, -1):
        running_return = (
            reference_rewards[step]
            + float(gamma) * running_return * (~done_mask[step]).float()
        )
        returns[step] = running_return
    return returns.mean()


def delayed_reference_progress(
    previous_score: float,
    next_score: float,
    normalization_scale: float,
) -> tuple[float, float]:
    """Return the real next-rollout objective change used by the outer policy."""
    if normalization_scale <= 0.0:
        raise ValueError("normalization_scale must be positive.")
    improvement = float(next_score) - float(previous_score)
    normalized = float(
        torch.tanh(torch.tensor(improvement / float(normalization_scale))).item()
    )
    return improvement, normalized


def block_reference_learning_speed(
    start_score: float,
    evaluation_score: float,
    inner_updates: int,
    normalization_scale: float,
) -> tuple[float, float]:
    """Measure fixed-objective ascent per inner-PPO update in one meta block."""
    if inner_updates <= 0:
        raise ValueError("inner_updates must be positive.")
    return delayed_reference_progress(
        float(start_score) / float(inner_updates),
        float(evaluation_score) / float(inner_updates),
        normalization_scale,
    )


def block_reference_learning_trend(
    rollout_scores: torch.Tensor,
    normalization_scale: float,
) -> tuple[float, float, float, float]:
    """Fit an environment-wise fixed-objective trend with sign confidence.

    The first dimension indexes successive policy snapshots. Remaining
    dimensions are independent environment samples. Fitting each environment
    before aggregation preserves the same mean slope while providing a standard
    error; statistically ambiguous signs are shrunk instead of being treated as
    certain outer-RL outcomes.
    """
    scores = torch.as_tensor(rollout_scores, dtype=torch.float32)
    if scores.ndim < 1 or scores.shape[0] < 2:
        raise ValueError("A meta block requires at least two policy scores.")
    if not torch.isfinite(scores).all():
        raise ValueError("Every meta-block score must be finite.")
    if normalization_scale <= 0.0:
        raise ValueError("normalization_scale must be positive.")
    steps = torch.arange(scores.shape[0], device=scores.device, dtype=scores.dtype)
    centered_steps = steps - steps.mean()
    denominator = torch.square(centered_steps).sum().clamp_min(1.0e-6)
    flat_scores = scores.reshape(scores.shape[0], -1)
    sample_slopes = torch.sum(
        centered_steps.unsqueeze(-1) * (flat_scores - flat_scores.mean(dim=0, keepdim=True)),
        dim=0,
    ) / denominator
    mean_slope = sample_slopes.mean()
    if sample_slopes.numel() > 1:
        standard_error = sample_slopes.std(unbiased=True) / sample_slopes.numel() ** 0.5
        z_score = torch.abs(mean_slope) / standard_error.clamp_min(1.0e-8)
        sign_confidence = torch.erf(z_score / 2.0**0.5)
    else:
        standard_error = torch.zeros_like(mean_slope)
        sign_confidence = torch.ones_like(mean_slope)
    normalized = torch.tanh(mean_slope / float(normalization_scale)) * sign_confidence
    return (
        float(mean_slope.item()),
        float(normalized.item()),
        float(standard_error.item()),
        float(sign_confidence.item()),
    )


def collapse_shared_rollout_allocation(
    context_contributions: torch.Tensor,
    allocator_contexts: torch.Tensor,
    allocator_samples: torch.Tensor,
    old_log_probabilities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate and collapse one shared allocator action from a PPO rollout."""
    if context_contributions.ndim != 2 or allocator_contexts.ndim != 2:
        raise ValueError("Rollout contexts must have shape [time, feature].")
    if allocator_samples.ndim != 2 or old_log_probabilities.ndim != 2:
        raise ValueError("Rollout allocator data must have shapes [time, reward] and [time, reward].")
    rollout_steps = context_contributions.shape[0]
    if not (
        allocator_contexts.shape[0]
        == allocator_samples.shape[0]
        == old_log_probabilities.shape[0]
        == rollout_steps
    ):
        raise ValueError("All rollout allocator tensors must have the same time dimension.")
    if rollout_steps == 0:
        raise ValueError("Cannot collapse an empty allocator rollout.")
    if old_log_probabilities.shape != allocator_samples.shape:
        raise ValueError("Allocator samples and component log probabilities must have identical shapes.")

    first_context = allocator_contexts[0]
    first_sample = allocator_samples[0]
    first_log_probability = old_log_probabilities[0]
    if not torch.equal(allocator_contexts, first_context.unsqueeze(0).expand_as(allocator_contexts)):
        raise ValueError("Allocator context changed inside one PPO rollout.")
    if not torch.equal(allocator_samples, first_sample.unsqueeze(0).expand_as(allocator_samples)):
        raise ValueError("Reward allocation changed inside one PPO rollout.")
    if not torch.equal(
        old_log_probabilities,
        first_log_probability.expand_as(old_log_probabilities),
    ):
        raise ValueError("Allocator log probability changed inside one PPO rollout.")
    return (
        context_contributions.mean(dim=0),
        first_context,
        first_sample,
        first_log_probability,
    )


def causal_progress_credit(
    reference_progress: float,
    progress_scale: float,
    ppo_advantages: torch.Tensor,
    *,
    advantage_modulation: float,
    progress_weight: float,
) -> torch.Tensor:
    """Distribute delayed outer progress without allowing advantage reward hacking."""
    if progress_scale <= 0.0:
        raise ValueError("Progress scale must be positive.")
    if not 0.0 <= advantage_modulation < 1.0:
        raise ValueError("Advantage modulation must lie in [0, 1).")
    standardized_advantage = (
        (ppo_advantages - ppo_advantages.mean())
        / ppo_advantages.std(unbiased=False).clamp_min(1.0e-6)
    )
    normalized_progress = torch.tanh(
        torch.as_tensor(
            reference_progress / progress_scale,
            device=ppo_advantages.device,
            dtype=ppo_advantages.dtype,
        )
    )
    return (
        float(progress_weight)
        * normalized_progress
        * (1.0 + float(advantage_modulation) * torch.tanh(standardized_advantage))
    )


def rollout_causal_progress_credit(
    reference_progress: float,
    progress_scale: float,
    ppo_advantages: torch.Tensor,
    *,
    advantage_modulation: float,
    progress_weight: float,
) -> torch.Tensor:
    """Return one signed credit for one rollout-level allocator action."""
    if progress_scale <= 0.0:
        raise ValueError("Progress scale must be positive.")
    if not 0.0 <= advantage_modulation < 1.0:
        raise ValueError("Advantage modulation must lie in [0, 1).")
    standardized_advantage = (
        (ppo_advantages - ppo_advantages.mean())
        / ppo_advantages.std(unbiased=False).clamp_min(1.0e-6)
    )
    advantage_confidence = torch.tanh(standardized_advantage.abs().mean())
    normalized_progress = torch.tanh(
        torch.as_tensor(
            reference_progress / progress_scale,
            device=ppo_advantages.device,
            dtype=ppo_advantages.dtype,
        )
    )
    return (
        float(progress_weight)
        * normalized_progress
        * (1.0 + float(advantage_modulation) * advantage_confidence)
    )


def componentwise_rollout_causal_progress_credit(
    reference_progress: float,
    progress_scale: float,
    ppo_advantages: torch.Tensor,
    reward_component_returns: torch.Tensor,
    *,
    advantage_weight: float,
    progress_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Modulate causal outer progress with PPO component attribution.

    Fixed-reference progress alone controls the sign. PPO/component alignment
    only redistributes its magnitude, so mutable PPO advantages cannot move the
    allocator when the paired candidates are indistinguishable or reverse the
    immutable objective's preference.
    """
    if progress_scale <= 0.0:
        raise ValueError("Progress scale must be positive.")
    if not 0.0 <= advantage_weight <= 1.0:
        raise ValueError("Advantage weight must lie in [0, 1].")
    if reward_component_returns.ndim != ppo_advantages.ndim + 1:
        raise ValueError("Reward component returns must add one reward dimension to PPO advantages.")
    if reward_component_returns.shape[:-1] != ppo_advantages.shape:
        raise ValueError("Reward component returns and PPO advantages must share rollout dimensions.")

    flat_advantages = ppo_advantages.reshape(-1)
    flat_components = reward_component_returns.reshape(-1, reward_component_returns.shape[-1])
    standardized_advantage = (
        (flat_advantages - flat_advantages.mean())
        / flat_advantages.std(unbiased=False).clamp_min(1.0e-6)
    )
    standardized_components = (
        (flat_components - flat_components.mean(dim=0, keepdim=True))
        / flat_components.std(dim=0, unbiased=False, keepdim=True).clamp_min(1.0e-6)
    )
    component_alignment = torch.mean(
        standardized_advantage.unsqueeze(-1) * standardized_components,
        dim=0,
    ).clamp(-1.0, 1.0)
    normalized_progress = torch.tanh(
        torch.as_tensor(
            reference_progress / progress_scale,
            device=ppo_advantages.device,
            dtype=ppo_advantages.dtype,
        )
    )
    component_credit = float(progress_weight) * normalized_progress * (
        1.0 + float(advantage_weight) * component_alignment
    )
    return component_credit, component_alignment


def statewise_rollout_causal_progress_credit(
    reference_progress: float,
    progress_scale: float,
    ppo_advantages: torch.Tensor,
    reward_component_returns: torch.Tensor,
    *,
    advantage_weight: float,
    progress_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign delayed fixed-objective credit to every state/reward pair.

    The measured next-rollout objective change remains the only source of the
    update sign. Standardized PPO advantage and component return provide a bounded
    local attribution factor, allowing the reward head to learn from the state
    feature that actually produced each reward rather than only an aggregate
    context.  The factor is strictly positive, so mutable PPO quantities cannot
    overturn the immutable objective result.
    """
    if progress_scale <= 0.0:
        raise ValueError("Progress scale must be positive.")
    if not 0.0 <= advantage_weight <= 1.0:
        raise ValueError("Advantage weight must lie in [0, 1].")
    if reward_component_returns.ndim != ppo_advantages.ndim + 1:
        raise ValueError("Reward component returns must add one reward dimension to PPO advantages.")
    if reward_component_returns.shape[:-1] != ppo_advantages.shape:
        raise ValueError("Reward component returns and PPO advantages must share rollout dimensions.")

    flat_advantages = ppo_advantages.reshape(-1)
    flat_components = reward_component_returns.reshape(-1, reward_component_returns.shape[-1])
    standardized_advantage = (
        (flat_advantages - flat_advantages.mean())
        / flat_advantages.std(unbiased=False).clamp_min(1.0e-6)
    )
    standardized_components = (
        (flat_components - flat_components.mean(dim=0, keepdim=True))
        / flat_components.std(dim=0, unbiased=False, keepdim=True).clamp_min(1.0e-6)
    )
    local_alignment = (
        standardized_advantage.unsqueeze(-1) * standardized_components
    ).clamp(-1.0, 1.0)
    component_alignment = local_alignment.mean(dim=0)
    normalized_progress = torch.tanh(
        torch.as_tensor(
            reference_progress / progress_scale,
            device=ppo_advantages.device,
            dtype=ppo_advantages.dtype,
        )
    )
    local_credit = float(progress_weight) * normalized_progress * (
        1.0 + float(advantage_weight) * local_alignment
    )
    return local_credit.reshape(reward_component_returns.shape), component_alignment


def componentwise_reference_alignment_credit(
    reference_returns: torch.Tensor,
    ppo_advantages: torch.Tensor,
    reward_component_returns: torch.Tensor,
    *,
    advantage_modulation: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return dense per-reward credit anchored to the immutable objective.

    Fixed-reference correlation determines the sign. PPO-advantage correlation
    only changes its magnitude, so mutable training rewards cannot reverse the
    outer objective.
    """
    if not 0.0 <= advantage_modulation < 1.0:
        raise ValueError("advantage_modulation must lie in [0, 1).")
    if reference_returns.shape != ppo_advantages.shape:
        raise ValueError("Reference returns and PPO advantages must have identical shapes.")
    if reward_component_returns.shape[:-1] != reference_returns.shape:
        raise ValueError("Reward component returns must add one reward dimension.")

    flat_reference = reference_returns.reshape(-1)
    flat_advantages = ppo_advantages.reshape(-1)
    flat_components = reward_component_returns.reshape(-1, reward_component_returns.shape[-1])

    def standardize(value: torch.Tensor, dim: int | None = None) -> torch.Tensor:
        return (value - value.mean(dim=dim, keepdim=dim is not None)) / value.std(
            dim=dim, unbiased=False, keepdim=dim is not None
        ).clamp_min(1.0e-6)

    standardized_reference = standardize(flat_reference)
    standardized_advantage = standardize(flat_advantages)
    standardized_components = standardize(flat_components, dim=0)
    reference_alignment = torch.mean(
        standardized_reference.unsqueeze(-1) * standardized_components,
        dim=0,
    ).clamp(-1.0, 1.0)
    ppo_alignment = torch.mean(
        standardized_advantage.unsqueeze(-1) * standardized_components,
        dim=0,
    ).clamp(-1.0, 1.0)
    dense_credit = reference_alignment * (
        1.0 + float(advantage_modulation) * ppo_alignment
    )
    return dense_credit, reference_alignment, ppo_alignment


def _masked_mean(value: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Return a numerically safe mean, including for an all-false mask."""
    if mask is None:
        return value.mean()
    mask = mask.to(device=value.device, dtype=value.dtype)
    try:
        mask = torch.broadcast_to(mask, value.shape)
    except RuntimeError as error:
        raise ValueError(
            f"Mask shape {tuple(mask.shape)} cannot be broadcast to value shape {tuple(value.shape)}."
        ) from error
    denominator = mask.sum().clamp_min(1.0)
    return (value * mask).sum() / denominator


class DepthFeatureEncoder(nn.Module):
    """Compact depth-history encoder that can be deployed without prediction heads.

    Accepted input shapes are ``[B, history, height, width]`` and
    ``[B, history * height * width]``. The output is a bounded latent vector,
    making an EMA copy suitable as a stable observation for the policy.
    """

    checkpoint_format_version = 2

    def __init__(
        self,
        depth_history_steps: int = 4,
        depth_height: int = 27,
        depth_width: int = 48,
        latent_dim: int = 32,
    ) -> None:
        super().__init__()
        if depth_history_steps <= 0 or depth_height <= 0 or depth_width <= 0 or latent_dim <= 0:
            raise ValueError("All depth encoder dimensions must be positive.")

        self.depth_history_steps = int(depth_history_steps)
        self.depth_height = int(depth_height)
        self.depth_width = int(depth_width)
        self.latent_dim = int(latent_dim)

        self.backbone = nn.Sequential(
            nn.Conv2d(self.depth_history_steps, 24, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, 24),
            nn.ELU(),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 48),
            nn.ELU(),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((2, 4)),
            nn.Flatten(start_dim=1),
        )
        self.projection = nn.Sequential(
            nn.Linear(96 * 2 * 4, 128),
            nn.ELU(),
            nn.Linear(128, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.Tanh(),
        )

    @property
    def input_dim(self) -> int:
        return self.depth_history_steps * self.depth_height * self.depth_width

    def get_config(self) -> dict[str, int]:
        return {
            "depth_history_steps": self.depth_history_steps,
            "depth_height": self.depth_height,
            "depth_width": self.depth_width,
            "latent_dim": self.latent_dim,
        }

    def _reshape_input(self, depth_history: torch.Tensor) -> torch.Tensor:
        if depth_history.ndim == 2:
            if depth_history.shape[-1] != self.input_dim:
                raise ValueError(
                    f"Flattened depth input has {depth_history.shape[-1]} values; expected {self.input_dim}."
                )
            return depth_history.reshape(
                -1,
                self.depth_history_steps,
                self.depth_height,
                self.depth_width,
            )
        if depth_history.ndim == 4:
            expected = (self.depth_history_steps, self.depth_height, self.depth_width)
            if tuple(depth_history.shape[1:]) != expected:
                raise ValueError(
                    f"Depth input shape is {tuple(depth_history.shape[1:])}; expected {expected}."
                )
            return depth_history
        raise ValueError(
            "Depth history must have shape [B, history, height, width] or [B, history*height*width]."
        )

    def forward(self, depth_history: torch.Tensor) -> torch.Tensor:
        depth_history = self._reshape_input(depth_history)
        return self.projection(self.backbone(depth_history))

    def save_checkpoint(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        """Save a self-contained encoder-only checkpoint for deployment."""
        payload = {
            "format_version": self.checkpoint_format_version,
            "encoder_config": self.get_config(),
            "encoder_state_dict": self.state_dict(),
            "extra": {} if extra is None else extra,
        }
        torch.save(payload, Path(path))

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        map_location: str | torch.device | None = None,
        strict: bool = True,
    ) -> tuple[DepthFeatureEncoder, dict[str, Any]]:
        """Construct an encoder directly from an encoder-only checkpoint."""
        payload = torch.load(Path(path), map_location=map_location)
        if not isinstance(payload, dict) or "encoder_config" not in payload:
            raise ValueError("Checkpoint is not a self-contained DepthFeatureEncoder checkpoint.")
        format_version = payload.get("format_version")
        if format_version != cls.checkpoint_format_version:
            raise ValueError(
                f"Unsupported depth encoder checkpoint format {format_version!r}; "
                f"expected {cls.checkpoint_format_version}."
            )
        if not isinstance(payload["encoder_config"], dict):
            raise ValueError("Depth encoder checkpoint encoder_config must be a dictionary.")
        encoder = cls(**payload["encoder_config"])
        state_dict = payload.get("encoder_state_dict", payload.get("state_dict"))
        if state_dict is None:
            raise ValueError("Encoder checkpoint contains no state dictionary.")
        encoder.load_state_dict(state_dict, strict=strict)
        return encoder, payload.get("extra", {})


class PredictiveFeasibilityModel(nn.Module):
    """Training-only action-conditioned future-feasibility predictor.

    ``online_depth_encoder`` receives gradients from the event and auxiliary
    future-state objectives. ``policy_depth_encoder`` is its exponential-moving
    average (EMA) copy and is the only component required by the deployed actor.

    Event predictions have shape ``[B, ensemble_size, 4]`` in the fixed order
    ``stable, grounded, trackable, low_slip``. ``predict`` also returns their
    mean, ensemble standard deviation, and a conservative probability
    ``mean - uncertainty_scale * std``.
    """

    checkpoint_format_version = 2

    def __init__(
        self,
        history_steps: int = 4,
        depth_height: int = 27,
        depth_width: int = 48,
        state_history_steps: int | None = None,
        state_dim: int = 28,
        action_dim: int = 6,
        event_dim: int = 4,
        latent_dim: int = 32,
        future_steps: int = 18,
        future_state_dim: int = 10,
        reward_dim: int = 4,
        ensemble_size: int = 3,
        fusion_hidden_dims: tuple[int, ...] = (128, 128),
        event_logit_bias: float = 1.4,
        ema_decay: float = 0.995,
        uncertainty_scale: float = 1.0,
        ensemble_bootstrap_probability: float = 0.8,
        depth_history_steps: int | None = None,
    ) -> None:
        super().__init__()
        if depth_history_steps is not None:
            history_steps = int(depth_history_steps)
        if state_history_steps is None:
            state_history_steps = int(history_steps)
        if state_history_steps <= 0 or state_dim <= 0 or action_dim <= 0:
            raise ValueError("State/action dimensions must be positive.")
        if future_steps <= 0 or ensemble_size <= 0 or reward_dim <= 0:
            raise ValueError("future_steps and ensemble_size must be positive.")
        if event_dim != len(EVENT_NAMES):
            raise ValueError(f"event_dim must be {len(EVENT_NAMES)} for the fixed event semantics {EVENT_NAMES}.")
        if future_state_dim != 10:
            raise ValueError("future_state_dim must be 10: quaternion(4) + linear velocity(3) + angular velocity(3).")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must lie in [0, 1).")
        if not fusion_hidden_dims:
            raise ValueError("fusion_hidden_dims cannot be empty.")
        if not 0.0 < ensemble_bootstrap_probability <= 1.0:
            raise ValueError("ensemble_bootstrap_probability must lie in (0, 1].")

        self.state_history_steps = int(state_history_steps)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.event_dim = int(event_dim)
        self.latent_dim = int(latent_dim)
        self.future_steps = int(future_steps)
        self.future_state_dim = int(future_state_dim)
        self.reward_dim = int(reward_dim)
        self.ensemble_size = int(ensemble_size)
        self.ema_decay = float(ema_decay)
        self.uncertainty_scale = float(uncertainty_scale)
        self.event_logit_bias = float(event_logit_bias)
        self.fusion_hidden_dims = tuple(int(dim) for dim in fusion_hidden_dims)
        self.ensemble_bootstrap_probability = float(ensemble_bootstrap_probability)

        self.online_depth_encoder = DepthFeatureEncoder(
            depth_history_steps=history_steps,
            depth_height=depth_height,
            depth_width=depth_width,
            latent_dim=latent_dim,
        )
        self.policy_depth_encoder = copy.deepcopy(self.online_depth_encoder)
        self.policy_depth_encoder.requires_grad_(False)

        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_history_steps * self.state_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_dim, 32),
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU(),
        )

        fusion_layers: list[nn.Module] = []
        previous_dim = self.latent_dim + 64 + 32
        for hidden_dim in self.fusion_hidden_dims:
            fusion_layers.extend((nn.Linear(previous_dim, hidden_dim), nn.ELU()))
            previous_dim = hidden_dim
        self.fusion = nn.Sequential(*fusion_layers)

        self.event_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(previous_dim, 64),
                    nn.ELU(),
                    nn.Linear(64, self.event_dim),
                )
                for _ in range(self.ensemble_size)
            ]
        )
        self.future_head = nn.Sequential(
            nn.Linear(previous_dim, 128),
            nn.ELU(),
            nn.Linear(128, self.future_steps * self.future_state_dim),
        )
        # The allocator consumes a stable, explicit state/action context rather
        # than the auxiliary event/future latent. The latter is continuously
        # rewritten by supervised learning; using it made reward weights change
        # even when the outer policy had not updated. Tanh bounds raw sensor
        # scales without introducing trainable shared features.
        self.reward_context_dim = self.state_history_steps * self.state_dim + self.action_dim
        # Independent output rows are essential for coordinate-wise outer RL:
        # updating one reward must not mutate a shared reward-head hidden layer
        # and silently move the other allocations.
        self.reward_head = nn.Sequential(
            nn.Linear(self.reward_context_dim, self.reward_dim)
        )
        self._initialize_output_heads()

    @property
    def depth_encoder(self) -> DepthFeatureEncoder:
        """Alias for the gradient-trained encoder, useful to optimizer builders."""
        return self.online_depth_encoder

    def get_config(self) -> dict[str, Any]:
        encoder_config = self.online_depth_encoder.get_config()
        return {
            "history_steps": encoder_config["depth_history_steps"],
            "depth_height": encoder_config["depth_height"],
            "depth_width": encoder_config["depth_width"],
            "latent_dim": encoder_config["latent_dim"],
            "state_history_steps": self.state_history_steps,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "event_dim": self.event_dim,
            "future_steps": self.future_steps,
            "future_state_dim": self.future_state_dim,
            "reward_dim": self.reward_dim,
            "reward_context_dim": self.reward_context_dim,
            "ensemble_size": self.ensemble_size,
            "fusion_hidden_dims": self.fusion_hidden_dims,
            "event_logit_bias": self.event_logit_bias,
            "ema_decay": self.ema_decay,
            "uncertainty_scale": self.uncertainty_scale,
            "ensemble_bootstrap_probability": self.ensemble_bootstrap_probability,
        }

    def _initialize_output_heads(self) -> None:
        for head in self.event_heads:
            output_layer = head[-1]
            assert isinstance(output_layer, nn.Linear)
            nn.init.normal_(output_layer.weight, mean=0.0, std=1.0e-3)
            nn.init.constant_(output_layer.bias, self.event_logit_bias)

        future_output = self.future_head[-1]
        assert isinstance(future_output, nn.Linear)
        nn.init.normal_(future_output.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(future_output.bias)
        with torch.no_grad():
            future_output.bias.view(self.future_steps, self.future_state_dim)[:, 0] = 1.0
        reward_output = self.reward_head[-1]
        assert isinstance(reward_output, nn.Linear)
        nn.init.normal_(reward_output.weight, mean=0.0, std=1.0e-3)
        # Direct reward weighting starts from the neutral multiplier one.  The
        # environment also blends from one during predictor warm-up.
        nn.init.ones_(reward_output.bias)

    def _reshape_state_history(self, state_history: torch.Tensor) -> torch.Tensor:
        expected_dim = self.state_history_steps * self.state_dim
        if state_history.ndim == 3:
            expected_shape = (self.state_history_steps, self.state_dim)
            if tuple(state_history.shape[1:]) != expected_shape:
                raise ValueError(
                    f"State history shape is {tuple(state_history.shape[1:])}; expected {expected_shape}."
                )
            return state_history.reshape(-1, expected_dim)
        if state_history.ndim == 2 and state_history.shape[-1] == expected_dim:
            return state_history
        raise ValueError(
            f"State history must have shape [B, {self.state_history_steps}, {self.state_dim}] "
            f"or [B, {expected_dim}]."
        )

    def encode_for_policy(self, depth_history: torch.Tensor, detach: bool = True) -> torch.Tensor:
        """Encode policy depth observations with the stable EMA encoder.

        The default detached output is appropriate for PPO rollouts: PPO trains
        the actor, while prediction losses train the online encoder and EMA keeps
        the actor-facing representation stable.
        """
        if detach:
            with torch.no_grad():
                return self.policy_depth_encoder(depth_history)
        return self.policy_depth_encoder(depth_history)

    def encode_for_training(self, depth_history: torch.Tensor) -> torch.Tensor:
        """Encode depth with the online encoder, retaining prediction gradients."""
        return self.online_depth_encoder(depth_history)

    def predict_from_latent(
        self,
        depth_latent: torch.Tensor,
        state_history: torch.Tensor,
        actions: torch.Tensor,
        conservative_beta: float | None = None,
        uncertainty_scale: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the action-conditioned heads from a precomputed depth latent."""
        if depth_latent.ndim != 2 or depth_latent.shape[-1] != self.latent_dim:
            raise ValueError(f"Depth latent must have shape [B, {self.latent_dim}].")
        state_history = self._reshape_state_history(state_history)
        if actions.ndim != 2 or actions.shape[-1] != self.action_dim:
            raise ValueError(f"Action must have shape [B, {self.action_dim}].")
        if depth_latent.shape[0] != state_history.shape[0] or depth_latent.shape[0] != actions.shape[0]:
            raise ValueError("Depth, state, and action batch sizes must match.")

        fused = self.fusion(
            torch.cat(
                (depth_latent, self.state_encoder(state_history), self.action_encoder(actions)),
                dim=-1,
            )
        )
        event_logits = torch.stack([head(fused) for head in self.event_heads], dim=1)
        event_probabilities = torch.sigmoid(event_logits)
        event_mean = event_probabilities.mean(dim=1)
        event_std = event_probabilities.std(dim=1, unbiased=False)
        if conservative_beta is not None and uncertainty_scale is not None:
            raise ValueError("Pass only one of conservative_beta and uncertainty_scale.")
        requested_scale = conservative_beta if conservative_beta is not None else uncertainty_scale
        scale = self.uncertainty_scale if requested_scale is None else float(requested_scale)
        conservative_probability = (event_mean - scale * event_std).clamp(0.0, 1.0)

        future_prediction = self.future_head(fused).view(-1, self.future_steps, self.future_state_dim)
        future_quaternion = F.normalize(future_prediction[..., :4], dim=-1, eps=1.0e-6)
        future_prediction = torch.cat((future_quaternion, future_prediction[..., 4:]), dim=-1)
        allocator_context = torch.tanh(torch.cat((state_history, actions), dim=-1))
        return {
            "depth_latent": depth_latent,
            # The outer reward allocator is optimized after PPO has computed
            # rollout advantages.  Retaining this compact, detached feature is
            # much cheaper than retaining full depth histories for the rollout.
            "fused_features": fused,
            # This context is independent of every auxiliary Predictor
            # parameter, so event/future optimizer steps cannot silently move
            # the reward policy between collection and outer update.
            "allocator_context": allocator_context,
            "event_logits": event_logits,
            "event_probabilities": event_probabilities,
            "event_mean": event_mean,
            "event_std": event_std,
            "mean_probability": event_mean,
            "std_probability": event_std,
            "conservative_probability": conservative_probability,
            "future_prediction": future_prediction,
            "reward_prediction": self.reward_head(allocator_context),
        }

    def predict(
        self,
        depth_history: torch.Tensor,
        state_history: torch.Tensor,
        actions: torch.Tensor,
        use_policy_encoder: bool = False,
        detach_encoder: bool = False,
        conservative_beta: float | None = None,
        uncertainty_scale: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict future prerequisites and auxiliary body states.

        Training normally uses the online encoder (the defaults). The explicit
        ``predict_for_gate`` path also uses online latents; the policy-encoder
        option is reserved for encoder diagnostics and deployment checks.
        """
        encoder = self.policy_depth_encoder if use_policy_encoder else self.online_depth_encoder
        if detach_encoder:
            with torch.no_grad():
                depth_latent = encoder(depth_history)
        else:
            depth_latent = encoder(depth_history)
        return self.predict_from_latent(
            depth_latent,
            state_history,
            actions,
            conservative_beta=conservative_beta,
            uncertainty_scale=uncertainty_scale,
        )

    @torch.no_grad()
    def predict_for_gate(
        self,
        depth_history: torch.Tensor,
        state_history: torch.Tensor,
        actions: torch.Tensor,
        conservative_beta: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run gate inference on the online latent distribution used to train heads."""

        depth_latent = self.encode_for_training(depth_history)
        return self.predict_from_latent(
            depth_latent,
            state_history,
            actions,
            conservative_beta=conservative_beta,
        )

    def compute_loss(
        self,
        prediction: dict[str, torch.Tensor],
        event_targets: torch.Tensor,
        future_targets: torch.Tensor,
        reward_targets: torch.Tensor | None = None,
        event_mask: torch.Tensor | None = None,
        future_mask: torch.Tensor | None = None,
        event_weight: float = 1.0,
        quaternion_weight: float = 1.0,
        linear_velocity_weight: float = 1.0,
        angular_velocity_weight: float = 0.5,
        reward_weight: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute masked event BCE and sign-invariant future-state losses.

        ``event_targets`` may be binary or soft and has shape ``[B, 4]``.
        ``future_targets`` has shape ``[B, future_steps, 10]``. Event masks may
        have shape ``[B]``/``[B, 4]``; future masks may have shape
        ``[B]``/``[B, future_steps]``. All component losses and ``total_loss``
        are returned for logging.
        """
        event_logits = prediction["event_logits"]
        future_prediction = prediction["future_prediction"]
        expected_event_shape = (event_logits.shape[0], self.event_dim)
        if tuple(event_targets.shape) != expected_event_shape:
            raise ValueError(
                f"Event targets have shape {tuple(event_targets.shape)}; expected {expected_event_shape}."
            )
        expected_future_shape = (future_prediction.shape[0], self.future_steps, self.future_state_dim)
        if tuple(future_targets.shape) != expected_future_shape:
            raise ValueError(
                f"Future targets have shape {tuple(future_targets.shape)}; expected {expected_future_shape}."
            )

        event_targets = event_targets.to(device=event_logits.device, dtype=event_logits.dtype)
        expanded_event_targets = event_targets.unsqueeze(1).expand_as(event_logits)
        event_loss_values = F.binary_cross_entropy_with_logits(
            event_logits,
            expanded_event_targets,
            reduction="none",
        )
        expanded_event_mask: torch.Tensor | None = None
        if event_mask is not None:
            if event_mask.ndim == 1:
                event_mask = event_mask[:, None]
            expanded_event_mask = event_mask[:, None, :]
        if self.training and self.ensemble_size > 1 and self.ensemble_bootstrap_probability < 1.0:
            bootstrap_mask = torch.rand(
                event_logits.shape[0],
                self.ensemble_size,
                1,
                device=event_logits.device,
                generator=generator,
            ) < self.ensemble_bootstrap_probability
            # Prevent a head from receiving an empty mini-batch.
            for head_index in range(self.ensemble_size):
                bootstrap_mask[head_index % event_logits.shape[0], head_index, 0] = True
            expanded_event_mask = (
                bootstrap_mask if expanded_event_mask is None else bootstrap_mask & expanded_event_mask.bool()
            )
        event_loss = _masked_mean(event_loss_values, expanded_event_mask)

        future_targets = future_targets.to(
            device=future_prediction.device,
            dtype=future_prediction.dtype,
        )
        predicted_quaternion = future_prediction[..., :4]
        target_quaternion = F.normalize(future_targets[..., :4], dim=-1, eps=1.0e-6)
        same_sign_error = torch.square(predicted_quaternion - target_quaternion).sum(dim=-1)
        opposite_sign_error = torch.square(predicted_quaternion + target_quaternion).sum(dim=-1)
        quaternion_error = torch.minimum(same_sign_error, opposite_sign_error)
        linear_velocity_error = torch.square(
            future_prediction[..., 4:7] - future_targets[..., 4:7]
        ).mean(dim=-1)
        angular_velocity_error = torch.square(
            future_prediction[..., 7:10] - future_targets[..., 7:10]
        ).mean(dim=-1)

        expanded_future_mask = future_mask
        if expanded_future_mask is not None:
            if expanded_future_mask.ndim == 1:
                expanded_future_mask = expanded_future_mask[:, None]
            if expanded_future_mask.ndim == 3 and expanded_future_mask.shape[-1] == 1:
                expanded_future_mask = expanded_future_mask.squeeze(-1)
        quaternion_loss = _masked_mean(quaternion_error, expanded_future_mask)
        linear_velocity_loss = _masked_mean(linear_velocity_error, expanded_future_mask)
        angular_velocity_loss = _masked_mean(angular_velocity_error, expanded_future_mask)
        reward_loss = torch.zeros((), device=future_prediction.device, dtype=future_prediction.dtype)
        if reward_targets is not None:
            reward_prediction = prediction["reward_prediction"]
            if reward_targets.shape != reward_prediction.shape:
                raise ValueError(
                    f"Reward targets have shape {tuple(reward_targets.shape)}; "
                    f"expected {tuple(reward_prediction.shape)}."
                )
            reward_loss = torch.square(
                reward_prediction - reward_targets.to(reward_prediction)
            ).mean()

        total_loss = (
            float(event_weight) * event_loss
            + float(quaternion_weight) * quaternion_loss
            + float(linear_velocity_weight) * linear_velocity_loss
            + float(angular_velocity_weight) * angular_velocity_loss
            + float(reward_weight) * reward_loss
        )

        with torch.no_grad():
            event_mean = prediction["event_mean"]
            brier_values = torch.square(event_mean - event_targets)
            metric_mask = event_mask
            if metric_mask is not None and metric_mask.ndim == 1:
                metric_mask = metric_mask[:, None]
            event_brier = _masked_mean(brier_values, metric_mask)
            event_accuracy_values = ((event_mean >= 0.5) == (event_targets >= 0.5)).to(event_mean.dtype)
            event_accuracy = _masked_mean(event_accuracy_values, metric_mask)
            ensemble_uncertainty = _masked_mean(prediction["event_std"], metric_mask)

        return {
            "total_loss": total_loss,
            "event_loss": event_loss,
            "quaternion_loss": quaternion_loss,
            "linear_velocity_loss": linear_velocity_loss,
            "angular_velocity_loss": angular_velocity_loss,
            "reward_loss": reward_loss,
            "event_brier": event_brier,
            "event_accuracy": event_accuracy,
            "ensemble_uncertainty": ensemble_uncertainty,
        }

    @torch.no_grad()
    def update_ema(self, decay: float | None = None) -> None:
        """Update the deployable policy encoder from the online encoder."""
        decay = self.ema_decay if decay is None else float(decay)
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must lie in [0, 1).")
        for policy_parameter, online_parameter in zip(
            self.policy_depth_encoder.parameters(),
            self.online_depth_encoder.parameters(),
            strict=True,
        ):
            policy_parameter.lerp_(online_parameter, 1.0 - decay)
        for policy_buffer, online_buffer in zip(
            self.policy_depth_encoder.buffers(),
            self.online_depth_encoder.buffers(),
            strict=True,
        ):
            if torch.is_floating_point(policy_buffer):
                policy_buffer.lerp_(online_buffer, 1.0 - decay)
            else:
                policy_buffer.copy_(online_buffer)

    @torch.no_grad()
    def synchronize_policy_encoder(self) -> None:
        """Hard-copy online encoder weights into the deployable EMA encoder."""
        self.policy_depth_encoder.load_state_dict(self.online_depth_encoder.state_dict())

    def save_full(
        self,
        path: str | Path,
        optimizer: torch.optim.Optimizer | None = None,
        allocator_optimizer: torch.optim.Optimizer | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Save training state, including both encoders and all prediction heads."""
        payload: dict[str, Any] = {
            "format_version": self.checkpoint_format_version,
            "model_config": self.get_config(),
            "model_state_dict": self.state_dict(),
            "extra": {} if extra is None else extra,
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        if allocator_optimizer is not None:
            payload["allocator_optimizer_state_dict"] = allocator_optimizer.state_dict()
        torch.save(payload, Path(path))

    def load_full(
        self,
        path: str | Path,
        optimizer: torch.optim.Optimizer | None = None,
        allocator_optimizer: torch.optim.Optimizer | None = None,
        map_location: str | torch.device | None = None,
        strict: bool = True,
        required_extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Restore a full training checkpoint and return its extra metadata.

        Required semantic metadata is validated before any model or optimizer
        state is mutated.  This keeps structurally compatible checkpoints from
        silently crossing incompatible outer-learning algorithms.
        """
        payload = torch.load(Path(path), map_location=map_location)
        if not isinstance(payload, dict):
            raise ValueError("Full predictor checkpoint must be a dictionary.")
        format_version = payload.get("format_version")
        if format_version != self.checkpoint_format_version:
            raise ValueError(
                f"Unsupported full predictor checkpoint format {format_version!r}; "
                f"expected {self.checkpoint_format_version}."
            )
        checkpoint_config = payload.get("model_config")
        if not isinstance(checkpoint_config, dict):
            raise ValueError("Full predictor checkpoint contains no valid model_config dictionary.")
        expected_config = self.get_config()
        if checkpoint_config != expected_config:
            raise ValueError(
                f"Full predictor checkpoint config {checkpoint_config} does not match model config {expected_config}."
            )
        extra = payload.get("extra", {})
        if not isinstance(extra, dict):
            raise ValueError("Full predictor checkpoint contains no valid extra metadata dictionary.")
        if required_extra is not None:
            mismatches = {
                key: (extra.get(key), expected_value)
                for key, expected_value in required_extra.items()
                if extra.get(key) != expected_value
            }
            if mismatches:
                raise ValueError(
                    "Full predictor checkpoint semantic metadata is incompatible: "
                    f"{mismatches}. Start a fresh predictor meta-training run."
                )
        state_dict = payload.get("model_state_dict", payload.get("state_dict"))
        if state_dict is None:
            raise ValueError("Full predictor checkpoint contains no model state dictionary.")
        self.load_state_dict(state_dict, strict=strict)
        if optimizer is not None and "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if allocator_optimizer is not None and "allocator_optimizer_state_dict" in payload:
            allocator_optimizer.load_state_dict(payload["allocator_optimizer_state_dict"])
        return extra

    def save_encoder(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        """Save only the EMA depth encoder used by the deployed actor."""
        self.policy_depth_encoder.save_checkpoint(path, extra=extra)

    def load_encoder(
        self,
        path: str | Path,
        map_location: str | torch.device | None = None,
        target: str = "policy",
        strict: bool = True,
    ) -> dict[str, Any]:
        """Load an encoder-only checkpoint into ``policy``, ``online``, or ``both``."""
        if target not in {"policy", "online", "both"}:
            raise ValueError("target must be 'policy', 'online', or 'both'.")
        encoder, extra = DepthFeatureEncoder.from_checkpoint(
            path,
            map_location=map_location,
            strict=strict,
        )
        expected_config = self.online_depth_encoder.get_config()
        if encoder.get_config() != expected_config:
            raise ValueError(
                f"Encoder checkpoint config {encoder.get_config()} does not match model config {expected_config}."
            )
        state_dict = encoder.state_dict()
        if target in {"policy", "both"}:
            self.policy_depth_encoder.load_state_dict(state_dict, strict=strict)
        if target in {"online", "both"}:
            self.online_depth_encoder.load_state_dict(state_dict, strict=strict)
        return extra


__all__ = [
    "EVENT_NAMES",
    "DepthFeatureEncoder",
    "PredictiveFeasibilityModel",
    "causal_progress_credit",
    "block_reference_learning_speed",
    "block_reference_learning_trend",
    "censored_diagonal_gaussian_component_log_prob",
    "componentwise_rollout_causal_progress_credit",
    "antithetic_reward_candidates",
    "componentwise_reference_alignment_credit",
    "statewise_rollout_causal_progress_credit",
    "collapse_shared_rollout_allocation",
    "diagonal_gaussian_component_log_prob",
    "diagonal_gaussian_log_prob",
    "delayed_reference_progress",
    "discounted_rollout_score",
    "fixed_reference_policy_improvement",
    "linear_trend",
    "rollout_causal_progress_credit",
    "straight_through_clamp",
    "winner_conditioned_pair_progress",
]
