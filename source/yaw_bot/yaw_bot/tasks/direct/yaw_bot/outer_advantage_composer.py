"""Outer-advantage guided reward composition primitives.

This module deliberately has no Isaac Lab dependency.  Reward composition and
its training loss can therefore be regression-tested without a simulator.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call

try:
    from .predictive_labels import DIRECT_REWARD_NAMES
except ImportError:  # Standalone CPU regression tests load this file directly.
    from predictive_labels import DIRECT_REWARD_NAMES


REWARD_GROUP_NAMES = ("stability", "contact_slip", "linear", "yaw", "regularization")
OUTER_CHECKPOINT_REQUIRED_KEYS = (
    "outer_composer_state_dict",
    "outer_rollout_composer_state_dict",
    "outer_critic_state_dict",
    "outer_group_rms_state_dict",
    "outer_composer_optimizer_state_dict",
    "outer_critic_optimizer_state_dict",
    "outer_beta_iteration",
    "outer_beta_total_iterations",
    "outer_beta",
    "outer_composer_updates",
    "outer_critic_updates",
    "outer_learning_auc",
)

# The task currently exposes 22 atomic terms (two more than an old design
# document stated).  Preserve every term and give every index exactly one owner.
REWARD_GROUP_INDICES = (
    (0, 1, 2, 8),                 # stability
    (9, 10),                      # contact / slip-related contact penalties
    (12, 13, 14, 15, 16, 20, 21),# linear command/progress
    (3, 11, 17, 18),              # yaw command/progress
    (4, 5, 6, 7, 19),             # regularization
)


def validate_reward_groups(
    reward_names: Sequence[str] = DIRECT_REWARD_NAMES,
    groups: Sequence[Sequence[int]] = REWARD_GROUP_INDICES,
) -> None:
    """Require a disjoint, exhaustive partition of the atomic rewards."""

    flat = [int(index) for group in groups for index in group]
    expected = list(range(len(reward_names)))
    if sorted(flat) != expected or len(flat) != len(set(flat)):
        raise ValueError(f"Reward groups must partition indices {expected}; got {flat}.")


validate_reward_groups()


@torch.no_grad()
def _distributed_all_reduce_in_place(
    tensor: torch.Tensor,
    *,
    op: dist.ReduceOp,
) -> None:
    """Run a collective on CPU when the default process group is Gloo."""
    if dist.get_backend() == "gloo" and tensor.device.type != "cpu":
        host_tensor = tensor.detach().cpu()
        dist.all_reduce(host_tensor, op=op)
        tensor.copy_(host_tensor.to(tensor.device))
    else:
        dist.all_reduce(tensor, op=op)


def reward_group_index_tensor(device: torch.device | str | None = None) -> torch.Tensor:
    """Return the group owner of every atomic reward."""

    result = torch.empty(len(DIRECT_REWARD_NAMES), dtype=torch.long, device=device)
    for group_index, atomic_indices in enumerate(REWARD_GROUP_INDICES):
        result[list(atomic_indices)] = group_index
    return result


class RunningGroupRMS(nn.Module):
    """EMA running RMS for the five raw group sums.

    RMS normalization intentionally does not subtract a mean: signs and zero
    retain their physical reward meaning.  The pre-update scale is used for the
    current transition and statistics are updated only afterwards.
    """

    def __init__(self, num_groups: int = 5, decay: float = 0.999, epsilon: float = 1.0e-6) -> None:
        super().__init__()
        if num_groups <= 0 or not 0.0 <= decay < 1.0 or epsilon <= 0.0:
            raise ValueError("Invalid RunningGroupRMS configuration.")
        self.num_groups = int(num_groups)
        self.decay = float(decay)
        self.epsilon = float(epsilon)
        self.register_buffer("mean_square", torch.ones(self.num_groups))
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))

    @property
    def rms(self) -> torch.Tensor:
        return torch.sqrt(self.mean_square.clamp_min(self.epsilon))

    @torch.no_grad()
    def update(self, group_rewards: torch.Tensor) -> None:
        if group_rewards.ndim < 2 or group_rewards.shape[-1] != self.num_groups:
            raise ValueError(f"Expected [..., {self.num_groups}] group rewards.")
        reduce_dims = tuple(range(group_rewards.ndim - 1))
        batch_square = torch.mean(torch.square(group_rewards.float()), dim=reduce_dims)
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            # Every rank executes one RMS update per simulator step. Averaging
            # the five sufficient statistics here is exact global-batch RMS
            # normalization and costs only five floats per environment step.
            _distributed_all_reduce_in_place(
                batch_square, op=dist.ReduceOp.SUM
            )
            batch_square.div_(dist.get_world_size())
        if int(self.updates.item()) == 0:
            self.mean_square.copy_(batch_square.clamp_min(self.epsilon))
        else:
            self.mean_square.mul_(self.decay).add_(batch_square, alpha=1.0 - self.decay)
        self.updates.add_(1)

    def normalize(self, group_rewards: torch.Tensor, *, update: bool = True) -> torch.Tensor:
        scale = self.rms.to(group_rewards)
        normalized = group_rewards / scale
        if update:
            self.update(group_rewards.detach())
        return normalized


def group_atomic_rewards(
    atomic_rewards: torch.Tensor,
    rms: RunningGroupRMS,
    *,
    update_rms: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return raw groups, normalized groups, and per-atomic normalized values."""

    if atomic_rewards.shape[-1] != len(DIRECT_REWARD_NAMES):
        raise ValueError(
            f"Expected {len(DIRECT_REWARD_NAMES)} atomic rewards, got {atomic_rewards.shape[-1]}."
        )
    raw_groups = torch.stack(
        [atomic_rewards[..., list(indices)].sum(dim=-1) for indices in REWARD_GROUP_INDICES],
        dim=-1,
    )
    scales = rms.rms.to(atomic_rewards)
    normalized_groups = raw_groups / scales
    owners = reward_group_index_tensor(atomic_rewards.device)
    normalized_atomic = atomic_rewards / scales[owners]
    if update_rms:
        rms.update(raw_groups.detach())
    return raw_groups, normalized_groups, normalized_atomic


class CenteredTanhComposer(nn.Module):
    """Map a detached predictive fused latent to five bounded mean-one weights."""

    def __init__(self, latent_dim: int, num_groups: int = 5, half_range: float = 0.4) -> None:
        super().__init__()
        if latent_dim <= 0 or num_groups <= 1 or not 0.0 < half_range < 1.0:
            raise ValueError("Invalid composer configuration.")
        self.latent_dim = int(latent_dim)
        self.num_groups = int(num_groups)
        self.half_range = float(half_range)
        self.network = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, self.num_groups),
        )
        output = self.network[-1]
        assert isinstance(output, nn.Linear)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def logits(self, fused_latent: torch.Tensor) -> torch.Tensor:
        if fused_latent.shape[-1] != self.latent_dim:
            raise ValueError(f"Expected latent dimension {self.latent_dim}.")
        return self.network(fused_latent.detach())

    def weights_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[-1] != self.num_groups:
            raise ValueError(f"Expected {self.num_groups} composer logits.")
        centered = torch.tanh(logits)
        centered = centered - centered.mean(dim=-1, keepdim=True)
        # Centering can double the tanh range. Rescale only when necessary so
        # every sample remains inside the promised exact bounds.
        divisor = centered.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
        return 1.0 + self.half_range * centered / divisor

    def forward(self, fused_latent: torch.Tensor) -> torch.Tensor:
        return self.weights_from_logits(self.logits(fused_latent))


class OuterCritic(nn.Module):
    """Training-only value function for the fixed outer reward."""

    def __init__(self, observation_dim: int) -> None:
        super().__init__()
        if observation_dim <= 0:
            raise ValueError("observation_dim must be positive.")
        self.observation_dim = int(observation_dim)
        self.value = nn.Sequential(
            nn.Linear(self.observation_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.value(observations).squeeze(-1)


class LIRPGOuterCritic(nn.Module):
    """Exact two-layer tanh ``v_ex`` architecture from the PPO-LIRPG source."""

    def __init__(self, observation_dim: int) -> None:
        super().__init__()
        if observation_dim <= 0:
            raise ValueError("observation_dim must be positive.")
        self.observation_dim = int(observation_dim)
        self.value = nn.Sequential(
            nn.Linear(self.observation_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.value(observations).squeeze(-1)


class LIRPGIntrinsicReward(nn.Module):
    """The observation-action scalar reward network used by PPO+LIRPG.

    This is a PyTorch port of the official MuJoCo implementation: two 64-unit
    tanh layers followed by a bounded scalar tanh output.  It deliberately
    receives the existing policy observation and action rather than adding any
    new environment feature.
    """

    def __init__(self, observation_dim: int, action_dim: int) -> None:
        super().__init__()
        if observation_dim <= 0 or action_dim <= 0:
            raise ValueError("LIRPG observation and action dimensions must be positive.")
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.reward = nn.Sequential(
            nn.Linear(self.observation_dim + self.action_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if observations.shape[:-1] != actions.shape[:-1]:
            raise ValueError("LIRPG observations and actions must share leading dimensions.")
        if observations.shape[-1] != self.observation_dim:
            raise ValueError(f"Expected {self.observation_dim} LIRPG observation values.")
        if actions.shape[-1] != self.action_dim:
            raise ValueError(f"Expected {self.action_dim} LIRPG action values.")
        return self.reward(torch.cat((observations.detach(), actions.detach()), dim=-1)).squeeze(-1)


def validate_static_group_weights(
    weights: Sequence[float],
    *,
    num_groups: int = len(REWARD_GROUP_NAMES),
    lower: float = 0.6,
    upper: float = 1.4,
    require_mean_one: bool = True,
) -> tuple[float, ...]:
    """Validate one state-independent five-group reward vector."""

    result = tuple(float(value) for value in weights)
    if len(result) != int(num_groups):
        raise ValueError(f"Expected {num_groups} static group weights, got {len(result)}.")
    if not all(torch.isfinite(torch.tensor(value)).item() for value in result):
        raise ValueError("Static group weights must be finite.")
    if any(value < float(lower) or value > float(upper) for value in result):
        raise ValueError(f"Static group weights must lie in [{lower}, {upper}].")
    if require_mean_one and abs(sum(result) / len(result) - 1.0) > 1.0e-6:
        raise ValueError("Static group weights must have mean one.")
    return result


def static_group_weight_tensor(
    weights: Sequence[float],
    reference: torch.Tensor,
) -> torch.Tensor:
    """Broadcast validated static weights over a rollout/environment batch."""

    values = validate_static_group_weights(weights)
    tensor = torch.as_tensor(values, device=reference.device, dtype=reference.dtype)
    return tensor.expand(*reference.shape[:-1], len(values))


def outer_reward(
    linear_tracking: torch.Tensor,
    yaw_tracking: torch.Tensor,
    terminated: torch.Tensor,
    actions: torch.Tensor,
    *,
    termination_penalty: float = 5.0,
    action_cost: float = 0.01,
) -> torch.Tensor:
    """Fixed simple objective: tracking + survival - failure - action cost."""

    terminated = terminated.to(linear_tracking)
    tracking = 0.5 * (linear_tracking + yaw_tracking)
    survival = 1.0 - terminated
    return (
        tracking
        + survival
        - float(termination_penalty) * terminated
        - float(action_cost) * torch.mean(torch.square(actions), dim=-1)
    )


def lirpg_actor_reward(
    extrinsic_reward: torch.Tensor,
    intrinsic_reward: torch.Tensor,
    *,
    extrinsic_coefficient: float = 0.01,
    intrinsic_coefficient: float = 1.0,
) -> torch.Tensor:
    """Compose the policy reward exactly as PPO-LIRPG does for MuJoCo."""

    if extrinsic_reward.shape != intrinsic_reward.shape:
        raise ValueError("Extrinsic and intrinsic LIRPG rewards must share a shape.")
    if extrinsic_coefficient < 0.0 or intrinsic_coefficient < 0.0:
        raise ValueError("LIRPG reward coefficients must be non-negative.")
    return (
        float(extrinsic_coefficient) * extrinsic_reward
        + float(intrinsic_coefficient) * intrinsic_reward
    )


def differentiable_mixed_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    final_value: torch.Tensor,
    dones: torch.Tensor,
    *,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """Reproduce the official PPO-LIRPG ``TD_MIX + COEF_MAT`` computation."""

    if rewards.ndim != 2 or values.shape != rewards.shape or dones.shape != rewards.shape:
        raise ValueError("LIRPG rewards, values and dones must share [T,N].")
    if final_value.shape != rewards.shape[1:]:
        raise ValueError("The final mixed value must have shape [N].")
    future = torch.zeros_like(final_value)
    reversed_advantages: list[torch.Tensor] = []
    for step in range(rewards.shape[0] - 1, -1, -1):
        not_done = 1.0 - dones[step].to(rewards)
        next_value = final_value if step == rewards.shape[0] - 1 else values[step + 1]
        delta = rewards[step] + float(gamma) * not_done * next_value - values[step]
        future = delta + float(gamma) * float(lam) * not_done * future
        reversed_advantages.append(future)
    return torch.stack(list(reversed(reversed_advantages)), dim=0)


def beta_schedule(
    iteration: int,
    total_iterations: int,
    *,
    maximum: float = 1.0,
    warmup_fraction: float = 0.2,
) -> float:
    """Warm from zero to full Composer control, then hold permanently."""

    if total_iterations <= 0 or iteration < 0 or not 0.0 <= maximum <= 1.0:
        raise ValueError("Invalid beta schedule arguments.")
    if not 0.0 < warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must lie in (0, 1).")
    progress = min(float(iteration) / float(max(total_iterations - 1, 1)), 1.0)
    if progress < warmup_fraction:
        return float(maximum) * progress / warmup_fraction
    return float(maximum)


def resolve_beta_schedule_horizon(
    saved_total_iterations: int,
    completed_iterations: int,
    additional_iterations: int,
) -> int:
    """Resolve the absolute beta horizon for a resumed training run.

    A short continuation inside the checkpoint's original horizon must retain
    that horizon.  If the requested continuation extends past it, however, the
    schedule must extend as well; otherwise a continuation would jump to full
    Composer deployment rather than preserving its original warmup progress.
    """

    if saved_total_iterations <= 0 or completed_iterations < 0 or additional_iterations <= 0:
        raise ValueError("Invalid beta resume horizon arguments.")
    requested_endpoint = int(completed_iterations) + int(additional_iterations)
    return max(int(saved_total_iterations), requested_endpoint)


def effective_composer_weights(
    group_weights: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Interpolate from uniform atomic-reward weights to Composer control."""

    if not 0.0 <= float(beta) <= 1.0:
        raise ValueError("Composer deployment beta must lie in [0, 1].")
    return 1.0 + float(beta) * (group_weights - 1.0)


def compose_actor_reward(
    fixed_internal_reward: torch.Tensor,
    normalized_group_rewards: torch.Tensor,
    group_weights: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Compose PPO reward from fixed safety terms and real normalized atomics."""

    if normalized_group_rewards.shape != group_weights.shape:
        raise ValueError("Group reward and weight shapes must match.")
    if fixed_internal_reward.shape != normalized_group_rewards.shape[:-1]:
        raise ValueError("Fixed internal reward must contain one value per sample.")
    weights = effective_composer_weights(group_weights, beta)
    return fixed_internal_reward + torch.mean(
        normalized_group_rewards * weights, dim=-1
    )


def select_actor_reward(
    fixed_outer_reward: torch.Tensor,
    fixed_internal_reward: torch.Tensor,
    normalized_group_rewards: torch.Tensor,
    group_weights: torch.Tensor,
    beta: float,
    *,
    outer_only: bool,
) -> torch.Tensor:
    """Select actor reward while preserving the original outer-only ablation."""

    if outer_only:
        return fixed_outer_reward
    return compose_actor_reward(
        fixed_internal_reward,
        normalized_group_rewards,
        group_weights,
        beta,
    )


def generalized_advantage(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    dones: torch.Tensor,
    *,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute time-major GAE and value targets without modifying inputs."""

    if rewards.shape != values.shape or rewards.shape != dones.shape or rewards.ndim != 2:
        raise ValueError("rewards, values and dones must have the same [T, N] shape.")
    if next_value.shape != rewards.shape[1:]:
        raise ValueError("next_value must contain one value per environment.")
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(next_value)
    following_value = next_value
    for step in range(rewards.shape[0] - 1, -1, -1):
        not_done = 1.0 - dones[step].to(rewards)
        delta = rewards[step] + float(gamma) * following_value * not_done - values[step]
        gae = delta + float(gamma) * float(lam) * not_done * gae
        advantages[step] = gae
        following_value = values[step]
    return advantages, advantages + values


def normalized_group_advantages(
    group_rewards: torch.Tensor,
    dones: torch.Tensor,
    *,
    gamma: float,
    lam: float,
    epsilon: float = 1.0e-6,
) -> torch.Tensor:
    """Zero-baseline GAE for each shaping group, standardized per group."""

    if group_rewards.ndim != 3 or group_rewards.shape[:2] != dones.shape:
        raise ValueError("Expected group_rewards [T,N,G] and dones [T,N].")
    advantages = torch.zeros_like(group_rewards)
    gae = torch.zeros_like(group_rewards[0])
    for step in range(group_rewards.shape[0] - 1, -1, -1):
        not_done = (1.0 - dones[step].to(group_rewards)).unsqueeze(-1)
        gae = group_rewards[step] + float(gamma) * float(lam) * not_done * gae
        advantages[step] = gae
    reduce_dims = (0, 1)
    mean = advantages.mean(dim=reduce_dims, keepdim=True)
    std = advantages.std(dim=reduce_dims, unbiased=False, keepdim=True).clamp_min(epsilon)
    return (advantages - mean) / std


def cross_split_credit_reliability(
    group_advantages: torch.Tensor,
    outer_advantages: torch.Tensor,
    *,
    minimum_signal: float = 0.05,
    epsilon: float = 1.0e-6,
) -> dict[str, torch.Tensor]:
    """Estimate whether outer credit agrees across independent environment halves.

    A composer update is useful only if the five-dimensional direction linking
    shaping groups to the fixed outer objective is reproducible.  Even/odd
    environment IDs form two disjoint estimators without shortening the
    rollout.  Reliability is high only when their correlation vectors point in
    the same direction *and* both contain a non-trivial signal.
    """

    if group_advantages.ndim != 3 or group_advantages.shape[:-1] != outer_advantages.shape:
        raise ValueError("Expected group advantages [T,N,G] and outer advantages [T,N].")
    if group_advantages.shape[1] < 2:
        raise ValueError("Cross-split credit reliability requires at least two environments.")
    if minimum_signal <= 0.0:
        raise ValueError("minimum_signal must be positive.")

    def correlation_vector(environment_slice: slice) -> torch.Tensor:
        groups = group_advantages[:, environment_slice].reshape(-1, group_advantages.shape[-1])
        outer = outer_advantages[:, environment_slice].reshape(-1)
        groups = (groups - groups.mean(dim=0, keepdim=True)) / groups.std(
            dim=0, unbiased=False, keepdim=True
        ).clamp_min(epsilon)
        outer = (outer - outer.mean()) / outer.std(unbiased=False).clamp_min(epsilon)
        return torch.mean(groups * outer.unsqueeze(-1), dim=0)

    first = correlation_vector(slice(0, None, 2))
    second = correlation_vector(slice(1, None, 2))
    direction_cosine = F.cosine_similarity(first, second, dim=0, eps=epsilon)
    group_count_scale = float(group_advantages.shape[-1]) ** 0.5
    signal = torch.minimum(torch.linalg.vector_norm(first), torch.linalg.vector_norm(second))
    signal = signal / group_count_scale
    reliability = direction_cosine.clamp(0.0, 1.0) * (signal / float(minimum_signal)).clamp(0.0, 1.0)
    return {
        "reliability": reliability.detach(),
        "direction_cosine": direction_cosine.detach(),
        "signal": signal.detach(),
        "first_direction": first.detach(),
        "second_direction": second.detach(),
    }


def differentiable_shaping_advantage(
    fixed_internal_rewards: torch.Tensor,
    normalized_group_rewards: torch.Tensor,
    group_weights: torch.Tensor,
    dones: torch.Tensor,
    *,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """Compute zero-baseline shaping GAE with the weight from each actual step.

    Weighting per-group GAEs afterwards is incorrect for a state-dependent
    composer because it applies ``w_t`` to rewards produced under future
    ``w_{t+1:}``.  Compose immediate rewards first, then propagate them through
    time while retaining gradients to every corresponding weight.
    """

    if normalized_group_rewards.shape != group_weights.shape:
        raise ValueError("Group rewards and weights must have identical [T,N,G] shapes.")
    if normalized_group_rewards.ndim != 3 or normalized_group_rewards.shape[:2] != dones.shape:
        raise ValueError("Expected group rewards [T,N,G] and dones [T,N].")
    if fixed_internal_rewards.shape != dones.shape:
        raise ValueError("Fixed internal rewards and dones must share [T,N].")
    shaping_rewards = fixed_internal_rewards + torch.mean(
        normalized_group_rewards * group_weights, dim=-1
    )
    future = torch.zeros_like(shaping_rewards[0])
    reversed_advantages: list[torch.Tensor] = []
    for step in range(shaping_rewards.shape[0] - 1, -1, -1):
        not_done = 1.0 - dones[step].to(shaping_rewards)
        future = shaping_rewards[step] + float(gamma) * float(lam) * not_done * future
        reversed_advantages.append(future)
    return torch.stack(list(reversed(reversed_advantages)), dim=0)


def composer_alignment_loss(
    composer: CenteredTanhComposer,
    fused_latents: torch.Tensor,
    normalized_group_rewards: torch.Tensor,
    outer_advantages: torch.Tensor,
    dones: torch.Tensor,
    *,
    gamma: float = 0.995,
    lam: float = 0.95,
    huber_weight: float = 1.0,
    weight_to_one: float = 1.0e-3,
    temporal_smoothness: float = 1.0e-2,
    credit_weight: float | torch.Tensor = 1.0,
    epsilon: float = 1.0e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Align shaping advantage with stop-gradient outer advantage."""

    if fused_latents.shape[:-1] != outer_advantages.shape:
        raise ValueError("Latents and outer advantages must share [T,N].")
    if normalized_group_rewards.shape[:-1] != outer_advantages.shape:
        raise ValueError("Group and outer advantages must share [T,N].")
    weights = composer(fused_latents.detach())
    shaping = differentiable_shaping_advantage(
        torch.zeros_like(outer_advantages),
        normalized_group_rewards.detach(),
        weights,
        dones,
        gamma=float(gamma),
        lam=float(lam),
    )
    outer = outer_advantages.detach()
    shaping_normalized = (shaping - shaping.mean()) / shaping.std(unbiased=False).clamp_min(epsilon)
    outer_normalized = (outer - outer.mean()) / outer.std(unbiased=False).clamp_min(epsilon)
    flat_shaping = shaping_normalized.reshape(-1)
    flat_outer = outer_normalized.reshape(-1)
    cosine = torch.sum(flat_shaping * flat_outer) / (
        torch.linalg.vector_norm(flat_shaping).clamp_min(epsilon)
        * torch.linalg.vector_norm(flat_outer).clamp_min(epsilon)
    )
    cosine_loss = 1.0 - cosine
    huber = F.smooth_l1_loss(shaping_normalized, outer_normalized)
    one_loss = torch.square(weights - 1.0).mean()
    if weights.shape[0] > 1:
        continuity = (1.0 - dones[:-1].to(weights)).unsqueeze(-1)
        smooth = (torch.square(weights[1:] - weights[:-1]) * continuity).sum() / (
            continuity.sum().clamp_min(1.0) * weights.shape[-1]
        )
    else:
        smooth = torch.zeros((), device=weights.device, dtype=weights.dtype)
    credit_weight_tensor = torch.as_tensor(
        credit_weight, device=weights.device, dtype=weights.dtype
    ).clamp(0.0, 1.0)
    total = (
        credit_weight_tensor * (cosine_loss + float(huber_weight) * huber)
        + float(weight_to_one) * one_loss
        + float(temporal_smoothness) * smooth
    )
    return total, {
        "cosine": cosine.detach(),
        "cosine_loss": cosine_loss.detach(),
        "huber_loss": huber.detach(),
        "weight_to_one_loss": one_loss.detach(),
        "temporal_smoothness_loss": smooth.detach(),
        "credit_weight": credit_weight_tensor.detach(),
        "weights": weights.detach(),
        "shaping_advantage": shaping.detach(),
    }


def _squashed_gaussian_log_probability(
    raw_mean: torch.Tensor,
    actions: torch.Tensor,
    standard_deviation: torch.Tensor,
    *,
    maximum_latent_mean: float,
) -> torch.Tensor:
    """Evaluate stored tanh actions under an actor mean without touching policy caches."""

    if maximum_latent_mean <= 0.0:
        raise ValueError("maximum_latent_mean must be positive.")
    if raw_mean.shape != actions.shape or standard_deviation.shape != actions.shape:
        raise ValueError("Actor mean, actions and standard deviation must share [B,A].")
    mean = float(maximum_latent_mean) * torch.tanh(
        raw_mean / float(maximum_latent_mean)
    )
    clipped_actions = actions.clamp(min=-1.0 + 1.0e-6, max=1.0 - 1.0e-6)
    latent_actions = torch.atanh(clipped_actions)
    distribution = torch.distributions.Normal(
        mean, standard_deviation.clamp_min(1.0e-6)
    )
    log_det_jacobian = torch.log(
        1.0 - torch.square(clipped_actions) + 1.0e-6
    )
    return (distribution.log_prob(latent_actions) - log_det_jacobian).sum(dim=-1)


def _ppo_surrogate_loss(
    new_log_probability: torch.Tensor,
    old_log_probability: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_param: float,
) -> torch.Tensor:
    """Standard clipped PPO policy loss for a flat transition batch."""

    if clip_param <= 0.0:
        raise ValueError("clip_param must be positive.")
    if (
        new_log_probability.shape != old_log_probability.shape
        or new_log_probability.shape != advantages.shape
    ):
        raise ValueError("PPO log probabilities and advantages must share [B].")
    ratio = torch.exp(new_log_probability - old_log_probability)
    unclipped = -advantages * ratio
    clipped = -advantages * ratio.clamp(
        1.0 - float(clip_param), 1.0 + float(clip_param)
    )
    return torch.maximum(unclipped, clipped).mean()


def _virtual_adam_parameters(
    actor: nn.Module,
    gradients: Sequence[torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> dict[str, torch.Tensor]:
    """Apply one differentiable Adam step without mutating actor or optimizer."""

    named_parameters = list(actor.named_parameters())
    if len(named_parameters) != len(gradients):
        raise ValueError("Actor parameter and gradient counts differ.")
    virtual: dict[str, torch.Tensor] = {}
    for (name, parameter), gradient in zip(named_parameters, gradients, strict=True):
        virtual[name] = _virtual_adam_parameter(parameter, gradient, optimizer)
    return virtual


def _virtual_adam_parameter(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> torch.Tensor:
    """Apply the exact first-order Adam expression used by PPO-LIRPG."""

    parameter_group = None
    for group in optimizer.param_groups:
        if any(candidate is parameter for candidate in group["params"]):
            parameter_group = group
            break
    if parameter_group is None:
        raise ValueError("Virtual-Adam parameter is absent from the PPO optimizer.")
    group = parameter_group
    betas = group.get("betas", (0.9, 0.999))
    beta1, beta2 = float(betas[0]), float(betas[1])
    epsilon = float(group.get("eps", 1.0e-8))
    learning_rate = float(group["lr"])
    weight_decay = float(group.get("weight_decay", 0.0))
    maximize = bool(group.get("maximize", False))
    if maximize:
        gradient = -gradient
    if weight_decay != 0.0:
        gradient = gradient + weight_decay * parameter

    state = optimizer.state.get(parameter, {})
    exp_avg = state.get("exp_avg")
    exp_avg_sq = state.get("exp_avg_sq")
    if exp_avg is None:
        exp_avg = torch.zeros_like(parameter)
    else:
        exp_avg = exp_avg.detach()
    if exp_avg_sq is None:
        exp_avg_sq = torch.zeros_like(parameter)
    else:
        exp_avg_sq = exp_avg_sq.detach()
    raw_step = state.get("step", 0)
    if isinstance(raw_step, torch.Tensor):
        step = int(raw_step.detach().item())
    else:
        step = int(raw_step)
    next_step = step + 1

    next_exp_avg = beta1 * exp_avg + (1.0 - beta1) * gradient
    # The official graph explicitly stop-gradients the squared gradient in v,
    # while retaining the reward gradient through the first moment m.
    next_exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * torch.square(
        gradient.detach()
    )
    bias_correction1 = 1.0 - beta1**next_step
    bias_correction2 = 1.0 - beta2**next_step
    corrected_learning_rate = (
        learning_rate * bias_correction2**0.5 / bias_correction1
    )
    return parameter - corrected_learning_rate * next_exp_avg / (
        torch.sqrt(next_exp_avg_sq) + epsilon
    )


def composer_meta_gradient_loss(
    composer: CenteredTanhComposer,
    actor: nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    fused_latents: torch.Tensor,
    fixed_internal_rewards: torch.Tensor,
    normalized_group_rewards: torch.Tensor,
    outer_advantages: torch.Tensor,
    actor_observations: torch.Tensor,
    actions: torch.Tensor,
    old_action_log_probabilities: torch.Tensor,
    action_standard_deviations: torch.Tensor,
    dones: torch.Tensor,
    *,
    gamma: float = 0.995,
    lam: float = 0.95,
    clip_param: float = 0.1,
    maximum_latent_mean: float = 4.0,
    weight_to_one: float = 1.0e-3,
    temporal_smoothness: float = 1.0e-2,
    epsilon: float = 1.0e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Differentiate through one virtual PPO update into the reward composer.

    Even environment IDs form the virtual inner-PPO batch. Odd IDs are held
    out and evaluate that virtual actor update using only stop-gradient outer
    advantage. Consequently the composer is rewarded for changing the *actor
    update* in a direction that improves the fixed objective, rather than for
    making shaping returns correlate with outer returns on the same samples.
    The real actor and its Adam state are never mutated.
    """

    if fused_latents.ndim != 3 or fused_latents.shape[:2] != outer_advantages.shape:
        raise ValueError("Expected fused latents [T,N,D] and outer advantages [T,N].")
    if normalized_group_rewards.shape != fused_latents.shape[:2] + (
        composer.num_groups,
    ):
        raise ValueError("Expected normalized group rewards [T,N,G].")
    if fixed_internal_rewards.shape != outer_advantages.shape:
        raise ValueError("Expected fixed internal rewards [T,N].")
    if actor_observations.shape[:2] != outer_advantages.shape:
        raise ValueError("Actor observations must share rollout [T,N] dimensions.")
    if actions.shape[:2] != outer_advantages.shape:
        raise ValueError("Actions must share rollout [T,N] dimensions.")
    if action_standard_deviations.shape != actions.shape:
        raise ValueError("Stored action standard deviations must match actions.")
    old_log_probability = old_action_log_probabilities.squeeze(-1)
    if old_log_probability.shape != outer_advantages.shape:
        raise ValueError("Old action log probabilities must have shape [T,N] or [T,N,1].")
    if dones.shape != outer_advantages.shape:
        raise ValueError("Dones and outer advantages must share [T,N].")
    if outer_advantages.shape[1] < 2:
        raise ValueError("Meta-gradient requires at least two environments.")

    weights = composer(fused_latents.detach())
    shaping_advantages = differentiable_shaping_advantage(
        fixed_internal_rewards.detach(),
        normalized_group_rewards.detach(),
        weights,
        dones.detach(),
        gamma=float(gamma),
        lam=float(lam),
    )
    outer = outer_advantages.detach()
    # Meta-training always evaluates full Composer control from the real
    # environment reward components. Deployment beta is deliberately absent:
    # it only controls how quickly the real PPO adopts already-trainable
    # Composer weights.
    combined_advantages = (
        shaping_advantages - shaping_advantages.mean()
    ) / shaping_advantages.std(unbiased=False).clamp_min(epsilon)
    normalized_outer = (outer - outer.mean()) / outer.std(
        unbiased=False
    ).clamp_min(epsilon)

    train_slice = slice(0, None, 2)
    validation_slice = slice(1, None, 2)

    def flatten_environment_slice(tensor: torch.Tensor, selection: slice) -> torch.Tensor:
        return tensor[:, selection].reshape(
            -1, *tensor.shape[2:]
        )

    train_observations = flatten_environment_slice(
        actor_observations.detach(), train_slice
    )
    train_actions = flatten_environment_slice(actions.detach(), train_slice)
    train_standard_deviations = flatten_environment_slice(
        action_standard_deviations.detach(), train_slice
    )
    train_old_log_probability = old_log_probability[:, train_slice].reshape(-1).detach()
    train_advantages = combined_advantages[:, train_slice].reshape(-1)

    actor_parameters = tuple(actor.parameters())
    raw_train_mean = actor(train_observations)
    train_log_probability = _squashed_gaussian_log_probability(
        raw_train_mean,
        train_actions,
        train_standard_deviations,
        maximum_latent_mean=float(maximum_latent_mean),
    )
    inner_policy_loss = _ppo_surrogate_loss(
        train_log_probability,
        train_old_log_probability,
        train_advantages,
        clip_param=float(clip_param),
    )
    inner_gradients = torch.autograd.grad(
        inner_policy_loss,
        actor_parameters,
        create_graph=True,
        retain_graph=True,
    )
    virtual_parameters = _virtual_adam_parameters(
        actor, inner_gradients, actor_optimizer
    )

    validation_observations = flatten_environment_slice(
        actor_observations.detach(), validation_slice
    )
    validation_actions = flatten_environment_slice(actions.detach(), validation_slice)
    validation_standard_deviations = flatten_environment_slice(
        action_standard_deviations.detach(), validation_slice
    )
    validation_old_log_probability = old_log_probability[:, validation_slice].reshape(
        -1
    ).detach()
    validation_outer_advantages = normalized_outer[:, validation_slice].reshape(-1)

    raw_validation_mean_before = actor(validation_observations)
    validation_log_probability_before = _squashed_gaussian_log_probability(
        raw_validation_mean_before,
        validation_actions,
        validation_standard_deviations,
        maximum_latent_mean=float(maximum_latent_mean),
    )
    outer_policy_loss_before = _ppo_surrogate_loss(
        validation_log_probability_before,
        validation_old_log_probability,
        validation_outer_advantages,
        clip_param=float(clip_param),
    )
    raw_validation_mean_after = functional_call(
        actor, virtual_parameters, (validation_observations,)
    )
    validation_log_probability_after = _squashed_gaussian_log_probability(
        raw_validation_mean_after,
        validation_actions,
        validation_standard_deviations,
        maximum_latent_mean=float(maximum_latent_mean),
    )
    outer_policy_loss_after = _ppo_surrogate_loss(
        validation_log_probability_after,
        validation_old_log_probability,
        validation_outer_advantages,
        clip_param=float(clip_param),
    )
    meta_policy_loss = outer_policy_loss_after - outer_policy_loss_before.detach()

    one_loss = torch.square(weights - 1.0).mean()
    if weights.shape[0] > 1:
        continuity = (1.0 - dones[:-1].to(weights)).unsqueeze(-1)
        smooth = (torch.square(weights[1:] - weights[:-1]) * continuity).sum() / (
            continuity.sum().clamp_min(1.0) * weights.shape[-1]
        )
    else:
        smooth = torch.zeros((), device=weights.device, dtype=weights.dtype)
    total = (
        meta_policy_loss
        + float(weight_to_one) * one_loss
        + float(temporal_smoothness) * smooth
    )

    with torch.no_grad():
        shaping_normalized = (
            shaping_advantages - shaping_advantages.mean()
        ) / shaping_advantages.std(unbiased=False).clamp_min(epsilon)
        advantage_cosine = F.cosine_similarity(
            shaping_normalized.reshape(-1),
            normalized_outer.reshape(-1),
            dim=0,
            eps=epsilon,
        )
    return total, {
        "meta_policy_loss": meta_policy_loss.detach(),
        "meta_outer_loss_before": outer_policy_loss_before.detach(),
        "meta_outer_loss_after": outer_policy_loss_after.detach(),
        "predicted_outer_improvement": (
            outer_policy_loss_before - outer_policy_loss_after
        ).detach(),
        "inner_policy_loss": inner_policy_loss.detach(),
        "advantage_cosine": advantage_cosine.detach(),
        "weight_to_one_loss": one_loss.detach(),
        "temporal_smoothness_loss": smooth.detach(),
        "weights": weights.detach(),
        "shaping_advantage": shaping_advantages.detach(),
    }


def lirpg_meta_gradient_loss(
    reward_model: LIRPGIntrinsicReward,
    actor: nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    fixed_internal_rewards: torch.Tensor,
    outer_advantages: torch.Tensor,
    actor_observations: torch.Tensor,
    actions: torch.Tensor,
    old_action_log_probabilities: torch.Tensor,
    action_standard_deviations: torch.Tensor,
    dones: torch.Tensor,
    *,
    gamma: float = 0.995,
    lam: float = 0.95,
    clip_param: float = 0.1,
    maximum_latent_mean: float = 4.0,
    intrinsic_coefficient: float = 1.0,
    reward_l2: float = 1.0e-4,
    temporal_smoothness: float = 1.0e-3,
    epsilon: float = 1.0e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Differentiate the outer objective through one PPO+LIRPG update.

    Unlike the proposed Composer's cross-environment holdout, this intentionally
    uses the full rollout for both the virtual mixed-reward update and the
    extrinsic evaluation, matching the official PPO+LIRPG algorithm.
    """

    rollout_shape = outer_advantages.shape
    if outer_advantages.ndim != 2 or rollout_shape[1] < 2:
        raise ValueError("LIRPG requires outer advantages [T,N] with N >= 2.")
    if fixed_internal_rewards.shape != rollout_shape or dones.shape != rollout_shape:
        raise ValueError("LIRPG rewards, dones and outer advantages must share [T,N].")
    if actor_observations.shape[:2] != rollout_shape or actions.shape[:2] != rollout_shape:
        raise ValueError("LIRPG observations/actions must share rollout dimensions.")
    if action_standard_deviations.shape != actions.shape:
        raise ValueError("Stored action standard deviations must match actions.")
    old_log_probability = old_action_log_probabilities.squeeze(-1)
    if old_log_probability.shape != rollout_shape:
        raise ValueError("Old action log probabilities must have shape [T,N] or [T,N,1].")

    intrinsic_rewards = reward_model(actor_observations.detach(), actions.detach())
    actor_rewards = lirpg_actor_reward(
        fixed_internal_rewards.detach(),
        intrinsic_rewards,
        intrinsic_coefficient=float(intrinsic_coefficient),
    )
    future = torch.zeros_like(actor_rewards[0])
    reversed_advantages: list[torch.Tensor] = []
    for step in range(actor_rewards.shape[0] - 1, -1, -1):
        not_done = 1.0 - dones[step].to(actor_rewards)
        future = actor_rewards[step] + float(gamma) * float(lam) * not_done * future
        reversed_advantages.append(future)
    intrinsic_advantages = torch.stack(
        list(reversed(reversed_advantages)), dim=0
    )
    normalized_inner = (
        intrinsic_advantages - intrinsic_advantages.mean()
    ) / intrinsic_advantages.std(unbiased=False).clamp_min(epsilon)
    outer = outer_advantages.detach()
    normalized_outer = (outer - outer.mean()) / outer.std(
        unbiased=False
    ).clamp_min(epsilon)

    train_slice = slice(None)
    validation_slice = slice(None)

    def flatten_environment_slice(tensor: torch.Tensor, selection: slice) -> torch.Tensor:
        return tensor[:, selection].reshape(-1, *tensor.shape[2:])

    train_observations = flatten_environment_slice(
        actor_observations.detach(), train_slice
    )
    train_actions = flatten_environment_slice(actions.detach(), train_slice)
    train_standard_deviations = flatten_environment_slice(
        action_standard_deviations.detach(), train_slice
    )
    train_old_log_probability = old_log_probability[:, train_slice].reshape(-1).detach()
    train_advantages = normalized_inner[:, train_slice].reshape(-1)
    actor_parameters = tuple(actor.parameters())
    train_log_probability = _squashed_gaussian_log_probability(
        actor(train_observations),
        train_actions,
        train_standard_deviations,
        maximum_latent_mean=float(maximum_latent_mean),
    )
    inner_policy_loss = _ppo_surrogate_loss(
        train_log_probability,
        train_old_log_probability,
        train_advantages,
        clip_param=float(clip_param),
    )
    inner_gradients = torch.autograd.grad(
        inner_policy_loss,
        actor_parameters,
        create_graph=True,
        retain_graph=True,
    )
    virtual_parameters = _virtual_adam_parameters(
        actor, inner_gradients, actor_optimizer
    )

    validation_observations = flatten_environment_slice(
        actor_observations.detach(), validation_slice
    )
    validation_actions = flatten_environment_slice(actions.detach(), validation_slice)
    validation_standard_deviations = flatten_environment_slice(
        action_standard_deviations.detach(), validation_slice
    )
    validation_old_log_probability = old_log_probability[:, validation_slice].reshape(
        -1
    ).detach()
    validation_outer_advantages = normalized_outer[:, validation_slice].reshape(-1)
    validation_log_probability_before = _squashed_gaussian_log_probability(
        actor(validation_observations),
        validation_actions,
        validation_standard_deviations,
        maximum_latent_mean=float(maximum_latent_mean),
    )
    outer_policy_loss_before = _ppo_surrogate_loss(
        validation_log_probability_before,
        validation_old_log_probability,
        validation_outer_advantages,
        clip_param=float(clip_param),
    )
    validation_log_probability_after = _squashed_gaussian_log_probability(
        functional_call(actor, virtual_parameters, (validation_observations,)),
        validation_actions,
        validation_standard_deviations,
        maximum_latent_mean=float(maximum_latent_mean),
    )
    outer_policy_loss_after = _ppo_surrogate_loss(
        validation_log_probability_after,
        validation_old_log_probability,
        validation_outer_advantages,
        clip_param=float(clip_param),
    )
    meta_policy_loss = outer_policy_loss_after - outer_policy_loss_before.detach()

    magnitude_loss = torch.square(intrinsic_rewards).mean()
    if intrinsic_rewards.shape[0] > 1:
        continuity = 1.0 - dones[:-1].to(intrinsic_rewards)
        smoothness_loss = (
            torch.square(intrinsic_rewards[1:] - intrinsic_rewards[:-1]) * continuity
        ).sum() / continuity.sum().clamp_min(1.0)
    else:
        smoothness_loss = torch.zeros_like(meta_policy_loss)
    total = (
        meta_policy_loss
        + float(reward_l2) * magnitude_loss
        + float(temporal_smoothness) * smoothness_loss
    )
    with torch.no_grad():
        advantage_cosine = F.cosine_similarity(
            normalized_inner.reshape(-1),
            normalized_outer.reshape(-1),
            dim=0,
            eps=epsilon,
        )
    return total, {
        "meta_policy_loss": meta_policy_loss.detach(),
        "meta_outer_loss_before": outer_policy_loss_before.detach(),
        "meta_outer_loss_after": outer_policy_loss_after.detach(),
        "predicted_outer_improvement": (
            outer_policy_loss_before - outer_policy_loss_after
        ).detach(),
        "inner_policy_loss": inner_policy_loss.detach(),
        "intrinsic_reward_mean": intrinsic_rewards.mean().detach(),
        "intrinsic_reward_std": intrinsic_rewards.std(unbiased=False).detach(),
        "intrinsic_reward_abs": intrinsic_rewards.abs().mean().detach(),
        "intrinsic_advantage_mean": intrinsic_advantages.mean().detach(),
        "advantage_cosine": advantage_cosine.detach(),
        "reward_l2_loss": magnitude_loss.detach(),
        "temporal_smoothness_loss": smoothness_loss.detach(),
    }


def official_lirpg_meta_gradient_loss(
    reward_model: LIRPGIntrinsicReward,
    actor: nn.Module,
    actor_log_std: nn.Parameter,
    actor_optimizer: torch.optim.Optimizer,
    extrinsic_rewards: torch.Tensor,
    mixed_values: torch.Tensor,
    final_mixed_value: torch.Tensor,
    outer_advantages: torch.Tensor,
    actor_observations: torch.Tensor,
    actions: torch.Tensor,
    old_action_log_probabilities: torch.Tensor,
    dones: torch.Tensor,
    batch_indices: torch.Tensor,
    *,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_param: float = 0.2,
    maximum_latent_mean: float = 4.0,
    extrinsic_coefficient: float = 0.01,
    intrinsic_coefficient: float = 1.0,
    epsilon: float = 1.0e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """PyTorch tensor adapter for the official ``ppo2.Model`` meta update.

    The official TensorFlow graph recomputes intrinsic rewards for the complete
    rollout on every PPO minibatch, forms mixed GAE from the frozen mixed value
    baseline, takes one virtual Adam actor step, and differentiates the clipped
    extrinsic PPO objective through that step into the reward network.
    """

    rollout_shape = extrinsic_rewards.shape
    if extrinsic_rewards.ndim != 2:
        raise ValueError("Official LIRPG expects rollout rewards [T,N].")
    for name, tensor in (
        ("mixed_values", mixed_values),
        ("outer_advantages", outer_advantages),
        ("dones", dones),
    ):
        if tensor.shape != rollout_shape:
            raise ValueError(f"{name} must match the rollout shape.")
    if final_mixed_value.shape != rollout_shape[1:]:
        raise ValueError("final_mixed_value must have shape [N].")
    if actor_observations.shape[:2] != rollout_shape or actions.shape[:2] != rollout_shape:
        raise ValueError("Actor observations/actions must share rollout dimensions.")
    old_log_probability = old_action_log_probabilities.squeeze(-1)
    if old_log_probability.shape != rollout_shape:
        raise ValueError("Old action log probabilities must match [T,N].")

    flat_size = int(extrinsic_rewards.numel())
    batch_indices = batch_indices.to(device=extrinsic_rewards.device, dtype=torch.long)
    if batch_indices.ndim != 1 or batch_indices.numel() == 0:
        raise ValueError("batch_indices must be a non-empty vector.")
    if int(batch_indices.min()) < 0 or int(batch_indices.max()) >= flat_size:
        raise ValueError("batch_indices lie outside the flattened rollout.")

    intrinsic_rewards = reward_model(
        actor_observations.detach(), actions.detach()
    )
    mixed_rewards = lirpg_actor_reward(
        extrinsic_rewards.detach(),
        intrinsic_rewards,
        extrinsic_coefficient=float(extrinsic_coefficient),
        intrinsic_coefficient=float(intrinsic_coefficient),
    )
    mixed_advantages = differentiable_mixed_gae(
        mixed_rewards,
        mixed_values.detach(),
        final_mixed_value.detach(),
        dones.detach(),
        gamma=float(gamma),
        lam=float(lam),
    )
    flat_observations = actor_observations.detach().flatten(0, 1)
    flat_actions = actions.detach().flatten(0, 1)
    flat_old_log_probability = old_log_probability.detach().flatten()
    observations_batch = flat_observations[batch_indices]
    actions_batch = flat_actions[batch_indices]
    old_log_probability_batch = flat_old_log_probability[batch_indices]
    mixed_advantages_batch = mixed_advantages.flatten()[batch_indices]
    normalized_mixed_batch = (
        mixed_advantages_batch - mixed_advantages_batch.mean()
    ) / mixed_advantages_batch.std(unbiased=False).clamp_min(epsilon)
    outer_advantages_batch = outer_advantages.detach().flatten()[batch_indices]
    normalized_outer_batch = (
        outer_advantages_batch - outer_advantages_batch.mean()
    ) / outer_advantages_batch.std(unbiased=False).clamp_min(epsilon)

    actor_parameters = tuple(actor.parameters())
    current_standard_deviations = actor_log_std.exp().expand_as(actions_batch)
    log_probability = _squashed_gaussian_log_probability(
        actor(observations_batch),
        actions_batch,
        current_standard_deviations,
        maximum_latent_mean=float(maximum_latent_mean),
    )
    inner_policy_loss = _ppo_surrogate_loss(
        log_probability,
        old_log_probability_batch,
        normalized_mixed_batch,
        clip_param=float(clip_param),
    )
    differentiable_policy_parameters = actor_parameters + (actor_log_std,)
    inner_gradients = torch.autograd.grad(
        inner_policy_loss,
        differentiable_policy_parameters,
        create_graph=True,
        retain_graph=True,
    )
    virtual_parameters = _virtual_adam_parameters(
        actor, inner_gradients[:-1], actor_optimizer
    )
    virtual_log_std = _virtual_adam_parameter(
        actor_log_std, inner_gradients[-1], actor_optimizer
    )
    virtual_log_probability = _squashed_gaussian_log_probability(
        functional_call(actor, virtual_parameters, (observations_batch,)),
        actions_batch,
        virtual_log_std.exp().expand_as(actions_batch),
        maximum_latent_mean=float(maximum_latent_mean),
    )
    outer_policy_loss = _ppo_surrogate_loss(
        virtual_log_probability,
        old_log_probability_batch,
        normalized_outer_batch,
        clip_param=float(clip_param),
    )
    with torch.no_grad():
        current_outer_loss = _ppo_surrogate_loss(
            log_probability.detach(),
            old_log_probability_batch,
            normalized_outer_batch,
            clip_param=float(clip_param),
        )
        cosine = F.cosine_similarity(
            normalized_mixed_batch.detach(),
            normalized_outer_batch,
            dim=0,
            eps=epsilon,
        )
    return outer_policy_loss, {
        "meta_policy_loss": outer_policy_loss.detach(),
        "meta_outer_loss_before": current_outer_loss.detach(),
        "meta_outer_loss_after": outer_policy_loss.detach(),
        "predicted_outer_improvement": (
            current_outer_loss - outer_policy_loss
        ).detach(),
        "inner_policy_loss": inner_policy_loss.detach(),
        "intrinsic_reward_mean": intrinsic_rewards.mean().detach(),
        "intrinsic_reward_std": intrinsic_rewards.std(unbiased=False).detach(),
        "intrinsic_reward_abs": intrinsic_rewards.abs().mean().detach(),
        "intrinsic_advantage_mean": mixed_advantages.mean().detach(),
        "advantage_cosine": cosine.detach(),
        "mixed_advantages_batch": normalized_mixed_batch.detach(),
        "mixed_returns_batch": (
            mixed_advantages_batch
            + mixed_values.detach().flatten()[batch_indices]
        ).detach(),
    }


def predictor_ready(sequence_age: torch.Tensor, horizon: int = 12) -> torch.Tensor:
    """Return which online sequences have exactly enough future supervision."""

    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    return sequence_age >= int(horizon)


def validate_outer_checkpoint_state(metadata: dict[str, object]) -> None:
    """Reject a resume sidecar missing any causal training state."""

    missing = [key for key in OUTER_CHECKPOINT_REQUIRED_KEYS if key not in metadata]
    if missing:
        raise ValueError(f"Outer-composer checkpoint is incomplete: {missing}.")


__all__ = [
    "REWARD_GROUP_NAMES",
    "REWARD_GROUP_INDICES",
    "OUTER_CHECKPOINT_REQUIRED_KEYS",
    "RunningGroupRMS",
    "CenteredTanhComposer",
    "OuterCritic",
    "LIRPGOuterCritic",
    "LIRPGIntrinsicReward",
    "beta_schedule",
    "resolve_beta_schedule_horizon",
    "compose_actor_reward",
    "effective_composer_weights",
    "select_actor_reward",
    "composer_alignment_loss",
    "composer_meta_gradient_loss",
    "lirpg_meta_gradient_loss",
    "official_lirpg_meta_gradient_loss",
    "lirpg_actor_reward",
    "differentiable_mixed_gae",
    "cross_split_credit_reliability",
    "differentiable_shaping_advantage",
    "generalized_advantage",
    "group_atomic_rewards",
    "normalized_group_advantages",
    "outer_reward",
    "predictor_ready",
    "reward_group_index_tensor",
    "static_group_weight_tensor",
    "validate_static_group_weights",
    "validate_reward_groups",
    "validate_outer_checkpoint_state",
]
