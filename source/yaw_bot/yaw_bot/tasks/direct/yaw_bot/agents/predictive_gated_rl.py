"""Deployment helpers plus the retained v2-v4 routing compatibility path.

The formal direct-weight task uses standard single-critic PPO. Multi-critic
classes in this module remain only for old checkpoint/numerical regression
coverage; they are not constructed by ``PredictiveGatedPPORunnerCfg``.
"""

from __future__ import annotations

import copy
import os
import warnings
from typing import Any

import torch

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, resolve_rnd_config, resolve_symmetry_config
from rsl_rl.networks import MLP, EmpiricalNormalization
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.storage import RolloutStorage


NUM_PREDICTIVE_TIERS = 4
PREDICTIVE_TIER_NAMES = ("stability", "contact", "motion", "tracking")
ACTOR_CHECKPOINT_FORMAT_VERSION = 1


def linear_learning_rate_schedule(
    iteration: int,
    start_iteration: int,
    end_iteration: int,
    initial_rate: float,
    final_rate: float,
) -> float:
    """Hold the PPO rate for acquisition, then decay it for consolidation."""

    if start_iteration < 0 or end_iteration <= start_iteration:
        raise ValueError("Learning-rate decay requires 0 <= start_iteration < end_iteration.")
    if initial_rate <= 0.0 or final_rate <= 0.0 or final_rate > initial_rate:
        raise ValueError("Learning rates must satisfy 0 < final_rate <= initial_rate.")
    progress = (int(iteration) - int(start_iteration)) / float(end_iteration - start_iteration)
    progress = max(0.0, min(1.0, progress))
    return float(initial_rate + progress * (final_rate - initial_rate))


@torch.no_grad()
def rollout_policy_kl(policy, observations, old_mu: torch.Tensor, old_sigma: torch.Tensor) -> float:
    """Measure old-to-current policy KL once over a complete rollout.

    Distributed training returns the worst rank rather than the average. A
    large update on either simulator shard must be rejected before the shared
    policy collects another rollout.
    """

    flat_observations = observations.flatten(0, 1)
    actor_observations = policy.get_actor_obs(flat_observations)
    actor_observations = policy.actor_obs_normalizer(actor_observations)
    policy._update_distribution(actor_observations)
    new_mu = policy.action_mean
    new_sigma = policy.action_std
    old_mu = old_mu.flatten(0, 1)
    old_sigma = old_sigma.flatten(0, 1)
    kl = torch.sum(
        torch.log(new_sigma / old_sigma + 1.0e-5)
        + (torch.square(old_sigma) + torch.square(old_mu - new_mu))
        / (2.0 * torch.square(new_sigma))
        - 0.5,
        dim=-1,
    )
    mean_kl_tensor = torch.mean(kl).detach().float().cpu()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(
            mean_kl_tensor, op=torch.distributed.ReduceOp.MAX
        )
    mean_kl = float(mean_kl_tensor.item())
    # ``ActorCritic._update_distribution`` expands the trainable scalar-std
    # parameter into a view.  This helper runs under no_grad, while a paired
    # population update may subsequently copy a winner back into this policy.
    # Keeping that no-grad view alive makes PyTorch reject the next logger
    # access after the underlying parameter was changed in-place.  The
    # distribution is only a cache, so detach it from model parameters after
    # measuring KL; the next ``act``/``evaluate`` call rebuilds it normally.
    policy.distribution = torch.distributions.Normal(
        new_mu.detach().clone(), new_sigma.detach().clone()
    )
    return mean_kl


def guarded_rollout_ppo_update(
    algorithm,
    *,
    update_callable=None,
    desired_kl: float,
    maximum_kl: float,
    minimum_rate: float,
    maximum_rate: float,
    adaptation_factor: float,
    backtrack_factor: float,
    maximum_backtracks: int,
) -> dict[str, float]:
    """Apply PPO only when its complete-rollout KL stays inside a trust region.

    PPO clipping constrains sampled likelihood ratios inside each minibatch; it
    does not bound the final policy after several epochs. This function keeps a
    pre-update policy/Adam snapshot, evaluates the final update against the
    complete rollout, and retries the same minibatch permutation at a lower
    learning rate when necessary. If all retries fail, no part of the rejected
    update is retained.
    """

    if desired_kl <= 0.0 or maximum_kl <= desired_kl:
        raise ValueError("Require 0 < desired_kl < maximum_kl.")
    if not 0.0 < minimum_rate <= maximum_rate:
        raise ValueError("Learning-rate bounds are invalid.")
    if adaptation_factor <= 1.0 or backtrack_factor <= 1.0:
        raise ValueError("Adaptation and backtrack factors must exceed one.")
    if maximum_backtracks < 0:
        raise ValueError("maximum_backtracks must be non-negative.")

    policy_state = copy.deepcopy(algorithm.policy.state_dict())
    optimizer_state = copy.deepcopy(algorithm.optimizer.state_dict())
    cpu_rng_state = torch.get_rng_state().clone()
    device_string = str(algorithm.device)
    cuda_rng_state = (
        torch.cuda.get_rng_state(algorithm.device).clone()
        if torch.cuda.is_available() and device_string.startswith("cuda")
        else None
    )
    full_storage_step = int(algorithm.storage.step)
    observations = algorithm.storage.observations
    old_mu = algorithm.storage.mu.detach().clone()
    old_sigma = algorithm.storage.sigma.detach().clone()
    pre_update_kl = rollout_policy_kl(
        algorithm.policy,
        observations,
        old_mu,
        old_sigma,
    )
    attempted_rate = min(
        maximum_rate, max(minimum_rate, float(algorithm.learning_rate))
    )
    candidate_kl = float("inf")
    loss_dict: dict[str, float] = {}
    accepted = False
    backtracks = 0

    def restore_pre_update_state(*, restore_rng: bool) -> None:
        algorithm.policy.load_state_dict(policy_state, strict=True)
        algorithm.optimizer.load_state_dict(optimizer_state)
        algorithm.storage.step = full_storage_step
        if restore_rng:
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, algorithm.device)

    for attempt in range(maximum_backtracks + 1):
        if attempt > 0:
            restore_pre_update_state(restore_rng=True)
        algorithm.learning_rate = attempted_rate
        for parameter_group in algorithm.optimizer.param_groups:
            parameter_group["lr"] = attempted_rate

        loss_dict = (
            algorithm.update() if update_callable is None else update_callable()
        )
        candidate_kl = rollout_policy_kl(
            algorithm.policy,
            observations,
            old_mu,
            old_sigma,
        )
        if torch.isfinite(torch.tensor(candidate_kl)) and candidate_kl <= maximum_kl:
            accepted = True
            break
        if attempt < maximum_backtracks:
            backtracks += 1
            attempted_rate = max(minimum_rate, attempted_rate / backtrack_factor)

    if accepted:
        next_rate = rollout_adaptive_learning_rate(
            attempted_rate,
            candidate_kl,
            desired_kl,
            minimum_rate=minimum_rate,
            maximum_rate=maximum_rate,
            factor=adaptation_factor,
        )
    else:
        restore_pre_update_state(restore_rng=True)
        algorithm.storage.clear()
        next_rate = max(minimum_rate, attempted_rate / backtrack_factor)

    algorithm.learning_rate = next_rate
    for parameter_group in algorithm.optimizer.param_groups:
        parameter_group["lr"] = next_rate
    loss_dict["rollout_kl"] = float(candidate_kl)
    loss_dict["rollout_pre_update_kl"] = float(pre_update_kl)
    loss_dict["rollout_update_accepted"] = float(accepted)
    loss_dict["rollout_trust_region_backtracks"] = float(backtracks)
    loss_dict["rollout_learning_rate_used"] = (
        float(attempted_rate) if accepted else 0.0
    )
    return loss_dict


def rollout_adaptive_learning_rate(
    current_rate: float,
    rollout_kl: float,
    desired_kl: float,
    *,
    minimum_rate: float,
    maximum_rate: float,
    factor: float,
) -> float:
    """Adjust LR at most once per PPO update from its complete-rollout KL."""

    if not 0.0 < minimum_rate <= maximum_rate:
        raise ValueError("Rollout-adaptive learning-rate bounds are invalid.")
    if desired_kl <= 0.0 or factor <= 1.0:
        raise ValueError("desired_kl must be positive and adaptation factor must exceed one.")
    rate = float(current_rate)
    if rollout_kl > 2.0 * desired_kl:
        rate /= factor
    elif 0.0 < rollout_kl < 0.5 * desired_kl:
        rate *= factor
    return max(minimum_rate, min(maximum_rate, rate))


def snapshot_rollout_observation(observation):
    """Clone an observation before the environment can reuse its buffer."""

    return observation.clone()


class ObservationSafePPO(PPO):
    """PPO that owns the observation paired with each sampled action.

    The direct environment reuses observation buffers. Upstream ``PPO.act``
    retains its input by reference until ``process_env_step``; without this
    snapshot, ``env.step`` can turn cached ``obs_t`` into ``obs_{t+1}`` while
    its action and old distribution still belong to time t.
    """

    def act(self, obs):
        actions = super().act(obs)
        self.transition.observations = snapshot_rollout_observation(obs)
        return actions


def bounded_action_standard_deviation(
    standard_deviation: torch.Tensor,
    minimum: float,
    maximum: float,
) -> torch.Tensor:
    """Apply a hard physical trust region to Gaussian exploration scale."""

    if minimum <= 0.0 or maximum <= minimum:
        raise ValueError("Noise bounds must satisfy 0 < minimum < maximum.")
    return standard_deviation.clamp(float(minimum), float(maximum))


def tanh_squashed_gaussian_log_prob(
    distribution: torch.distributions.Normal, actions: torch.Tensor
) -> torch.Tensor:
    """Return log p(tanh(z)=action) for a diagonal Gaussian latent ``z``."""

    clipped_actions = actions.clamp(min=-1.0 + 1.0e-6, max=1.0 - 1.0e-6)
    latent_actions = torch.atanh(clipped_actions)
    log_det_jacobian = torch.log(1.0 - torch.square(clipped_actions) + 1.0e-6)
    return (distribution.log_prob(latent_actions) - log_det_jacobian).sum(dim=-1)


def smoothly_bound_latent_mean(raw_mean: torch.Tensor, maximum: float) -> torch.Tensor:
    """Keep a tanh-Gaussian mean inside its numerically invertible latent range."""

    if maximum <= 0.0:
        raise ValueError("maximum latent mean must be positive.")
    return float(maximum) * torch.tanh(raw_mean / float(maximum))


def route_multi_critic_advantages(
    tier_advantages: torch.Tensor,
    tier_gates: torch.Tensor,
    *,
    normalize: bool,
) -> torch.Tensor:
    """Route raw per-tier advantages into the scalar actor signal."""

    if tier_advantages.shape != tier_gates.shape or tier_advantages.shape[-1] != NUM_PREDICTIVE_TIERS:
        raise ValueError("tier_advantages and tier_gates must have matching [..., 4] shapes.")
    routed_advantages = tier_advantages
    if normalize:
        reduction_dims = tuple(range(tier_advantages.ndim - 1))
        tier_mean = tier_advantages.mean(dim=reduction_dims, keepdim=True)
        tier_std = tier_advantages.std(dim=reduction_dims, keepdim=True, unbiased=False)
        routed_advantages = (tier_advantages - tier_mean) / (tier_std + 1.0e-8)
    actor_advantages = torch.sum(tier_gates * routed_advantages, dim=-1, keepdim=True)
    if normalize:
        actor_advantages = (actor_advantages - actor_advantages.mean()) / (
            actor_advantages.std(unbiased=False) + 1.0e-8
        )
    return actor_advantages


class ActorOnlyInferencePolicy(torch.nn.Module):
    """Deployment wrapper that never constructs prediction heads or critics."""

    def __init__(
        self,
        actor_obs_groups: list[str],
        num_actor_obs: int,
        num_actions: int,
        actor_hidden_dims=(256, 128, 64),
        activation: str = "elu",
        actor_obs_normalization: bool = False,
        state_dependent_std: bool = False,
        action_squash: bool = False,
        maximum_latent_mean: float | None = None,
    ) -> None:
        super().__init__()
        self.actor_obs_groups = list(actor_obs_groups)
        self.num_actor_obs = int(num_actor_obs)
        self.num_actions = int(num_actions)
        self.actor_hidden_dims = list(actor_hidden_dims)
        self.activation = activation
        self.actor_obs_normalization = bool(actor_obs_normalization)
        self.state_dependent_std = bool(state_dependent_std)
        self.action_squash = bool(action_squash)
        self.maximum_latent_mean = (
            None if maximum_latent_mean is None else float(maximum_latent_mean)
        )
        if self.maximum_latent_mean is not None and self.maximum_latent_mean <= 0.0:
            raise ValueError("maximum_latent_mean must be positive when provided.")
        output_dims = [2, num_actions] if self.state_dependent_std else num_actions
        self.actor = MLP(num_actor_obs, output_dims, list(actor_hidden_dims), activation)
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

    def _actor_observation(self, obs) -> torch.Tensor:
        return torch.cat([obs[group] for group in self.actor_obs_groups], dim=-1)

    def act_inference(self, obs) -> torch.Tensor:
        actor_observation = self.actor_obs_normalizer(self._actor_observation(obs))
        output = self.actor(actor_observation)
        mean = output[..., 0, :] if self.state_dependent_std else output
        if self.action_squash and self.maximum_latent_mean is not None:
            mean = smoothly_bound_latent_mean(mean, self.maximum_latent_mean)
        return torch.tanh(mean) if self.action_squash else mean

    def forward(self, obs) -> torch.Tensor:
        return self.act_inference(obs)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        del dones

    def get_config(self) -> dict[str, Any]:
        """Return architecture fields that must match an actor sidecar."""

        config = {
            "actor_obs_groups": self.actor_obs_groups,
            "num_actor_obs": self.num_actor_obs,
            "num_actions": self.num_actions,
            "actor_hidden_dims": self.actor_hidden_dims,
            "activation": self.activation,
            "state_dependent_std": self.state_dependent_std,
            "actor_obs_normalization": self.actor_obs_normalization,
        }
        # Preserve exact compatibility with format-v1 checkpoints written
        # before bounded-action policies existed.
        if self.action_squash:
            config["action_squash"] = True
            if self.maximum_latent_mean is not None:
                config["maximum_latent_mean"] = self.maximum_latent_mean
        return config

    def load_deployment_checkpoint(
        self,
        checkpoint_path: str,
        map_location: str | torch.device | None = "cpu",
    ) -> int:
        """Load an actor-only sidecar without deserializing critics or optimizer state."""

        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("Actor-only checkpoint must be a dictionary.")
        if payload.get("format_version") != ACTOR_CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported actor-only checkpoint format {payload.get('format_version')!r}; "
                f"expected {ACTOR_CHECKPOINT_FORMAT_VERSION}."
            )
        checkpoint_config = payload.get("actor_config")
        runtime_config = self.get_config()
        # Format-v1 actors written before latent-mean bounding must retain
        # their original tanh(raw_mean) inference behavior.
        if (
            isinstance(checkpoint_config, dict)
            and "maximum_latent_mean" not in checkpoint_config
            and "maximum_latent_mean" in runtime_config
        ):
            legacy_runtime_config = dict(runtime_config)
            legacy_runtime_config.pop("maximum_latent_mean")
            if checkpoint_config == legacy_runtime_config:
                self.maximum_latent_mean = None
                runtime_config = self.get_config()
        if checkpoint_config != runtime_config:
            raise ValueError(
                f"Actor-only checkpoint config {checkpoint_config} does not match "
                f"runtime config {runtime_config}."
            )
        self.actor.load_state_dict(payload["actor_state_dict"], strict=True)
        normalizer_state = payload.get("actor_normalizer_state_dict", {})
        self.actor_obs_normalizer.load_state_dict(normalizer_state, strict=True)
        self.eval()
        return int(payload.get("iter", 0))

    def load_training_checkpoint(
        self,
        checkpoint_path: str,
        map_location: str | torch.device | None = None,
    ) -> int:
        """Load only actor/normalizer tensors from a native RSL-RL checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        policy_state = checkpoint.get("model_state_dict")
        if policy_state is None:
            raise KeyError(f"Checkpoint contains no model_state_dict: {checkpoint_path}")

        actor_state = {
            key.removeprefix("actor."): value
            for key, value in policy_state.items()
            if key.startswith("actor.")
        }
        if not actor_state:
            raise KeyError(f"Checkpoint contains no actor parameters: {checkpoint_path}")
        self.actor.load_state_dict(actor_state, strict=True)

        normalizer_state = {
            key.removeprefix("actor_obs_normalizer."): value
            for key, value in policy_state.items()
            if key.startswith("actor_obs_normalizer.")
        }
        if normalizer_state:
            self.actor_obs_normalizer.load_state_dict(normalizer_state, strict=True)
        self.eval()
        return int(checkpoint.get("iter", 0))


def save_actor_only_checkpoint(
    policy: torch.nn.Module,
    path: str,
    actor_config: dict[str, Any],
    *,
    iteration: int,
    checkpoint_group_id: str,
) -> None:
    """Save the deployable actor and normalizer, excluding critics and optimizer."""

    if not hasattr(policy, "actor") or not hasattr(policy, "actor_obs_normalizer"):
        raise TypeError("Training policy does not expose actor and actor_obs_normalizer modules.")
    payload = {
        "format_version": ACTOR_CHECKPOINT_FORMAT_VERSION,
        "actor_config": actor_config,
        "actor_state_dict": policy.actor.state_dict(),
        "actor_normalizer_state_dict": policy.actor_obs_normalizer.state_dict(),
        "iter": int(iteration),
        "checkpoint_group_id": checkpoint_group_id,
    }
    torch.save(payload, path)


class BoundedStdActorCritic(ActorCritic):
    """Single-critic actor with bounded exploration and physically bounded actions.

    The environment accepts actions in [-1, 1].  Sampling an unconstrained
    Gaussian and clipping it in the environment makes PPO evaluate the density
    of an action different from the one that was simulated; once means drift
    outside the range, clipping also creates an absorbing all-saturated policy.
    This actor samples in latent Gaussian space and maps it through ``tanh``
    before the transition is stored.  The inverse transform and Jacobian are
    used when PPO recomputes action log-probabilities.
    """

    def __init__(
        self,
        *args,
        minimum_noise_std: float = 0.05,
        maximum_noise_std: float = 1.05,
        maximum_latent_mean: float = 4.0,
        noise_std_type: str = "log",
        action_squash: bool = True,
        **kwargs,
    ) -> None:
        if noise_std_type != "log":
            raise ValueError("BoundedStdActorCritic requires log-space standard deviation.")
        if minimum_noise_std <= 0.0 or maximum_noise_std <= minimum_noise_std:
            raise ValueError("Noise bounds must satisfy 0 < minimum < maximum.")
        if maximum_latent_mean <= 0.0:
            raise ValueError("maximum_latent_mean must be positive.")
        self.minimum_noise_std = float(minimum_noise_std)
        self.maximum_noise_std = float(maximum_noise_std)
        self.maximum_latent_mean = float(maximum_latent_mean)
        self.action_squash = bool(action_squash)
        super().__init__(*args, noise_std_type=noise_std_type, **kwargs)
        # A single non-finite PPO minibatch must not poison Adam's learned
        # log-standard-deviation (or any actor tensor) permanently.
        for parameter in self.parameters():
            parameter.register_hook(self._finite_gradient)

    @staticmethod
    def _finite_gradient(gradient: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)

    def _update_distribution(self, obs) -> None:
        # Keep the trainable log-scale itself finite before exponentiating it.
        # Clamping only the derived std is insufficient: NaN survives clamp and
        # crashes Normal.sample on the following rollout.
        lower_log_std = float(torch.log(torch.tensor(self.minimum_noise_std)).item())
        upper_log_std = float(torch.log(torch.tensor(self.maximum_noise_std)).item())
        with torch.no_grad():
            self.log_std.nan_to_num_(nan=0.0, posinf=upper_log_std, neginf=lower_log_std)
            self.log_std.clamp_(min=lower_log_std, max=upper_log_std)
        raw_mean = self.actor(obs)
        mean = (
            smoothly_bound_latent_mean(raw_mean, self.maximum_latent_mean)
            if self.action_squash
            else raw_mean
        )
        bounded_std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = torch.distributions.Normal(mean, bounded_std)

    def act(self, obs, **kwargs) -> torch.Tensor:
        """Sample an action in latent space, then apply the bounded actuator map."""

        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        self._update_distribution(obs)
        action = self.distribution.sample()
        return torch.tanh(action) if self.action_squash else action

    def act_inference(self, obs) -> torch.Tensor:
        """Return the deterministic action using the same map as training."""

        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        if self.state_dependent_std:
            mean = self.actor(obs)[..., 0, :]
        else:
            mean = self.actor(obs)
        if self.action_squash:
            mean = smoothly_bound_latent_mean(mean, self.maximum_latent_mean)
        return torch.tanh(mean) if self.action_squash else mean

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Evaluate executed tanh-Gaussian actions under the latent policy."""

        if not self.action_squash:
            return super().get_actions_log_prob(actions)
        # Rollout actions are tanh outputs, hence open-interval values in exact
        # arithmetic.  The clamp only protects checkpoint/replay round-off.
        return tanh_squashed_gaussian_log_prob(self.distribution, actions)


class MultiCriticActorCritic(ActorCritic):
    """Feed-forward actor with one independent value head per reward tier."""

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims=(256, 256, 256),
        critic_hidden_dims=(256, 256, 256),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        num_value_heads: int = NUM_PREDICTIVE_TIERS,
        **kwargs,
    ) -> None:
        if num_value_heads != NUM_PREDICTIVE_TIERS:
            raise ValueError(
                f"MultiCriticActorCritic requires exactly {NUM_PREDICTIVE_TIERS} value heads, "
                f"got {num_value_heads}."
            )

        super().__init__(
            obs,
            obs_groups,
            num_actions,
            actor_obs_normalization=actor_obs_normalization,
            critic_obs_normalization=critic_obs_normalization,
            actor_hidden_dims=list(actor_hidden_dims),
            critic_hidden_dims=list(critic_hidden_dims),
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs,
        )

        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            if len(obs[obs_group].shape) != 2:
                raise AssertionError("MultiCriticActorCritic only supports 1D critic observations.")
            num_critic_obs += obs[obs_group].shape[-1]

        # Replacing ``critic`` removes the single value MLP registered by the
        # base class, while retaining all of its actor/distribution and
        # observation-normalization behavior.
        self.critic = torch.nn.ModuleList(
            [
                MLP(num_critic_obs, 1, list(critic_hidden_dims), activation)
                for _ in range(NUM_PREDICTIVE_TIERS)
            ]
        )
        self.num_value_heads = NUM_PREDICTIVE_TIERS
        print(f"Critic MLPs ({', '.join(PREDICTIVE_TIER_NAMES)}): {self.critic}")

    def evaluate(self, obs, **kwargs) -> torch.Tensor:
        """Return four independently estimated values in tier order."""

        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        return torch.cat([value_head(critic_obs) for value_head in self.critic], dim=-1)


class PredictiveGatedRolloutStorage(RolloutStorage):
    """Rollout storage with ungated GAE and routed multi-critic advantages.

    This is not standard PPO/GAE over a scalar gated reward: each raw tier has
    its own critic and GAE, and the current-step gates route those advantages
    into the actor objective.
    """

    class Transition(RolloutStorage.Transition):
        """RSL-RL transition extended with predictive-gating tensors."""

        def __init__(self) -> None:
            super().__init__()
            self.tier_rewards = None
            self.tier_gates = None

    def __init__(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        obs,
        actions_shape,
        device="cpu",
        num_tiers: int = NUM_PREDICTIVE_TIERS,
    ) -> None:
        if training_type != "rl":
            raise ValueError("PredictiveGatedRolloutStorage is only available for RL training.")
        if num_tiers != NUM_PREDICTIVE_TIERS:
            raise ValueError(
                f"PredictiveGatedRolloutStorage requires {NUM_PREDICTIVE_TIERS} tiers, got {num_tiers}."
            )

        super().__init__(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            device,
        )
        self.num_tiers = num_tiers

        tier_shape = (num_transitions_per_env, num_envs, num_tiers)
        self.tier_rewards = torch.zeros(*tier_shape, device=self.device)
        self.tier_gates = torch.zeros(*tier_shape, device=self.device)
        self.values = torch.zeros(*tier_shape, device=self.device)
        self.returns = torch.zeros(*tier_shape, device=self.device)
        self.tier_advantages = torch.zeros(*tier_shape, device=self.device)

        # RSL-RL's PPO generator consumes ``advantages`` as the actor signal.
        # It remains scalar even though values and returns are per-tier.
        self.advantages = torch.zeros(
            num_transitions_per_env,
            num_envs,
            1,
            device=self.device,
        )

        # Explicit aliases make the storage contract self-documenting without
        # changing the field names expected by RSL-RL's mini-batch generators.
        self.tier_values = self.values
        self.tier_returns = self.returns

    def add_transitions(self, transition: Transition) -> None:
        """Store a transition after validating its four-tier payload."""

        expected_shape = (self.num_envs, self.num_tiers)
        if transition.tier_rewards is None:
            raise RuntimeError("Transition is missing predictive_gating['tier_rewards'].")
        if transition.tier_gates is None:
            raise RuntimeError("Transition is missing predictive_gating['tier_gates'].")
        if tuple(transition.tier_rewards.shape) != expected_shape:
            raise ValueError(
                f"tier_rewards must have shape {expected_shape}, got {tuple(transition.tier_rewards.shape)}."
            )
        if tuple(transition.tier_gates.shape) != expected_shape:
            raise ValueError(
                f"tier_gates must have shape {expected_shape}, got {tuple(transition.tier_gates.shape)}."
            )

        step = self.step
        super().add_transitions(transition)
        self.tier_rewards[step].copy_(transition.tier_rewards)
        self.tier_gates[step].copy_(transition.tier_gates)

    def compute_returns(
        self,
        last_values: torch.Tensor,
        gamma: float,
        lam: float,
        normalize_advantage: bool = True,
    ) -> None:
        """Compute independent tier GAE, then route only the actor signal."""

        expected_shape = (self.num_envs, self.num_tiers)
        if tuple(last_values.shape) != expected_shape:
            raise ValueError(f"last_values must have shape {expected_shape}, got {tuple(last_values.shape)}.")

        gae = torch.zeros_like(last_values)
        for step in reversed(range(self.num_transitions_per_env)):
            next_values = last_values if step == self.num_transitions_per_env - 1 else self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = (
                self.tier_rewards[step]
                + next_is_not_terminal * gamma * next_values
                - self.values[step]
            )
            gae = delta + next_is_not_terminal * gamma * lam * gae
            self.returns[step] = gae + self.values[step]

        self.tier_advantages.copy_(self.returns - self.values)

        # This is the only operation in storage where predictive gates are
        # applied.  Critics therefore learn the original, ungated tier returns.
        # Reward tiers have intentionally different physical units and scales.
        self.advantages.copy_(
            route_multi_critic_advantages(
                self.tier_advantages,
                self.tier_gates,
                normalize=normalize_advantage,
            )
        )


class PredictiveGatedPPO(ObservationSafePPO):
    """PPO that consumes per-tier rewards and predicted gates from extras."""

    def __init__(self, policy, *args, num_tiers: int = NUM_PREDICTIVE_TIERS, **kwargs) -> None:
        if num_tiers != NUM_PREDICTIVE_TIERS:
            raise ValueError(f"PredictiveGatedPPO requires {NUM_PREDICTIVE_TIERS} tiers, got {num_tiers}.")
        if getattr(policy, "num_value_heads", None) != num_tiers:
            raise ValueError(
                f"PredictiveGatedPPO expected a policy with {num_tiers} value heads, "
                f"got {getattr(policy, 'num_value_heads', None)}."
            )
        self.num_tiers = num_tiers
        super().__init__(policy, *args, **kwargs)
        if self.rnd is not None:
            raise ValueError(
                "PredictiveGatedPPO does not support RND rewards because an intrinsic reward tier "
                "has not been defined. Set algorithm.rnd_cfg to None."
            )
        self.transition = PredictiveGatedRolloutStorage.Transition()

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape) -> None:
        """Create the four-tier rollout storage using RSL-RL's 3.1.2 API."""

        self.storage = PredictiveGatedRolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            self.device,
            num_tiers=self.num_tiers,
        )

    def process_env_step(self, obs, rewards, dones, extras) -> None:
        """Read the strict predictive-gating payload and record the step."""

        self.policy.update_normalization(obs)

        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        if extras is None or "predictive_gating" not in extras:
            raise KeyError(
                "PredictiveGatedPPO requires extras['predictive_gating'] with "
                "'tier_rewards' and 'tier_gates'."
            )
        predictive_gating = extras["predictive_gating"]
        try:
            tier_rewards_raw = predictive_gating["tier_rewards"]
            tier_gates_raw = predictive_gating["tier_gates"]
        except (KeyError, TypeError) as exc:
            raise KeyError(
                "extras['predictive_gating'] must contain 'tier_rewards' and 'tier_gates'."
            ) from exc

        num_envs = self.transition.values.shape[0]
        tier_rewards = self._as_tier_tensor(tier_rewards_raw, "tier_rewards", num_envs)
        tier_gates = self._as_tier_tensor(tier_gates_raw, "tier_gates", num_envs)

        if not torch.isfinite(tier_rewards).all():
            raise ValueError("predictive_gating['tier_rewards'] contains NaN or Inf values.")
        if not torch.isfinite(tier_gates).all():
            raise ValueError("predictive_gating['tier_gates'] contains NaN or Inf values.")
        if torch.any(tier_gates < 0.0) or torch.any(tier_gates > 1.0):
            gate_min = tier_gates.min().item()
            gate_max = tier_gates.max().item()
            raise ValueError(f"tier_gates must lie in [0, 1], got range [{gate_min}, {gate_max}].")

        # Time-limit bootstrapping is performed independently for every critic.
        # The gates are intentionally absent from this calculation.
        if "time_outs" in extras:
            time_outs = torch.as_tensor(
                extras["time_outs"],
                device=self.device,
                dtype=tier_rewards.dtype,
            ).reshape(-1)
            if time_outs.numel() != num_envs:
                raise ValueError(
                    f"time_outs must contain one value per environment ({num_envs}), got {time_outs.numel()}."
                )
            tier_rewards = tier_rewards + self.gamma * self.transition.values * time_outs.unsqueeze(-1)

        self.transition.tier_rewards = tier_rewards.detach()
        self.transition.tier_gates = tier_gates.detach()

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def _as_tier_tensor(self, value: Any, name: str, num_envs: int) -> torch.Tensor:
        """Convert an extras value to the canonical ``[num_envs, 4]`` shape."""

        tensor = torch.as_tensor(value, device=self.device, dtype=torch.float32)
        if tensor.ndim == 3 and tensor.shape[-1] == 1:
            tensor = tensor.squeeze(-1)
        if tensor.ndim == 1 and num_envs == 1 and tensor.numel() == self.num_tiers:
            tensor = tensor.unsqueeze(0)

        expected_shape = (num_envs, self.num_tiers)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}.")
        return tensor


class CpuSynchronizedPPO(ObservationSafePPO):
    """PPO whose distributed collectives use CPU buffers with a Gloo group.

    Isaac PhysX owns long-lived CUDA work in each process. On the tested
    dual-5080 driver stack, launching NCCL collectives beside that work causes
    an illegal memory access. Forward/backward remain entirely GPU-local; only
    the flattened gradient and initial state cross the CPU boundary.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.is_multi_gpu and self.schedule == "adaptive":
            raise ValueError(
                "CPU-synchronized distributed PPO requires a fixed learning-rate "
                "schedule; adaptive KL uses CUDA collectives inside upstream RSL-RL."
            )

    def broadcast_parameters(self) -> None:
        if not self.is_multi_gpu:
            return
        if self.gpu_global_rank == 0:
            policy_state = {
                name: tensor.detach().cpu()
                for name, tensor in self.policy.state_dict().items()
            }
            rnd_state = (
                {
                    name: tensor.detach().cpu()
                    for name, tensor in self.rnd.predictor.state_dict().items()
                }
                if self.rnd
                else None
            )
        else:
            policy_state = None
            rnd_state = None
        state_objects = [policy_state, rnd_state]
        torch.distributed.broadcast_object_list(state_objects, src=0)
        self.policy.load_state_dict(state_objects[0], strict=True)
        if self.rnd:
            self.rnd.predictor.load_state_dict(state_objects[1], strict=True)

    def reduce_parameters(self) -> None:
        if not self.is_multi_gpu:
            return
        parameters = list(self.policy.parameters())
        if self.rnd:
            parameters.extend(self.rnd.parameters())
        gradients = [
            parameter.grad
            for parameter in parameters
            if parameter.grad is not None
        ]
        if not gradients:
            return
        flat_cpu = torch.cat(
            [gradient.detach().reshape(-1).cpu() for gradient in gradients]
        )
        torch.distributed.all_reduce(
            flat_cpu, op=torch.distributed.ReduceOp.SUM
        )
        flat_cpu.div_(self.gpu_world_size)
        offset = 0
        for gradient in gradients:
            size = gradient.numel()
            gradient.copy_(
                flat_cpu[offset : offset + size]
                .view_as(gradient)
                .to(gradient.device)
            )
            offset += size


class BoundedStdOnPolicyRunner(OnPolicyRunner):
    """Construct standard single-critic PPO with bounded Gaussian exploration."""

    def _configure_multi_gpu(self) -> None:
        """Use Gloo because NCCL conflicts with this Isaac PhysX GPU stack."""
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))
        expected_device = f"cuda:{self.gpu_local_rank}"
        if self.device != expected_device:
            raise ValueError(
                f"Device '{self.device}' does not match local rank device '{expected_device}'."
            )
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError("Local rank must be smaller than world size.")
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError("Global rank must be smaller than world size.")
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,
            "local_rank": self.gpu_local_rank,
            "world_size": self.gpu_world_size,
        }
        torch.cuda.set_device(self.gpu_local_rank)
        backend = os.environ.get("YAWBOT_DISTRIBUTED_BACKEND", "gloo")
        if backend != "gloo":
            raise ValueError(
                "YawBot distributed training currently requires the Gloo backend."
            )
        torch.distributed.init_process_group(
            backend=backend,
            rank=self.gpu_global_rank,
            world_size=self.gpu_world_size,
        )

    def _construct_algorithm(self, obs) -> PPO:
        self.alg_cfg = resolve_rnd_config(
            copy.deepcopy(self.alg_cfg),
            obs,
            self.cfg["obs_groups"],
            self.env,
        )
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        policy_cfg = copy.deepcopy(self.policy_cfg)
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Set "
                "normalization in the policy config.",
                DeprecationWarning,
            )
            policy_cfg.setdefault(
                "actor_obs_normalization", self.cfg["empirical_normalization"]
            )
            policy_cfg.setdefault(
                "critic_obs_normalization", self.cfg["empirical_normalization"]
            )
        policy_cfg.pop("class_name", None)
        actor_critic = BoundedStdActorCritic(
            obs,
            self.cfg["obs_groups"],
            self.env.num_actions,
            **policy_cfg,
        ).to(self.device)

        algorithm_cfg = dict(self.alg_cfg)
        algorithm_class = algorithm_cfg.pop("class_name", "PPO")
        if algorithm_class != "PPO":
            raise ValueError(
                "BoundedStdOnPolicyRunner retains standard PPO and cannot construct "
                f"{algorithm_class}."
            )
        algorithm_type = (
            CpuSynchronizedPPO
            if self.multi_gpu_cfg is not None
            else ObservationSafePPO
        )
        algorithm = algorithm_type(
            actor_critic,
            device=self.device,
            multi_gpu_cfg=self.multi_gpu_cfg,
            **algorithm_cfg,
        )
        algorithm.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )
        return algorithm


class PredictiveGatedOnPolicyRunner(OnPolicyRunner):
    """RSL-RL runner that constructs the predictive-gated PPO stack."""

    def _construct_algorithm(self, obs) -> PredictiveGatedPPO:
        """Construct custom policy, algorithm, and storage via the 3.1.2 hook."""

        self.alg_cfg = resolve_rnd_config(
            copy.deepcopy(self.alg_cfg),
            obs,
            self.cfg["obs_groups"],
            self.env,
        )
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        policy_cfg = copy.deepcopy(self.policy_cfg)
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Set "
                "`actor_obs_normalization` and `critic_obs_normalization` in the policy config.",
                DeprecationWarning,
            )
            if policy_cfg.get("actor_obs_normalization") is None:
                policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if policy_cfg.get("critic_obs_normalization") is None:
                policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        # This runner explicitly selects its custom classes; class_name remains
        # accepted in standard Isaac Lab configs solely for registry compatibility.
        policy_cfg.pop("class_name", None)
        configured_num_heads = policy_cfg.pop("num_value_heads", NUM_PREDICTIVE_TIERS)
        if configured_num_heads != NUM_PREDICTIVE_TIERS:
            raise ValueError(
                f"The predictive-gated runner requires {NUM_PREDICTIVE_TIERS} value heads, "
                f"got {configured_num_heads}."
            )

        actor_critic = MultiCriticActorCritic(
            obs,
            self.cfg["obs_groups"],
            self.env.num_actions,
            num_value_heads=NUM_PREDICTIVE_TIERS,
            **policy_cfg,
        ).to(self.device)

        # A resolved symmetry configuration may contain the live environment,
        # which must not be deep-copied.  The algorithm only mutates nested RND
        # configuration during construction; RND is rejected above, so a
        # shallow top-level copy is sufficient here.
        algorithm_cfg = dict(self.alg_cfg)
        algorithm_cfg.pop("class_name", None)
        configured_num_tiers = algorithm_cfg.pop("num_tiers", NUM_PREDICTIVE_TIERS)
        if configured_num_tiers != NUM_PREDICTIVE_TIERS:
            raise ValueError(
                f"The predictive-gated runner requires {NUM_PREDICTIVE_TIERS} reward tiers, "
                f"got {configured_num_tiers}."
            )

        algorithm = PredictiveGatedPPO(
            actor_critic,
            device=self.device,
            num_tiers=NUM_PREDICTIVE_TIERS,
            multi_gpu_cfg=self.multi_gpu_cfg,
            **algorithm_cfg,
        )
        algorithm.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )
        return algorithm


__all__ = [
    "ACTOR_CHECKPOINT_FORMAT_VERSION",
    "ActorOnlyInferencePolicy",
    "BoundedStdActorCritic",
    "BoundedStdOnPolicyRunner",
    "CpuSynchronizedPPO",
    "MultiCriticActorCritic",
    "ObservationSafePPO",
    "PredictiveGatedOnPolicyRunner",
    "PredictiveGatedPPO",
    "PredictiveGatedRolloutStorage",
    "bounded_action_standard_deviation",
    "guarded_rollout_ppo_update",
    "linear_learning_rate_schedule",
    "rollout_adaptive_learning_rate",
    "rollout_policy_kl",
    "route_multi_critic_advantages",
    "tanh_squashed_gaussian_log_prob",
    "smoothly_bound_latent_mean",
    "save_actor_only_checkpoint",
    "snapshot_rollout_observation",
]
