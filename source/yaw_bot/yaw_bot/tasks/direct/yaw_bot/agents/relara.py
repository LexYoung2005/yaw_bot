"""PPO adapter for the official ICML 2024 ReLara reward agent.

The network topology and SAC update below follow ``mahaozhe/ReLara`` commit
``b384b0d7676bb9ef0d935b6464a606c6b5b0c596`` (MIT). The policy agent is
deliberately supplied by RSL-RL PPO so reward-learning methods can be compared
without changing the yaw_bot controller optimizer.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """The residual block used verbatim by the official reward agent."""

    def __init__(self, width: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        outputs = F.relu(self.fc1(inputs))
        outputs = self.fc2(outputs)
        return F.relu(outputs + residual)


class ReLaraRewardActor(nn.Module):
    """Official residual Gaussian actor producing one reward in [-scale, scale]."""

    def __init__(self, context_dim: int, reward_scale: float = 1.0, block_num: int = 3):
        super().__init__()
        if context_dim <= 0 or reward_scale <= 0.0:
            raise ValueError("ReLara context_dim and reward_scale must be positive.")
        self.context_dim = int(context_dim)
        self.reward_scale = float(reward_scale)
        self.fc1 = nn.Linear(self.context_dim, 256)
        self.hidden_blocks = nn.ModuleList(
            [ResidualBlock(256) for _ in range(int(block_num))]
        )
        self.fc2 = nn.Linear(256, 128)
        self.fc_mean = nn.Linear(128, 1)
        self.fc_logstd = nn.Linear(128, 1)
        self.log_std_max = 2.0
        self.log_std_min = -5.0

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if context.shape[-1] != self.context_dim:
            raise ValueError(
                f"Expected ReLara context dim {self.context_dim}, got {context.shape[-1]}."
            )
        outputs = self.fc1(context)
        for block in self.hidden_blocks:
            outputs = block(outputs)
        outputs = F.relu(self.fc2(outputs))
        mean = self.fc_mean(outputs)
        log_std = torch.tanh(self.fc_logstd(outputs))
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1.0)
        return mean, log_std

    def get_action(
        self,
        context: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self(context)
        std = log_std.exp()
        if noise is None:
            noise = torch.randn_like(mean)
        latent = mean + std * noise
        squashed = torch.tanh(latent)
        action = squashed * self.reward_scale
        normal = torch.distributions.Normal(mean, std)
        log_probability = normal.log_prob(latent)
        log_probability -= torch.log(
            self.reward_scale * (1.0 - squashed.square()) + 1.0e-6
        )
        deterministic = torch.tanh(mean) * self.reward_scale
        return action, log_probability.sum(dim=-1, keepdim=True), deterministic


class ReLaraRewardQNetwork(nn.Module):
    """Official residual twin-Q member for ``Q((s,a), proposed_reward)``."""

    def __init__(self, context_dim: int, block_num: int = 3):
        super().__init__()
        self.context_dim = int(context_dim)
        self.fc1 = nn.Linear(self.context_dim + 1, 256)
        self.hidden_blocks = nn.ModuleList(
            [ResidualBlock(256) for _ in range(int(block_num))]
        )
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, context: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
        if reward.ndim == context.ndim - 1:
            reward = reward.unsqueeze(-1)
        outputs = torch.cat((context, reward), dim=-1)
        outputs = self.fc1(outputs)
        for block in self.hidden_blocks:
            outputs = block(outputs)
        return self.fc3(F.relu(self.fc2(outputs)))


class ReLaraReplayBuffer:
    """CPU circular replay with the exact Reward-Agent transition semantics."""

    def __init__(self, capacity: int, context_dim: int):
        if capacity <= 0:
            raise ValueError("ReLara replay capacity must be positive.")
        self.capacity = int(capacity)
        self.context_dim = int(context_dim)
        self.contexts = torch.empty(self.capacity, self.context_dim)
        self.next_contexts = torch.empty_like(self.contexts)
        self.proposed_rewards = torch.empty(self.capacity, 1)
        self.environment_rewards = torch.empty(self.capacity, 1)
        self.dones = torch.empty(self.capacity, 1)
        self.position = 0
        self.size = 0

    def add(
        self,
        contexts: torch.Tensor,
        next_contexts: torch.Tensor,
        proposed_rewards: torch.Tensor,
        environment_rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        count = int(contexts.shape[0])
        if count == 0:
            return
        if count > self.capacity:
            start = count - self.capacity
            contexts = contexts[start:]
            next_contexts = next_contexts[start:]
            proposed_rewards = proposed_rewards[start:]
            environment_rewards = environment_rewards[start:]
            dones = dones[start:]
            count = self.capacity
        tensors = (
            contexts.detach().to(device="cpu", dtype=torch.float32),
            next_contexts.detach().to(device="cpu", dtype=torch.float32),
            proposed_rewards.detach().reshape(-1, 1).to(device="cpu", dtype=torch.float32),
            environment_rewards.detach().reshape(-1, 1).to(device="cpu", dtype=torch.float32),
            dones.detach().reshape(-1, 1).to(device="cpu", dtype=torch.float32),
        )
        indices = (torch.arange(count) + self.position) % self.capacity
        for storage, values in zip(
            (
                self.contexts,
                self.next_contexts,
                self.proposed_rewards,
                self.environment_rewards,
                self.dones,
            ),
            tensors,
            strict=True,
        ):
            storage[indices] = values
        self.position = (self.position + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def sample(
        self,
        batch_size: int,
        device: torch.device,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if self.size < batch_size:
            raise RuntimeError(
                f"ReLara replay has {self.size} samples, needs {batch_size}."
            )
        indices = torch.randint(self.size, (batch_size,), generator=generator)
        return tuple(
            storage[indices].to(device=device, non_blocking=True)
            for storage in (
                self.contexts,
                self.next_contexts,
                self.proposed_rewards,
                self.environment_rewards,
                self.dones,
            )
        )


@dataclass(frozen=True)
class ReLaraConfig:
    gamma: float = 0.99
    reward_scale: float = 1.0
    beta: float = 0.2
    replay_capacity: int = 1_000_000
    batch_size: int = 512
    learning_starts: int = 5_000
    actor_lr: float = 3.0e-4
    critic_lr: float = 1.0e-3
    alpha_lr: float = 1.0e-4
    policy_frequency: int = 2
    target_frequency: int = 1
    tau: float = 0.005
    # Official autotune path initializes log_alpha=0 (alpha=1).
    initial_alpha: float = 1.0
    alpha_autotune: bool = True


class ReLaraRewardAgent(nn.Module):
    """Official Reward-Agent SAC, decoupled from the yaw_bot policy optimizer."""

    def __init__(
        self,
        context_dim: int,
        config: ReLaraConfig,
        *,
        device: torch.device | str,
        seed: int,
    ):
        super().__init__()
        self.context_dim = int(context_dim)
        self.config = config
        self.device = torch.device(device)
        self.actor = ReLaraRewardActor(context_dim, config.reward_scale).to(self.device)
        self.rollout_actor = copy.deepcopy(self.actor).requires_grad_(False).eval()
        self.q1 = ReLaraRewardQNetwork(context_dim).to(self.device)
        self.q2 = ReLaraRewardQNetwork(context_dim).to(self.device)
        self.q1_target = copy.deepcopy(self.q1).requires_grad_(False)
        self.q2_target = copy.deepcopy(self.q2).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            [*self.q1.parameters(), *self.q2.parameters()], lr=config.critic_lr
        )
        self.log_alpha = nn.Parameter(
            torch.tensor(
                float(torch.log(torch.tensor(config.initial_alpha))),
                device=self.device,
            )
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)
        self.target_entropy = -1.0
        self.replay = ReLaraReplayBuffer(config.replay_capacity, context_dim)
        self.collection_samples = 0
        self.gradient_steps = 0
        self.private_generator = torch.Generator(device=self.device)
        self.private_generator.manual_seed(int(seed))
        self.replay_generator = torch.Generator(device="cpu")
        self.replay_generator.manual_seed(int(seed) + 1)

    @property
    def alpha(self) -> torch.Tensor:
        if self.config.alpha_autotune:
            return self.log_alpha.exp()
        return torch.tensor(self.config.initial_alpha, device=self.device)

    @torch.no_grad()
    def freeze_for_rollout(self) -> None:
        self.rollout_actor.load_state_dict(self.actor.state_dict())
        self.rollout_actor.requires_grad_(False).eval()

    @torch.no_grad()
    def propose(self, contexts: torch.Tensor) -> torch.Tensor:
        if self.collection_samples < self.config.learning_starts:
            return (
                2.0
                * torch.rand(
                    (*contexts.shape[:-1], 1),
                    device=contexts.device,
                    generator=self.private_generator,
                )
                - 1.0
            ) * self.config.reward_scale
        noise = torch.randn(
            (*contexts.shape[:-1], 1),
            device=contexts.device,
            generator=self.private_generator,
        )
        reward, _, _ = self.rollout_actor.get_action(contexts.detach(), noise=noise)
        return reward

    def add_rollout(
        self,
        contexts: torch.Tensor,
        next_contexts: torch.Tensor,
        proposed_rewards: torch.Tensor,
        environment_rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        self.replay.add(
            contexts.reshape(-1, self.context_dim),
            next_contexts.reshape(-1, self.context_dim),
            proposed_rewards.reshape(-1, 1),
            environment_rewards.reshape(-1, 1),
            dones.reshape(-1, 1),
        )
        self.collection_samples += int(contexts.numel() // self.context_dim)

    def optimize(self, updates: int) -> dict[str, float]:
        if self.replay.size < max(self.config.learning_starts, self.config.batch_size):
            return {
                "replay_size": float(self.replay.size),
                "gradient_steps": float(self.gradient_steps),
                "reward_agent_active": 0.0,
            }
        totals: dict[str, float] = {}
        for _ in range(int(updates)):
            metrics = self._optimize_once()
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + value
        if not totals:
            return {}
        metrics = {name: value / updates for name, value in totals.items()}
        metrics.update(
            replay_size=float(self.replay.size),
            gradient_steps=float(self.gradient_steps),
            reward_agent_active=1.0,
        )
        return metrics

    def _noise(self, shape: torch.Size) -> torch.Tensor:
        return torch.randn(shape, device=self.device, generator=self.private_generator)

    def _optimize_once(self) -> dict[str, float]:
        context, next_context, proposed, environment_reward, done = self.replay.sample(
            self.config.batch_size,
            self.device,
            generator=self.replay_generator,
        )
        with torch.no_grad():
            next_reward, next_log_pi, _ = self.actor.get_action(
                next_context, noise=self._noise(torch.Size((next_context.shape[0], 1)))
            )
            minimum_target_q = torch.minimum(
                self.q1_target(next_context, next_reward),
                self.q2_target(next_context, next_reward),
            ) - self.alpha.detach() * next_log_pi
            q_target = environment_reward + (
                1.0 - done
            ) * self.config.gamma * minimum_target_q

        q1_value = self.q1(context, proposed)
        q2_value = self.q2(context, proposed)
        q1_loss = F.mse_loss(q1_value, q_target)
        q2_loss = F.mse_loss(q2_value, q_target)
        critic_loss = q1_loss + q2_loss
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss_value = 0.0
        alpha_loss_value = 0.0
        if self.gradient_steps % self.config.policy_frequency == 0:
            for _ in range(self.config.policy_frequency):
                reward_action, log_pi, _ = self.actor.get_action(
                    context, noise=self._noise(torch.Size((context.shape[0], 1)))
                )
                minimum_q = torch.minimum(
                    self.q1(context, reward_action), self.q2(context, reward_action)
                )
                actor_loss = (self.alpha.detach() * log_pi - minimum_q).mean()
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                self.actor_optimizer.step()
                actor_loss_value += float(actor_loss.detach())

                if self.config.alpha_autotune:
                    with torch.no_grad():
                        _, fresh_log_pi, _ = self.actor.get_action(
                            context,
                            noise=self._noise(torch.Size((context.shape[0], 1))),
                        )
                    alpha_loss = (
                        -self.log_alpha.exp()
                        * (fresh_log_pi + self.target_entropy)
                    ).mean()
                    self.alpha_optimizer.zero_grad(set_to_none=True)
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                    alpha_loss_value += float(alpha_loss.detach())
            actor_loss_value /= self.config.policy_frequency
            alpha_loss_value /= self.config.policy_frequency

        if self.gradient_steps % self.config.target_frequency == 0:
            with torch.no_grad():
                for source, target in (
                    (self.q1, self.q1_target),
                    (self.q2, self.q2_target),
                ):
                    for source_parameter, target_parameter in zip(
                        source.parameters(), target.parameters(), strict=True
                    ):
                        target_parameter.lerp_(source_parameter, self.config.tau)
        self.gradient_steps += 1
        return {
            "q1_value": float(q1_value.detach().mean()),
            "q2_value": float(q2_value.detach().mean()),
            "q1_loss": float(q1_loss.detach()),
            "q2_loss": float(q2_loss.detach()),
            "actor_loss": actor_loss_value,
            "alpha": float(self.alpha.detach()),
            "alpha_loss": alpha_loss_value,
        }

    def training_state_dict(self) -> dict[str, object]:
        """Match official checkpoints (networks only) while retaining optimizers."""
        return {
            "actor": self.actor.state_dict(),
            "rollout_actor": self.rollout_actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "collection_samples": self.collection_samples,
            "gradient_steps": self.gradient_steps,
            "private_generator_state": self.private_generator.get_state(),
            "replay_generator_state": self.replay_generator.get_state(),
        }

    def load_training_state_dict(self, state: dict[str, object]) -> None:
        for name in ("actor", "rollout_actor", "q1", "q2", "q1_target", "q2_target"):
            getattr(self, name).load_state_dict(state[name])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        with torch.no_grad():
            self.log_alpha.copy_(torch.as_tensor(state["log_alpha"], device=self.device))
        self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        self.collection_samples = int(state["collection_samples"])
        self.gradient_steps = int(state["gradient_steps"])
        if "private_generator_state" in state:
            self.private_generator.set_state(state["private_generator_state"])
        if "replay_generator_state" in state:
            self.replay_generator.set_state(state["replay_generator_state"])


def relara_policy_reward(
    environment_reward: torch.Tensor,
    proposed_reward: torch.Tensor,
    *,
    beta: float = 0.2,
) -> torch.Tensor:
    """Official policy reward ``r_E + beta * r_S``."""
    if proposed_reward.ndim == environment_reward.ndim + 1:
        proposed_reward = proposed_reward.squeeze(-1)
    if environment_reward.shape != proposed_reward.shape:
        raise ValueError("ReLara environmental and proposed rewards must share a shape.")
    return environment_reward + float(beta) * proposed_reward


__all__ = [
    "ReLaraConfig",
    "ReLaraReplayBuffer",
    "ReLaraRewardActor",
    "ReLaraRewardAgent",
    "ReLaraRewardQNetwork",
    "ResidualBlock",
    "relara_policy_reward",
]
