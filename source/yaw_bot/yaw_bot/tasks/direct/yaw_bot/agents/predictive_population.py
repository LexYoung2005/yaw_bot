"""Two-branch population PPO for causal reward-allocation meta-training."""

from __future__ import annotations

import copy

import torch

from ..predictive_feasibility import (
    discounted_rollout_score,
    linear_trend,
    winner_conditioned_pair_progress,
)


class AntitheticPopulationPPO:
    """Train antithetic reward candidates on independent environment halves.

    Both branches begin each generation with identical policy and optimizer
    state, then control disjoint environment populations for several complete
    rollouts. Only the difference between their immutable-objective learning
    slopes credits the shared reward allocator.
    """

    def __init__(self, algorithm, task, cfg, num_steps_per_env: int) -> None:
        if task.num_envs < 2 or task.num_envs % 2 != 0:
            raise ValueError("Population allocator requires an even number of environments.")
        self.task = task
        self.cfg = cfg
        self.num_steps_per_env = int(num_steps_per_env)
        self.half_envs = task.num_envs // 2
        self.generation_rollouts = int(cfg.predictive_allocator_meta_rollouts)
        if self.generation_rollouts < 2:
            raise ValueError("Population allocator needs at least two rollouts per generation.")

        observation_template = algorithm.storage.observations[0]
        action_shape = algorithm.storage.actions_shape
        self.positive = algorithm
        self.negative = copy.deepcopy(algorithm)
        self.positive.init_storage(
            "rl", self.half_envs, self.num_steps_per_env, observation_template[: self.half_envs], action_shape
        )
        self.negative.init_storage(
            "rl", self.half_envs, self.num_steps_per_env, observation_template[self.half_envs :], action_shape
        )
        self.rnd = None
        self._step_meta: list[dict[str, torch.Tensor]] = []
        self._generation: list[dict[str, object]] = []
        signs = torch.ones(task.num_envs, device=task.device)
        signs[self.half_envs :] = -1.0
        task.set_predictive_allocator_candidate_signs(signs)

    @property
    def policy(self):
        return self.positive.policy

    @property
    def optimizer(self):
        return self.positive.optimizer

    @property
    def learning_rate(self) -> float:
        return float(self.positive.learning_rate)

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        self.positive.learning_rate = float(value)
        self.negative.learning_rate = float(value)

    @property
    def gamma(self) -> float:
        return float(self.positive.gamma)

    def act(self, obs):
        # Common random numbers make the candidate comparison causal: both
        # branches receive the same Gaussian action noise. Previously the
        # second branch consumed a different global RNG segment even while the
        # policies were identical at generation start.
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state(self.task.device)
            if torch.cuda.is_available() and str(self.task.device).startswith("cuda")
            else None
        )
        positive_actions = self.positive.act(obs[: self.half_envs])
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state, self.task.device)
        negative_actions = self.negative.act(obs[self.half_envs :])
        return torch.cat((positive_actions, negative_actions), dim=0)

    @staticmethod
    def _branch_extras(extras: dict, start: int, stop: int) -> dict:
        branch = {}
        if "time_outs" in extras:
            branch["time_outs"] = extras["time_outs"][start:stop]
        return branch

    def process_env_step(self, obs, rewards, dones, extras) -> None:
        meta = extras.get("predictive_meta")
        if meta is None:
            raise RuntimeError("Population PPO requires predictive_meta on every environment step.")
        self._step_meta.append(
            {
                "context_contribution": meta["context_contribution"].detach(),
                "allocator_context": meta["allocator_context"].detach(),
                "allocator_mean": meta["allocator_mean"].detach(),
                "allocator_sample": meta["allocator_sample"].detach(),
                "old_log_probability": meta["allocator_log_probability"].detach(),
                "state_allocator_context": meta["state_allocator_context"].detach(),
                "state_allocator_sample": meta["state_allocator_sample"].detach(),
                "state_allocator_log_probability": meta[
                    "state_allocator_log_probability"
                ].detach(),
                "reference_rewards": meta["reference_rewards"].detach(),
                "reward_components": meta["reward_components"].detach(),
                "dones": dones.detach().clone(),
            }
        )
        self.positive.process_env_step(
            obs[: self.half_envs],
            rewards[: self.half_envs],
            dones[: self.half_envs],
            self._branch_extras(extras, 0, self.half_envs),
        )
        self.negative.process_env_step(
            obs[self.half_envs :],
            rewards[self.half_envs :],
            dones[self.half_envs :],
            self._branch_extras(extras, self.half_envs, self.task.num_envs),
        )

    def compute_returns(self, obs) -> None:
        self.positive.compute_returns(obs[: self.half_envs])
        self.negative.compute_returns(obs[self.half_envs :])

    @staticmethod
    def _mean_losses(positive: dict[str, float], negative: dict[str, float]) -> dict[str, float]:
        keys = positive.keys() & negative.keys()
        return {key: 0.5 * (float(positive[key]) + float(negative[key])) for key in keys}

    @staticmethod
    def _clone_branch(source, target) -> None:
        with torch.no_grad():
            target.policy.load_state_dict(source.policy.state_dict())
        target.optimizer.load_state_dict(copy.deepcopy(source.optimizer.state_dict()))
        target.learning_rate = float(source.learning_rate)

    def update(self) -> dict[str, float]:
        if len(self._step_meta) != self.num_steps_per_env:
            raise RuntimeError("Population PPO received incomplete rollout metadata.")
        reference_rewards = torch.stack([step["reference_rewards"] for step in self._step_meta])
        reward_components = torch.stack([step["reward_components"] for step in self._step_meta])
        dones = torch.stack([step["dones"] for step in self._step_meta])
        next_context = torch.stack([step["context_contribution"] for step in self._step_meta]).mean(0)
        state_allocator_context = torch.stack(
            [step["state_allocator_context"] for step in self._step_meta]
        )
        state_allocator_sample = torch.stack(
            [step["state_allocator_sample"] for step in self._step_meta]
        )
        state_allocator_log_probability = torch.stack(
            [step["state_allocator_log_probability"] for step in self._step_meta]
        )
        first = self._step_meta[0]
        for step in self._step_meta[1:]:
            for key in ("allocator_context", "allocator_mean", "allocator_sample", "old_log_probability"):
                if not torch.equal(step[key], first[key]):
                    raise RuntimeError(f"Population allocator {key} changed inside a generation rollout.")

        positive_advantages = self.positive.storage.advantages.detach().clone().squeeze(-1)
        negative_advantages = self.negative.storage.advantages.detach().clone().squeeze(-1)
        positive_score = discounted_rollout_score(
            reference_rewards[:, : self.half_envs],
            dones[:, : self.half_envs].reshape(reference_rewards[:, : self.half_envs].shape),
            float(self.cfg.predictive_allocator_gamma),
        )
        negative_score = discounted_rollout_score(
            reference_rewards[:, self.half_envs :],
            dones[:, self.half_envs :].reshape(reference_rewards[:, self.half_envs :].shape),
            float(self.cfg.predictive_allocator_gamma),
        )
        # Reuse the same minibatch permutation stream for both candidate PPOs.
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state(self.task.device)
            if torch.cuda.is_available() and str(self.task.device).startswith("cuda")
            else None
        )
        positive_loss = self.positive.update()
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state, self.task.device)
        negative_loss = self.negative.update()
        loss_dict = self._mean_losses(positive_loss, negative_loss)
        self._generation.append(
            {
                "next_context": next_context,
                "allocator_context": first["allocator_context"],
                "allocator_sample": first["allocator_sample"],
                "old_log_probability": first["old_log_probability"],
                "positive_reference_rewards": reference_rewards[:, : self.half_envs],
                "negative_reference_rewards": reference_rewards[:, self.half_envs :],
                "positive_components": reward_components[:, : self.half_envs],
                "negative_components": reward_components[:, self.half_envs :],
                "positive_dones": dones[:, : self.half_envs],
                "negative_dones": dones[:, self.half_envs :],
                "positive_advantages": positive_advantages,
                "negative_advantages": negative_advantages,
                "positive_score": positive_score,
                "negative_score": negative_score,
                "positive_allocator_context": state_allocator_context[:, : self.half_envs],
                "negative_allocator_context": state_allocator_context[:, self.half_envs :],
                "positive_allocator_sample": state_allocator_sample[:, : self.half_envs],
                "negative_allocator_sample": state_allocator_sample[:, self.half_envs :],
                "positive_allocator_log_probability": state_allocator_log_probability[
                    :, : self.half_envs
                ],
                "negative_allocator_log_probability": state_allocator_log_probability[
                    :, self.half_envs :
                ],
            }
        )
        self._step_meta.clear()
        loss_dict["reward_allocator/population_generation_progress"] = (
            len(self._generation) / self.generation_rollouts
        )
        if len(self._generation) < self.generation_rollouts:
            return loss_dict

        positive_scores = torch.stack([block["positive_score"] for block in self._generation])
        negative_scores = torch.stack([block["negative_score"] for block in self._generation])
        positive_slope = float(linear_trend(positive_scores).item())
        negative_slope = float(linear_trend(negative_scores).item())
        pair_progress = 0.5 * (positive_slope - negative_slope)
        positive_wins = positive_slope >= negative_slope
        winner_key = "positive" if positive_wins else "negative"
        winner_sign = 1.0 if positive_wins else -1.0
        # Metadata below comes from the selected branch. Its sample direction
        # already changes sign when the negative branch wins, so express credit
        # in that same action frame. The old signed positive-minus-negative gap
        # reversed both signs and moved the allocator toward the loser.
        winner_progress = winner_conditioned_pair_progress(pair_progress, winner_sign)
        winner_normalized_progress = float(
            torch.tanh(
                torch.tensor(
                    winner_progress / float(self.cfg.predictive_reference_progress_scale)
                )
            ).item()
        )
        winner = self.positive if positive_wins else self.negative
        loser = self.negative if positive_wins else self.positive
        self._clone_branch(winner, loser)

        allocator_metrics = self.task.update_predictive_reward_allocator(
            allocator_context=torch.cat(
                [block[f"{winner_key}_allocator_context"] for block in self._generation], dim=0
            ),
            allocator_sample=torch.cat(
                [block[f"{winner_key}_allocator_sample"] for block in self._generation], dim=0
            ),
            old_log_probability=torch.cat(
                [
                    block[f"{winner_key}_allocator_log_probability"]
                    for block in self._generation
                ],
                dim=0,
            ),
            behavior_reference_rewards=torch.cat(
                [block[f"{winner_key}_reference_rewards"] for block in self._generation], dim=0
            ),
            behavior_dones=torch.cat(
                [block[f"{winner_key}_dones"] for block in self._generation], dim=0
            ),
            behavior_reward_components=torch.cat(
                [block[f"{winner_key}_components"] for block in self._generation], dim=0
            ),
            ppo_advantages=torch.cat(
                [block[f"{winner_key}_advantages"] for block in self._generation], dim=0
            ),
            reference_improvement=winner_progress,
            normalized_reference_improvement=winner_normalized_progress,
            allocation_blend=self.task.predictive_allocator_rollout_blend(),
        )
        aggregate_context = torch.stack([block["next_context"] for block in self._generation]).mean(0)
        self.task.begin_predictive_allocator_rollout(aggregate_context, explore=True)
        self._generation.clear()
        allocator_metrics.update(
            {
                "population_positive_slope": positive_slope,
                "population_negative_slope": negative_slope,
                "population_slope_gap": positive_slope - negative_slope,
                "population_positive_wins": float(positive_wins),
                "population_winner_progress": winner_progress,
            }
        )
        loss_dict.update({f"reward_allocator/{key}": value for key, value in allocator_metrics.items()})
        return loss_dict


__all__ = ["AntitheticPopulationPPO"]
