"""CPU regression tests for Outer-Advantage Guided Reward Composition."""

from __future__ import annotations

import copy
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = REPOSITORY_ROOT / "source/yaw_bot/yaw_bot/tasks/direct/yaw_bot"
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(MODULE_DIR / "agents"))

from official_lirpg import official_lirpg_ppo_update  # noqa: E402
from outer_advantage_composer import (  # noqa: E402
    OUTER_CHECKPOINT_REQUIRED_KEYS,
    CenteredTanhComposer,
    LIRPGIntrinsicReward,
    LIRPGOuterCritic,
    OuterCritic,
    RunningGroupRMS,
    beta_schedule,
    compose_actor_reward,
    composer_alignment_loss,
    composer_meta_gradient_loss,
    cross_split_credit_reliability,
    differentiable_mixed_gae,
    differentiable_shaping_advantage,
    effective_composer_weights,
    lirpg_actor_reward,
    lirpg_meta_gradient_loss,
    official_lirpg_meta_gradient_loss,
    predictor_ready,
    resolve_beta_schedule_horizon,
    select_actor_reward,
    static_group_weight_tensor,
    validate_outer_checkpoint_state,
    validate_static_group_weights,
)


class OuterAdvantageComposerTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def _meta_batch(self):
        time_steps, environments = 5, 6
        actor = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.ELU(),
            torch.nn.Linear(8, 2),
        )
        actor_log_std = torch.nn.Parameter(torch.full((2,), math.log(0.7)))
        actor_optimizer = torch.optim.Adam(
            [*actor.parameters(), actor_log_std], lr=1.0e-3
        )
        observations = torch.randn(time_steps, environments, 4)
        standard_deviations = torch.full((time_steps, environments, 2), 0.7)
        with torch.no_grad():
            raw_mean = actor(observations)
            mean = 4.0 * torch.tanh(raw_mean / 4.0)
            latent_actions = mean + 0.1 * torch.randn_like(mean)
            actions = torch.tanh(latent_actions)
            log_det = torch.log(1.0 - torch.square(actions) + 1.0e-6)
            old_log_probability = (
                torch.distributions.Normal(mean, standard_deviations).log_prob(
                    latent_actions
                )
                - log_det
            ).sum(dim=-1)
        return {
            "actor": actor,
            "actor_log_std": actor_log_std,
            "actor_optimizer": actor_optimizer,
            "observations": observations,
            "actions": actions,
            "old_log_probability": old_log_probability,
            "standard_deviations": standard_deviations,
            "latents": torch.randn(time_steps, environments, 16),
            "fixed_internal": torch.randn(time_steps, environments),
            "groups": torch.randn(time_steps, environments, 5),
            "outer_advantages": torch.randn(time_steps, environments),
            "dones": torch.zeros(time_steps, environments),
        }

    def _meta_loss(
        self,
        composer: CenteredTanhComposer,
        batch: dict[str, object],
        outer_advantages: torch.Tensor | None = None,
    ):
        return composer_meta_gradient_loss(
            composer,
            batch["actor"],
            batch["actor_optimizer"],
            batch["latents"],
            batch["fixed_internal"],
            batch["groups"],
            batch["outer_advantages"]
            if outer_advantages is None
            else outer_advantages,
            batch["observations"],
            batch["actions"],
            batch["old_log_probability"],
            batch["standard_deviations"],
            batch["dones"],
        )

    def test_beta_zero_is_numerically_uniform_internal_reward(self) -> None:
        groups = torch.randn(32, 5)
        weights = torch.rand(32, 5) * 0.8 + 0.6
        fixed = torch.randn(32)
        actual = compose_actor_reward(fixed, groups, weights, beta=0.0)
        torch.testing.assert_close(
            actual, fixed + groups.mean(dim=-1), rtol=0.0, atol=0.0
        )

    def test_outer_only_ablation_returns_original_outer_reward_exactly(self) -> None:
        outer = torch.randn(32)
        groups = torch.randn(32, 5)
        weights = torch.rand(32, 5) * 0.8 + 0.6
        actual = select_actor_reward(
            outer,
            torch.randn(32),
            groups,
            weights,
            beta=1.0,
            outer_only=True,
        )
        self.assertIs(actual, outer)
        torch.testing.assert_close(actual, outer, rtol=0.0, atol=0.0)

    def test_fixed_internal_safety_terms_penalize_termination(self) -> None:
        groups = torch.zeros(2, 5)
        weights = torch.ones(2, 5)
        fixed = torch.tensor([1.0, -15.0])
        actual = compose_actor_reward(fixed, groups, weights, beta=1.0)
        torch.testing.assert_close(actual, fixed, rtol=0.0, atol=0.0)
        self.assertGreater(float(actual[0]), float(actual[1]))

    def test_resume_inside_saved_beta_horizon_preserves_schedule(self) -> None:
        self.assertEqual(resolve_beta_schedule_horizon(800, 400, 200), 800)

    def test_beta_warms_to_one_and_never_decays(self) -> None:
        self.assertEqual(beta_schedule(0, 100), 0.0)
        self.assertAlmostEqual(beta_schedule(10, 100), 10.0 / 19.8)
        self.assertEqual(beta_schedule(20, 100), 1.0)
        self.assertEqual(beta_schedule(99, 100), 1.0)

    def test_resume_past_saved_beta_horizon_extends_schedule(self) -> None:
        self.assertEqual(resolve_beta_schedule_horizon(800, 800, 700), 1500)

    def test_shaping_scale_is_invariant_to_group_count(self) -> None:
        groups = torch.ones(3, 5)
        weights = torch.ones(3, 5)
        fixed = torch.full((3,), 2.0)
        actual = compose_actor_reward(fixed, groups, weights, beta=0.5)
        torch.testing.assert_close(actual, torch.full((3,), 3.0))

    def test_beta_interpolates_weights_without_scaling_reward_channel(self) -> None:
        weights = torch.tensor([[1.4, 1.2, 1.0, 0.8, 0.6]])
        actual = effective_composer_weights(weights, beta=0.5)
        torch.testing.assert_close(
            actual, torch.tensor([[1.2, 1.1, 1.0, 0.9, 0.8]])
        )

    def test_weights_initialize_at_one_and_remain_mean_one_and_bounded(self) -> None:
        composer = CenteredTanhComposer(16)
        latent = torch.randn(128, 16)
        initialized = composer(latent)
        torch.testing.assert_close(initialized, torch.ones_like(initialized), rtol=0.0, atol=0.0)
        with torch.no_grad():
            for parameter in composer.parameters():
                parameter.normal_(0.0, 3.0)
        weights = composer(latent)
        torch.testing.assert_close(
            weights.mean(dim=-1), torch.ones(128), rtol=1.0e-6, atol=1.0e-6
        )
        self.assertGreaterEqual(float(weights.min()), 0.6 - 1.0e-6)
        self.assertLessEqual(float(weights.max()), 1.4 + 1.0e-6)

    def test_a1_uniform_and_a2_static_weights_use_same_group_shape(self) -> None:
        groups = torch.randn(7, 11, 5)
        uniform = static_group_weight_tensor((1, 1, 1, 1, 1), groups)
        static = static_group_weight_tensor((1.2, 0.8, 1.4, 0.7, 0.9), groups)
        self.assertEqual(uniform.shape, groups.shape)
        torch.testing.assert_close(uniform, torch.ones_like(groups))
        torch.testing.assert_close(
            static[0, 0], torch.tensor([1.2, 0.8, 1.4, 0.7, 0.9])
        )
        self.assertEqual(
            validate_static_group_weights((1, 1, 1, 1, 1)),
            (1.0, 1.0, 1.0, 1.0, 1.0),
        )
        with self.assertRaises(ValueError):
            validate_static_group_weights((1.0, 1.0))
        with self.assertRaises(ValueError):
            validate_static_group_weights((1.4, 1.4, 1.4, 1.4, 1.4))

    def test_lirpg_reward_network_matches_bounded_observation_action_principle(self) -> None:
        reward_model = LIRPGIntrinsicReward(4, 2)
        observations = torch.randn(5, 6, 4, requires_grad=True)
        actions = torch.randn(5, 6, 2, requires_grad=True)
        intrinsic = reward_model(observations, actions)
        self.assertEqual(intrinsic.shape, (5, 6))
        self.assertLessEqual(float(intrinsic.max()), 1.0)
        self.assertGreaterEqual(float(intrinsic.min()), -1.0)
        intrinsic.sum().backward()
        self.assertIsNone(observations.grad)
        self.assertIsNone(actions.grad)

    def test_lirpg_actor_reward_matches_official_extrinsic_mixture(self) -> None:
        extrinsic = torch.tensor([1.0, -15.0])
        intrinsic = torch.tensor([0.25, 0.25])
        actual = lirpg_actor_reward(extrinsic, intrinsic)
        torch.testing.assert_close(actual, torch.tensor([0.26, 0.10]))

    def test_official_lirpg_mixed_gae_matches_discounted_return_without_value(self) -> None:
        rewards = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        advantages = differentiable_mixed_gae(
            rewards,
            torch.zeros_like(rewards),
            torch.zeros(2),
            torch.zeros_like(rewards),
            gamma=1.0,
            lam=1.0,
        )
        torch.testing.assert_close(
            advantages,
            torch.tensor([[9.0, 12.0], [8.0, 10.0], [5.0, 6.0]]),
        )

    def test_lirpg_outer_advantage_changes_only_reward_model_gradient(self) -> None:
        batch = self._meta_batch()
        reward_model = LIRPGIntrinsicReward(4, 2)
        actor_before = copy.deepcopy(batch["actor"].state_dict())

        def gradient(outer_advantages: torch.Tensor) -> torch.Tensor:
            loss, _ = official_lirpg_meta_gradient_loss(
                reward_model,
                batch["actor"],
                batch["actor_log_std"],
                batch["actor_optimizer"],
                batch["fixed_internal"],
                torch.zeros_like(batch["fixed_internal"]),
                torch.zeros(batch["fixed_internal"].shape[1]),
                outer_advantages,
                batch["observations"],
                batch["actions"],
                batch["old_log_probability"],
                batch["dones"],
                torch.arange(batch["fixed_internal"].numel()),
            )
            gradients = torch.autograd.grad(loss, tuple(reward_model.parameters()))
            return torch.cat([value.reshape(-1) for value in gradients])

        outer = batch["outer_advantages"]
        first = gradient(outer)
        second = gradient(torch.flip(outer, dims=(0,)))
        self.assertGreater(float(torch.linalg.vector_norm(first - second)), 1.0e-8)
        self.assertEqual(actor_before.keys(), batch["actor"].state_dict().keys())
        for name, value in actor_before.items():
            torch.testing.assert_close(value, batch["actor"].state_dict()[name])
        self.assertTrue(all(parameter.grad is None for parameter in batch["actor"].parameters()))

    def test_official_lirpg_updates_reward_on_every_minibatch(self) -> None:
        from tensordict import TensorDict

        time_steps, environments, observation_dim, action_dim = 3, 4, 4, 2

        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.actor = torch.nn.Sequential(
                    torch.nn.Linear(observation_dim, 8),
                    torch.nn.Tanh(),
                    torch.nn.Linear(8, action_dim),
                )
                self.critic = torch.nn.Sequential(
                    torch.nn.Linear(observation_dim, 8),
                    torch.nn.Tanh(),
                    torch.nn.Linear(8, 1),
                )
                self.log_std = torch.nn.Parameter(torch.full((action_dim,), -0.3))
                self.is_recurrent = False
                self.distribution = None

            def act(self, observations):
                raw = self.actor(observations["policy"])
                mean = 4.0 * torch.tanh(raw / 4.0)
                std = self.log_std.exp().expand_as(mean)
                self.distribution = torch.distributions.Normal(mean, std)
                return torch.tanh(self.distribution.rsample())

            def get_actions_log_prob(self, actions):
                latent = torch.atanh(actions.clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6))
                return (
                    self.distribution.log_prob(latent)
                    - torch.log(1.0 - torch.square(actions) + 1.0e-6)
                ).sum(dim=-1)

            def evaluate(self, observations):
                return self.critic(observations["critic"])

            @property
            def action_mean(self):
                return self.distribution.mean

            @property
            def action_std(self):
                return self.distribution.stddev

            @property
            def entropy(self):
                return self.distribution.entropy().sum(dim=-1)

        observations = torch.randn(
            time_steps, environments, observation_dim
        )
        observation_dict = TensorDict(
            {"policy": observations, "critic": observations.clone()},
            batch_size=[time_steps, environments],
        )
        policy = Policy()
        with torch.no_grad():
            raw_mean = policy.actor(observations)
            mean = 4.0 * torch.tanh(raw_mean / 4.0)
            sigma = policy.log_std.exp().expand_as(mean)
            latent_actions = mean + 0.1 * torch.randn_like(mean)
            actions = torch.tanh(latent_actions)
            old_log_probability = (
                torch.distributions.Normal(mean, sigma).log_prob(latent_actions)
                - torch.log(1.0 - torch.square(actions) + 1.0e-6)
            ).sum(dim=-1, keepdim=True)

        class Storage:
            def __init__(self):
                self.num_transitions_per_env = time_steps
                self.num_envs = environments
                self.step = time_steps
                self.observations = observation_dict
                self.actions = actions
                self.values = torch.zeros(time_steps, environments, 1)
                self.returns = torch.randn(time_steps, environments, 1)
                self.actions_log_prob = old_log_probability
                self.mu = mean
                self.sigma = sigma

            def clear(self):
                self.step = 0

        class Algorithm:
            def __init__(self):
                self.device = torch.device("cpu")
                self.policy = policy
                self.storage = Storage()
                self.optimizer = torch.optim.Adam(
                    policy.parameters(), lr=3.0e-4
                )
                self.num_learning_epochs = 2
                self.num_mini_batches = 2
                self.gamma = 0.99
                self.lam = 0.95
                self.clip_param = 0.2
                self.desired_kl = None
                self.schedule = "fixed"
                self.learning_rate = 3.0e-4
                self.use_clipped_value_loss = True
                self.value_loss_coef = 0.5
                self.entropy_coef = 0.0
                self.max_grad_norm = 0.5
                self.is_multi_gpu = False
                self.rnd = None
                self.symmetry = None

        algorithm = Algorithm()
        reward_model = LIRPGIntrinsicReward(observation_dim, action_dim)
        reward_optimizer = torch.optim.Adam(
            reward_model.parameters(), lr=1.0e-4, eps=1.0e-5
        )
        outer_critic = LIRPGOuterCritic(observation_dim)
        outer_critic_optimizer = torch.optim.Adam(
            outer_critic.parameters(), lr=1.0e-4, eps=1.0e-5
        )
        reward_before = copy.deepcopy(reward_model.state_dict())
        outer_critic_before = copy.deepcopy(outer_critic.state_dict())
        policy_before = copy.deepcopy(policy.state_dict())
        ppo_metrics, meta_metrics, update_count = official_lirpg_ppo_update(
            algorithm,
            reward_model=reward_model,
            reward_optimizer=reward_optimizer,
            outer_critic=outer_critic,
            outer_critic_optimizer=outer_critic_optimizer,
            extrinsic_rewards=torch.randn(time_steps, environments),
            outer_advantages=torch.randn(time_steps, environments),
            outer_returns=torch.randn(time_steps, environments),
            outer_values=torch.zeros(time_steps, environments),
            critic_observations=observations,
            actor_observations=observations,
            actions=actions,
            old_action_log_probabilities=old_log_probability,
            dones=torch.zeros(time_steps, environments),
            final_mixed_value=torch.zeros(environments, 1),
            actor_maximum_latent_mean=4.0,
        )
        self.assertEqual(update_count, 4)
        self.assertEqual(algorithm.storage.step, 0)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in ppo_metrics.values()))
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in meta_metrics.values()))
        self.assertTrue(
            any(
                not torch.equal(value, reward_before[name])
                for name, value in reward_model.state_dict().items()
            )
        )
        self.assertTrue(
            any(
                not torch.equal(value, policy_before[name])
                for name, value in policy.state_dict().items()
            )
        )
        self.assertTrue(
            any(
                not torch.equal(value, outer_critic_before[name])
                for name, value in outer_critic.state_dict().items()
            )
        )

    def test_composer_update_does_not_update_actor_or_predictor_encoder(self) -> None:
        batch = self._meta_batch()
        actor = batch["actor"]
        encoder = torch.nn.Linear(8, 16)
        composer = CenteredTanhComposer(16)
        actor_before = copy.deepcopy(actor.state_dict())
        encoder_before = copy.deepcopy(encoder.state_dict())
        batch["latents"] = encoder(torch.randn(5, 6, 8))
        loss, _ = self._meta_loss(composer, batch)
        optimizer = torch.optim.Adam(composer.parameters(), lr=1.0e-3)
        optimizer.zero_grad(set_to_none=True)
        gradients = torch.autograd.grad(loss, tuple(composer.parameters()))
        for parameter, gradient in zip(
            composer.parameters(), gradients, strict=True
        ):
            parameter.grad = gradient
        optimizer.step()
        self.assertTrue(all(parameter.grad is None for parameter in actor.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in encoder.parameters()))
        for name, value in actor.state_dict().items():
            torch.testing.assert_close(value, actor_before[name])
        for name, value in encoder.state_dict().items():
            torch.testing.assert_close(value, encoder_before[name])

    def test_outer_advantage_changes_composer_gradient(self) -> None:
        composer = CenteredTanhComposer(16)
        batch = self._meta_batch()

        def gradient(outer: torch.Tensor) -> torch.Tensor:
            composer.zero_grad(set_to_none=True)
            loss, _ = self._meta_loss(composer, batch, outer)
            gradients = torch.autograd.grad(loss, tuple(composer.parameters()))
            return torch.cat([gradient.reshape(-1) for gradient in gradients]).clone()

        outer = torch.randn(5, 6)
        first = gradient(outer)
        second = gradient(torch.flip(outer, dims=(0,)))
        self.assertGreater(float(torch.linalg.vector_norm(first)), 1.0e-7)
        self.assertGreater(float(torch.linalg.vector_norm(first - second)), 1.0e-7)

    def test_meta_loss_predicts_outer_effect_after_virtual_actor_update(self) -> None:
        composer = CenteredTanhComposer(16)
        batch = self._meta_batch()
        actor_before = copy.deepcopy(batch["actor"].state_dict())
        optimizer_before = copy.deepcopy(batch["actor_optimizer"].state_dict())
        loss, metrics = self._meta_loss(composer, batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(metrics["meta_outer_loss_after"]))
        torch.testing.assert_close(
            metrics["predicted_outer_improvement"],
            metrics["meta_outer_loss_before"] - metrics["meta_outer_loss_after"],
        )
        for name, value in batch["actor"].state_dict().items():
            torch.testing.assert_close(value, actor_before[name])
        self.assertEqual(batch["actor_optimizer"].state_dict(), optimizer_before)

    def test_meta_gradient_remains_active_independently_of_deployment_beta(self) -> None:
        composer = CenteredTanhComposer(16)
        batch = self._meta_batch()
        loss, _ = composer_meta_gradient_loss(
            composer,
            batch["actor"],
            batch["actor_optimizer"],
            batch["latents"],
            batch["fixed_internal"],
            batch["groups"],
            batch["outer_advantages"],
            batch["observations"],
            batch["actions"],
            batch["old_log_probability"],
            batch["standard_deviations"],
            batch["dones"],
            weight_to_one=0.0,
            temporal_smoothness=0.0,
        )
        gradients = torch.autograd.grad(loss, tuple(composer.parameters()))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0.0)

    def test_cross_split_credit_accepts_reproducible_direction(self) -> None:
        groups = torch.randn(12, 64, 5)
        outer = 1.5 * groups[..., 0] - 0.75 * groups[..., 3] + 0.05 * torch.randn(12, 64)
        credit = cross_split_credit_reliability(groups, outer)
        self.assertGreater(float(credit["direction_cosine"]), 0.95)
        self.assertGreater(float(credit["reliability"]), 0.9)

    def test_cross_split_credit_rejects_conflicting_direction(self) -> None:
        groups = torch.randn(12, 64, 5)
        outer = torch.empty(12, 64)
        outer[:, 0::2] = groups[:, 0::2, 0]
        outer[:, 1::2] = -groups[:, 1::2, 0]
        credit = cross_split_credit_reliability(groups, outer)
        self.assertLess(float(credit["direction_cosine"]), -0.9)
        self.assertEqual(float(credit["reliability"]), 0.0)

    def test_zero_credit_disables_alignment_gradient(self) -> None:
        composer = CenteredTanhComposer(16)
        latent = torch.randn(8, 7, 16)
        with torch.no_grad():
            for parameter in composer.parameters():
                parameter.normal_(0.0, 0.1)
        loss, metrics = composer_alignment_loss(
            composer,
            latent,
            torch.randn(8, 7, 5),
            torch.randn(8, 7),
            torch.zeros(8, 7),
            weight_to_one=0.0,
            temporal_smoothness=0.0,
            credit_weight=0.0,
        )
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(float(metrics["credit_weight"]), 0.0)

    def test_state_dependent_weights_are_applied_before_temporal_credit(self) -> None:
        rewards = torch.tensor(
            [
                [[[2.0], [10.0]]],
                [[[3.0], [20.0]]],
            ]
        ).squeeze(1)
        weights = torch.tensor(
            [
                [[[0.5], [0.1]]],
                [[[2.0], [0.2]]],
            ]
        ).squeeze(1)
        dones = torch.zeros(2, 2)
        actual = differentiable_shaping_advantage(
            torch.zeros(2, 2),
            rewards,
            weights,
            dones,
            gamma=0.9,
            lam=0.8,
        )
        immediate = torch.tensor([[1.0, 1.0], [6.0, 4.0]])
        expected = torch.stack((immediate[0] + 0.9 * 0.8 * immediate[1], immediate[1]))
        torch.testing.assert_close(actual, expected)

    def test_updated_phi_cannot_change_cached_rollout_reward(self) -> None:
        composer = CenteredTanhComposer(16)
        optimizer = torch.optim.Adam(composer.parameters(), lr=1.0e-2)
        batch = self._meta_batch()
        latent = batch["latents"]
        groups = batch["groups"]
        with torch.no_grad():
            cached = compose_actor_reward(
                batch["fixed_internal"], groups, composer(latent), beta=0.5
            ).clone()
        loss, _ = self._meta_loss(composer, batch)
        optimizer.zero_grad(set_to_none=True)
        gradients = torch.autograd.grad(loss, tuple(composer.parameters()))
        for parameter, gradient in zip(
            composer.parameters(), gradients, strict=True
        ):
            parameter.grad = gradient
        optimizer.step()
        torch.testing.assert_close(cached, cached.clone(), rtol=0.0, atol=0.0)
        next_rollout = compose_actor_reward(
            batch["fixed_internal"], groups, composer(latent), beta=0.5
        )
        self.assertFalse(torch.equal(cached, next_rollout))

    def test_predictor_becomes_ready_after_twelve_steps(self) -> None:
        ages = torch.tensor([0, 11, 12, 13])
        expected = torch.tensor([False, False, True, True])
        torch.testing.assert_close(predictor_ready(ages, horizon=12), expected)

    def test_checkpoint_round_trip_contains_all_training_state(self) -> None:
        composer = CenteredTanhComposer(16)
        rollout_composer = copy.deepcopy(composer)
        critic = OuterCritic(12)
        rms = RunningGroupRMS()
        rms.update(torch.randn(32, 5))
        composer_optimizer = torch.optim.Adam(composer.parameters(), lr=3.0e-4)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1.0e-3)
        payload = {
            "outer_composer_state_dict": composer.state_dict(),
            "outer_rollout_composer_state_dict": rollout_composer.state_dict(),
            "outer_critic_state_dict": critic.state_dict(),
            "outer_group_rms_state_dict": rms.state_dict(),
            "outer_composer_optimizer_state_dict": composer_optimizer.state_dict(),
            "outer_critic_optimizer_state_dict": critic_optimizer.state_dict(),
            "outer_beta_iteration": 37,
            "outer_beta_total_iterations": 800,
            "outer_beta": 0.5,
            "outer_composer_updates": 37,
            "outer_critic_updates": 37,
            "outer_learning_auc": 12.5,
        }
        validate_outer_checkpoint_state(payload)
        self.assertEqual(set(OUTER_CHECKPOINT_REQUIRED_KEYS), set(payload))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "outer.pt"
            torch.save(payload, path)
            restored = torch.load(path, weights_only=False)
        validate_outer_checkpoint_state(restored)
        restored_rms = RunningGroupRMS()
        restored_rms.load_state_dict(restored["outer_group_rms_state_dict"])
        torch.testing.assert_close(restored_rms.mean_square, rms.mean_square)
        self.assertEqual(restored["outer_beta_iteration"], 37)
        self.assertEqual(restored["outer_beta_total_iterations"], 800)


if __name__ == "__main__":
    unittest.main()
