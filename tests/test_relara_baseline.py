"""CPU regression tests for the ICML 2024 ReLara-PPO adapter."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = REPOSITORY_ROOT / "source/yaw_bot/yaw_bot/tasks/direct/yaw_bot/agents"
sys.path.insert(0, str(AGENT_DIR))

from relara import (  # noqa: E402
    ReLaraConfig,
    ReLaraReplayBuffer,
    ReLaraRewardActor,
    ReLaraRewardAgent,
    ReLaraRewardQNetwork,
    relara_policy_reward,
)


class ReLaraBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)

    def test_official_residual_network_shapes_and_bounds(self) -> None:
        actor = ReLaraRewardActor(9)
        critic = ReLaraRewardQNetwork(9)
        context = torch.randn(32, 9)
        proposed, log_probability, deterministic = actor.get_action(context)
        self.assertEqual(proposed.shape, (32, 1))
        self.assertEqual(log_probability.shape, (32, 1))
        self.assertEqual(critic(context, proposed).shape, (32, 1))
        self.assertLessEqual(float(proposed.max()), 1.0)
        self.assertGreaterEqual(float(proposed.min()), -1.0)
        self.assertLessEqual(float(deterministic.max()), 1.0)
        self.assertGreaterEqual(float(deterministic.min()), -1.0)
        self.assertEqual(len(actor.hidden_blocks), 3)
        self.assertEqual(len(critic.hidden_blocks), 3)

    def test_policy_reward_is_official_environment_plus_beta_shaping(self) -> None:
        environment = torch.tensor([1.0, -5.0])
        proposed = torch.tensor([0.5, -0.5])
        actual = relara_policy_reward(environment, proposed, beta=0.2)
        torch.testing.assert_close(actual, torch.tensor([1.1, -5.1]))

    def test_replay_preserves_reward_agent_transition_fields(self) -> None:
        replay = ReLaraReplayBuffer(capacity=8, context_dim=3)
        context = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        next_context = context + 100.0
        proposed = torch.arange(4, dtype=torch.float32)
        environment = proposed + 10.0
        done = torch.tensor([0.0, 1.0, 0.0, 1.0])
        replay.add(context, next_context, proposed, environment, done)
        self.assertEqual(replay.size, 4)
        torch.testing.assert_close(replay.contexts[:4], context)
        torch.testing.assert_close(replay.next_contexts[:4], next_context)
        torch.testing.assert_close(replay.proposed_rewards[:4, 0], proposed)
        torch.testing.assert_close(replay.environment_rewards[:4, 0], environment)
        torch.testing.assert_close(replay.dones[:4, 0], done)

    def test_random_warmup_then_frozen_actor_rollout(self) -> None:
        config = ReLaraConfig(
            replay_capacity=64,
            batch_size=8,
            learning_starts=8,
        )
        agent = ReLaraRewardAgent(5, config, device="cpu", seed=9)
        context = torch.randn(8, 5)
        warmup = agent.propose(context)
        self.assertTrue(torch.all(warmup <= 1.0))
        self.assertTrue(torch.all(warmup >= -1.0))
        agent.collection_samples = 8
        with torch.no_grad():
            for parameter in agent.actor.parameters():
                parameter.add_(1.0)
        # Collection remains on phi_k until the explicit rollout boundary.
        before_activation = copy.deepcopy(agent.rollout_actor.state_dict())
        self.assertTrue(
            all(
                torch.equal(value, before_activation[name])
                for name, value in agent.rollout_actor.state_dict().items()
            )
        )
        agent.freeze_for_rollout()
        self.assertTrue(
            all(
                torch.equal(value, agent.actor.state_dict()[name])
                for name, value in agent.rollout_actor.state_dict().items()
            )
        )
        active = agent.propose(context)
        self.assertEqual(active.shape, (8, 1))
        self.assertTrue(torch.all(active <= 1.0))
        self.assertTrue(torch.all(active >= -1.0))

    def test_reward_agent_update_does_not_touch_policy_parameters(self) -> None:
        config = ReLaraConfig(
            replay_capacity=128,
            batch_size=8,
            learning_starts=8,
        )
        agent = ReLaraRewardAgent(5, config, device="cpu", seed=11)
        policy = torch.nn.Linear(4, 2)
        policy_before = copy.deepcopy(policy.state_dict())
        context = torch.randn(16, 5)
        agent.add_rollout(
            context,
            torch.randn_like(context),
            torch.tanh(torch.randn(16, 1)),
            torch.randn(16, 1),
            torch.zeros(16, 1),
        )
        actor_before = copy.deepcopy(agent.actor.state_dict())
        metrics = agent.optimize(2)
        self.assertEqual(metrics["reward_agent_active"], 1.0)
        self.assertTrue(
            any(
                not torch.equal(value, actor_before[name])
                for name, value in agent.actor.state_dict().items()
            )
        )
        for name, value in policy.state_dict().items():
            torch.testing.assert_close(value, policy_before[name])

    def test_training_state_round_trip_excludes_replay_like_official_source(self) -> None:
        config = ReLaraConfig(
            replay_capacity=32,
            batch_size=4,
            learning_starts=4,
        )
        source = ReLaraRewardAgent(3, config, device="cpu", seed=2)
        source.collection_samples = 15
        source.gradient_steps = 7
        state = source.training_state_dict()
        self.assertNotIn("replay", state)
        restored = ReLaraRewardAgent(3, config, device="cpu", seed=3)
        restored.load_training_state_dict(state)
        self.assertEqual(restored.collection_samples, 15)
        self.assertEqual(restored.gradient_steps, 7)
        for name, value in source.actor.state_dict().items():
            torch.testing.assert_close(value, restored.actor.state_dict()[name])


if __name__ == "__main__":
    unittest.main()
