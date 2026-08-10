"""Contract tests linking the public implementation to the manuscript."""

from __future__ import annotations

import ast
import csv
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "source/yaw_bot/yaw_bot/tasks/direct/yaw_bot"
sys.path.insert(0, str(MODULE_DIR))

from outer_advantage_composer import (  # noqa: E402
    REWARD_GROUP_INDICES,
    REWARD_GROUP_NAMES,
    CenteredTanhComposer,
    OuterCritic,
    RunningGroupRMS,
    outer_reward,
)
from pose_predictor import DepthPosePredictor  # noqa: E402
from predictive_feasibility import PredictiveFeasibilityModel  # noqa: E402
from predictive_labels import DIRECT_REWARD_NAMES  # noqa: E402


class PaperCorrespondenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = json.loads((ROOT / "configs/paper_experiments.json").read_text(encoding="utf-8"))

    def test_twenty_two_rewards_have_one_group_each(self) -> None:
        self.assertEqual(len(DIRECT_REWARD_NAMES), 22)
        self.assertEqual(
            REWARD_GROUP_NAMES,
            ("stability", "contact_slip", "linear", "yaw", "regularization"),
        )
        flattened = [index for group in REWARD_GROUP_INDICES for index in group]
        self.assertEqual(sorted(flattened), list(range(22)))
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_group_semantics_match_the_manuscript(self) -> None:
        grouped_names = [{DIRECT_REWARD_NAMES[index] for index in indices} for indices in REWARD_GROUP_INDICES]
        self.assertEqual(
            grouped_names[0],
            {
                "angle_penalty",
                "roll_pitch_rate_penalty",
                "projected_gravity",
                "vertical_velocity_penalty",
            },
        )
        self.assertEqual(
            grouped_names[1],
            {"wheel_contact", "wheel_air_penalty"},
        )
        self.assertEqual(
            grouped_names[3],
            {
                "yaw_rate_penalty",
                "wheel_yaw_tracking",
                "yaw_direction",
                "body_yaw_tracking",
            },
        )
        self.assertEqual(
            grouped_names[4],
            {
                "action_rate_penalty",
                "action_magnitude_penalty",
                "stillness_penalty",
                "servo_motion_penalty",
                "stop_motion_penalty",
            },
        )

    def test_composer_is_128_128_64_5_and_exactly_bounded_mean_one(self) -> None:
        composer = CenteredTanhComposer(128)
        linear_layers = [layer for layer in composer.network if isinstance(layer, torch.nn.Linear)]
        self.assertEqual(
            [(layer.in_features, layer.out_features) for layer in linear_layers],
            [(128, 128), (128, 64), (64, 5)],
        )
        with torch.no_grad():
            for parameter in composer.parameters():
                parameter.normal_(mean=0.0, std=4.0)
        weights = composer(torch.randn(256, 128))
        torch.testing.assert_close(
            weights.mean(dim=-1),
            torch.ones(256),
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        self.assertGreaterEqual(float(weights.min()), 0.6 - 1.0e-6)
        self.assertLessEqual(float(weights.max()), 1.4 + 1.0e-6)

    def test_predictive_network_dimensions_match_supplement_table_one(self) -> None:
        model = PredictiveFeasibilityModel(
            history_steps=4,
            depth_height=54,
            depth_width=96,
            state_dim=28,
            action_dim=6,
            event_dim=4,
            latent_dim=32,
            future_steps=12,
            future_state_dim=10,
            reward_dim=22,
            ensemble_size=3,
            fusion_hidden_dims=(128, 128),
        )
        self.assertEqual(model.online_depth_encoder.latent_dim, 32)
        self.assertEqual(model.state_encoder[2].out_features, 64)
        self.assertEqual(model.action_encoder[2].out_features, 32)
        self.assertEqual(model.fusion[-2].out_features, 128)
        self.assertEqual(len(model.event_heads), 3)
        self.assertEqual(model.future_head[-1].out_features, 120)

        pose = DepthPosePredictor(54, 96, 25, 5, 5, (256, 128))
        self.assertEqual(pose.head[-1].out_features, 50)

    def test_outer_critic_and_fixed_task_objective_match_equation_one(self) -> None:
        critic = OuterCritic(75)
        linear_layers = [layer for layer in critic.value if isinstance(layer, torch.nn.Linear)]
        self.assertEqual(
            [(layer.in_features, layer.out_features) for layer in linear_layers],
            [(75, 256), (256, 128), (128, 1)],
        )
        linear = torch.tensor([0.8, 0.2])
        yaw = torch.tensor([0.4, 0.6])
        terminated = torch.tensor([False, True])
        actions = torch.tensor([[1.0] * 6, [0.5] * 6])
        expected = torch.tensor([1.59, -4.6025])
        torch.testing.assert_close(
            outer_reward(linear, yaw, terminated, actions),
            expected,
        )

    def test_rms_uses_decay_point_999_without_centering(self) -> None:
        rms = RunningGroupRMS(decay=0.999)
        values = torch.tensor([[2.0, -3.0, 4.0, -5.0, 6.0]])
        normalized = rms.normalize(values)
        torch.testing.assert_close(normalized, values)
        torch.testing.assert_close(
            rms.mean_square,
            torch.tensor([4.0, 9.0, 16.0, 25.0, 36.0]),
        )

    def test_protocol_matches_paper_training_and_evaluation_counts(self) -> None:
        training = self.protocol["training"]
        self.assertEqual(training["num_envs"], 512)
        self.assertEqual(training["rollout_steps"], 24)
        self.assertEqual(training["iterations"], 1500)
        self.assertEqual(training["physics_hz"], 120)
        self.assertEqual(training["control_hz"], 60)
        self.assertEqual(self.protocol["selection"]["trailing_window"], 100)
        self.assertEqual(self.protocol["evaluation"]["seeds"], [211, 3301, 7907])
        self.assertEqual(
            self.protocol["evaluation"]["action_saturation_threshold"],
            0.98,
        )

    def test_static_task_default_is_the_paper_vector(self) -> None:
        config_path = MODULE_DIR / "yaw_bot_env_cfg.py"
        tree = ast.parse(config_path.read_text(encoding="utf-8"))
        value = None
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "YawBotStaticRewardPPOEnvCfg":
                for statement in node.body:
                    if isinstance(statement, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id == "outer_static_group_weights"
                        for target in statement.targets
                    ):
                        value = ast.literal_eval(statement.value)
        self.assertEqual(value, (1.2, 0.8, 1.4, 0.7, 0.9))

    def test_exactly_six_paper_tasks_are_public(self) -> None:
        registration = (MODULE_DIR / "__init__.py").read_text(encoding="utf-8")
        registered_ids = [
            node.value
            for node in ast.walk(ast.parse(registration))
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.startswith("Template-Yaw-Bot-")
        ]
        expected = {
            self.protocol[method]["task"] for method in ("yaw", "outer", "uniform", "static", "lirpg", "relara")
        }
        self.assertEqual(set(registered_ids), expected)
        self.assertEqual(len(registered_ids), 6)

    def test_machine_readable_reward_table_matches_source_names_and_scales(self) -> None:
        reward_table = json.loads((ROOT / "configs/reward_terms.json").read_text(encoding="utf-8"))
        terms = reward_table["atomic_terms"]
        self.assertEqual([term["index"] for term in terms], list(range(22)))
        self.assertEqual([term["name"] for term in terms], list(DIRECT_REWARD_NAMES))
        group_by_index = {
            index: group_name
            for group_name, indices in zip(REWARD_GROUP_NAMES, REWARD_GROUP_INDICES, strict=True)
            for index in indices
        }
        self.assertEqual(
            [term["group"] for term in terms],
            [group_by_index[index] for index in range(22)],
        )
        source = (MODULE_DIR / "yaw_bot_env_cfg.py").read_text(encoding="utf-8")
        expected_scale_literals = {
            "angle_penalty": "rew_scale_angle = -0.2",
            "roll_pitch_rate_penalty": "rew_scale_ang_vel = -0.03",
            "projected_gravity": "rew_scale_projected_gravity = 0.1",
            "yaw_rate_penalty": "rew_scale_yaw_ang_vel = -0.05",
            "action_rate_penalty": "rew_scale_joint_action_rate = -1.0e-4",
            "action_magnitude_penalty": "rew_scale_action_magnitude = -0.02",
            "stillness_penalty": "rew_scale_pre_stage3_still = -2.0",
            "servo_motion_penalty": "rew_scale_pre_stage3_servo_motion = -0.02",
            "vertical_velocity_penalty": "rew_scale_vertical_vel = -1.0",
            "wheel_contact": "rew_scale_wheel_contact = 1.0",
            "wheel_air_penalty": "rew_scale_wheel_air = -2.5",
            "wheel_yaw_tracking": "rew_scale_track_wheel_yaw = 2.0",
            "wheel_linear_tracking": "rew_scale_track_wheel_lin = 4.0",
            "body_linear_progress": "rew_scale_forward_vel = 8.0",
            "wrong_direction_penalty": "rew_scale_backward_vel = -6.0",
            "wheel_linear_progress": "rew_scale_forward_progress = 3.0",
            "command_direction": "rew_scale_direction = 3.0",
            "yaw_direction": "rew_scale_yaw_direction = 1.0",
            "body_yaw_tracking": "rew_scale_track_yaw_vel = 6.0",
            "stop_motion_penalty": "rew_scale_command_stop_motion = -2.0",
            "body_linear_tracking": "rew_scale_track_lin_vel = 8.0",
            "planar_position_penalty": "rew_scale_planar_position_error = -2.0",
        }
        self.assertEqual(set(expected_scale_literals), set(DIRECT_REWARD_NAMES))
        for literal in expected_scale_literals.values():
            self.assertIn(literal, source)

    def test_machine_readable_task_interface_matches_source_configuration(self) -> None:
        interface = json.loads((ROOT / "configs/task_interface.json").read_text(encoding="utf-8"))
        self.assertEqual(interface["timing"]["physics_hz"], 120)
        self.assertEqual(interface["timing"]["action_decimation"], 2)
        self.assertEqual(interface["timing"]["control_hz"], 60)
        self.assertEqual(interface["actions"]["dimension"], 6)
        self.assertEqual(interface["policy_observation"]["dimension"], 75)
        self.assertEqual(
            interface["commands"]["linear_velocity_range"],
            [-1.0, 1.0],
        )
        self.assertEqual(interface["depth_camera"]["height"], 54)
        self.assertEqual(interface["depth_camera"]["width"], 96)
        self.assertEqual(
            interface["robot_material_startup_randomization"]["static_friction_range"],
            [0.8, 1.6],
        )
        self.assertEqual(
            interface["parallel_leg_kinematics"]["nominal_knee_degrees"],
            87.176,
        )

        source = (MODULE_DIR / "yaw_bot_env_cfg.py").read_text(encoding="utf-8")
        for literal in (
            "decimation = 2",
            "action_space = 6",
            '"policy": 25 + pose_prediction_dim',
            "command_lin_vel_x_range = (-1.0, 1.0)",
            "branch_hip_action_scale = 0.35",
            "mapped_hip_action_scale = 0.35",
            "wheel_action_scale = 10.0",
            "static_friction=1.4",
            "dynamic_friction=1.2",
            '"static_friction_range": (0.8, 1.6)',
            '"dynamic_friction_range": (0.7, 1.3)',
        ):
            self.assertIn(literal, source)

    def test_environment_manifest_contains_supplement_hardware(self) -> None:
        environment = json.loads((ROOT / "configs/environment.json").read_text(encoding="utf-8"))
        self.assertEqual(environment["hardware"]["cpu"], "AMD Ryzen 9 9950X")
        self.assertEqual(environment["hardware"]["memory_gb"], 64)
        self.assertEqual(environment["hardware"]["gpus"]["count"], 2)
        self.assertEqual(
            environment["hardware"]["gpus"]["model"],
            "NVIDIA GeForce RTX 5080",
        )
        self.assertEqual(environment["software"]["isaac_sim"], "5.1")
        self.assertEqual(environment["software"]["rsl_rl_lib"], "3.1.2")

    def test_anonymized_per_seed_data_recomputes_submitted_aggregate(self) -> None:
        metric_names = (
            "Evaluation/fixed_outer_reward",
            "mean_episode_length",
            "Evaluation/command_success",
            "Evaluation/termination_rate",
            "Diagnostics/lin_cmd_success_rate",
            "Diagnostics/yaw_cmd_signed_success_rate",
            "Evaluation/action_saturation_rate",
            "Diagnostics/planar_position_error",
        )
        with (ROOT / "results/evaluation_aggregate.csv").open(newline="", encoding="utf-8") as stream:
            aggregate_rows = {row["method"]: row for row in csv.DictReader(stream)}
        for method in ("yaw", "outer", "uniform", "static", "lirpg", "relara"):
            label = self.protocol[method]["label"]
            payloads = [
                json.loads((ROOT / "results/evaluation_json" / f"{method}_seed{seed}.json").read_text(encoding="utf-8"))
                for seed in self.protocol["evaluation"]["seeds"]
            ]
            self.assertTrue(
                all(payload["checkpoint"].startswith("ANONYMIZED_SELECTED_CHECKPOINT/") for payload in payloads)
            )
            row = aggregate_rows[label]
            for metric_name in metric_names:
                values = np.asarray(
                    [
                        payload["mean_episode_length"]
                        if metric_name == "mean_episode_length"
                        else payload["metrics"][metric_name]
                        for payload in payloads
                    ],
                    dtype=np.float64,
                )
                self.assertAlmostEqual(
                    float(row[f"{metric_name}_mean"]),
                    float(values.mean()),
                    places=12,
                )
                self.assertAlmostEqual(
                    float(row[f"{metric_name}_sd"]),
                    float(values.std(ddof=1)),
                    places=12,
                )

    def test_submitted_training_archive_is_complete_and_release_safe(self) -> None:
        archive = np.load(ROOT / "results/submitted_training_curves.npz")
        self.assertEqual(len(archive.files), 141)
        self.assertTrue(all(array.shape == (1500,) for array in archive.values()))
        self.assertTrue(all("/" not in key and "2026-" not in key for key in archive.files))
        for method in ("yaw", "outer", "uniform", "static", "lirpg", "relara"):
            for seed in self.protocol[method]["seeds"]:
                self.assertIn(
                    f"training__{method}__{seed}__task_reward",
                    archive.files,
                )


if __name__ == "__main__":
    unittest.main()
