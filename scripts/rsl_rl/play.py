# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
from depth_camera_visualizer import add_depth_camera_visualization_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--max_steps", type=int, default=None, help="Exit cleanly after this many playback steps.")
parser.add_argument(
    "--evaluation",
    action="store_true",
    default=False,
    help="Run deterministic deployment-only evaluation with automatic commands.",
)
parser.add_argument(
    "--evaluation_output",
    type=str,
    default=None,
    help="JSON file receiving independent play-evaluation metrics.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--follow-camera",
    action="store_true",
    default=False,
    help="Enable a viewer camera that follows the robot during playback.",
)
parser.add_argument(
    "--follow-camera-offset",
    type=float,
    nargs=3,
    default=(1.8, 1.8, 1.0),
    metavar=("X", "Y", "Z"),
    help="Camera offset from the robot base in world coordinates when follow camera is enabled.",
)
parser.add_argument(
    "--follow-camera-lookat-offset",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.2),
    metavar=("X", "Y", "Z"),
    help="Look-at offset from the robot base in world coordinates when follow camera is enabled.",
)
add_depth_camera_visualization_args(parser)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# The policy observation contains depth-camera data, including in headless playback.
args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")
RSL_RL_VERSION = "3.1.2"
if version.parse(installed_version) != version.parse(RSL_RL_VERSION):
    raise RuntimeError(
        f"YawBot playback is validated with rsl-rl-lib=={RSL_RL_VERSION}; installed version is {installed_version}."
    )

"""Rest everything follows."""

import time

import gymnasium as gym
import numpy as np
import torch
import yaw_bot.tasks  # noqa: F401
from yaw_bot.tasks.direct.yaw_bot.agents.predictive_gated_rl import ActorOnlyInferencePolicy
from depth_camera_visualizer import DepthCameraVisualizer
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab.sim as sim_utils
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.math import quat_from_euler_xyz

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
)

try:
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
except ImportError:
    def handle_deprecated_rsl_rl_cfg(agent_cfg, _installed_version):
        """Compatibility shim for Isaac Lab versions that removed this helper."""
        return agent_cfg
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from checkpoint_group import find_latest_complete_checkpoint, validate_checkpoint_group


class WsAdSe2Keyboard(Se2Keyboard):
    """SE(2) keyboard using W/S for linear velocity and A/D for yaw."""

    def _create_key_bindings(self):
        self._INPUT_KEY_MAPPING = {
            # forward / backward
            "W": np.asarray([-1.0, 0.0, 0.0]) * self.v_x_sensitivity,
            "S": np.asarray([1.0, 0.0, 0.0]) * self.v_x_sensitivity,
            # yaw left / right
            "A": np.asarray([0.0, 0.0, 1.0]) * self.omega_z_sensitivity,
            "D": np.asarray([0.0, 0.0, -1.0]) * self.omega_z_sensitivity,
        }

    def __str__(self) -> str:
        msg = f"Keyboard Controller for SE(2): {self.__class__.__name__}\n"
        msg += f"\tKeyboard name: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tReset all commands: L\n"
        msg += "\tMove forward   (along x-axis): W\n"
        msg += "\tMove backward  (along x-axis): S\n"
        msg += "\tYaw positively (along z-axis): A\n"
        msg += "\tYaw negatively (along z-axis): D"
        return msg


class TargetPositionVectorVisualizer:
    """Display the planar vector from the robot to its integrated command target."""

    def __init__(self, env):
        self.env = env
        vector_cfg = GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/CommandedPositionVector")
        vector_cfg.markers["arrow"].scale = (1.0, 0.04, 0.04)
        self.vector_marker = VisualizationMarkers(vector_cfg)
        self.target_marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/CommandedPositionTarget",
                markers={
                    "target": sim_utils.SphereCfg(
                        radius=0.04,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.8, 0.0),
                            roughness=0.8,
                        ),
                    )
                },
            )
        )

    def update(self):
        root_pos_w = self.env.robot.data.root_pos_w[0]
        target_xy_w = self.env._commanded_position_w[0]
        target_vector_w = target_xy_w - root_pos_w[:2]
        vector_length = torch.linalg.vector_norm(target_vector_w)
        vector_yaw = torch.atan2(target_vector_w[1], target_vector_w[0]).reshape(1)
        zeros = torch.zeros_like(vector_yaw)

        marker_height = root_pos_w[2] + 0.25
        vector_origin = torch.stack([root_pos_w[0], root_pos_w[1], marker_height]).reshape(1, 3)
        target_position = torch.stack([target_xy_w[0], target_xy_w[1], marker_height]).reshape(1, 3)
        vector_scale = torch.stack(
            [vector_length.clamp(min=1.0e-3), torch.ones_like(vector_length), torch.ones_like(vector_length)]
        ).reshape(1, 3)
        self.vector_marker.visualize(
            translations=vector_origin,
            orientations=quat_from_euler_xyz(zeros, zeros, vector_yaw),
            scales=vector_scale,
        )
        self.target_marker.visualize(translations=target_position)


def pose_predictor_checkpoint_path(policy_checkpoint_path: str) -> str:
    checkpoint_dir = os.path.dirname(policy_checkpoint_path)
    checkpoint_name = os.path.basename(policy_checkpoint_path)
    return os.path.join(checkpoint_dir, f"pose_predictor_{checkpoint_name}")


def existing_pose_predictor_checkpoint_path(policy_checkpoint_path: str) -> str:
    """Resolve new and legacy predictor checkpoint names."""
    checkpoint_path = pose_predictor_checkpoint_path(policy_checkpoint_path)
    if os.path.isfile(checkpoint_path):
        return checkpoint_path
    checkpoint_root, checkpoint_ext = os.path.splitext(policy_checkpoint_path)
    return f"{checkpoint_root}_pose_predictor{checkpoint_ext}"


def deployable_depth_encoder_checkpoint_path(policy_checkpoint_path: str) -> str:
    checkpoint_dir = os.path.dirname(policy_checkpoint_path)
    checkpoint_name = os.path.basename(policy_checkpoint_path)
    return os.path.join(checkpoint_dir, f"depth_encoder_{checkpoint_name}")


def deployable_actor_checkpoint_path(policy_checkpoint_path: str) -> str:
    checkpoint_dir = os.path.dirname(policy_checkpoint_path)
    checkpoint_name = os.path.basename(policy_checkpoint_path)
    return os.path.join(checkpoint_dir, f"actor_{checkpoint_name}")


def resolve_play_checkpoint(log_root_path: str, agent_cfg: RslRlBaseRunnerCfg) -> str:
    """Resolve the latest policy checkpoint without matching auxiliary models."""
    run_pattern = agent_cfg.load_run
    if run_pattern in (None, "-1", ".*"):
        run_pattern = ".*"

    checkpoint_pattern = agent_cfg.load_checkpoint
    if checkpoint_pattern in (None, "-1", ".*", "model_.*.pt"):
        checkpoint_pattern = r"model_\d+\.pt"

    return get_checkpoint_path(log_root_path, run_pattern, checkpoint_pattern)


def load_policy_checkpoint(runner, checkpoint_path: str) -> None:
    """Load native RSL-RL 3.1.2 and the project's older split checkpoints."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        try:
            runner.load(checkpoint_path, load_optimizer=False, map_location=runner.device)
        except Exception as exc:
            raise RuntimeError(f"Failed to load native RSL-RL 3.1.2 checkpoint {checkpoint_path}.") from exc
        return

    actor_state = checkpoint.get("actor_state_dict")
    critic_state = checkpoint.get("critic_state_dict")
    if actor_state is None or critic_state is None:
        raise KeyError(
            f"Checkpoint contains neither native model_state_dict nor legacy actor/critic state: {checkpoint_path}"
        )
    policy = runner.alg.policy
    combined_state = {}
    for key, value in actor_state.items():
        if key == "distribution.std_param":
            combined_state["std"] = value
        elif key.startswith("mlp."):
            combined_state[f"actor.{key.removeprefix('mlp.')}"] = value
    for key, value in critic_state.items():
        if key.startswith("mlp."):
            combined_state[f"critic.{key.removeprefix('mlp.')}"] = value
    try:
        policy.load_state_dict(combined_state, strict=True)
    except Exception as exc:
        raise RuntimeError(
            f"Legacy checkpoint {checkpoint_path} is incompatible with the current observation/policy architecture."
        ) from exc
    runner.current_learning_iteration = int(checkpoint.get("iter", 0))
    print(f"[INFO]: Loaded split legacy checkpoint at iteration {runner.current_learning_iteration}.")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = (
        int(args_cli.num_envs) if args_cli.evaluation and args_cli.num_envs else 1
    )
    if not args_cli.evaluation:
        env_cfg.episode_length_s = 1.0e6
    env_cfg.disable_termination = False
    env_cfg.use_velocity_commands = True
    env_cfg.resample_commands = bool(args_cli.evaluation)
    env_cfg.pose_predictor_train = False
    predictive_gating_enabled = bool(getattr(env_cfg, "predictive_gating_enable", False))
    if predictive_gating_enabled:
        # This switch is read during env construction: playback creates only
        # the EMA depth encoder, never the event/future heads or their queues.
        env_cfg.predictive_feasibility_train = False

    # Manual demonstrations suppress pushes.  Independent evaluation keeps the
    # training-time disturbance process unchanged for a matched task protocol.
    if not args_cli.evaluation and hasattr(env_cfg, "events") and hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        if predictive_gating_enabled:
            run_pattern = ".*" if agent_cfg.load_run in (None, "-1", ".*") else agent_cfg.load_run
            resume_path = find_latest_complete_checkpoint(log_root_path, run_pattern)
        else:
            resume_path = resolve_play_checkpoint(log_root_path, agent_cfg)

    if predictive_gating_enabled:
        validate_checkpoint_group(resume_path)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if predictive_gating_enabled:
        encoder_path = deployable_depth_encoder_checkpoint_path(resume_path)
        if not env.unwrapped.load_deployable_depth_encoder(encoder_path):
            raise FileNotFoundError(
                f"Predictive-gated playback requires its encoder-only sidecar: {encoder_path}"
            )
        pose_path = existing_pose_predictor_checkpoint_path(resume_path)
        if not env.unwrapped.load_pose_predictor(pose_path, load_optimizer=False):
            raise FileNotFoundError(
                f"Predictive-gated playback requires its actor-side pose predictor: {pose_path}"
            )
    else:
        pose_path = existing_pose_predictor_checkpoint_path(resume_path)
        if not env.unwrapped.load_pose_predictor(pose_path, load_optimizer=False):
            raise FileNotFoundError(f"Baseline playback requires its pose predictor sidecar: {pose_path}")

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    keyboard = None
    if not args_cli.evaluation:
        keyboard = WsAdSe2Keyboard(
            Se2KeyboardCfg(
                v_x_sensitivity=max(abs(env_cfg.command_lin_vel_x_range[0]), abs(env_cfg.command_lin_vel_x_range[1])),
                v_y_sensitivity=0.0,
                omega_z_sensitivity=max(abs(env_cfg.command_yaw_vel_range[0]), abs(env_cfg.command_yaw_vel_range[1])),
                sim_device=env.unwrapped.device,
            )
        )
        print(keyboard)
        print("[INFO] Manual command control enabled: W/S for forward-backward, A/D for yaw, L to reset commands.")
    else:
        if args_cli.max_steps is None or args_cli.max_steps <= 0:
            raise ValueError("--evaluation requires a positive --max_steps.")
        if not args_cli.evaluation_output:
            raise ValueError("--evaluation requires --evaluation_output.")
        print(
            "[INFO] Independent deployment evaluation enabled: "
            f"num_envs={env.unwrapped.num_envs}, steps={args_cli.max_steps}, "
            f"seed={agent_cfg.seed}."
        )
    if args_cli.follow_camera:
        print(
            "[INFO] Follow camera enabled: "
            f"offset={tuple(args_cli.follow_camera_offset)}, "
            f"lookat_offset={tuple(args_cli.follow_camera_lookat_offset)}."
        )

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    loaded_iteration = 0
    if predictive_gating_enabled:
        # Construct only the deployable actor and selectively load its tensors;
        # no critic or predictive head exists in the playback process.
        policy_nn = ActorOnlyInferencePolicy(
            actor_obs_groups=agent_cfg.obs_groups["policy"],
            num_actor_obs=env_cfg.predictive_policy_observation_dim,
            num_actions=env.num_actions,
            actor_hidden_dims=agent_cfg.policy.actor_hidden_dims,
            activation=agent_cfg.policy.activation,
            actor_obs_normalization=agent_cfg.policy.actor_obs_normalization,
            state_dependent_std=agent_cfg.policy.state_dependent_std,
            action_squash=bool(getattr(agent_cfg.policy, "action_squash", False)),
            maximum_latent_mean=float(
                getattr(agent_cfg.policy, "maximum_latent_mean", 4.0)
            ),
        ).to(env.unwrapped.device)
        actor_path = deployable_actor_checkpoint_path(resume_path)
        loaded_iteration = policy_nn.load_deployment_checkpoint(
            actor_path, map_location="cpu"
        )
        policy = policy_nn
        print(f"[INFO]: Loaded actor-only policy at iteration {loaded_iteration}.")
    else:
        # load previously trained baseline model
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        load_policy_checkpoint(runner, resume_path)
        loaded_iteration = int(runner.current_learning_iteration)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        # extract the neural network for RSL-RL < 4.0 to reset recurrent states
        if version.parse(installed_version) < version.parse("4.0.0"):
            if version.parse(installed_version) >= version.parse("2.3.0"):
                policy_nn = runner.alg.policy
            else:
                policy_nn = runner.alg.actor_critic

    # export the trained policy to JIT and ONNX formats
    # Note: Disabled by default to prevent segmentation faults (闪退) during ONNX tracing
    # export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    #
    # if version.parse(installed_version) >= version.parse("4.0.0"):
    #     # use the new export functions for rsl-rl >= 4.0.0
    #     runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
    #     runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
    # else:
    #     # extract the neural network for rsl-rl < 4.0.0
    #     if version.parse(installed_version) >= version.parse("2.3.0"):
    #         policy_nn = runner.alg.policy
    #     else:
    #         policy_nn = runner.alg.actor_critic
    #
    #     # extract the normalizer
    #     if hasattr(policy_nn, "actor_obs_normalizer"):
    #         normalizer = policy_nn.actor_obs_normalizer
    #     elif hasattr(policy_nn, "student_obs_normalizer"):
    #         normalizer = policy_nn.student_obs_normalizer
    #     else:
    #         normalizer = None
    #
    #     # export to JIT and ONNX
    #     export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    #     export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    follow_camera_offset = torch.tensor(args_cli.follow_camera_offset, dtype=torch.float32)
    follow_camera_lookat_offset = torch.tensor(args_cli.follow_camera_lookat_offset, dtype=torch.float32)

    def update_follow_camera():
        """Keep the viewer camera centered near the robot base."""
        if not args_cli.follow_camera:
            return
        root_pos_w = env.unwrapped.robot.data.root_pos_w[0].detach().cpu()
        camera_eye = (root_pos_w + follow_camera_offset).tolist()
        camera_target = (root_pos_w + follow_camera_lookat_offset).tolist()
        env.unwrapped.sim.set_camera_view(camera_eye, camera_target)

    # reset environment
    obs = env.get_observations()
    if predictive_gating_enabled:
        repeated_obs = env.get_observations()
        for observation_group in ("policy", "critic"):
            if not torch.equal(obs[observation_group], repeated_obs[observation_group]):
                raise RuntimeError(
                    f"Predictive observation group {observation_group!r} changed across repeated reads of one step."
                )
    update_follow_camera()
    target_position_visualizer = None
    if not args_cli.evaluation:
        target_position_visualizer = TargetPositionVectorVisualizer(env.unwrapped)
        target_position_visualizer.update()
        print("[INFO] Target position vector enabled: green arrow points from the robot to the yellow target.")
    depth_visualizer = DepthCameraVisualizer(
        env.unwrapped,
        enabled=args_cli.visualize_depth_camera,
        interval=args_cli.depth_visualization_interval,
        window_name="YawBot play depth camera",
        output_dir=os.path.join(log_dir, "depth_camera") if args_cli.visualize_depth_camera else None,
    )
    depth_visualizer.update(force=True)
    timestep = 0
    last_printed_command = None
    evaluation_metric_sums: dict[str, float] = {}
    evaluation_metric_counts: dict[str, int] = {}
    evaluation_episode_lengths: list[float] = []
    evaluation_current_lengths = torch.zeros(
        env.unwrapped.num_envs, device=env.unwrapped.device, dtype=torch.long
    )
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            if not args_cli.evaluation:
                teleop_command = keyboard.advance()
                env.unwrapped._commands[:, 0] = teleop_command[0]
                env.unwrapped._commands[:, 1] = teleop_command[2]
                env.unwrapped._command_time_left.fill_(1.0e6)
                command_start = env_cfg.policy_command_observation_start
                obs["policy"][:, command_start : command_start + 2] = env.unwrapped._commands

                current_command = (float(teleop_command[0].item()), float(teleop_command[2].item()))
                if current_command != last_printed_command:
                    print(
                        f"[CMD] v_x={current_command[0]: .3f} m/s, yaw={current_command[1]: .3f} rad/s",
                        flush=True,
                    )
                    last_printed_command = current_command

            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, extras = env.step(actions)
            update_follow_camera()
            if target_position_visualizer is not None:
                target_position_visualizer.update()
            depth_visualizer.update()
            if args_cli.evaluation:
                evaluation_current_lengths.add_(1)
                completed = dones.bool().reshape(-1)
                if torch.any(completed):
                    evaluation_episode_lengths.extend(
                        evaluation_current_lengths[completed].detach().cpu().float().tolist()
                    )
                    evaluation_current_lengths[completed] = 0
                for metric_name, metric_value in extras.get("log", {}).items():
                    if metric_name.startswith("Evaluation/") or metric_name in {
                        "Diagnostics/lin_cmd_success_rate",
                        "Diagnostics/yaw_cmd_signed_success_rate",
                        "Diagnostics/forward_cmd_success_rate",
                        "Diagnostics/stop_cmd_success_rate",
                        "Diagnostics/planar_position_error",
                    }:
                        value = float(metric_value)
                        evaluation_metric_sums[metric_name] = (
                            evaluation_metric_sums.get(metric_name, 0.0) + value
                        )
                        evaluation_metric_counts[metric_name] = (
                            evaluation_metric_counts.get(metric_name, 0) + 1
                        )
            # reset recurrent states for episodes that have terminated
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)
        timestep += 1
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        if args_cli.max_steps is not None and timestep >= args_cli.max_steps:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    depth_visualizer.close()

    if args_cli.evaluation:
        output_path = os.path.abspath(args_cli.evaluation_output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        evaluation_result = {
            "task": args_cli.task,
            "checkpoint": os.path.abspath(resume_path),
            "checkpoint_iteration": int(loaded_iteration),
            "evaluation_seed": int(agent_cfg.seed),
            "num_envs": int(env.unwrapped.num_envs),
            "steps": int(timestep),
            "completed_episodes": len(evaluation_episode_lengths),
            "mean_episode_length": (
                float(np.mean(evaluation_episode_lengths))
                if evaluation_episode_lengths
                else None
            ),
            "metrics": {
                name: evaluation_metric_sums[name] / evaluation_metric_counts[name]
                for name in sorted(evaluation_metric_sums)
            },
        }
        with open(output_path, "w", encoding="utf-8") as output_stream:
            json.dump(evaluation_result, output_stream, indent=2, ensure_ascii=False)
        print(f"[INFO] Wrote independent evaluation metrics to {output_path}")

    # close the simulator
    if keyboard is not None:
        del keyboard
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
