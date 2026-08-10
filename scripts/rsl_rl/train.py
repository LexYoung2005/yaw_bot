# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import os
import re
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
from depth_camera_visualizer import add_depth_camera_visualization_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--static_reward_weights",
    type=float,
    nargs=5,
    metavar=("STABILITY", "CONTACT", "LINEAR", "YAW", "REGULARIZATION"),
    default=None,
    help="A2 fixed weights for the five RMS-normalized reward groups.",
)
parser.add_argument(
    "--predictor_pretrain",
    action="store_true",
    default=False,
    help="Meta-train the predictor with a disposable inner PPO; retain the predictor for a fresh policy run.",
)
parser.add_argument(
    "--pretrained_predictor",
    type=str,
    default=None,
    help="Full predictor sidecar to load into a fresh (non-resumed) PPO run.",
)
parser.add_argument(
    "--continue_predictor",
    type=str,
    default=None,
    help="Load a predictor and its optimizers into a fresh disposable PPO meta-training trial.",
)
parser.add_argument(
    "--resume_curriculum_stage",
    type=int,
    choices=range(1, 5),
    default=None,
    help="Curriculum stage to use when resuming a legacy checkpoint without saved curriculum state.",
)
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
add_depth_camera_visualization_args(parser)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# A distributed GUI launch should create one interactive viewport, not one
# window per rank.  Secondary ranks still keep cameras enabled because depth is
# part of the policy observation, but render them off-screen.
distributed_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
if args_cli.distributed and distributed_local_rank != 0:
    args_cli.headless = True
    args_cli.visualize_depth_camera = False
if args_cli.distributed:
    # Keep simulation and network compute on each GPU while synchronizing the
    # compact training state through CPU Gloo collectives.
    os.environ.setdefault("YAWBOT_DISTRIBUTED_BACKEND", "gloo")

# The policy observation contains depth-camera data, including in headless training.
args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# Experiments are validated against one exact RSL-RL build.
RSL_RL_VERSION = "3.1.2"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) != version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import logging
import time
import types
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner
from depth_camera_visualizer import DepthCameraVisualizer

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

try:
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
except ImportError:
    def handle_deprecated_rsl_rl_cfg(agent_cfg, _installed_version):
        """Compatibility shim for Isaac Lab versions that removed this helper."""
        return agent_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

import yaw_bot.tasks  # noqa: F401
from yaw_bot.tasks.direct.yaw_bot.agents.predictive_gated_rl import (
    BoundedStdOnPolicyRunner,
    PredictiveGatedOnPolicyRunner,
)
from yaw_bot.tasks.direct.yaw_bot.agents.predictive_gated_rl import guarded_rollout_ppo_update
from yaw_bot.tasks.direct.yaw_bot.agents.predictive_gated_rl import linear_learning_rate_schedule
from yaw_bot.tasks.direct.yaw_bot.agents.predictive_gated_rl import rollout_adaptive_learning_rate
from yaw_bot.tasks.direct.yaw_bot.agents.predictive_gated_rl import rollout_policy_kl
from yaw_bot.tasks.direct.yaw_bot.outer_advantage_composer import (
    resolve_beta_schedule_horizon,
)
from yaw_bot.tasks.direct.yaw_bot.agents.predictive_gated_rl import save_actor_only_checkpoint
from yaw_bot.tasks.direct.yaw_bot.agents.predictive_population import AntitheticPopulationPPO
from yaw_bot.tasks.direct.yaw_bot.agents.official_lirpg import (
    official_lirpg_ppo_update,
)
from yaw_bot.tasks.direct.yaw_bot.predictive_feasibility import block_reference_learning_trend

from checkpoint_group import find_latest_complete_checkpoint, save_checkpoint_group, validate_checkpoint_group

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


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


def predictive_feasibility_checkpoint_path(policy_checkpoint_path: str) -> str:
    checkpoint_dir = os.path.dirname(policy_checkpoint_path)
    checkpoint_name = os.path.basename(policy_checkpoint_path)
    return os.path.join(checkpoint_dir, f"predictive_feasibility_{checkpoint_name}")


def deployable_depth_encoder_checkpoint_path(policy_checkpoint_path: str) -> str:
    checkpoint_dir = os.path.dirname(policy_checkpoint_path)
    checkpoint_name = os.path.basename(policy_checkpoint_path)
    return os.path.join(checkpoint_dir, f"depth_encoder_{checkpoint_name}")


def deployable_actor_checkpoint_path(policy_checkpoint_path: str) -> str:
    checkpoint_dir = os.path.dirname(policy_checkpoint_path)
    checkpoint_name = os.path.basename(policy_checkpoint_path)
    return os.path.join(checkpoint_dir, f"actor_{checkpoint_name}")


def resolve_resume_checkpoint(log_root_path: str, agent_cfg: RslRlBaseRunnerCfg) -> str:
    """Resolve the latest policy checkpoint without matching auxiliary models."""
    run_pattern = agent_cfg.load_run
    if run_pattern in (None, "-1", ".*"):
        run_pattern = ".*"
    elif os.path.isdir(os.path.join(log_root_path, run_pattern)):
        # ``get_checkpoint_path`` treats load_run as a prefix regex. In a
        # distributed experiment that made an explicit rank-0 run name also
        # match its ``_rank1`` worker directory, which intentionally contains
        # no policy checkpoint. Prefer the exact directory when it exists.
        run_pattern = rf"^{re.escape(run_pattern)}$"

    checkpoint_pattern = agent_cfg.load_checkpoint
    if checkpoint_pattern in (None, "-1", ".*", "model_.*.pt"):
        checkpoint_pattern = r"model_\d+\.pt"

    return get_checkpoint_path(log_root_path, run_pattern, checkpoint_pattern)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    predictive_gating_enabled = bool(getattr(env_cfg, "predictive_gating_enable", False))
    predictor_modes = (
        bool(args_cli.predictor_pretrain),
        bool(args_cli.pretrained_predictor),
        bool(args_cli.continue_predictor),
    )
    if any(predictor_modes) and not predictive_gating_enabled:
        raise ValueError("Predictor pretraining/loading requires the predictive-gated task.")
    if sum(predictor_modes) > 1:
        raise ValueError(
            "--predictor_pretrain, --continue_predictor, and --pretrained_predictor are mutually exclusive."
        )
    if (args_cli.pretrained_predictor or args_cli.continue_predictor) and agent_cfg.resume:
        raise ValueError("A predictor sidecar with a fresh PPO cannot be combined with --resume.")
    outer_composer_enabled = bool(
        getattr(env_cfg, "outer_advantage_composer_enable", False)
    )
    reward_composition_mode = str(
        getattr(env_cfg, "outer_reward_composition_mode", "composer")
    ).lower()
    if bool(getattr(env_cfg, "training_launch_paused", False)):
        raise RuntimeError(
            f"Training launch for reward mode {reward_composition_mode!r} is paused "
            "while its official-source implementation is being integrated."
        )
    if args_cli.static_reward_weights is not None:
        if reward_composition_mode != "static":
            raise ValueError(
                "--static_reward_weights is valid only for "
                "Template-Yaw-Bot-Static-Reward-PPO-Direct-v0."
            )
        from yaw_bot.tasks.direct.yaw_bot.outer_advantage_composer import (
            validate_static_group_weights,
        )

        env_cfg.outer_static_group_weights = validate_static_group_weights(
            args_cli.static_reward_weights
        )
    if outer_composer_enabled and any(predictor_modes):
        raise ValueError(
            "The default Outer-Advantage mode learns its H=12 predictor online and does not "
            "use dedicated predictor pretraining. Select "
            "Template-Yaw-Bot-Predictive-Meta-Direct-v0 for the retained legacy workflow."
        )
    if predictive_gating_enabled and args_cli.distributed and not outer_composer_enabled:
        raise ValueError(
            "Distributed training is supported for the default Outer-Advantage "
            "composer, but not for the retained legacy predictor/meta-population path."
        )
    if reward_composition_mode == "relara" and args_cli.distributed:
        raise ValueError(
            "The source-faithful ReLara replay/SAC Reward Agent currently supports "
            "one GPU only; distributed ranks would otherwise train divergent replay buffers."
        )

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = os.environ.get(
        "YAWBOT_DISTRIBUTED_RUN_TIMESTAMP",
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not
    # change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    if args_cli.distributed and app_launcher.local_rank != 0:
        # Only rank zero logs and checkpoints in RSL-RL. A separate worker
        # directory prevents config/git-diff writers from racing rank zero.
        log_dir += f"_rank{app_launcher.local_rank}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        if predictive_gating_enabled and agent_cfg.load_checkpoint in (None, "-1", ".*", "model_.*.pt"):
            run_pattern = ".*" if agent_cfg.load_run in (None, "-1", ".*") else agent_cfg.load_run
            resume_path = find_latest_complete_checkpoint(log_root_path, run_pattern)
        else:
            resume_path = resolve_resume_checkpoint(log_root_path, agent_cfg)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    depth_visualizer = DepthCameraVisualizer(
        env.unwrapped,
        enabled=args_cli.visualize_depth_camera,
        interval=args_cli.depth_visualization_interval,
        window_name="YawBot train depth camera",
        output_dir=os.path.join(log_dir, "depth_camera") if args_cli.visualize_depth_camera else None,
    )
    if args_cli.visualize_depth_camera:
        original_env_step = env.step

        def step_with_depth_visualization(*args, **kwargs):
            result = original_env_step(*args, **kwargs)
            depth_visualizer.update()
            return result

        env.step = step_with_depth_visualization
        depth_visualizer.update(force=True)

    start_time = time.time()

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "BoundedStdOnPolicyRunner":
        runner = BoundedStdOnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
        )
    elif agent_cfg.class_name == "PredictiveGatedOnPolicyRunner":
        runner = PredictiveGatedOnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
        )
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    if reward_composition_mode == "lirpg":
        # OpenAI Baselines' PPO-LIRPG uses Adam epsilon 1e-5 for the policy,
        # intrinsic reward, and extrinsic-value optimizers.
        for parameter_group in runner.alg.optimizer.param_groups:
            parameter_group["eps"] = 1.0e-5

    # Keep task-owned training state synchronized with each policy checkpoint.
    original_runner_save = runner.save
    if predictive_gating_enabled:
        def save_with_predictive_model(self, path: str, infos: dict | None = None):
            checkpoint_infos = dict(infos) if infos is not None else {}
            actor_config = {
                "actor_obs_groups": list(agent_cfg.obs_groups["policy"]),
                "num_actor_obs": int(env_cfg.predictive_policy_observation_dim),
                "num_actions": int(env.num_actions),
                "actor_hidden_dims": list(agent_cfg.policy.actor_hidden_dims),
                "activation": agent_cfg.policy.activation,
                "state_dependent_std": bool(agent_cfg.policy.state_dependent_std),
                "actor_obs_normalization": bool(agent_cfg.policy.actor_obs_normalization),
                "action_squash": bool(getattr(agent_cfg.policy, "action_squash", False)),
                "maximum_latent_mean": float(
                    getattr(agent_cfg.policy, "maximum_latent_mean", 4.0)
                ),
            }

            def write_policy(temporary_path: str, group_id: str) -> None:
                policy_infos = dict(checkpoint_infos)
                policy_infos["yaw_bot_checkpoint_group_id"] = group_id
                original_runner_save(temporary_path, policy_infos)

            def write_predictor(temporary_path: str, group_id: str) -> None:
                env.unwrapped.save_predictive_feasibility_full(temporary_path, group_id)

            def write_encoder(temporary_path: str, group_id: str) -> None:
                env.unwrapped.save_deployable_depth_encoder(temporary_path, group_id)

            def write_pose_predictor(temporary_path: str, group_id: str) -> None:
                env.unwrapped.save_pose_predictor(temporary_path, group_id)

            def write_actor(temporary_path: str, group_id: str) -> None:
                save_actor_only_checkpoint(
                    self.alg.policy,
                    temporary_path,
                    actor_config,
                    iteration=self.current_learning_iteration,
                    checkpoint_group_id=group_id,
                )

            save_checkpoint_group(
                path,
                {
                    "policy": (path, write_policy),
                    "predictor": (predictive_feasibility_checkpoint_path(path), write_predictor),
                    "encoder": (deployable_depth_encoder_checkpoint_path(path), write_encoder),
                    "pose": (pose_predictor_checkpoint_path(path), write_pose_predictor),
                    "actor": (deployable_actor_checkpoint_path(path), write_actor),
                },
            )

        runner.save = types.MethodType(save_with_predictive_model, runner)
    else:
        def save_with_pose_predictor(self, path: str, infos: dict | None = None):
            checkpoint_infos = dict(infos) if infos is not None else {}
            checkpoint_infos["yaw_bot_curriculum_state"] = env.unwrapped.get_curriculum_state()
            original_runner_save(path, checkpoint_infos)
            env.unwrapped.save_pose_predictor(pose_predictor_checkpoint_path(path))

        runner.save = types.MethodType(save_with_pose_predictor, runner)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        if predictive_gating_enabled:
            validate_checkpoint_group(resume_path)
        # load previously trained model
        checkpoint_infos = runner.load(resume_path)
        if predictive_gating_enabled:
            # Optimizer state restoration also restores the checkpoint's
            # parameter-group LR.  Make a fixed schedule genuinely authoritative
            # so pre-fix checkpoints cannot silently keep the collapsed 1e-5 LR.
            if agent_cfg.algorithm.schedule == "fixed" and not bool(
                getattr(agent_cfg, "rollout_adaptive_schedule", False)
            ):
                configured_learning_rate = float(agent_cfg.algorithm.learning_rate)
                restored_learning_rates = [
                    float(group["lr"]) for group in runner.alg.optimizer.param_groups
                ]
                for group in runner.alg.optimizer.param_groups:
                    group["lr"] = configured_learning_rate
                runner.alg.learning_rate = configured_learning_rate
                if any(
                    abs(rate - configured_learning_rate) > 1.0e-12
                    for rate in restored_learning_rates
                ):
                    print(
                        "[YawBot] Overrode restored PPO learning rate(s) "
                        f"{restored_learning_rates} with fixed configured rate {configured_learning_rate}."
                    )
            predictive_path = predictive_feasibility_checkpoint_path(resume_path)
            if not env.unwrapped.load_predictive_feasibility(predictive_path, load_optimizer=True):
                raise FileNotFoundError(
                    f"Predictive-gated policy requires its training sidecar: {predictive_path}"
                )
            pose_path = existing_pose_predictor_checkpoint_path(resume_path)
            if not env.unwrapped.load_pose_predictor(pose_path, load_optimizer=True):
                raise FileNotFoundError(
                    f"Predictive-gated policy requires its actor-side pose predictor: {pose_path}"
                )
        else:
            pose_path = existing_pose_predictor_checkpoint_path(resume_path)
            if not env.unwrapped.load_pose_predictor(pose_path, load_optimizer=True):
                raise FileNotFoundError(f"Baseline resume requires its pose predictor sidecar: {pose_path}")
            curriculum_state = checkpoint_infos.get("yaw_bot_curriculum_state") if checkpoint_infos else None
            if not env.unwrapped.load_curriculum_state(curriculum_state):
                if args_cli.resume_curriculum_stage is not None:
                    env.unwrapped.load_curriculum_state(
                        {
                            "stage": args_cli.resume_curriculum_stage,
                            "active_stage": args_cli.resume_curriculum_stage,
                            "last_unlock": args_cli.resume_curriculum_stage,
                        }
                    )
                    print("[YawBot] Applied --resume_curriculum_stage for a legacy checkpoint.")
                else:
                    print(
                        "[YawBot] Checkpoint has no curriculum state (legacy checkpoint); curriculum starts from stage 1. "
                        "Use --resume_curriculum_stage to override it."
                    )

    if args_cli.pretrained_predictor:
        predictor_path = os.path.abspath(args_cli.pretrained_predictor)
        if not env.unwrapped.load_predictive_feasibility(predictor_path, load_optimizer=False):
            raise FileNotFoundError(f"Pretrained predictor not found: {predictor_path}")
        env.unwrapped.reset_predictive_allocator_learning_context()
        env.unwrapped.freeze_predictive_updates()
        print(
            "[YawBot] Initialized a fresh PPO policy with a frozen pretrained predictor: "
            f"{predictor_path}"
        )

    if args_cli.continue_predictor:
        predictor_path = os.path.abspath(args_cli.continue_predictor)
        if not env.unwrapped.load_predictive_feasibility(predictor_path, load_optimizer=True):
            raise FileNotFoundError(f"Predictor continuation checkpoint not found: {predictor_path}")
        if not env.unwrapped.predictive_allocator_updates_enabled():
            raise RuntimeError("The continuation checkpoint has predictor updates disabled.")
        env.unwrapped.reset_predictive_allocator_learning_context()
        print(
            "[YawBot] Continuing predictor meta-training with retained predictor/optimizer state "
            f"and a fresh disposable PPO: {predictor_path}"
        )

    if args_cli.predictor_pretrain or args_cli.continue_predictor:
        print(
            "[YawBot] Predictor meta-pretraining: the inner PPO remains trainable so the reward "
            "allocator receives real learning-speed and advantage feedback. This PPO is disposable; "
            "start the final policy with --pretrained_predictor."
        )

    if args_cli.distributed:
        env.unwrapped.synchronize_distributed_training_state(initial=True)
        if app_launcher.local_rank == 0:
            display_description = (
                "all ranks run headless"
                if args_cli.headless
                else "rank 0 keeps the viewport and secondary ranks run off-screen"
            )
            print(
                "[YawBot] Distributed task state synchronized across "
                f"{runner.gpu_world_size} GPUs; {display_description}."
            )

    if predictive_gating_enabled and not outer_composer_enabled:
        env.unwrapped.begin_predictive_allocator_rollout(
            explore=env.unwrapped.predictive_allocator_updates_enabled()
        )

    outer_total_iterations = int(agent_cfg.max_iterations)
    if outer_composer_enabled and agent_cfg.resume:
        saved_outer_total_iterations = env.unwrapped.outer_beta_total_iterations()
        outer_total_iterations = resolve_beta_schedule_horizon(
            saved_outer_total_iterations,
            env.unwrapped.outer_beta_iteration(),
            int(agent_cfg.max_iterations),
        )
        if outer_total_iterations > saved_outer_total_iterations:
            print(
                "[YawBot] Extended outer beta horizon from "
                f"{saved_outer_total_iterations} to {outer_total_iterations} iterations "
                "to cover the requested continuation."
            )
    if outer_composer_enabled:
        initial_beta = env.unwrapped.begin_outer_advantage_rollout(outer_total_iterations)
        print(
            "[YawBot] Reward composition mode "
            f"{reward_composition_mode!r} enabled with the unchanged yaw_bot "
            "environment and PPO; online H=12 diagnostics remain active, "
            f"initial beta={initial_beta:.4f}."
        )

    population_allocator_enabled = bool(
        getattr(env_cfg, "predictive_allocator_population_enable", False)
    )
    if outer_composer_enabled:
        outer_rollout_meta: list[dict[str, torch.Tensor]] = []
        final_outer_critic_observation: torch.Tensor | None = None
        final_mixed_critic_observation: torch.Tensor | None = None
        original_process_env_step = runner.alg.process_env_step

        def process_env_step_with_outer_composer(algorithm, obs, rewards, dones, extras):
            nonlocal final_outer_critic_observation, final_mixed_critic_observation
            original_process_env_step(obs, rewards, dones, extras)
            meta = extras.get("outer_composer")
            if meta is None:
                raise RuntimeError("Outer-composer step did not provide training metadata.")
            time_outs = extras.get("time_outs")
            if time_outs is None:
                time_outs = torch.zeros_like(dones)
            outer_rollout_meta.append(
                {
                    "fused_latent": meta["fused_latent"].detach(),
                    "normalized_group_rewards": meta[
                        "normalized_group_rewards"
                    ].detach(),
                    "fixed_outer_reward": meta["fixed_outer_reward"].detach(),
                    "fixed_internal_reward": meta["fixed_internal_reward"].detach(),
                    "cached_actor_reward": meta["cached_actor_reward"].detach(),
                    "relara_proposed_reward": meta[
                        "relara_proposed_reward"
                    ].detach(),
                    "dones": dones.detach().clone(),
                    "time_outs": time_outs.detach().clone(),
                }
            )
            final_mixed_critic_observation = obs["critic"].detach().clone()
            if reward_composition_mode == "lirpg":
                with torch.no_grad():
                    final_outer_critic_observation = (
                        algorithm.policy.actor_obs_normalizer(obs["policy"])
                        .detach()
                        .clone()
                    )
            else:
                final_outer_critic_observation = (
                    final_mixed_critic_observation.detach().clone()
                )

        runner.alg.process_env_step = types.MethodType(
            process_env_step_with_outer_composer, runner.alg
        )
        original_outer_algorithm_update = runner.alg.update

        def update_with_outer_composer(algorithm):
            nonlocal final_outer_critic_observation, final_mixed_critic_observation
            if len(outer_rollout_meta) != runner.num_steps_per_env:
                raise RuntimeError(
                    "Outer composer received "
                    f"{len(outer_rollout_meta)} transitions; expected {runner.num_steps_per_env}."
                )
            if final_outer_critic_observation is None:
                raise RuntimeError("Outer composer has no final critic observation.")
            if final_mixed_critic_observation is None:
                raise RuntimeError("PPO has no final mixed-critic observation.")
            # Predictor and pose-predictor use local online labels during the
            # rollout. Average their parameters and Adam moments once at this
            # safe boundary before computing phi_{k+1}.
            if args_cli.distributed:
                env.unwrapped.synchronize_distributed_training_state(initial=False)
            # The PPO storage contains the observations that generated each
            # already-cached scalar reward. Updating phi below cannot mutate it.
            cached_actor_rewards = torch.stack(
                [step["cached_actor_reward"] for step in outer_rollout_meta]
            )
            stored_actor_rewards = algorithm.storage.rewards.squeeze(-1).detach().clone()
            stored_actions = algorithm.storage.actions.detach()
            stored_latent_means = algorithm.storage.mu.detach()
            inverse_actions = torch.atanh(
                stored_actions.clamp(min=-1.0 + 1.0e-6, max=1.0 - 1.0e-6)
            )
            action_density_metrics = {
                "exact_action_boundary_rate": float(
                    (torch.abs(stored_actions) >= 1.0).float().mean().item()
                ),
                "latent_mean_abs_max": float(torch.abs(stored_latent_means).amax().item()),
                "inverse_action_mean_gap_max": float(
                    torch.abs(inverse_actions - stored_latent_means).amax().item()
                ),
                "old_action_log_prob_abs_max": float(
                    torch.abs(algorithm.storage.actions_log_prob).amax().item()
                ),
            }
            # RSL-RL adds only timeout bootstrap to storage. For non-timeout
            # transitions exact equality catches accidental in-rollout reward mutation.
            time_outs = torch.stack([step["time_outs"] for step in outer_rollout_meta])
            non_timeout = time_outs <= 0
            if not torch.allclose(
                stored_actor_rewards[non_timeout],
                cached_actor_rewards[non_timeout],
                atol=1.0e-6,
                rtol=1.0e-6,
            ):
                raise RuntimeError("Cached rollout reward changed before the composer update.")
            with torch.no_grad():
                meta_actor_observations = algorithm.policy.actor_obs_normalizer(
                    algorithm.storage.observations["policy"]
                ).detach()
            if reward_composition_mode == "relara":
                raw_actor_observations = algorithm.storage.observations[
                    "policy"
                ].detach()
                with torch.no_grad():
                    final_actor_action = algorithm.policy.act_inference(
                        {"policy": final_mixed_critic_observation}
                    ).detach()
                relara_metrics = env.unwrapped.update_relara_reward_agent(
                    actor_observations=raw_actor_observations,
                    actor_actions=algorithm.storage.actions.detach(),
                    final_actor_observation=final_mixed_critic_observation.detach(),
                    final_actor_action=final_actor_action,
                    proposed_rewards=torch.stack(
                        [
                            step["relara_proposed_reward"]
                            for step in outer_rollout_meta
                        ]
                    ),
                    fixed_outer_rewards=torch.stack(
                        [step["fixed_outer_reward"] for step in outer_rollout_meta]
                    ),
                    dones=torch.stack(
                        [step["dones"] for step in outer_rollout_meta]
                    ),
                )
                if not torch.equal(
                    algorithm.storage.rewards.squeeze(-1), stored_actor_rewards
                ):
                    raise RuntimeError(
                        "ReLara update mutated rollout-k PPO rewards."
                    )
                outer_rollout_meta.clear()
                final_outer_critic_observation = None
                final_mixed_critic_observation = None
                loss_dict = guarded_rollout_ppo_update(
                    algorithm,
                    update_callable=original_outer_algorithm_update,
                    desired_kl=float(agent_cfg.rollout_adaptive_desired_kl),
                    maximum_kl=float(agent_cfg.rollout_trust_region_maximum_kl),
                    minimum_rate=float(agent_cfg.rollout_adaptive_min_learning_rate),
                    maximum_rate=float(agent_cfg.rollout_adaptive_max_learning_rate),
                    adaptation_factor=float(agent_cfg.rollout_adaptive_factor),
                    backtrack_factor=float(agent_cfg.rollout_trust_region_backtrack_factor),
                    maximum_backtracks=int(
                        agent_cfg.rollout_trust_region_maximum_backtracks
                    ),
                )
                loss_dict.update(
                    {
                        f"relara/{name}": float(value)
                        for name, value in relara_metrics.items()
                    }
                )
                return loss_dict
            outer_metrics = env.unwrapped.update_outer_advantage_composer(
                fused_latents=torch.stack(
                    [step["fused_latent"] for step in outer_rollout_meta]
                ),
                normalized_group_rewards=torch.stack(
                    [step["normalized_group_rewards"] for step in outer_rollout_meta]
                ),
                fixed_internal_rewards=torch.stack(
                    [step["fixed_internal_reward"] for step in outer_rollout_meta]
                ),
                fixed_outer_rewards=torch.stack(
                    [step["fixed_outer_reward"] for step in outer_rollout_meta]
                ),
                dones=torch.stack([step["dones"] for step in outer_rollout_meta]),
                time_outs=time_outs,
                critic_observations=(
                    meta_actor_observations
                    if reward_composition_mode == "lirpg"
                    else algorithm.storage.observations["critic"]
                ),
                final_critic_observation=final_outer_critic_observation,
                actor=algorithm.policy.actor,
                actor_optimizer=algorithm.optimizer,
                actor_observations=meta_actor_observations,
                actor_actions=algorithm.storage.actions,
                actor_old_log_probabilities=algorithm.storage.actions_log_prob,
                actor_action_standard_deviations=algorithm.storage.sigma,
                actor_maximum_latent_mean=float(
                    algorithm.policy.maximum_latent_mean
                ),
                ppo_clip_param=float(algorithm.clip_param),
                gamma=float(algorithm.gamma),
                lam=float(algorithm.lam),
                defer_lirpg_reward_update=reward_composition_mode == "lirpg",
            )
            if reward_composition_mode == "lirpg":
                outer_advantages = outer_metrics.pop("_outer_advantages_tensor")
                outer_returns = outer_metrics.pop("_outer_returns_tensor")
                outer_values = outer_metrics.pop("_outer_values_tensor")
                with torch.no_grad():
                    final_mixed_value = algorithm.policy.critic(
                        algorithm.policy.critic_obs_normalizer(
                            final_mixed_critic_observation
                        )
                    ).detach()
                fixed_outer_rewards = torch.stack(
                    [step["fixed_outer_reward"] for step in outer_rollout_meta]
                )
                ppo_metrics, lirpg_metrics, meta_updates = (
                    official_lirpg_ppo_update(
                        algorithm,
                        reward_model=env.unwrapped.lirpg_intrinsic_reward,
                        reward_optimizer=env.unwrapped._lirpg_optimizer,
                        outer_critic=env.unwrapped.outer_critic,
                        outer_critic_optimizer=env.unwrapped._outer_critic_optimizer,
                        extrinsic_rewards=fixed_outer_rewards,
                        outer_advantages=outer_advantages,
                        outer_returns=outer_returns,
                        outer_values=outer_values,
                        critic_observations=meta_actor_observations,
                        actor_observations=meta_actor_observations,
                        actions=algorithm.storage.actions,
                        old_action_log_probabilities=algorithm.storage.actions_log_prob,
                        dones=torch.stack(
                            [step["dones"] for step in outer_rollout_meta]
                        ),
                        final_mixed_value=final_mixed_value,
                        actor_maximum_latent_mean=float(
                            algorithm.policy.maximum_latent_mean
                        ),
                        extrinsic_coefficient=float(
                            env_cfg.lirpg_extrinsic_coefficient
                        ),
                        intrinsic_coefficient=float(
                            env_cfg.lirpg_intrinsic_coefficient
                        ),
                        reward_gradient_clip=float(
                            env_cfg.lirpg_gradient_clip
                        ),
                        outer_critic_gradient_clip=float(
                            env_cfg.lirpg_gradient_clip
                        ),
                    )
                )
                lirpg_metrics.update(action_density_metrics)
                outer_metrics.update(lirpg_metrics)
                env.unwrapped.complete_official_lirpg_rollout(
                    {
                        name: float(value)
                        for name, value in outer_metrics.items()
                    },
                    meta_updates=meta_updates,
                )
                if not torch.equal(
                    algorithm.storage.rewards.squeeze(-1), stored_actor_rewards
                ) and algorithm.storage.step != 0:
                    raise RuntimeError(
                        "Official LIRPG unexpectedly retained mutable rollout storage."
                    )
                outer_rollout_meta.clear()
                final_outer_critic_observation = None
                final_mixed_critic_observation = None
                ppo_metrics.update(
                    {
                        f"outer_composer/{name}": float(value)
                        for name, value in outer_metrics.items()
                    }
                )
                return ppo_metrics
            outer_metrics.update(action_density_metrics)
            # Assert the central causal invariant after phi_{k+1} exists.
            if not torch.equal(algorithm.storage.rewards.squeeze(-1), stored_actor_rewards):
                raise RuntimeError("Composer update mutated rollout-k PPO rewards.")
            outer_rollout_meta.clear()
            final_outer_critic_observation = None
            final_mixed_critic_observation = None
            loss_dict = guarded_rollout_ppo_update(
                algorithm,
                update_callable=original_outer_algorithm_update,
                desired_kl=float(agent_cfg.rollout_adaptive_desired_kl),
                maximum_kl=float(agent_cfg.rollout_trust_region_maximum_kl),
                minimum_rate=float(agent_cfg.rollout_adaptive_min_learning_rate),
                maximum_rate=float(agent_cfg.rollout_adaptive_max_learning_rate),
                adaptation_factor=float(agent_cfg.rollout_adaptive_factor),
                backtrack_factor=float(agent_cfg.rollout_trust_region_backtrack_factor),
                maximum_backtracks=int(
                    agent_cfg.rollout_trust_region_maximum_backtracks
                ),
            )
            loss_dict.update(
                {f"outer_composer/{name}": value for name, value in outer_metrics.items()}
            )
            return loss_dict

        runner.alg.update = types.MethodType(update_with_outer_composer, runner.alg)

    elif (
        predictive_gating_enabled
        and env.unwrapped.predictive_allocator_updates_enabled()
        and population_allocator_enabled
    ):
        runner.alg = AntitheticPopulationPPO(
            runner.alg,
            env.unwrapped,
            env_cfg,
            runner.num_steps_per_env,
        )
        print(
            "[YawBot] Population meta-training: independent +residual/-residual PPO branches "
            f"use {env.unwrapped.num_envs // 2} environments each for "
            f"{env_cfg.predictive_allocator_meta_rollouts} rollouts per generation."
        )

    elif predictive_gating_enabled and env.unwrapped.predictive_allocator_updates_enabled():
        predictive_rollout_meta: list[dict[str, torch.Tensor]] = []
        meta_block: list[dict[str, object]] = []
        evaluation_rollout_pending = False
        warmup_rollouts_remaining = int(
            getattr(env_cfg, "predictive_allocator_score_bootstrap_rollouts", 1)
        )
        meta_block_inner_updates = int(env_cfg.predictive_allocator_meta_rollouts)
        if warmup_rollouts_remaining < 0:
            raise ValueError("predictive_allocator_score_bootstrap_rollouts must be non-negative.")
        if meta_block_inner_updates <= 0:
            raise ValueError("predictive_allocator_meta_rollouts must be positive.")
        original_process_env_step = runner.alg.process_env_step

        def process_env_step_with_predictive_meta(algorithm, obs, rewards, dones, extras):
            original_process_env_step(obs, rewards, dones, extras)
            meta = extras.get("predictive_meta")
            if meta is None:
                raise RuntimeError("Predictive training step did not provide allocator metadata.")
            predictive_rollout_meta.append(
                {
                    "context_contribution": meta["context_contribution"].detach(),
                    "allocator_context": meta["allocator_context"].detach(),
                    "allocator_sample": meta["allocator_sample"].detach(),
                    "allocator_log_probability": meta[
                        "allocator_log_probability"
                    ].detach(),
                    "allocator_residual": meta["allocator_residual"].detach(),
                    "allocator_coordinate": int(meta["allocator_coordinate"]),
                    "reference_rewards": meta["reference_rewards"].detach(),
                    "reward_components": meta["reward_components"].detach(),
                    "dones": dones.detach().clone(),
                }
            )

        runner.alg.process_env_step = types.MethodType(
            process_env_step_with_predictive_meta, runner.alg
        )
        original_predictive_algorithm_update = runner.alg.update

        def update_with_reward_allocator(algorithm):
            nonlocal evaluation_rollout_pending
            nonlocal warmup_rollouts_remaining
            if len(predictive_rollout_meta) != runner.num_steps_per_env:
                raise RuntimeError(
                    "Reward allocator received "
                    f"{len(predictive_rollout_meta)} transitions; expected {runner.num_steps_per_env}."
                )
            next_context = torch.stack(
                [step["context_contribution"] for step in predictive_rollout_meta], dim=0
            ).mean(dim=0)
            reference_rewards = torch.stack(
                [step["reference_rewards"] for step in predictive_rollout_meta], dim=0
            )
            reward_components = torch.stack(
                [step["reward_components"] for step in predictive_rollout_meta], dim=0
            )
            allocator_context = predictive_rollout_meta[0]["allocator_context"]
            allocator_sample = predictive_rollout_meta[0]["allocator_sample"]
            allocator_log_probability = predictive_rollout_meta[0][
                "allocator_log_probability"
            ]
            allocator_residual = predictive_rollout_meta[0]["allocator_residual"]
            for step in predictive_rollout_meta[1:]:
                if not torch.equal(step["allocator_context"], allocator_context):
                    raise RuntimeError("Aggregate allocator context changed inside one rollout.")
                if not torch.equal(step["allocator_sample"], allocator_sample):
                    raise RuntimeError("Aggregate allocator action changed inside one rollout.")
                if not torch.equal(step["allocator_residual"], allocator_residual):
                    raise RuntimeError("Allocator residual changed inside one rollout.")
            dones = torch.stack([step["dones"] for step in predictive_rollout_meta], dim=0)
            allocator_coordinates = {
                int(step["allocator_coordinate"]) for step in predictive_rollout_meta
            }
            if len(allocator_coordinates) != 1:
                raise RuntimeError("Allocator coordinate changed inside one PPO rollout.")
            current_allocator_coordinate = allocator_coordinates.pop()
            # Continuing-task average reward is invariant to the arbitrary PPO
            # rollout boundary. Keep one score per environment so the outer
            # trend can estimate uncertainty across 512 physical trajectories.
            current_scores = reference_rewards.mean(dim=0)
            current_score = float(current_scores.mean().item())
            current_advantages = algorithm.storage.advantages.detach().clone().squeeze(-1)
            current_allocation_blend = env.unwrapped.predictive_allocator_rollout_blend()
            current_transition = {
                "allocator_context": allocator_context,
                "allocator_sample": allocator_sample,
                "old_log_probability": allocator_log_probability,
                "allocator_residual": allocator_residual,
                "reference_rewards": reference_rewards,
                "dones": dones,
                "reward_components": reward_components,
                "ppo_advantages": current_advantages,
                "allocation_blend": current_allocation_blend,
                "allocator_coordinate": current_allocator_coordinate,
                "reference_score": current_score,
                "reference_scores": current_scores,
            }
            predictive_rollout_meta.clear()
            allocator_metrics: dict[str, float] = {}

            if warmup_rollouts_remaining > 0:
                loss_dict = original_predictive_algorithm_update()
                warmup_rollouts_remaining -= 1
                env.unwrapped.begin_predictive_allocator_rollout(
                    next_context, explore=True, resample_exploration=True
                )
                return loss_dict

            if not evaluation_rollout_pending:
                meta_block.append(current_transition)
                loss_dict = original_predictive_algorithm_update()
                evaluation_rollout_pending = len(meta_block) >= meta_block_inner_updates
                env.unwrapped.begin_predictive_allocator_rollout(
                    next_context,
                    explore=not evaluation_rollout_pending,
                    resample_exploration=False,
                )
                loss_dict["reward_allocator/meta_block_progress"] = (
                    len(meta_block) / meta_block_inner_updates
                )
                return loss_dict

            # This rollout evaluates the policy after a fixed block of inner
            # updates. Do not update PPO from it: otherwise its reward action
            # would be in flight while the outer policy changes.
            algorithm.storage.clear()
            loss_dict = {"reward_allocator/evaluation_only": 1.0}
            if len(meta_block) != meta_block_inner_updates:
                raise RuntimeError("Incomplete reward-allocator meta block at evaluation.")
            coordinates = {int(block["allocator_coordinate"]) for block in meta_block}
            if len(coordinates) != 1:
                raise RuntimeError("Allocator coordinate changed inside one meta block.")
            allocation_blends = {float(block["allocation_blend"]) for block in meta_block}
            if len(allocation_blends) != 1:
                raise RuntimeError("Allocator blend changed inside one meta block.")
            block_residual = meta_block[0]["allocator_residual"]
            if any(
                not torch.equal(block["allocator_residual"], block_residual)
                for block in meta_block[1:]
            ):
                raise RuntimeError("Allocator residual changed inside one meta block.")
            block_scores = torch.stack(
                [block["reference_scores"] for block in meta_block]
                + [current_scores],
                dim=0,
            )
            (
                reference_improvement,
                normalized_improvement,
                slope_standard_error,
                slope_sign_confidence,
            ) = block_reference_learning_trend(
                block_scores, float(env_cfg.predictive_reference_progress_scale)
            )
            allocator_metrics = env.unwrapped.update_predictive_reward_allocator(
                allocator_context=meta_block[0]["allocator_context"],
                allocator_sample=meta_block[0]["allocator_sample"],
                old_log_probability=meta_block[0]["old_log_probability"],
                behavior_reference_rewards=torch.cat(
                    [block["reference_rewards"] for block in meta_block], dim=0
                ),
                behavior_dones=torch.cat(
                    [block["dones"] for block in meta_block], dim=0
                ),
                behavior_reward_components=torch.cat(
                    [block["reward_components"] for block in meta_block], dim=0
                ),
                ppo_advantages=torch.cat(
                    [block["ppo_advantages"] for block in meta_block], dim=0
                ),
                reference_improvement=reference_improvement,
                normalized_reference_improvement=normalized_improvement,
                allocation_blend=allocation_blends.pop(),
                allocator_coordinate=coordinates.pop(),
            )
            allocator_metrics["block_start_score"] = float(block_scores[0].mean().item())
            allocator_metrics["block_evaluation_score"] = current_score
            allocator_metrics["block_learning_speed"] = reference_improvement
            allocator_metrics["block_slope_standard_error"] = slope_standard_error
            allocator_metrics["block_slope_sign_confidence"] = slope_sign_confidence
            centered_steps = (
                torch.arange(block_scores.shape[0], device=block_scores.device, dtype=block_scores.dtype)
                - 0.5 * (block_scores.shape[0] - 1)
            )
            per_environment_slopes = torch.sum(
                centered_steps.unsqueeze(-1)
                * (block_scores - block_scores.mean(dim=0, keepdim=True)),
                dim=0,
            ) / torch.square(centered_steps).sum().clamp_min(1.0e-6)
            allocator_metrics["block_score_fit_residual"] = float(
                torch.mean(
                    torch.square(
                        block_scores
                        - (
                            block_scores.mean(dim=0, keepdim=True)
                            + centered_steps.unsqueeze(-1)
                            * per_environment_slopes.unsqueeze(0)
                        )
                    )
                ).sqrt().item()
            )
            meta_block.clear()
            evaluation_rollout_pending = False
            env.unwrapped.begin_predictive_allocator_rollout(
                next_context, explore=True, resample_exploration=True
            )
            loss_dict.update(
                {f"reward_allocator/{name}": value for name, value in allocator_metrics.items()}
            )
            return loss_dict

        runner.alg.update = types.MethodType(update_with_reward_allocator, runner.alg)

    elif (
        predictive_gating_enabled
        and getattr(env_cfg, "predictive_allocator_train", False)
        and args_cli.pretrained_predictor
    ):
        # A frozen predictor still adapts its deterministic reward allocation to
        # the preceding rollout context. Only its parameters are frozen.
        frozen_context_contributions: list[torch.Tensor] = []
        original_process_env_step = runner.alg.process_env_step

        def process_env_step_with_frozen_predictor(algorithm, obs, rewards, dones, extras):
            original_process_env_step(obs, rewards, dones, extras)
            meta = extras.get("predictive_meta")
            if meta is None:
                raise RuntimeError("Frozen predictor step did not provide context metadata.")
            frozen_context_contributions.append(meta["context_contribution"].detach())

        runner.alg.process_env_step = types.MethodType(
            process_env_step_with_frozen_predictor, runner.alg
        )
        original_frozen_algorithm_update = runner.alg.update

        def update_with_frozen_predictor_context(algorithm):
            if len(frozen_context_contributions) != runner.num_steps_per_env:
                raise RuntimeError("Frozen predictor received an incomplete rollout context.")
            next_context = torch.stack(frozen_context_contributions).mean(dim=0)
            frozen_context_contributions.clear()
            loss_dict = original_frozen_algorithm_update()
            env.unwrapped.begin_predictive_allocator_rollout(next_context, explore=False)
            return loss_dict

        runner.alg.update = types.MethodType(update_with_frozen_predictor_context, runner.alg)

    if args_cli.distributed and not outer_composer_enabled:
        original_distributed_algorithm_update = runner.alg.update

        def update_with_distributed_task_state(algorithm):
            env.unwrapped.synchronize_distributed_training_state(initial=False)
            return original_distributed_algorithm_update()

        runner.alg.update = types.MethodType(
            update_with_distributed_task_state, runner.alg
        )

    if predictive_gating_enabled and not outer_composer_enabled and bool(
        getattr(agent_cfg, "rollout_adaptive_schedule", False)
    ):
        def install_rollout_adaptive_schedule(algorithm) -> None:
            original_update = algorithm.update
            minimum_rate = float(agent_cfg.rollout_adaptive_min_learning_rate)
            maximum_rate = float(agent_cfg.rollout_adaptive_max_learning_rate)
            factor = float(agent_cfg.rollout_adaptive_factor)
            desired_kl = float(agent_cfg.rollout_adaptive_desired_kl)
            acquisition_rollouts = int(
                getattr(agent_cfg, "rollout_adaptive_acquisition_rollouts", 0)
            )
            completed_rollouts = 0

            def update_once_per_rollout(inner_algorithm):
                nonlocal completed_rollouts
                loss_dict = original_update()
                evaluation_only = bool(
                    loss_dict.get("reward_allocator/evaluation_only", 0.0)
                )
                if evaluation_only:
                    loss_dict["rollout_kl"] = 0.0
                    return loss_dict
                rollout_kl = rollout_policy_kl(
                    inner_algorithm.policy,
                    inner_algorithm.storage.observations,
                    inner_algorithm.storage.mu,
                    inner_algorithm.storage.sigma,
                )
                if completed_rollouts < acquisition_rollouts:
                    learning_rate = min(
                        maximum_rate, inner_algorithm.learning_rate * factor
                    )
                else:
                    learning_rate = rollout_adaptive_learning_rate(
                        inner_algorithm.learning_rate,
                        rollout_kl,
                        desired_kl,
                        minimum_rate=minimum_rate,
                        maximum_rate=maximum_rate,
                        factor=factor,
                    )
                completed_rollouts += 1
                inner_algorithm.learning_rate = learning_rate
                for parameter_group in inner_algorithm.optimizer.param_groups:
                    parameter_group["lr"] = learning_rate
                loss_dict["rollout_kl"] = rollout_kl
                return loss_dict

            algorithm.update = types.MethodType(update_once_per_rollout, algorithm)

        if isinstance(runner.alg, AntitheticPopulationPPO):
            install_rollout_adaptive_schedule(runner.alg.positive)
            install_rollout_adaptive_schedule(runner.alg.negative)
        else:
            install_rollout_adaptive_schedule(runner.alg)

    if (
        predictive_gating_enabled
        and agent_cfg.algorithm.schedule == "fixed"
        and hasattr(agent_cfg, "learning_rate_decay_start_iteration")
    ):
        original_algorithm_update = runner.alg.update
        initial_learning_rate = float(agent_cfg.algorithm.learning_rate)
        decay_start = int(agent_cfg.learning_rate_decay_start_iteration)
        decay_end = int(agent_cfg.learning_rate_decay_end_iteration)
        final_learning_rate = float(agent_cfg.learning_rate_final)

        def update_with_learning_rate_consolidation(algorithm):
            learning_rate = linear_learning_rate_schedule(
                runner.current_learning_iteration,
                decay_start,
                decay_end,
                initial_learning_rate,
                final_learning_rate,
            )
            algorithm.learning_rate = learning_rate
            for parameter_group in algorithm.optimizer.param_groups:
                parameter_group["lr"] = learning_rate
            return original_algorithm_update()

        runner.alg.update = types.MethodType(update_with_learning_rate_consolidation, runner.alg)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dependency_record_path = os.path.join(log_dir, "params", "dependency_versions.json")
    with open(dependency_record_path, "w", encoding="utf-8") as dependency_stream:
        json.dump({"rsl-rl-lib": installed_version}, dependency_stream, indent=2, sort_keys=True)
        dependency_stream.write("\n")

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    depth_visualizer.close()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
