# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg, RayCasterCameraCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

from yaw_bot.robots import YAW_BOT_CFG

YAW_BOT_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    # Tune terrain features to the yaw_bot footprint (~0.19 m x 0.11 m).
    # The default Isaac Lab examples are sized for much larger robots, which
    # makes this platform drive a long distance before encountering elevation changes.
    size=(1.6, 1.6),
    border_width=2.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    color_scheme="height",
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.008, 0.04),
            step_width=0.12,
            platform_width=0.35,
            border_width=0.15,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.008, 0.04),
            step_width=0.12,
            platform_width=0.35,
            border_width=0.15,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.15, grid_height_range=(0.008, 0.035), platform_width=0.35
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.005, 0.02), noise_step=0.005, border_width=0.1
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.12), platform_width=0.35, border_width=0.1
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.12), platform_width=0.35, border_width=0.1
        ),
    },
)

@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.6), # 减小范围
            "dynamic_friction_range": (0.7, 1.3), # 减小范围
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Body"),
            "mass_distribution_params": (-0.1, 0.2), # 减小质量增减的范围
            "operation": "add",
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 7.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {
                "x": (-0.2, 0.2), # 减小推力范围
                "y": (-0.2, 0.2), # 减小推力范围
                "z": (-0.05, 0.05), # 减小推力范围
                "roll": (-0.05, 0.05), # 减小角度扰动范围
                "pitch": (-0.05, 0.05), # 减小角度扰动范围
                "yaw": (-0.15, 0.15), # 减小角度扰动范围
            },
        },
    )

@configclass
class YawBotEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 60.0
    disable_termination = False
    events: EventCfg = EventCfg()

    # spaces
    action_space = 6
    depth_observation_height = 54
    depth_observation_width = 96
    pose_predictor_history_steps = 5
    pose_predictor_future_steps = 5
    pose_predictor_state_dim = 25
    pose_predictor_output_dim = 10
    pose_prediction_dim = pose_predictor_future_steps * pose_predictor_output_dim
    policy_command_observation_start = 13
    observation_space = {
        "policy": 25 + pose_prediction_dim,
        "depth": [1, depth_observation_height, depth_observation_width],
    }
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.4,
            dynamic_friction=1.2,
            restitution=0.0,
        ),
    )

    # robot
    robot_cfg: ArticulationCfg = YAW_BOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512,
        env_spacing=0.5,
        replicate_physics=True,
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=YAW_BOT_ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.4,
            dynamic_friction=1.2,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    reset_height_offset = 0.04

    # joint names
    body_link_name = "Body"
    left_wheel_body_name = "L_wheel"
    right_wheel_body_name = "R_wheel"
    left_hip_joint_name = "Body_r_1"
    left_knee_joint_name = "L_leg1_r_4"
    left_wheel_joint_name = "L_leg2_r_7"
    right_hip_joint_name = "Body_r_8"
    right_knee_joint_name = "R_leg1_r_9"
    right_wheel_joint_name = "R_leg2_r_10"

    # default leg posture is now defined by branch hip angle a and mapped hip angle b
    default_branch_hip_angle = 0.8726646259971648
    default_mapped_hip_angle = 0.6981317007977318

    # observation noise
    enable_imu_noise = True
    imu_quat_noise_std = 0.01
    imu_ang_vel_noise_std = 0.08
    imu_projected_gravity_noise_std = 0.03

    # velocity command settings
    use_velocity_commands = True
    use_fixed_velocity_command = False
    fixed_command_lin_vel_x = 0.2
    fixed_command_yaw_vel = 0.0
    command_lin_vel_x_range = (-1.0, 1.0)
    command_yaw_vel_range = (-1.0, 1.0)
    command_resample_time_range = (5.0, 10.0)
    resample_commands = True
    velocity_command_curriculum_start_stage = 3
    command_lin_vel_x_min_abs = 0.15
    command_yaw_vel_min_abs = 0.35
    command_yaw_probability = 0.5
    command_stop_probability = 0.15
    command_stop_threshold = 0.05
    command_stop_yaw_rate_threshold = 0.10
    command_tracking_sigma_lin = 0.15
    command_tracking_sigma_yaw = 0.3
    command_tracking_lin_error_threshold = 0.20
    command_tracking_yaw_error_threshold = 0.35
    command_tracking_upright_sigma = 0.08
    command_tracking_stability_sigma = 0.5
    command_tracking_gravity_sigma = 0.05

    # action ranges
    branch_hip_action_scale = 0.35
    mapped_hip_action_scale = 0.35
    wheel_action_scale = 10.0
    mapped_hip_lower_limit = 0.0
    mapped_hip_upper_limit = 1.3962634015954636
    wheel_radius = 0.0325

    # non-wheel contact termination
    termination_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/(Body|L_leg1|L_leg2|R_leg1|R_leg2)",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
    )
    # wheel-ground contact used by rewards and curriculum gates
    wheel_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/(L_wheel|R_wheel)",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
    )
    body_imu: ImuCfg = ImuCfg(
        prim_path="/World/envs/env_.*/Robot/Body",
        update_period=0.0,
        debug_vis=False,
        offset=ImuCfg.OffsetCfg(
            # Approximate Body geometric center from the URDF mesh bounding box.
            pos=(0.0, -0.0075, -0.0109),
        ),
    )
    depth_camera: RayCasterCameraCfg = RayCasterCameraCfg(
        prim_path="/World/envs/env_.*/Robot/Body",
        # Only the shared terrain participates in depth generation. Robots from
        # this or any other cloned environment are excluded by construction.
        mesh_prim_paths=["/World/ground"],
        update_period=0.0,
        data_types=["distance_to_image_plane"],
        depth_clipping_behavior="max",
        max_distance=2.0,
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=18.0,
            horizontal_aperture=28.0,
            height=108,
            width=192,
        ),
        # The robot's semantic forward direction is +Y. In the world camera
        # convention, this faces +Y and pitches the camera down by about 15 degrees.
        offset=RayCasterCameraCfg.OffsetCfg(
            pos=(0.0, 0.035, 0.025),
            rot=(0.70105738, -0.09229596, 0.09229596, 0.70105738),
            convention="world",
        ),
    )
    depth_max_distance = 2.0
    pose_predictor_enabled = True
    pose_predictor_hidden_dims = (256, 128)
    pose_predictor_learning_rate = 1.0e-3
    pose_predictor_train = True
    pose_predictor_train_interval = 4
    pose_predictor_batch_size = 1024
    pose_predictor_gradient_clip = 1.0
    pose_predictor_linear_velocity_loss_weight = 0.25
    pose_predictor_angular_velocity_loss_weight = 0.1
    termination_contact_force_threshold = 0.1
    termination_body_only = False
    termination_max_gravity_xy_error = None
    wheel_contact_normal_force_threshold = 1.0

    # reward scales
    rew_scale_alive = 1.0
    # Keep the proven 7.17 reward basis.  The learned allocator fine-tunes these
    # terms; it must not also be asked to compensate for an unvalidated rescale
    # of the complete inner-PPO objective.
    rew_scale_terminated = -15.0
    rew_scale_angle = -0.2
    rew_scale_ang_vel = -0.03
    rew_scale_yaw_ang_vel = -0.05
    rew_scale_projected_gravity = 0.1
    rew_scale_joint_action_rate = -1.0e-4
    # New in the direct allocator.  Keep it weak so hip corrections required
    # for balance are not suppressed before the allocator has learned.
    rew_scale_action_magnitude = -0.02
    rew_scale_vertical_vel = -1.0
    rew_scale_pre_stage3_still = -2.0
    rew_scale_pre_stage3_servo_motion = -0.02
    rew_scale_command_stop_motion = -2.0
    rew_scale_wheel_contact = 1.0
    rew_scale_track_lin_vel = 8.0
    rew_scale_track_yaw_vel = 6.0
    rew_scale_track_wheel_lin = 4.0
    rew_scale_track_wheel_yaw = 2.0
    rew_scale_planar_position_error = -2.0
    rew_scale_forward_vel = 8.0
    rew_scale_forward_progress = 3.0
    rew_scale_backward_vel = -6.0
    rew_scale_direction = 3.0
    rew_scale_yaw_direction = 1.0
    rew_scale_wheel_air = -2.5
    forward_velocity_cap = 0.35
    forward_progress_cap = 0.25
    stability_gravity_error_threshold = 0.08
    stability_ang_vel_threshold = 3.0
    wheel_speed_gate_threshold = 0.03
    body_speed_gate_threshold = 0.03
    curriculum_enable = True
    curriculum_unlock_rate = 0.7
    curriculum_stage2_min_episode_ratio = 0.85
    curriculum_ema_alpha = 0.02
    curriculum_window_episodes = 10
    curriculum_warmup_episodes = 20
    curriculum_check_interval_episodes = 10
    reward_gate_enable = True


@configclass
class YawBotPredictiveGatedEnvCfg(YawBotEnvCfg):
    """Deployable observations with training-only predictive prerequisite gating.

    The actor receives hardware-available proprioception plus the retained
    depth-to-future-pose prediction. Future-event heads and Composer state are
    training-only.
    """

    # Method switch.  The original task keeps this disabled and remains checkpoint
    # compatible with the pose-prediction baseline.
    predictive_gating_enable = True
    # Preserve the proven 7.17 actor-side future-pose observation. This model
    # is separate from the training-only reward allocator below.
    pose_predictor_enabled = True
    pose_predictor_train = True
    # Direct allocation means no hand-written stage or reward gate. The
    # original PPO task retains those mechanisms as a separate baseline.
    curriculum_enable = False
    reward_gate_enable = False
    # Use the same crisp failure definition that taught the baseline to avoid
    # resting on its knees/leg frame.
    termination_body_only = False
    termination_max_gravity_xy_error = None

    # Four frames at the 60 Hz control rate and a 0.2 s prediction horizon.  The
    # reduced image size keeps the on-policy future-supervision queue bounded.
    # Keep the actor-side pose predictor on the validated 7.17 image geometry.
    depth_observation_height = 54
    depth_observation_width = 96
    predictive_history_steps = 4
    predictive_future_steps = 12
    predictive_state_dim = 28
    # Keep the inner PPO observation on the proven 7.17 proprioceptive basis.
    # The Predictor may use the richer 28-D state internally, but its online
    # auxiliary updates must not replace the Actor's state representation.
    predictive_actor_state_dim = 25
    predictive_action_dim = 6
    predictive_event_dim = 4
    predictive_reward_dim = 22
    predictive_future_state_dim = 10
    predictive_depth_latent_dim = 32
    predictive_ensemble_size = 3
    predictive_ensemble_bootstrap_probability = 0.8
    # Predictor-side stochastic work must never advance PPO's global Torch RNG.
    # Keeping a private stream makes same-seed reward ablations start from the
    # same Actor and preserves the Actor's action/minibatch sampling sequence.
    predictive_random_seed = 314159

    # Reward allocation is training-side machinery and never enters the Actor.
    # Actor input remains the proven 7.17 25-D state + 50-D future-pose output.
    # critic: policy observation + privileged linear velocity/contact/slip (6)
    predictive_actor_pose_prediction_dim = 50
    predictive_policy_observation_dim = (
        predictive_actor_state_dim + predictive_actor_pose_prediction_dim
    )
    predictive_critic_privileged_dim = 0
    policy_command_observation_start = 13
    observation_space = {
        "policy": predictive_policy_observation_dim,
        "critic": predictive_policy_observation_dim + predictive_critic_privileged_dim,
    }

    # Auxiliary predictor optimization and conservative soft gating.
    predictive_feasibility_train = True
    # Default architecture: online H=12 supervision plus outer-advantage
    # guided five-group composition.  The historical 22-D stochastic allocator
    # remains available through YawBotPredictiveMetaAllocatorEnvCfg below.
    outer_advantage_composer_enable = True
    # The simulator, PPO and observation/action paths stay identical across
    # reward-learning baselines. Only this training-time reward principle
    # changes: composer, uniform, static, or lirpg.
    outer_reward_composition_mode = "composer"
    outer_static_group_weights = (1.0, 1.0, 1.0, 1.0, 1.0)
    predictive_allocator_train = False
    # The active path differentiates through one virtual PPO step and evaluates
    # it on a disjoint environment half using fixed-objective advantage. The old
    # two-branch population remains available only as an explicit diagnostic.
    predictive_allocator_population_enable = False
    predictive_fixed_weight_control_use_baseline_reward = False
    predictive_fixed_initial_weights_control = False
    predictive_learning_rate = 5.0e-4
    predictive_weight_decay = 1.0e-5
    predictive_train_interval = 4
    predictive_batch_size = 512
    predictive_gradient_clip = 1.0
    predictive_ema_decay = 0.995
    predictive_uncertainty_beta = 1.0
    outer_composer_learning_rate = 3.0e-4
    outer_composer_train = True
    outer_critic_learning_rate = 1.0e-3
    outer_critic_learning_epochs = 3
    outer_critic_gradient_clip = 1.0
    outer_composer_gradient_clip = 1.0
    outer_composer_weight_half_range = 0.4
    # The composer is optimized through one virtual Adam/PPO actor update:
    # even environment IDs provide the inner shaping update and odd IDs
    # evaluate its fixed-outer effect. No same-rollout reward-alignment loss,
    # REINFORCE credit, or meta-population is used.
    outer_composer_weight_to_one = 1.0e-3
    outer_composer_temporal_smoothness = 1.0e-2
    lirpg_learning_rate = 1.0e-4
    lirpg_gradient_clip = 0.5
    # Official MuJoCo PPO-LIRPG uses 0.01 * extrinsic + 1.0 * intrinsic.
    lirpg_extrinsic_coefficient = 0.01
    lirpg_intrinsic_coefficient = 1.0
    lirpg_reward_l2 = 0.0
    lirpg_temporal_smoothness = 0.0
    # ICML 2024 ReLara Reward-Agent defaults.  The policy-agent SAC from the
    # author repository is replaced only by the common yaw_bot PPO controller.
    relara_implementation_version = 1
    relara_reward_scale = 1.0
    relara_beta = 0.2
    relara_gamma = 0.99
    relara_replay_capacity = 1_000_000
    relara_batch_size = 512
    relara_learning_starts = 5_000
    relara_actor_learning_rate = 3.0e-4
    relara_critic_learning_rate = 1.0e-3
    relara_alpha_learning_rate = 1.0e-4
    relara_policy_frequency = 2
    relara_target_frequency = 1
    relara_tau = 0.005
    # Official autotuning initializes log_alpha=0, hence alpha=1 regardless of
    # the unused --ra-alpha fallback value.
    relara_initial_alpha = 1.0
    relara_alpha_autotune = True
    outer_group_rms_decay = 0.999
    # Beta only controls deployment interpolation from uniform internal
    # rewards to Composer weights. Meta-training itself always uses beta=1.
    outer_beta_maximum = 1.0
    outer_beta_warmup_fraction = 0.2
    outer_only_actor_reward = False
    outer_reward_termination_penalty = 5.0
    outer_reward_action_cost = 0.01
    predictive_reward_weight_min = 0.0
    predictive_reward_weight_max = 2.00
    # Initial allocator policy, not a runtime gate or a phase schedule.  This is
    # the canonical PPO's verified balance objective expressed as one point in
    # the 22-D weight space: the eight balance terms start at 1 and every other
    # term remains present at the common lower bound.  Commands are still active
    # and every value remains continuously trainable from iteration zero.
    predictive_reward_initial_weights = (
        1.00,  # angle_penalty
        1.00,  # roll_pitch_rate_penalty
        1.00,  # projected_gravity
        1.00,  # yaw_rate_penalty
        1.00,  # action_rate_penalty
        0.00,  # action_magnitude_penalty
        1.00,  # stillness_penalty
        1.00,  # servo_motion_penalty
        1.00,  # vertical_velocity_penalty
        0.00,  # wheel_contact
        0.00,  # wheel_air_penalty
        0.00,  # wheel_yaw_tracking
        0.00,  # wheel_linear_tracking
        0.00,  # body_linear_progress
        0.00,  # wrong_direction_penalty
        0.00,  # wheel_linear_progress
        0.00,  # command_direction
        0.00,  # yaw_direction
        0.00,  # body_yaw_tracking
        0.00,  # stop_motion_penalty
        0.00,  # body_linear_tracking
        0.00,  # planar_position_penalty
    )
    # Direct allocation starts immediately. There is no hand-written reward
    # phase and no hidden warm-up gate.
    predictive_gate_warmup_control_steps = 0
    predictive_gate_ramp_control_steps = 0
    predictive_event_loss_weight = 1.0
    predictive_future_loss_weight = 0.25
    # Hold one contextual reward policy fixed for four inner updates, then use
    # a PPO-free rollout as the final policy snapshot of the outer meta block.
    predictive_allocator_meta_rollouts = 4
    predictive_allocator_learning_rate = 5.0e-4
    # Keep boundary exploration local: half the 22-D prior is deliberately at
    # the lower bound, so a 0.10 residual asymmetrically switched on many motion
    # terms even before the allocator had evidence.
    predictive_allocator_exploration_std = 0.02
    # The reward policy is genuinely 22-D: every meta block explores all reward
    # coordinates instead of spending 22 blocks on a serial coordinate sweep.
    predictive_allocator_coordinate_exploration = False
    # The deterministic head still allocates all 22 weights state by state, but
    # one shared 22-D exploration residual is held for the complete meta block.
    # This gives inner PPO a stationary reward function and gives the outer loop
    # one identifiable causal action instead of transition-frequency reward noise.
    predictive_allocator_statewise_exploration = False
    # The first rollout starts with every environment freshly reset and its
    # fixed score is not sampled from the steady on-policy episode-age
    # distribution. Discard only S1-S0; reward allocation and PPO still run.
    predictive_allocator_score_bootstrap_rollouts = 1
    predictive_allocator_gamma = 0.995
    predictive_allocator_ppo_credit = 0.25
    predictive_allocator_progress_credit = 1.00
    # Population credit is the difference between two eight-rollout learning
    # slopes, whose natural scale is comparable to the earlier block slope.
    predictive_reference_progress_scale = 0.25
    predictive_allocator_importance_clip = 0.20
    predictive_allocator_regularization = 1.0e-3
    predictive_allocator_context_ema_decay = 0.50

    # Immutable reference objective used only to judge the allocator.  These
    # coefficients are deliberately not part of DIRECT_REWARD_NAMES and cannot
    # be changed by the predictor.
    reference_alive_weight = 1.0
    # One fixed final objective; no coefficient changes with iteration/state.
    reference_terminal_penalty = -15.0
    reference_stability_weight = 0.50
    reference_grounded_weight = 0.25
    reference_tracking_weight = 0.50
    reference_low_slip_weight = 0.25
    reference_action_penalty = 0.10
    predictive_slip_threshold = 0.12
    predictive_contact_force_scale = 20.0
    predictive_slip_scale = 0.35

    # Legacy baseline scales remain configured for log/checkpoint compatibility.
    # With reward_gate_enable=False their stage-dependent path is inactive.
    rew_scale_pre_stage3_still = -2.0
    rew_scale_pre_stage3_servo_motion = -0.02

    # Cast only the pixels consumed by the predictive task. The baseline keeps
    # its higher-resolution ray camera for backward compatibility.
    depth_camera = YawBotEnvCfg().depth_camera.replace(
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=18.0,
            horizontal_aperture=28.0,
            height=depth_observation_height,
            width=depth_observation_width,
        )
    )


@configclass
class YawBotOuterOnlyPPOEnvCfg(YawBotPredictiveGatedEnvCfg):
    """Strict actor-reward ablation for the outer-advantage composer.

    Everything except the reward delivered to PPO is kept identical to the
    predictive/composer task.  PPO receives exactly the same fixed simple outer
    reward as the original ablation for every transition; online predictor
    targets and training-only diagnostics remain active so this is still a
    one-variable comparison rather than a different environment or PPO
    implementation.  ``outer_only_actor_reward`` is an explicit compatibility
    switch because beta now interpolates internal-reward weights only.
    """

    outer_beta_maximum = 0.0
    outer_composer_train = False
    outer_only_actor_reward = True


@configclass
class YawBotPredictiveFixedWeightsEnvCfg(YawBotPredictiveGatedEnvCfg):
    """Single-variable ablation: identical predictive task with all weights fixed at one."""

    predictive_allocator_train = False
    outer_advantage_composer_enable = False
    # A real fixed-weight control must reproduce the baseline reward exactly;
    # otherwise the three newly proposed atomic terms confound the ablation.
    predictive_fixed_weight_control_use_baseline_reward = True


@configclass
class YawBotUniformRewardPPOEnvCfg(YawBotPredictiveGatedEnvCfg):
    """A1: fixed unit weights on the exact five normalized reward groups."""

    outer_reward_composition_mode = "uniform"
    outer_static_group_weights = (1.0, 1.0, 1.0, 1.0, 1.0)
    outer_composer_train = False
    outer_beta_maximum = 1.0
    outer_beta_warmup_fraction = 0.2


@configclass
class YawBotStaticRewardPPOEnvCfg(YawBotPredictiveGatedEnvCfg):
    """Paper Static control with one state-independent five-group vector."""

    outer_reward_composition_mode = "static"
    outer_static_group_weights = (1.2, 0.8, 1.4, 0.7, 0.9)
    outer_composer_train = False
    outer_beta_maximum = 1.0
    outer_beta_warmup_fraction = 0.2


@configclass
class YawBotLIRPGPPOEnvCfg(YawBotPredictiveGatedEnvCfg):
    """A3: PPO+LIRPG scalar intrinsic reward with held-out outer credit."""

    # The source-faithful adapter is regression-tested; formal launches are
    # explicitly enabled by the experiment queue.
    training_launch_paused = False
    lirpg_implementation_version = 2
    outer_reward_composition_mode = "lirpg"
    outer_composer_train = True
    # LIRPG learns its scalar reward from rollout zero; it has no deployment
    # interpolation parameter in the original algorithm.
    outer_beta_maximum = 1.0


@configclass
class YawBotReLaraPPOEnvCfg(YawBotPredictiveGatedEnvCfg):
    """ICML 2024 ReLara Reward Agent with the common yaw_bot PPO policy."""

    outer_reward_composition_mode = "relara"
    outer_composer_train = False
    # ReLara has a fixed r_E + beta*r_S mixture, not Composer's beta schedule.
    outer_beta_maximum = 0.2


@configclass
class YawBotPredictiveBalanceFixedEnvCfg(YawBotPredictiveGatedEnvCfg):
    """Diagnostic control: freeze the direct reward head at its balance prior."""

    predictive_allocator_train = False
    outer_advantage_composer_enable = False
    predictive_fixed_initial_weights_control = True


@configclass
class YawBotPredictiveMetaAllocatorEnvCfg(YawBotPredictiveGatedEnvCfg):
    """Legacy dedicated predictor/meta-population reward allocator."""

    outer_advantage_composer_enable = False
    predictive_allocator_train = True
