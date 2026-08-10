# Paper-to-Code Correspondence

The map below is intentionally explicit: every method component and reported
protocol choice has one canonical implementation location.

| Manuscript item | Canonical implementation |
|---|---|
| 75-D actor observation (25-D state + 50-D forecast) | `yaw_bot_env_cfg.py`, `pose_predictor.py`, and `YawBotEnv._get_observations` |
| Six normalized actions and parallel-leg mapping | `yaw_bot_env.py::_pre_physics_step`, `_apply_action`, and `_parallel_knee_angle` |
| 120 Hz physics / 60 Hz control | `YawBotEnvCfg.sim.dt` and `YawBotEnvCfg.decimation` |
| Four-frame training-only predictive history | `YawBotPredictiveGatedEnvCfg.predictive_history_steps` |
| 32/64/32-D depth/state/action encoders and 128-D fusion | `predictive_feasibility.py::PredictiveFeasibilityModel` |
| Auxiliary event ensemble and 12-step future-state head | `PredictiveFeasibilityModel.event_heads` and `.future_head` |
| 22 atomic rewards | `predictive_labels.py::DIRECT_REWARD_NAMES` and `YawBotEnv._get_rewards` |
| Exhaustive five-group partition, Eq. (2) | `outer_advantage_composer.py::REWARD_GROUP_INDICES` and `group_atomic_rewards` |
| RMS decay 0.999, no mean subtraction, pre-update scale | `RunningGroupRMS` |
| 128-64 Composer with zero output initialization | `CenteredTanhComposer` |
| Bounded mean-one projection, Eqs. (3)-(4) | `CenteredTanhComposer.weights_from_logits` |
| 20% deployment warmup, Eq. (5) | `beta_schedule` and `effective_composer_weights` |
| Fixed safety reward and composed actor reward, Eq. (6) | `compose_actor_reward` and `YawBotEnv._get_rewards` |
| Compose-before-GAE recursion, Eq. (7) | `differentiable_shaping_advantage` |
| Even/odd environment split | `composer_meta_gradient_loss` |
| Functional one-step Adam, Eq. (8) | `_virtual_adam_parameters` and `_virtual_adam_parameter` |
| Environment-split task-advantage objective, Eq. (9) | `composer_meta_gradient_loss` |
| Immutable task objective, Eq. (1) | `outer_reward` |
| Outer critic (75-256-128-1) | `OuterCritic` |
| Actor/critic (75-256-128-64 outputs) | `agents/rsl_rl_ppo_cfg.py` |
| Outer-only, Uniform, Static controls | the corresponding classes in `yaw_bot_env_cfg.py` |
| Static vector (1.2, 0.8, 1.4, 0.7, 0.9) | `YawBotStaticRewardPPOEnvCfg.outer_static_group_weights` |
| PPO-LIRPG adapter | `agents/official_lirpg.py` and `LIRPGPPORunnerCfg` |
| ReLara-PPO adapter | `agents/relara.py` and `YawBotReLaraPPOEnvCfg` |
| 512 envs, 24 steps, 1500 iterations, three seeds | `configs/paper_experiments.json` |
| Trailing-100 checkpoint selection | `scripts/rsl_rl/select_best_reward_checkpoint.py` |
| Three fresh evaluation seeds and 7200 evaluation steps | `configs/paper_experiments.json` and `scripts/rsl_rl/play.py` |
| Eight reported metrics, including saturation at `abs(a) >= 0.98` | `YawBotEnv._get_rewards` evaluation diagnostics and `scripts/plot_results.py` |
| Complete 22-term expressions and scales | `configs/reward_terms.json` |
| Exact review hardware and software versions | `configs/environment.json` |
| Figure 3 Composer-weight analysis | `scripts/plot_composer_weights.py` and `results/submitted_training_curves.npz` |
| Figure 4 training-curve analysis | `scripts/plot_training_curves.py` and `results/submitted_training_curves.npz` |
| Figure 5 eight-metric analysis | `scripts/plot_results.py` and `results/evaluation_json/` |
| Terrain material and robot-side friction randomization | `PROTOCOL_NOTES.md` and `EventCfg` |
| Machine-readable control, observation, sensing, commands, and kinematics | `configs/task_interface.json` |

## Reward groups

The tuple order is the order in `DIRECT_REWARD_NAMES`.

- Stability: angle, roll/pitch rate, projected gravity, vertical velocity.
- Contact/slip: wheel contact and wheel-air penalty.
- Linear: wheel/body tracking, body/wheel progress, wrong direction, signed
  direction, and planar-position penalty.
- Yaw: yaw-rate penalty, wheel/body yaw tracking, and signed yaw direction.
- Regularization: action rate, action magnitude, stillness, servo motion, and
  stop-motion penalty.

`tests/test_paper_correspondence.py` checks this partition, the projection
bounds, network dimensions, task objective, static vector, and paper protocol.
