"""Pure tensor helpers for predictive-feasibility labels and gates.

Keeping these operations independent from Isaac Lab makes their semantics easy
to test without constructing a simulator.
"""

from __future__ import annotations

import torch


DIRECT_REWARD_NAMES = (
    "angle_penalty",
    "roll_pitch_rate_penalty",
    "projected_gravity",
    "yaw_rate_penalty",
    "action_rate_penalty",
    "action_magnitude_penalty",
    "stillness_penalty",
    "servo_motion_penalty",
    "vertical_velocity_penalty",
    "wheel_contact",
    "wheel_air_penalty",
    "wheel_yaw_tracking",
    "wheel_linear_tracking",
    "body_linear_progress",
    "wrong_direction_penalty",
    "wheel_linear_progress",
    "command_direction",
    "yaw_direction",
    "body_yaw_tracking",
    "stop_motion_penalty",
    "body_linear_tracking",
    "planar_position_penalty",
)


def command_trackable_label(
    commands: torch.Tensor,
    forward_velocity: torch.Tensor,
    yaw_velocity: torch.Tensor,
    *,
    stop_threshold: float,
    linear_error_threshold: float,
    yaw_error_threshold: float,
    stop_linear_threshold: float,
    stop_yaw_threshold: float,
) -> torch.Tensor:
    """Return whether linear and yaw commands are simultaneously trackable.

    A command component below ``stop_threshold`` is considered inactive and is
    not required to match a non-zero target.  If both components are inactive,
    the stricter standstill thresholds are used instead.
    """

    if commands.ndim != 2 or commands.shape[-1] != 2:
        raise ValueError(f"commands must have shape [B, 2], got {tuple(commands.shape)}.")
    if forward_velocity.shape != commands[:, 0].shape or yaw_velocity.shape != commands[:, 1].shape:
        raise ValueError("Velocity tensors must contain one scalar per command row.")

    linear_command = commands[:, 0]
    yaw_command = commands[:, 1]
    active_linear = torch.abs(linear_command) > float(stop_threshold)
    active_yaw = torch.abs(yaw_command) > float(stop_threshold)
    stop_command = ~active_linear & ~active_yaw

    linear_success = (~active_linear) | (
        torch.abs(forward_velocity - linear_command) <= float(linear_error_threshold)
    )
    yaw_success = (~active_yaw) | (torch.abs(yaw_velocity - yaw_command) <= float(yaw_error_threshold))
    moving_success = (~stop_command) & linear_success & yaw_success
    stop_success = stop_command & (torch.abs(forward_velocity) <= float(stop_linear_threshold)) & (
        torch.abs(yaw_velocity) <= float(stop_yaw_threshold)
    )
    return moving_success | stop_success


def stable_event_label(stage2_stable: torch.Tensor, reset_terminated: torch.Tensor) -> torch.Tensor:
    """Exclude true failures from the future-stability target; timeouts stay valid."""

    if stage2_stable.shape != reset_terminated.shape:
        raise ValueError("stage2_stable and reset_terminated must have the same shape.")
    return stage2_stable.bool() & ~reset_terminated.bool()


def aggregate_prerequisite_targets(
    event_sequence: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Aggregate every atomic event as a masked horizon-occupancy target.

    Requiring stable or grounded to hold at every step produced virtually no
    positive grounded windows for the wheel-legged robot.  Occupancy retains
    short contact interruptions while still penalizing persistently infeasible
    trajectories.
    """

    if event_sequence.ndim != 3 or event_sequence.shape[-1] != 4:
        raise ValueError(f"event_sequence must have shape [B, H, 4], got {tuple(event_sequence.shape)}.")
    if valid_mask is None:
        valid_mask = torch.ones(event_sequence.shape[:2], device=event_sequence.device, dtype=torch.bool)
    if valid_mask.shape != event_sequence.shape[:2]:
        raise ValueError(
            f"valid_mask must have shape {tuple(event_sequence.shape[:2])}, got {tuple(valid_mask.shape)}."
        )
    if torch.any(valid_mask.sum(dim=1) == 0):
        raise ValueError("Every sequence must contain at least one valid outcome.")

    mask = valid_mask.unsqueeze(-1)
    occupancy_sum = torch.where(mask, event_sequence, torch.zeros_like(event_sequence)).sum(dim=1)
    return occupancy_sum / valid_mask.sum(dim=1, keepdim=True).to(event_sequence.dtype)


def product_tnorm_tier_gates(probabilities: torch.Tensor) -> torch.Tensor:
    """Build fuzzy prerequisite gates from atomic marginal probabilities.

    This product t-norm is a soft conjunction, not an estimate of a calibrated
    joint probability.
    """

    if probabilities.ndim != 2 or probabilities.shape[-1] != 4:
        raise ValueError(f"probabilities must have shape [B, 4], got {tuple(probabilities.shape)}.")
    ones = torch.ones_like(probabilities[:, 0])
    return torch.stack(
        (
            ones,
            probabilities[:, 0],
            probabilities[:, 0] * probabilities[:, 1],
            probabilities.prod(dim=-1),
        ),
        dim=-1,
    )


def geometric_mean_tier_gates(probabilities: torch.Tensor) -> torch.Tensor:
    """Build length-normalized prerequisite gates from atomic marginals.

    A raw product shrinks exponentially as prerequisites are added.  Taking the
    cumulative geometric mean keeps every prerequisite influential without
    penalizing later tiers merely because they contain more factors.  The
    cumulative minimum preserves the required non-increasing tier order.

    Trackability is deliberately an auxiliary prediction rather than a gate for
    the tracking reward itself.  Otherwise a policy must already track a command
    before it receives a useful tracking gradient.  Tier 4 therefore adds the
    low-slip physical prerequisite to stable and grounded.
    """

    if probabilities.ndim != 2 or probabilities.shape[-1] != 4:
        raise ValueError(f"probabilities must have shape [B, 4], got {tuple(probabilities.shape)}.")
    probabilities = probabilities.clamp(0.0, 1.0)
    gate1 = torch.ones_like(probabilities[:, 0])
    gate2 = probabilities[:, 0]
    gate3 = torch.sqrt(probabilities[:, 0] * probabilities[:, 1])
    gate3 = torch.minimum(gate2, gate3)
    gate4 = torch.pow(probabilities[:, 0] * probabilities[:, 1] * probabilities[:, 3], 1.0 / 3.0)
    gate4 = torch.minimum(gate3, gate4)
    return torch.stack((gate1, gate2, gate3, gate4), dim=-1)


def command_aligned_velocities(
    linear_command: torch.Tensor,
    forward_velocity: torch.Tensor,
    wheel_surface_speed: torch.Tensor,
    *,
    stop_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Express body and wheel motion in the requested linear-command direction.

    Positive values mean motion with the command for both forward and reverse
    commands.  The returned mask excludes stop and pure-yaw commands from linear
    progress rewards.
    """

    if linear_command.shape != forward_velocity.shape or linear_command.shape != wheel_surface_speed.shape:
        raise ValueError("Command and velocity tensors must have identical shapes.")
    active_linear = torch.abs(linear_command) > float(stop_threshold)
    direction = torch.sign(linear_command)
    return active_linear, direction * forward_velocity, direction * wheel_surface_speed


def differential_drive_yaw_proxy(semantic_wheel_velocity: torch.Tensor) -> torch.Tensor:
    """Return the positive-yaw differential wheel-speed proxy.

    This robot moves forward along body +Y, with the left wheel on body +X.
    From ``v = omega x r``, positive +Z yaw therefore requires the left wheel
    to move faster than the right wheel.
    """

    if semantic_wheel_velocity.ndim != 2 or semantic_wheel_velocity.shape[-1] != 2:
        raise ValueError(
            f"semantic_wheel_velocity must have shape [B, 2], got {tuple(semantic_wheel_velocity.shape)}."
        )
    return 0.5 * (semantic_wheel_velocity[:, 0] - semantic_wheel_velocity[:, 1])


def linear_warmup_blend(progress_steps: int, warmup_steps: int, ramp_steps: int) -> float:
    """Return a linear blend after a fixed warm-up period.

    ``progress_steps`` is intentionally generic.  Predictive gate activation uses
    one global control-step counter rather than predictor optimizer calls, since
    termination-triggered partial-window updates become much more frequent when
    many environments are simulated in parallel.
    """

    if progress_steps <= warmup_steps:
        return 0.0
    return max(0.0, min(1.0, (progress_steps - warmup_steps) / float(max(1, ramp_steps))))


def optimizer_warmup_blend(optimizer_steps: int, warmup_steps: int, ramp_steps: int) -> float:
    """Backward-compatible alias for older callers and checkpoints."""

    return linear_warmup_blend(optimizer_steps, warmup_steps, ramp_steps)


def ordered_partial_horizon_indices(
    newest_index: int,
    sequence_lengths: torch.Tensor,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return oldest positions, ordered circular indices, and validity mask."""

    if horizon <= 0 or sequence_lengths.ndim != 1:
        raise ValueError("horizon must be positive and sequence_lengths must be one-dimensional.")
    if torch.any(sequence_lengths < 1) or torch.any(sequence_lengths > horizon):
        raise ValueError("sequence_lengths must lie in [1, horizon].")
    oldest_indices = (int(newest_index) - sequence_lengths + 1) % horizon
    offsets = torch.arange(horizon, device=sequence_lengths.device).unsqueeze(0)
    ordered_indices = (oldest_indices.unsqueeze(1) + offsets) % horizon
    valid_mask = offsets < sequence_lengths.unsqueeze(1)
    return oldest_indices, ordered_indices, valid_mask


__all__ = [
    "DIRECT_REWARD_NAMES",
    "aggregate_prerequisite_targets",
    "command_aligned_velocities",
    "command_trackable_label",
    "differential_drive_yaw_proxy",
    "geometric_mean_tier_gates",
    "linear_warmup_blend",
    "optimizer_warmup_blend",
    "ordered_partial_horizon_indices",
    "product_tnorm_tier_gates",
    "stable_event_label",
]
