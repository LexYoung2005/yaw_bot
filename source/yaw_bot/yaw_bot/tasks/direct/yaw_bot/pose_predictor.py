from __future__ import annotations

import torch
from torch import nn


class DepthPosePredictor(nn.Module):
    """Predict future body orientation, linear velocity, and angular velocity."""

    def __init__(
        self,
        depth_observation_height: int,
        depth_observation_width: int,
        state_observation_dim: int,
        history_steps: int,
        prediction_steps: int,
        hidden_dims: tuple[int, ...],
    ) -> None:
        super().__init__()
        output_dim = prediction_steps * 10
        self.depth_observation_height = depth_observation_height
        self.depth_observation_width = depth_observation_width
        self.state_observation_dim = state_observation_dim
        self.history_steps = history_steps

        self.encoder = nn.Sequential(
            nn.Conv2d(history_steps, 16, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(start_dim=1),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(history_steps * state_observation_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
        )

        fusion_layers: list[nn.Module] = []
        previous_dim = 32 * 4 * 4 + 64
        for hidden_dim in hidden_dims:
            fusion_layers.extend([nn.Linear(previous_dim, hidden_dim), nn.ELU()])
            previous_dim = hidden_dim
        output_layer = nn.Linear(previous_dim, output_dim)
        fusion_layers.append(output_layer)
        self.head = nn.Sequential(*fusion_layers)
        self.prediction_steps = prediction_steps

        # Start from a stationary identity-orientation forecast with zero velocities.
        nn.init.normal_(output_layer.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(output_layer.bias)
        with torch.no_grad():
            output_layer.bias.view(prediction_steps, 10)[:, 0] = 1.0

    def forward(self, depth_history: torch.Tensor, state_history: torch.Tensor) -> torch.Tensor:
        depth_history = depth_history.view(
            -1,
            self.history_steps,
            self.depth_observation_height,
            self.depth_observation_width,
        )
        state_history = state_history.view(-1, self.history_steps * self.state_observation_dim)
        fused_features = torch.cat(
            [self.encoder(depth_history), self.state_encoder(state_history)],
            dim=-1,
        )
        prediction = self.head(fused_features).view(-1, self.prediction_steps, 10)
        quaternion = torch.nn.functional.normalize(prediction[..., :4], dim=-1, eps=1.0e-6)
        return torch.cat([quaternion, prediction[..., 4:]], dim=-1)


def future_state_prediction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    linear_velocity_weight: float,
    angular_velocity_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute sign-invariant quaternion loss and velocity regression losses."""
    predicted_quaternion = prediction[..., :4]
    target_quaternion = target[..., :4]
    same_sign_error = torch.sum(torch.square(predicted_quaternion - target_quaternion), dim=-1)
    opposite_sign_error = torch.sum(torch.square(predicted_quaternion + target_quaternion), dim=-1)
    quaternion_loss = torch.mean(torch.minimum(same_sign_error, opposite_sign_error))
    linear_velocity_loss = torch.mean(torch.square(prediction[..., 4:7] - target[..., 4:7]))
    angular_velocity_loss = torch.mean(torch.square(prediction[..., 7:10] - target[..., 7:10]))
    total_loss = (
        quaternion_loss
        + linear_velocity_weight * linear_velocity_loss
        + angular_velocity_weight * angular_velocity_loss
    )
    return total_loss, quaternion_loss, linear_velocity_loss, angular_velocity_loss
