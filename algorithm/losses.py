"""Masked Baseline V1 position and physical-velocity losses."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _validate_inputs(
    prediction: torch.Tensor,
    target: torch.Tensor,
    timestamps: torch.Tensor,
    sequence_mask: torch.Tensor,
    target_valid_mask: torch.Tensor,
) -> None:
    if prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError(f"prediction must have shape [B, T, 3], got {tuple(prediction.shape)}")
    if target.shape != prediction.shape:
        raise ValueError("target must match prediction shape")
    if timestamps.shape != prediction.shape[:2]:
        raise ValueError("timestamps must have shape [B, T]")
    if sequence_mask.shape != timestamps.shape or target_valid_mask.shape != timestamps.shape:
        raise ValueError("sequence_mask and target_valid_mask must have shape [B, T]")
    if sequence_mask.dtype != torch.bool or target_valid_mask.dtype != torch.bool:
        raise ValueError("masks must have bool dtype")
    if not torch.isfinite(timestamps[sequence_mask]).all():
        raise ValueError("timestamps at real timesteps must be finite")


def velocity_pair_data(
    timestamps: torch.Tensor,
    sequence_mask: torch.Tensor,
    target_valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return valid adjacent-pair mask and dt; reject non-increasing real time."""
    real_pairs = sequence_mask[:, :-1] & sequence_mask[:, 1:]
    dt = timestamps[:, 1:] - timestamps[:, :-1]
    if torch.any(dt[real_pairs] <= 0):
        bad = torch.nonzero(real_pairs & (dt <= 0), as_tuple=False)[0].tolist()
        raise ValueError(f"neck timestamps must be strictly increasing; invalid pair at batch/time={bad}")
    pair_mask = real_pairs & target_valid_mask[:, :-1] & target_valid_mask[:, 1:]
    return pair_mask, dt


class BaselineLoss(nn.Module):
    """Per-fragment reduction of SmoothL1 position plus L1 velocity."""

    def __init__(
        self,
        position_weight: float = 1.0,
        velocity_weight: float = 0.1,
        position_beta: float = 0.05,
    ) -> None:
        super().__init__()
        if position_weight < 0 or velocity_weight < 0 or position_beta <= 0:
            raise ValueError("loss weights must be non-negative and position_beta must be positive")
        self.position_weight = position_weight
        self.velocity_weight = velocity_weight
        self.position_beta = position_beta

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        timestamps: torch.Tensor,
        sequence_mask: torch.Tensor,
        target_valid_mask: torch.Tensor,
    ) -> dict[str, Any]:
        _validate_inputs(prediction, target, timestamps, sequence_mask, target_valid_mask)
        supervision_mask = sequence_mask & target_valid_mask
        pair_mask, dt = velocity_pair_data(timestamps, sequence_mask, target_valid_mask)

        position_values: list[torch.Tensor] = []
        velocity_values: list[torch.Tensor] = []
        for batch_index in range(prediction.shape[0]):
            selected_prediction = prediction[batch_index][supervision_mask[batch_index]]
            selected_target = target[batch_index][supervision_mask[batch_index]]
            if selected_target.numel():
                if not torch.isfinite(selected_target).all() or not torch.isfinite(selected_prediction).all():
                    raise ValueError(f"non-finite supervised position at batch index {batch_index}")
                position_values.append(
                    F.smooth_l1_loss(
                        selected_prediction,
                        selected_target,
                        beta=self.position_beta,
                        reduction="mean",
                    )
                )

            selected_pairs = pair_mask[batch_index]
            if torch.any(selected_pairs):
                # Select valid endpoints before arithmetic so invalid target NaNs
                # can never enter subtraction or division.
                left_prediction = prediction[batch_index, :-1][selected_pairs]
                right_prediction = prediction[batch_index, 1:][selected_pairs]
                left_target = target[batch_index, :-1][selected_pairs]
                right_target = target[batch_index, 1:][selected_pairs]
                selected_dt = dt[batch_index][selected_pairs].to(prediction.dtype).unsqueeze(1)
                tensors = (left_prediction, right_prediction, left_target, right_target, selected_dt)
                if not all(torch.isfinite(value).all() for value in tensors):
                    raise ValueError(f"non-finite supervised velocity data at batch index {batch_index}")
                prediction_velocity = (right_prediction - left_prediction) / selected_dt
                target_velocity = (right_target - left_target) / selected_dt
                velocity_values.append(F.l1_loss(prediction_velocity, target_velocity))

        zero = prediction.sum() * 0.0
        position_loss = torch.stack(position_values).mean() if position_values else zero
        velocity_loss = torch.stack(velocity_values).mean() if velocity_values else zero
        total = self.position_weight * position_loss + self.velocity_weight * velocity_loss
        return {
            "loss": total,
            "position_loss": position_loss,
            "velocity_loss": velocity_loss,
            "num_position_samples": len(position_values),
            "num_velocity_samples": len(velocity_values),
        }
