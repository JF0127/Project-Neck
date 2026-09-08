"""NaN-safe Baseline V1 motion metrics on valid physical timesteps."""

from __future__ import annotations

import math

import torch

from .losses import _validate_inputs, velocity_pair_data


class MotionMetricAccumulator:
    """Accumulate per-fragment MAEs so long fragments do not dominate."""

    _POSITION_NAMES = ("position_mae", "roll_mae", "pitch_mae", "yaw_mae")

    def __init__(self) -> None:
        self.sums = {name: 0.0 for name in (*self._POSITION_NAMES, "velocity_mae")}
        self.position_samples = 0
        self.velocity_samples = 0

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        timestamps: torch.Tensor,
        sequence_mask: torch.Tensor,
        target_valid_mask: torch.Tensor,
    ) -> None:
        _validate_inputs(prediction, target, timestamps, sequence_mask, target_valid_mask)
        supervision_mask = sequence_mask & target_valid_mask
        pair_mask, dt = velocity_pair_data(timestamps, sequence_mask, target_valid_mask)

        with torch.no_grad():
            for batch_index in range(prediction.shape[0]):
                selected_prediction = prediction[batch_index][supervision_mask[batch_index]]
                selected_target = target[batch_index][supervision_mask[batch_index]]
                if selected_target.numel():
                    if not torch.isfinite(selected_target).all() or not torch.isfinite(selected_prediction).all():
                        raise ValueError(f"non-finite supervised position at batch index {batch_index}")
                    error = (selected_prediction - selected_target).abs()
                    self.sums["position_mae"] += error.mean().item()
                    self.sums["roll_mae"] += error[:, 0].mean().item()
                    self.sums["pitch_mae"] += error[:, 1].mean().item()
                    self.sums["yaw_mae"] += error[:, 2].mean().item()
                    self.position_samples += 1

                selected_pairs = pair_mask[batch_index]
                if torch.any(selected_pairs):
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
                    self.sums["velocity_mae"] += (
                        prediction_velocity - target_velocity
                    ).abs().mean().item()
                    self.velocity_samples += 1

    def compute(self) -> dict[str, float | int]:
        result: dict[str, float | int] = {}
        for name in self._POSITION_NAMES:
            result[name] = (
                self.sums[name] / self.position_samples
                if self.position_samples
                else math.nan
            )
        result["velocity_mae"] = (
            self.sums["velocity_mae"] / self.velocity_samples
            if self.velocity_samples
            else math.nan
        )
        result["num_position_samples"] = self.position_samples
        result["num_velocity_samples"] = self.velocity_samples
        return result


def motion_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    timestamps: torch.Tensor,
    sequence_mask: torch.Tensor,
    target_valid_mask: torch.Tensor,
) -> dict[str, float | int]:
    accumulator = MotionMetricAccumulator()
    accumulator.update(prediction, target, timestamps, sequence_mask, target_valid_mask)
    return accumulator.compute()
