from __future__ import annotations

import unittest

import torch

from algorithm.losses import BaselineLoss
from algorithm.metrics import motion_metrics


class BaselineLossTests(unittest.TestCase):
    def setUp(self) -> None:
        self.criterion = BaselineLoss(position_weight=1.0, velocity_weight=0.1, position_beta=0.05)

    def _evaluate(self, prediction, target, timestamps, sequence_mask=None, valid_mask=None):
        batch, steps, _ = prediction.shape
        if sequence_mask is None:
            sequence_mask = torch.ones(batch, steps, dtype=torch.bool)
        if valid_mask is None:
            valid_mask = torch.ones(batch, steps, dtype=torch.bool)
        return self.criterion(prediction, target, timestamps, sequence_mask, valid_mask)

    def test_perfect_prediction_has_zero_position_loss(self) -> None:
        target = torch.randn(2, 5, 3)
        timestamps = torch.tensor([[0.0, 0.1, 0.3, 0.5, 0.9]]).expand(2, -1)
        result = self._evaluate(target.clone(), target, timestamps)
        torch.testing.assert_close(result["position_loss"], torch.tensor(0.0))

    def test_perfect_linear_trajectory_has_zero_velocity_loss(self) -> None:
        timestamps = torch.tensor([[0.0, 0.1, 0.4, 1.0]], dtype=torch.float64)
        target = (timestamps.unsqueeze(-1) * torch.tensor([1.0, -2.0, 0.5])).float()
        result = self._evaluate(target.clone(), target, timestamps)
        torch.testing.assert_close(result["velocity_loss"], torch.tensor(0.0))

    def test_invalid_nan_does_not_pollute_loss(self) -> None:
        prediction = torch.zeros(1, 4, 3, requires_grad=True)
        target = torch.tensor([[[0.0, 0.0, 0.0], [float("nan")] * 3, [0.2] * 3, [0.3] * 3]])
        timestamps = torch.tensor([[0.0, 0.2, 0.5, 0.9]], dtype=torch.float64)
        valid = torch.tensor([[True, False, True, True]])
        result = self._evaluate(prediction, target, timestamps, valid_mask=valid)
        self.assertTrue(torch.isfinite(result["loss"]))
        metrics = motion_metrics(prediction.detach(), target, timestamps, torch.ones_like(valid), valid)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["position_mae"])))
        self.assertTrue(torch.isfinite(torch.tensor(metrics["velocity_mae"])))
        result["loss"].backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_velocity_does_not_cross_invalid_gap(self) -> None:
        timestamps = torch.tensor([[0.0, 0.1, 0.2, 0.4, 0.8]], dtype=torch.float64)
        target = torch.zeros(1, 5, 3)
        prediction = torch.tensor([[[0.0] * 3, [0.0] * 3, [9.0] * 3, [4.0] * 3, [4.0] * 3]])
        valid = torch.tensor([[True, True, False, True, True]])
        result = self._evaluate(prediction, target, timestamps, valid_mask=valid)
        torch.testing.assert_close(result["velocity_loss"], torch.tensor(0.0))

    def test_padding_is_excluded(self) -> None:
        prediction = torch.zeros(1, 4, 3)
        target = torch.tensor([[[0.0] * 3, [0.0] * 3, [float("nan")] * 3, [float("nan")] * 3]])
        timestamps = torch.tensor([[0.0, 0.1, float("nan"), float("nan")]], dtype=torch.float64)
        sequence = torch.tensor([[True, True, False, False]])
        valid = torch.tensor([[True, True, False, False]])
        result = self._evaluate(prediction, target, timestamps, sequence, valid)
        self.assertTrue(torch.isfinite(result["loss"]))
        self.assertEqual(result["num_position_samples"], 1)
        self.assertEqual(result["num_velocity_samples"], 1)

    def test_nonuniform_timestamps_produce_physical_velocity(self) -> None:
        timestamps = torch.tensor([[0.0, 0.1, 0.6, 1.4]], dtype=torch.float64)
        target = (2.0 * timestamps).unsqueeze(-1).expand(-1, -1, 3).float()
        prediction = timestamps.unsqueeze(-1).expand(-1, -1, 3).float()
        result = self._evaluate(prediction, target, timestamps)
        torch.testing.assert_close(result["velocity_loss"], torch.tensor(1.0), rtol=1e-6, atol=1e-6)

    def test_non_increasing_real_timestamp_raises(self) -> None:
        values = torch.zeros(1, 3, 3)
        timestamps = torch.tensor([[0.0, 0.2, 0.2]], dtype=torch.float64)
        with self.assertRaises(ValueError):
            self._evaluate(values, values, timestamps)

    def test_metrics_have_expected_units_and_channels(self) -> None:
        timestamps = torch.tensor([[0.0, 0.5, 1.5]], dtype=torch.float64)
        target = torch.zeros(1, 3, 3)
        prediction = torch.tensor([[[0.0, 0.0, 0.0], [0.5, 1.0, 1.5], [1.5, 3.0, 4.5]]])
        metrics = motion_metrics(
            prediction,
            target,
            timestamps,
            torch.ones(1, 3, dtype=torch.bool),
            torch.ones(1, 3, dtype=torch.bool),
        )
        self.assertAlmostEqual(metrics["roll_mae"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["pitch_mae"], 4.0 / 3.0)
        self.assertAlmostEqual(metrics["yaw_mae"], 2.0)
        self.assertAlmostEqual(metrics["position_mae"], 4.0 / 3.0)
        self.assertAlmostEqual(metrics["velocity_mae"], 2.0)


if __name__ == "__main__":
    unittest.main()
