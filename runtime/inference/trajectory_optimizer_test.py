"""Pure-software tests for fixed-length trajectory optimization."""
from __future__ import annotations

import math
import unittest

import numpy as np

from .trajectory_optimizer import (
    TrajectoryOptimizer,
    TrajectoryOptimizerConfig,
    trajectory_metrics,
)

FPS = 30.0


def _roughness(trajectory: np.ndarray) -> float:
    return float(np.sum(np.square(np.diff(trajectory, n=2, axis=0))))


class TrajectoryOptimizerTest(unittest.TestCase):
    def test_constant_trajectory_remains_constant(self) -> None:
        trajectory = np.tile(
            np.asarray((0.1, -0.2, 0.3), dtype=np.float64),
            (30, 1),
        )

        optimized = np.asarray(TrajectoryOptimizer().optimize(trajectory, FPS))

        np.testing.assert_allclose(optimized, trajectory, rtol=0.0, atol=1e-15)
        for axis_metrics in trajectory_metrics(optimized, FPS).values():
            self.assertEqual(axis_metrics["peak_velocity_deg_s"], 0.0)
            self.assertEqual(axis_metrics["peak_acceleration_deg_s2"], 0.0)
            self.assertEqual(axis_metrics["peak_jerk_deg_s3"], 0.0)

    def test_endpoints_and_frame_count_are_preserved(self) -> None:
        trajectory = np.zeros((45, 3), dtype=np.float64)
        trajectory[5:20, 0] = math.radians(8.0)
        trajectory[12:32, 1] = math.radians(-6.0)
        trajectory[20:40, 2] = math.radians(5.0)
        original_first = tuple(trajectory[0])
        original_last = tuple(trajectory[-1])

        optimized = TrajectoryOptimizer().optimize(trajectory, FPS)

        self.assertEqual(len(optimized), len(trajectory))
        self.assertEqual(optimized[0], original_first)
        self.assertEqual(optimized[-1], original_last)

    def test_binomial_smoothing_reduces_high_frequency_jitter(self) -> None:
        trajectory = np.zeros((41, 3), dtype=np.float64)
        alternating = np.where(np.arange(39) % 2 == 0, 1.0, -1.0)
        trajectory[1:-1, 0] = np.radians(2.0 * alternating)
        optimizer = TrajectoryOptimizer(
            TrajectoryOptimizerConfig(
                smoothing_passes=2,
                max_velocity_deg_s=10_000.0,
                max_acceleration_deg_s2=1_000_000.0,
                projection_iterations=8,
            )
        )
        before_metrics = trajectory_metrics(trajectory, FPS)["roll"]

        optimized = np.asarray(optimizer.optimize(trajectory, FPS))
        after_metrics = trajectory_metrics(optimized, FPS)["roll"]

        self.assertLess(_roughness(optimized), _roughness(trajectory) * 0.1)
        self.assertLess(
            after_metrics["peak_jerk_deg_s3"],
            before_metrics["peak_jerk_deg_s3"] * 0.1,
        )

    def test_velocity_limit(self) -> None:
        limit = 20.0
        trajectory = np.zeros((61, 3), dtype=np.float64)
        trajectory[10:40, 0] = math.radians(12.0)
        optimizer = TrajectoryOptimizer(
            TrajectoryOptimizerConfig(
                smoothing_passes=0,
                max_velocity_deg_s=limit,
                max_acceleration_deg_s2=100_000.0,
                projection_iterations=16,
            )
        )
        self.assertGreater(
            trajectory_metrics(trajectory, FPS)["roll"]["peak_velocity_deg_s"],
            limit,
        )

        optimized = optimizer.optimize(trajectory, FPS)
        peak = trajectory_metrics(optimized, FPS)["roll"]["peak_velocity_deg_s"]

        self.assertLessEqual(peak, limit + 1e-6)

    def test_acceleration_limit(self) -> None:
        limit = 120.0
        trajectory = np.zeros((61, 3), dtype=np.float64)
        trajectory[15:45, 1] = math.radians(10.0)
        optimizer = TrajectoryOptimizer(
            TrajectoryOptimizerConfig(
                smoothing_passes=0,
                max_velocity_deg_s=10_000.0,
                max_acceleration_deg_s2=limit,
                projection_iterations=128,
            )
        )
        self.assertGreater(
            trajectory_metrics(trajectory, FPS)["pitch"][
                "peak_acceleration_deg_s2"
            ],
            limit,
        )

        optimized = optimizer.optimize(trajectory, FPS)
        peak = trajectory_metrics(optimized, FPS)["pitch"][
            "peak_acceleration_deg_s2"
        ]

        self.assertLessEqual(peak, limit + 1e-4)

    def test_output_is_finite_and_nonfinite_input_is_rejected(self) -> None:
        trajectory = np.zeros((20, 3), dtype=np.float64)
        trajectory[5:15] = (0.05, -0.04, 0.03)
        optimized = np.asarray(TrajectoryOptimizer().optimize(trajectory, FPS))
        self.assertTrue(np.isfinite(optimized).all())

        for invalid in (math.nan, math.inf, -math.inf):
            bad = trajectory.copy()
            bad[3, 1] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                TrajectoryOptimizer().optimize(bad, FPS)

    def test_limits_apply_independently_to_all_three_axes(self) -> None:
        velocity_limit = 30.0
        acceleration_limit = 180.0
        trajectory = np.zeros((61, 3), dtype=np.float64)
        trajectory[10:30, 0] = math.radians(12.0)
        trajectory[15:40, 1] = math.radians(-10.0)
        trajectory[20:50, 2] = math.radians(8.0)
        optimizer = TrajectoryOptimizer(
            TrajectoryOptimizerConfig(
                smoothing_passes=1,
                max_velocity_deg_s=velocity_limit,
                max_acceleration_deg_s2=acceleration_limit,
                projection_iterations=128,
            )
        )

        optimized = optimizer.optimize(trajectory, FPS)
        metrics = trajectory_metrics(optimized, FPS)

        for axis in ("roll", "pitch", "yaw"):
            with self.subTest(axis=axis):
                self.assertLessEqual(
                    metrics[axis]["peak_velocity_deg_s"],
                    velocity_limit + 1e-6,
                )
                self.assertLessEqual(
                    metrics[axis]["peak_acceleration_deg_s2"],
                    acceleration_limit + 1e-4,
                )


if __name__ == "__main__":
    unittest.main()
