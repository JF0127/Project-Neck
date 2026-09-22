"""Deterministic fixed-length optimization for absolute RPY trajectories."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np

AXES = ("roll", "pitch", "yaw")
TRAJECTORY_FPS = 30.0
BINOMIAL_KERNEL = np.asarray((1.0, 4.0, 6.0, 4.0, 1.0), dtype=np.float64) / 16.0

DEFAULT_SMOOTHING_PASSES = 1
DEFAULT_MAX_VELOCITY_DEG_S = 60.0
DEFAULT_MAX_ACCELERATION_DEG_S2 = 500.0
DEFAULT_PROJECTION_ITERATIONS = 64
_PROJECTION_TOLERANCE_RAD = 1e-10


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{field} must be finite")
    return converted


@dataclass(frozen=True)
class TrajectoryOptimizerConfig:
    """Readable engineering parameters for the first optimizer version."""

    smoothing_passes: int = DEFAULT_SMOOTHING_PASSES
    max_velocity_deg_s: float = DEFAULT_MAX_VELOCITY_DEG_S
    max_acceleration_deg_s2: float = DEFAULT_MAX_ACCELERATION_DEG_S2
    projection_iterations: int = DEFAULT_PROJECTION_ITERATIONS

    def __post_init__(self) -> None:
        if (
            isinstance(self.smoothing_passes, bool)
            or not isinstance(self.smoothing_passes, int)
            or self.smoothing_passes < 0
        ):
            raise ValueError("smoothing_passes must be a non-negative integer")
        if (
            isinstance(self.projection_iterations, bool)
            or not isinstance(self.projection_iterations, int)
            or self.projection_iterations < 1
        ):
            raise ValueError("projection_iterations must be a positive integer")
        max_velocity = _finite_float(
            self.max_velocity_deg_s, "max_velocity_deg_s"
        )
        max_acceleration = _finite_float(
            self.max_acceleration_deg_s2, "max_acceleration_deg_s2"
        )
        if max_velocity <= 0.0:
            raise ValueError("max_velocity_deg_s must be greater than zero")
        if max_acceleration <= 0.0:
            raise ValueError("max_acceleration_deg_s2 must be greater than zero")
        object.__setattr__(self, "max_velocity_deg_s", max_velocity)
        object.__setattr__(self, "max_acceleration_deg_s2", max_acceleration)


def _trajectory_array(
    trajectory: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    try:
        values = np.asarray(trajectory, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("trajectory must contain numeric values") from exc
    if values.ndim != 2 or values.shape[1:] != (3,) or len(values) == 0:
        raise ValueError("trajectory must have shape [T, 3] with T greater than zero")
    if not np.isfinite(values).all():
        raise ValueError("trajectory contains NaN or Inf")
    return values.copy()


def _validated_fps(fps: float) -> float:
    value = _finite_float(fps, "fps")
    if value <= 0.0:
        raise ValueError("fps must be greater than zero")
    return value


def trajectory_metrics(
    trajectory: Sequence[Sequence[float]] | np.ndarray,
    fps: float,
) -> dict[str, dict[str, float]]:
    """Return per-axis peak velocity, acceleration and jerk in degree units."""
    values = _trajectory_array(trajectory)
    rate = _validated_fps(fps)
    derivatives = (
        np.diff(values, n=1, axis=0) * rate,
        np.diff(values, n=2, axis=0) * rate**2,
        np.diff(values, n=3, axis=0) * rate**3,
    )
    metric_names = (
        "peak_velocity_deg_s",
        "peak_acceleration_deg_s2",
        "peak_jerk_deg_s3",
    )
    result: dict[str, dict[str, float]] = {axis: {} for axis in AXES}
    for metric_name, derivative in zip(metric_names, derivatives):
        if len(derivative):
            peaks = np.max(np.abs(derivative), axis=0)
        else:
            peaks = np.zeros(3, dtype=np.float64)
        peaks_deg = np.degrees(peaks)
        for axis_index, axis in enumerate(AXES):
            result[axis][metric_name] = float(peaks_deg[axis_index])
    return result


def _smooth_positions(values: np.ndarray, passes: int) -> np.ndarray:
    if passes == 0 or len(values) < 2:
        return values.copy()
    first = values[0].copy()
    last = values[-1].copy()
    result = values.copy()
    for _ in range(passes):
        padded = np.pad(result, ((2, 2), (0, 0)), mode="edge")
        result = sum(
            BINOMIAL_KERNEL[index] * padded[index : index + len(result)]
            for index in range(len(BINOMIAL_KERNEL))
        )
        result[0] = first
        result[-1] = last
    return result


def _forward_velocity_projection(values: np.ndarray, max_step: float) -> np.ndarray:
    """Project left-to-right while retaining enough reach for the fixed end."""
    result = values.copy()
    frame_count = len(result)
    if frame_count < 2:
        return result
    first = float(result[0])
    last = float(result[-1])
    if abs(last - first) > (frame_count - 1) * max_step + _PROJECTION_TOLERANCE_RAD:
        raise ValueError(
            "fixed trajectory endpoints are infeasible under max_velocity_deg_s"
        )
    for index in range(1, frame_count - 1):
        remaining_steps = frame_count - 1 - index
        lower = max(
            result[index - 1] - max_step,
            last - remaining_steps * max_step,
        )
        upper = min(
            result[index - 1] + max_step,
            last + remaining_steps * max_step,
        )
        result[index] = min(upper, max(lower, result[index]))
    result[0] = first
    result[-1] = last
    return result


def _project_velocity(values: np.ndarray, max_step: float) -> np.ndarray:
    result = values.copy()
    for axis in range(3):
        projected = _forward_velocity_projection(result[:, axis], max_step)
        projected = _forward_velocity_projection(projected[::-1], max_step)[::-1]
        result[:, axis] = projected
    return result


def _project_acceleration(values: np.ndarray, max_second_difference: float) -> np.ndarray:
    """Gauss-Seidel projection of each interior position onto curvature bounds."""
    result = values.copy()
    if len(result) < 3:
        return result
    for indices in (range(1, len(result) - 1), range(len(result) - 2, 0, -1)):
        for index in indices:
            neighbor_sum = result[index - 1] + result[index + 1]
            lower = (neighbor_sum - max_second_difference) * 0.5
            upper = (neighbor_sum + max_second_difference) * 0.5
            result[index] = np.minimum(upper, np.maximum(lower, result[index]))
    return result


class TrajectoryOptimizer:
    """Smooth and project one fixed-length absolute 30 Hz RPY trajectory."""

    def __init__(
        self,
        config: TrajectoryOptimizerConfig | None = None,
    ) -> None:
        self.config = config or TrajectoryOptimizerConfig()
        if not isinstance(self.config, TrajectoryOptimizerConfig):
            raise TypeError("config must be a TrajectoryOptimizerConfig")

    def optimize(
        self,
        trajectory: Sequence[Sequence[float]] | np.ndarray,
        fps: float = TRAJECTORY_FPS,
    ) -> tuple[tuple[float, float, float], ...]:
        rate = _validated_fps(fps)
        if rate != TRAJECTORY_FPS:
            raise ValueError("TrajectoryOptimizer requires exactly 30 Hz")
        original = _trajectory_array(trajectory)
        first = original[0].copy()
        last = original[-1].copy()
        result = _smooth_positions(original, self.config.smoothing_passes)

        max_velocity_step = math.radians(
            self.config.max_velocity_deg_s
        ) / rate
        max_second_difference = math.radians(
            self.config.max_acceleration_deg_s2
        ) / rate**2

        for _ in range(self.config.projection_iterations):
            result = _project_velocity(result, max_velocity_step)
            result = _project_acceleration(result, max_second_difference)
            result = _project_velocity(result, max_velocity_step)
            acceleration_excess = (
                np.max(np.abs(np.diff(result, n=2, axis=0)))
                - max_second_difference
                if len(result) >= 3
                else 0.0
            )
            if acceleration_excess <= _PROJECTION_TOLERANCE_RAD:
                break

        result[0] = first
        result[-1] = last
        if not np.isfinite(result).all():
            raise RuntimeError("trajectory optimization produced NaN or Inf")
        velocity_excess = (
            np.max(np.abs(np.diff(result, axis=0))) - max_velocity_step
            if len(result) >= 2
            else 0.0
        )
        acceleration_excess = (
            np.max(np.abs(np.diff(result, n=2, axis=0)))
            - max_second_difference
            if len(result) >= 3
            else 0.0
        )
        if velocity_excess > _PROJECTION_TOLERANCE_RAD:
            raise RuntimeError("velocity projection did not converge")
        if acceleration_excess > _PROJECTION_TOLERANCE_RAD:
            raise RuntimeError("acceleration projection did not converge")

        return tuple(
            tuple(float(component) for component in frame) for frame in result
        )
