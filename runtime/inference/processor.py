"""Common pure-software processing for all Motion backend outputs."""
from __future__ import annotations

import math
from typing import Sequence

from ..contracts import FinalTrajectory, MotionOutput
from .base import MOTION_FPS, validate_motion_output
from .trajectory_optimizer import TrajectoryOptimizer

NEUTRAL_RPY = (0.0, 0.0, 0.0)
_VALID_STATES = {"speaking", "listening", "silent"}


def _rpy(value: Sequence[float], name: str) -> tuple[float, float, float]:
    try:
        if len(value) != 3:
            raise ValueError
        converted = tuple(float(component) for component in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain exactly three numbers") from exc
    if not all(math.isfinite(component) for component in converted):
        raise ValueError(f"{name} contains NaN or Inf")
    return converted  # type: ignore[return-value]


def _interpolate(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    weight: float,
) -> tuple[float, float, float]:
    t = min(1.0, max(0.0, float(weight)))
    minimum_jerk = 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5
    return tuple(
        before + (after - before) * minimum_jerk
        for before, after in zip(start, end)
    )  # type: ignore[return-value]


class MotionProcessor:
    """Convert offsets to absolute RPY and optimize the complete trajectory.

    It starts at the supplied measured pose, reaches the backend's first
    absolute target, preserves all later model frames, returns to the kinematic
    neutral pose ``[0, 0, 0]``, then applies fixed-length kinematic optimization.
    """

    def __init__(
        self,
        start_transition_frames: int = 3,
        neutral_return_frames: int = 24,
        optimizer: TrajectoryOptimizer | None = None,
    ) -> None:
        if start_transition_frames < 1:
            raise ValueError("start_transition_frames must be at least 1")
        if neutral_return_frames < 1:
            raise ValueError("neutral_return_frames must be at least 1")
        if optimizer is not None and not isinstance(optimizer, TrajectoryOptimizer):
            raise TypeError("optimizer must be a TrajectoryOptimizer")
        self.start_transition_frames = start_transition_frames
        self.neutral_return_frames = neutral_return_frames
        self.optimizer = optimizer or TrajectoryOptimizer()

    @staticmethod
    def offset_to_absolute(
        output: MotionOutput,
        start_rpy: Sequence[float],
    ) -> tuple[tuple[float, float, float], ...]:
        validated = validate_motion_output(output)
        start = _rpy(start_rpy, "start_rpy")
        return tuple(
            tuple(start[axis] + frame[axis] for axis in range(3))
            for frame in validated.rpy_offset
        )  # type: ignore[return-value]

    def process(
        self,
        output: MotionOutput,
        start_rpy: Sequence[float],
    ) -> FinalTrajectory:
        start = _rpy(start_rpy, "start_rpy")
        absolute = self.offset_to_absolute(output, start)

        first_target = absolute[0]
        trajectory: list[tuple[float, float, float]] = [start]
        if first_target != start:
            for step in range(1, self.start_transition_frames + 1):
                trajectory.append(
                    _interpolate(start, first_target, step / self.start_transition_frames)
                )
        trajectory.extend(absolute[1:])
        states = ["speaking"] * len(trajectory)

        last = trajectory[-1]
        if last != NEUTRAL_RPY:
            for step in range(1, self.neutral_return_frames + 1):
                trajectory.append(
                    _interpolate(last, NEUTRAL_RPY, step / self.neutral_return_frames)
                )
                states.append("silent")

        optimized = self.optimizer.optimize(trajectory, MOTION_FPS)
        result = FinalTrajectory(
            rpy=optimized,
            fps=MOTION_FPS,
            states=tuple(states),
            duration_sec=len(optimized) / MOTION_FPS,
        )
        self.validate_final(result)
        return result

    @staticmethod
    def validate_final(trajectory: FinalTrajectory) -> None:
        if float(trajectory.fps) != MOTION_FPS:
            raise ValueError("final trajectory fps must be exactly 30")
        frames = tuple(_rpy(frame, "final trajectory frame") for frame in trajectory.rpy)
        if not frames or len(frames) != len(trajectory.states):
            raise ValueError("final trajectory frames/states must be non-empty and equal length")
        if any(state not in _VALID_STATES for state in trajectory.states):
            raise ValueError("final trajectory contains an invalid state")
        expected_duration = len(frames) / MOTION_FPS
        if not math.isfinite(trajectory.duration_sec) or not math.isclose(
            trajectory.duration_sec, expected_duration, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("final trajectory duration does not match its frame count")
