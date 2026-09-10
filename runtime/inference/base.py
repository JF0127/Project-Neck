"""Minimal replaceable Motion backend interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Sequence

from ..contracts import MotionOutput, MotionRequest

MOTION_FPS = 30.0


def _rpy_frames(values: Sequence[Sequence[float]]) -> tuple[tuple[float, float, float], ...]:
    frames: list[tuple[float, float, float]] = []
    for index, frame in enumerate(values):
        try:
            if len(frame) != 3:
                raise ValueError
            converted = tuple(float(component) for component in frame)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"motion frame {index} must contain exactly three numbers") from exc
        if not all(math.isfinite(component) for component in converted):
            raise ValueError(f"motion frame {index} contains NaN or Inf")
        frames.append(converted)  # type: ignore[arg-type]
    if not frames:
        raise ValueError("motion output must contain at least one frame")
    return tuple(frames)


def validate_motion_output(output: MotionOutput) -> MotionOutput:
    """Validate and normalize the fixed Motion output contract."""

    if not isinstance(output, MotionOutput):
        raise TypeError("Motion backend must return MotionOutput")
    try:
        fps = float(output.fps)
    except (TypeError, ValueError) as exc:
        raise ValueError("motion fps must be numeric") from exc
    if not math.isfinite(fps) or fps != MOTION_FPS:
        raise ValueError("motion fps must be exactly 30")
    return MotionOutput(rpy_offset=_rpy_frames(output.rpy_offset), fps=MOTION_FPS)


class MotionBackend(ABC):
    """A backend only infers offsets; common physical processing lives elsewhere."""

    def infer(self, request: MotionRequest) -> MotionOutput:
        return validate_motion_output(self._infer(request))

    @abstractmethod
    def _infer(self, request: MotionRequest) -> MotionOutput:
        """Produce one backend-specific prediction without Runtime mutation."""
