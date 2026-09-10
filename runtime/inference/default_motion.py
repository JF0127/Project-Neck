"""Default reply text and head motion used when generation fails."""

from __future__ import annotations

import math

from ..contracts import MotionOutput
from .base import MOTION_FPS

DEFAULT_GENERATION_FALLBACK_TEXT = "抱歉，我没听清楚，请再说一遍。"
DEFAULT_SHAKE_AMPLITUDE_DEG = 3.0
DEFAULT_SHAKE_HZ = 1.0


def default_motion_output(
    duration_sec: float,
    fps: float = MOTION_FPS,
) -> MotionOutput:
    """Small left-right yaw shake that starts and ends near neutral."""
    if not math.isfinite(duration_sec) or duration_sec <= 0.0:
        raise ValueError("duration_sec must be a positive finite number")
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be a positive finite number")

    frame_count = max(2, int(round(duration_sec * fps)))
    amplitude_rad = math.radians(DEFAULT_SHAKE_AMPLITUDE_DEG)
    frames = tuple(
        (
            0.0,
            0.0,
            amplitude_rad
            * math.sin(2.0 * math.pi * DEFAULT_SHAKE_HZ * index / fps),
        )
        for index in range(frame_count)
    )
    return MotionOutput(rpy_offset=frames, fps=fps)
