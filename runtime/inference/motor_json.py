"""Serialize a Runtime FinalTrajectory into the frozen Motor JSON schema."""

from __future__ import annotations

from ..contracts import FinalTrajectory
from .processor import MotionProcessor


def final_trajectory_to_motor_document(
    trajectory: FinalTrajectory,
    name: str,
) -> dict:
    """Return ``{name,fps,unit,order,trajectory,states}`` for the Motor socket."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("motor document name must be a non-empty string")
    MotionProcessor.validate_final(trajectory)
    return {
        "name": name,
        "fps": float(trajectory.fps),
        "unit": "radian",
        "order": ["roll", "pitch", "yaw"],
        "trajectory": [list(frame) for frame in trajectory.rpy],
        "states": list(trajectory.states),
    }
