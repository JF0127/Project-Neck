"""Lightweight safety validation for high-level speaking MotionPlan objects."""
from __future__ import annotations

from dataclasses import dataclass
import math

from .motion_plan import MotionPlan

ACTION_PRIMARY_AXES = {
    "nod": "pitch",
    "turn": "yaw",
    "tilt": "roll",
    "shake": "yaw",
}
DEFAULT_MAX_AMPLITUDE_DEG = 5.0
DEFAULT_MINIMUM_GAP_SEC = 0.0
DEFAULT_DURATION_TOLERANCE_SEC = 1.0 / 60.0


class MotionPlanValidationError(ValueError):
    """A structurally valid MotionPlan violates V2 planner constraints."""


@dataclass(frozen=True)
class MotionPlanValidatorConfig:
    max_amplitude_deg: float = DEFAULT_MAX_AMPLITUDE_DEG
    minimum_gap_sec: float = DEFAULT_MINIMUM_GAP_SEC
    duration_tolerance_sec: float = DEFAULT_DURATION_TOLERANCE_SEC

    def __post_init__(self) -> None:
        for name in (
            "max_amplitude_deg",
            "minimum_gap_sec",
            "duration_tolerance_sec",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            converted = float(value)
            if not math.isfinite(converted) or converted < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, converted)
        if self.max_amplitude_deg <= 0.0:
            raise ValueError("max_amplitude_deg must be greater than zero")


def validate_motion_plan(
    plan: MotionPlan,
    *,
    expected_duration_sec: float | None = None,
    config: MotionPlanValidatorConfig | None = None,
) -> MotionPlan:
    """Validate action vocabulary, axis compatibility, safety and overlap."""
    if not isinstance(plan, MotionPlan):
        raise TypeError("plan must be a MotionPlan")
    settings = config or MotionPlanValidatorConfig()
    if not isinstance(settings, MotionPlanValidatorConfig):
        raise TypeError("config must be a MotionPlanValidatorConfig")
    if expected_duration_sec is not None:
        if (
            isinstance(expected_duration_sec, bool)
            or not isinstance(expected_duration_sec, (int, float))
            or not math.isfinite(float(expected_duration_sec))
            or float(expected_duration_sec) <= 0.0
        ):
            raise MotionPlanValidationError(
                "expected_duration_sec must be positive and finite"
            )
        if abs(plan.duration_sec - float(expected_duration_sec)) > settings.duration_tolerance_sec:
            raise MotionPlanValidationError(
                "plan duration_sec does not match robot speech duration"
            )

    previous_end: float | None = None
    for index, segment in enumerate(plan.segments):
        required_axis = ACTION_PRIMARY_AXES.get(segment.action)
        if required_axis is None:
            raise MotionPlanValidationError(
                f"segments[{index}].action is not one of {sorted(ACTION_PRIMARY_AXES)}"
            )
        if segment.primary_axis != required_axis:
            raise MotionPlanValidationError(
                f"segments[{index}] action {segment.action!r} requires "
                f"primary_axis={required_axis!r}"
            )
        if abs(segment.amplitude_deg) > settings.max_amplitude_deg:
            raise MotionPlanValidationError(
                f"segments[{index}].amplitude_deg exceeds "
                f"{settings.max_amplitude_deg:g} degrees"
            )
        if segment.end_sec > plan.duration_sec:
            raise MotionPlanValidationError(
                f"segments[{index}].end_sec exceeds plan duration"
            )
        if previous_end is not None:
            gap = segment.start_sec - previous_end
            if gap < settings.minimum_gap_sec - 1e-9:
                raise MotionPlanValidationError(
                    f"segments[{index}] overlaps or violates minimum gap"
                )
        previous_end = segment.end_sec
    return plan
