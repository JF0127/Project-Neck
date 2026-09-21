"""Compile sparse degree-valued gesture actions into MotionOutput."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from ..contracts import MotionOutput
from .base import MOTION_FPS

MAX_AXIS_AMPLITUDE_DEG = 5.0
MIN_ACTION_DURATION_SEC = 0.7
MIN_NEUTRAL_GAP_SEC = 0.2
_AXIS_NAMES = ("roll", "pitch", "yaw")
_ACTION_FIELDS = {"start", "end", *_AXIS_NAMES}


class MotionPlanError(ValueError):
    """A sparse motion plan cannot be parsed or safely compiled."""


@dataclass(frozen=True)
class SparseMotionAction:
    """One temporary gesture amplitude expressed in degrees."""

    start: float
    end: float
    roll: float
    pitch: float
    yaw: float

    def as_dict(self) -> dict[str, float]:
        return {
            "start": self.start,
            "end": self.end,
            "roll": self.roll,
            "pitch": self.pitch,
            "yaw": self.yaw,
        }


@dataclass(frozen=True)
class RejectedMotionAction:
    """One locally rejected action and its validation reason."""

    index: int
    reason: str
    value: Any

    def as_dict(self) -> dict[str, Any]:
        return {"index": self.index, "reason": self.reason, "value": self.value}


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MotionPlanError(f"{field} must be a number")
    converted = float(value)
    if not math.isfinite(converted):
        raise MotionPlanError(f"{field} must be finite")
    return converted


def _parse_action(
    value: Any,
    index: int,
    duration_sec: float,
) -> SparseMotionAction:
    if not isinstance(value, Mapping) or set(value) != _ACTION_FIELDS:
        raise MotionPlanError(
            "must contain exactly start,end,roll,pitch,yaw"
        )
    start = _finite_number(value["start"], "start")
    end = _finite_number(value["end"], "end")
    if start < 0.0 or end > duration_sec:
        raise MotionPlanError(
            f"interval [{start}, {end}] is outside [0, {duration_sec}]"
        )
    if start >= end:
        raise MotionPlanError("start must be less than end")
    if end - start < MIN_ACTION_DURATION_SEC:
        raise MotionPlanError(
            f"duration must be at least {MIN_ACTION_DURATION_SEC:g}s"
        )

    axes = tuple(_finite_number(value[axis], axis) for axis in _AXIS_NAMES)
    for axis, component in zip(_AXIS_NAMES, axes):
        if abs(component) > MAX_AXIS_AMPLITUDE_DEG:
            raise MotionPlanError(
                f"{axis}={component} exceeds the "
                f"{MAX_AXIS_AMPLITUDE_DEG:g} degree limit"
            )
    return SparseMotionAction(start, end, *axes)


def parse_motion_plan_with_rejections(
    document: Any,
    duration_sec: float,
) -> tuple[tuple[SparseMotionAction, ...], tuple[RejectedMotionAction, ...]]:
    """Parse valid gestures while locally rejecting repairable bad actions."""
    duration = _finite_number(duration_sec, "duration_sec")
    if duration <= 0.0:
        raise MotionPlanError("duration_sec must be greater than zero")
    if not isinstance(document, Mapping) or set(document) != {"actions"}:
        raise MotionPlanError("motion plan must contain only the 'actions' field")
    raw_actions = document["actions"]
    if not isinstance(raw_actions, list):
        raise MotionPlanError("motion plan actions must be a list")

    candidates: list[tuple[int, SparseMotionAction]] = []
    rejected: list[RejectedMotionAction] = []
    for index, value in enumerate(raw_actions):
        try:
            action = _parse_action(value, index, duration)
        except MotionPlanError as exc:
            rejected.append(RejectedMotionAction(index, str(exc), value))
            continue
        candidates.append((index, action))

    candidates.sort(key=lambda item: (item[1].start, item[1].end, item[0]))
    accepted: list[SparseMotionAction] = []
    for original_index, action in candidates:
        if accepted:
            gap = action.start - accepted[-1].end
            if gap < MIN_NEUTRAL_GAP_SEC - 1e-9:
                reason = (
                    "overlaps the previous accepted action"
                    if gap < 0.0
                    else f"neutral gap must be at least {MIN_NEUTRAL_GAP_SEC:g}s"
                )
                rejected.append(
                    RejectedMotionAction(
                        original_index,
                        reason,
                        raw_actions[original_index],
                    )
                )
                continue
        accepted.append(action)

    rejected.sort(key=lambda item: item.index)
    return tuple(accepted), tuple(rejected)


def parse_motion_plan(
    document: Any,
    duration_sec: float,
) -> tuple[SparseMotionAction, ...]:
    """Return the valid subset of a strict ``{"actions": [...]}`` plan."""
    actions, _ = parse_motion_plan_with_rejections(document, duration_sec)
    return actions


def _minimum_jerk(value: float) -> float:
    """Quintic smoothstep with zero endpoint velocity and acceleration."""
    t = min(1.0, max(0.0, value))
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5


def _gesture_scale(progress: float) -> float:
    """40% rise, 20% peak hold, 40% return to baseline."""
    value = min(1.0, max(0.0, progress))
    if value < 0.4:
        return _minimum_jerk(value / 0.4)
    if value <= 0.6:
        return 1.0
    return 1.0 - _minimum_jerk((value - 0.6) / 0.4)


def compile_motion_plan(
    actions: Sequence[SparseMotionAction],
    duration_sec: float,
    fps: float = MOTION_FPS,
) -> MotionOutput:
    """Compile independent pulse gestures to 30 fps relative RPY radians."""
    duration = _finite_number(duration_sec, "duration_sec")
    if duration <= 0.0:
        raise MotionPlanError("duration_sec must be greater than zero")
    if float(fps) != MOTION_FPS:
        raise MotionPlanError("motion compiler fps must be exactly 30")

    frame_count = max(2, int(round(duration * MOTION_FPS)))
    frames = [[0.0, 0.0, 0.0] for _ in range(frame_count)]

    previous_end: float | None = None
    for index, action in enumerate(actions):
        if action.end - action.start < MIN_ACTION_DURATION_SEC:
            raise MotionPlanError(
                f"action {index} duration must be at least "
                f"{MIN_ACTION_DURATION_SEC:g}s"
            )
        if previous_end is not None:
            gap = action.start - previous_end
            if gap < MIN_NEUTRAL_GAP_SEC - 1e-9:
                raise MotionPlanError(
                    f"action {index} neutral gap must be at least "
                    f"{MIN_NEUTRAL_GAP_SEC:g}s"
                )
        amplitudes_deg = (action.roll, action.pitch, action.yaw)
        for axis, component in zip(_AXIS_NAMES, amplitudes_deg):
            if not math.isfinite(component) or abs(component) > MAX_AXIS_AMPLITUDE_DEG:
                raise MotionPlanError(
                    f"action {index}.{axis} exceeds the "
                    f"{MAX_AXIS_AMPLITUDE_DEG:g} degree limit"
                )

        start_frame = min(
            frame_count - 2,
            max(0, int(round(action.start * MOTION_FPS))),
        )
        end_frame = min(
            frame_count - 1,
            max(start_frame + 1, int(round(action.end * MOTION_FPS))),
        )
        span = end_frame - start_frame
        amplitudes_rad = tuple(math.radians(value) for value in amplitudes_deg)
        for frame_index in range(start_frame, end_frame + 1):
            progress = (frame_index - start_frame) / span
            scale = _gesture_scale(progress)
            frames[frame_index] = [
                amplitudes_rad[axis] * scale for axis in range(3)
            ]
        previous_end = action.end

    return MotionOutput(
        rpy_offset=tuple(tuple(frame) for frame in frames),
        fps=MOTION_FPS,
    )
