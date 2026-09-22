"""Immutable high-level Motion plan data structures."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

_PRIMARY_AXES = frozenset({"roll", "pitch", "yaw"})
_SEGMENT_FIELDS = {
    "start_sec",
    "end_sec",
    "action",
    "primary_axis",
    "amplitude_deg",
    "reason",
}
_PLAN_FIELDS = {"mode", "duration_sec", "segments"}


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{field} must be finite")
    return converted


def _require_fields(
    value: Any,
    expected: set[str],
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if set(value) != expected:
        raise ValueError(f"{name} must contain exactly {sorted(expected)}")
    return value


@dataclass(frozen=True)
class MotionSegment:
    """One high-level speaking-motion segment on the plan timeline."""

    start_sec: float
    end_sec: float
    action: str
    primary_axis: str
    amplitude_deg: float
    reason: str

    def __post_init__(self) -> None:
        start_sec = _finite_number(self.start_sec, "start_sec")
        end_sec = _finite_number(self.end_sec, "end_sec")
        amplitude_deg = _finite_number(self.amplitude_deg, "amplitude_deg")
        if start_sec < 0.0:
            raise ValueError("start_sec must be greater than or equal to zero")
        if end_sec <= start_sec:
            raise ValueError("end_sec must be greater than start_sec")
        if not isinstance(self.action, str) or not self.action.strip():
            raise ValueError("action must be a non-empty string")
        if (
            not isinstance(self.primary_axis, str)
            or self.primary_axis not in _PRIMARY_AXES
        ):
            raise ValueError("primary_axis must be roll, pitch or yaw")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")

        object.__setattr__(self, "start_sec", start_sec)
        object.__setattr__(self, "end_sec", end_sec)
        object.__setattr__(self, "amplitude_deg", amplitude_deg)

    def to_dict(self) -> dict[str, str | float]:
        return {
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "action": self.action,
            "primary_axis": self.primary_axis,
            "amplitude_deg": self.amplitude_deg,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MotionSegment":
        document = _require_fields(value, _SEGMENT_FIELDS, "MotionSegment")
        return cls(
            start_sec=document["start_sec"],
            end_sec=document["end_sec"],
            action=document["action"],
            primary_axis=document["primary_axis"],
            amplitude_deg=document["amplitude_deg"],
            reason=document["reason"],
        )


@dataclass(frozen=True)
class MotionPlan:
    """A complete high-level speaking-motion plan for one audio duration."""

    mode: str
    duration_sec: float
    segments: tuple[MotionSegment, ...]

    def __post_init__(self) -> None:
        duration_sec = _finite_number(self.duration_sec, "duration_sec")
        if duration_sec <= 0.0:
            raise ValueError("duration_sec must be greater than zero")
        if self.mode != "speaking":
            raise ValueError("mode must be 'speaking'")
        if not isinstance(self.segments, (list, tuple)):
            raise TypeError("segments must be a sequence of MotionSegment objects")
        segments = tuple(self.segments)
        for index, segment in enumerate(segments):
            if not isinstance(segment, MotionSegment):
                raise TypeError(f"segments[{index}] must be a MotionSegment")
            if segment.end_sec > duration_sec:
                raise ValueError(
                    f"segments[{index}].end_sec must not exceed duration_sec"
                )

        object.__setattr__(self, "duration_sec", duration_sec)
        object.__setattr__(self, "segments", segments)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "duration_sec": self.duration_sec,
            "segments": [segment.to_dict() for segment in self.segments],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MotionPlan":
        document = _require_fields(value, _PLAN_FIELDS, "MotionPlan")
        raw_segments = document["segments"]
        if not isinstance(raw_segments, (list, tuple)):
            raise TypeError("MotionPlan segments must be a list or tuple")
        segments = tuple(MotionSegment.from_dict(item) for item in raw_segments)
        return cls(
            mode=document["mode"],
            duration_sec=document["duration_sec"],
            segments=segments,
        )
