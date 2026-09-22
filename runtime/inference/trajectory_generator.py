"""Deterministic Continuous Motion Generator V3 for speaking MotionPlans."""
from __future__ import annotations

from dataclasses import dataclass
import math

from ..contracts import MotionOutput
from .base import MOTION_FPS
from .motion_plan import MotionPlan, MotionSegment
from .motion_plan_validator import validate_motion_plan

# Conservative V3 engineering defaults, not a final Motion Design Specification.
NOD_PHASE_RATIOS = (0.45, 0.10, 0.45)
TURN_PHASE_RATIOS = (0.35, 0.30, 0.35)
TILT_PHASE_RATIOS = (0.30, 0.40, 0.30)
SHAKE_PHASE_RATIOS = (0.25, 0.10, 0.30, 0.10, 0.25)
DEFAULT_SHAKE_REVERSE_RATIO = 0.8
DEFAULT_BASE_ROLL_AMPLITUDE_DEG = 0.35
DEFAULT_BASE_PITCH_AMPLITUDE_DEG = 0.45
DEFAULT_BASE_YAW_AMPLITUDE_DEG = 0.50
DEFAULT_BASE_ANCHOR_INTERVALS_SEC = (0.95, 1.30, 0.85, 1.15, 1.45, 1.05, 1.25)
DEFAULT_BASE_TRANSITION_RATIOS = (0.72, 0.84, 0.68, 0.80, 0.76, 0.88, 0.70)
DEFAULT_MAX_COMPOSED_OFFSET_DEG = 5.0

# Normalized, deliberately irregular targets. Repetition only starts after roughly
# ten seconds, longer than the normal short spoken response.
_BASE_TARGET_PATTERN = (
    (0.65, -0.45, 0.50),
    (0.65, -0.80, 0.85),
    (-0.40, -0.15, 0.25),
    (-0.85, 0.50, 0.25),
    (-0.15, 0.15, -0.55),
    (0.45, -0.55, -0.80),
    (0.10, -0.20, -0.10),
    (-0.55, 0.70, 0.65),
    (0.75, 0.35, 0.40),
    (0.25, -0.10, -0.35),
    (-0.30, -0.65, -0.70),
)
_AXIS_INDEX = {"roll": 0, "pitch": 1, "yaw": 2}


def _finite_float(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _positive_tuple(values: tuple[float, ...], name: str) -> tuple[float, ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{name} must be a non-empty tuple")
    converted = tuple(_finite_float(value, name) for value in values)
    if any(value <= 0.0 for value in converted):
        raise ValueError(f"{name} values must be greater than zero")
    return converted


@dataclass(frozen=True)
class TrajectoryGeneratorConfig:
    shake_reverse_ratio: float = DEFAULT_SHAKE_REVERSE_RATIO
    base_roll_amplitude_deg: float = DEFAULT_BASE_ROLL_AMPLITUDE_DEG
    base_pitch_amplitude_deg: float = DEFAULT_BASE_PITCH_AMPLITUDE_DEG
    base_yaw_amplitude_deg: float = DEFAULT_BASE_YAW_AMPLITUDE_DEG
    base_anchor_intervals_sec: tuple[float, ...] = DEFAULT_BASE_ANCHOR_INTERVALS_SEC
    base_transition_ratios: tuple[float, ...] = DEFAULT_BASE_TRANSITION_RATIOS
    max_composed_offset_deg: float = DEFAULT_MAX_COMPOSED_OFFSET_DEG

    def __post_init__(self) -> None:
        reverse_ratio = _finite_float(
            self.shake_reverse_ratio, "shake_reverse_ratio"
        )
        if not 0.0 < reverse_ratio <= 1.0:
            raise ValueError("shake_reverse_ratio must be in (0, 1]")
        object.__setattr__(self, "shake_reverse_ratio", reverse_ratio)

        amplitudes = []
        for name in (
            "base_roll_amplitude_deg",
            "base_pitch_amplitude_deg",
            "base_yaw_amplitude_deg",
        ):
            value = _finite_float(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
            amplitudes.append(value)

        intervals = _positive_tuple(
            self.base_anchor_intervals_sec, "base_anchor_intervals_sec"
        )
        transition_ratios = _positive_tuple(
            self.base_transition_ratios, "base_transition_ratios"
        )
        if any(value > 1.0 for value in transition_ratios):
            raise ValueError("base_transition_ratios values must not exceed one")
        object.__setattr__(self, "base_anchor_intervals_sec", intervals)
        object.__setattr__(self, "base_transition_ratios", transition_ratios)

        max_offset = _finite_float(
            self.max_composed_offset_deg, "max_composed_offset_deg"
        )
        if max_offset <= 0.0:
            raise ValueError("max_composed_offset_deg must be greater than zero")
        if any(value > max_offset for value in amplitudes):
            raise ValueError("base amplitudes must not exceed max_composed_offset_deg")
        object.__setattr__(self, "max_composed_offset_deg", max_offset)


def _minimum_jerk(value: float) -> float:
    t = min(1.0, max(0.0, float(value)))
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5


def _transition(start: float, end: float, progress: float) -> float:
    return start + (end - start) * _minimum_jerk(progress)


def _single_peak_modulation_scale(
    progress: float,
    ratios: tuple[float, float, float],
) -> float:
    """Return a local excursion envelope; zero means the current base carrier."""
    rise, hold, fall = ratios
    if progress < rise:
        return _minimum_jerk(progress / rise)
    if progress <= rise + hold:
        return 1.0
    return 1.0 - _minimum_jerk((progress - rise - hold) / fall)


def _shake(progress: float, reverse_ratio: float) -> float:
    rise, positive_hold, cross, negative_hold, return_phase = SHAKE_PHASE_RATIOS
    boundary_1 = rise
    boundary_2 = boundary_1 + positive_hold
    boundary_3 = boundary_2 + cross
    boundary_4 = boundary_3 + negative_hold
    if progress < boundary_1:
        return _transition(0.0, 1.0, progress / rise)
    if progress <= boundary_2:
        return 1.0
    if progress < boundary_3:
        return _transition(
            1.0,
            -reverse_ratio,
            (progress - boundary_2) / cross,
        )
    if progress <= boundary_4:
        return -reverse_ratio
    return _transition(
        -reverse_ratio,
        0.0,
        (progress - boundary_4) / return_phase,
    )


def _primitive_modulation_scale(
    segment: MotionSegment,
    progress: float,
    config: TrajectoryGeneratorConfig,
) -> float:
    if segment.action == "nod":
        return _single_peak_modulation_scale(progress, NOD_PHASE_RATIOS)
    if segment.action == "turn":
        return _single_peak_modulation_scale(progress, TURN_PHASE_RATIOS)
    if segment.action == "tilt":
        return _single_peak_modulation_scale(progress, TILT_PHASE_RATIOS)
    if segment.action == "shake":
        return _shake(progress, config.shake_reverse_ratio)
    raise ValueError(f"unsupported action: {segment.action}")


def _base_target(index: int, config: TrajectoryGeneratorConfig) -> tuple[float, ...]:
    normalized = _BASE_TARGET_PATTERN[index % len(_BASE_TARGET_PATTERN)]
    amplitudes_deg = (
        config.base_roll_amplitude_deg,
        config.base_pitch_amplitude_deg,
        config.base_yaw_amplitude_deg,
    )
    return tuple(
        math.radians(scale * amplitude)
        for scale, amplitude in zip(normalized, amplitudes_deg)
    )


def _generate_base_flow(
    duration_sec: float,
    frame_count: int,
    config: TrajectoryGeneratorConfig,
) -> list[list[float]]:
    """Generate low-frequency deterministic RPY carrier motion."""
    anchors: list[tuple[float, tuple[float, ...], float]] = [
        (0.0, (0.0, 0.0, 0.0), 1.0)
    ]
    anchor_time = 0.0
    interval_index = 0
    while anchor_time < duration_sec:
        interval = config.base_anchor_intervals_sec[
            interval_index % len(config.base_anchor_intervals_sec)
        ]
        anchor_time = min(duration_sec, anchor_time + interval)
        transition_ratio = config.base_transition_ratios[
            interval_index % len(config.base_transition_ratios)
        ]
        anchors.append(
            (anchor_time, _base_target(interval_index, config), transition_ratio)
        )
        interval_index += 1

    frames: list[list[float]] = []
    anchor_index = 1
    for frame_index in range(frame_count):
        timestamp = min(duration_sec, frame_index / MOTION_FPS)
        while (
            anchor_index < len(anchors) - 1
            and timestamp > anchors[anchor_index][0]
        ):
            anchor_index += 1
        start_time, start_pose, _ = anchors[anchor_index - 1]
        end_time, end_pose, transition_ratio = anchors[anchor_index]
        span = end_time - start_time
        if span <= 0.0:
            pose = end_pose
        else:
            interval_progress = (timestamp - start_time) / span
            transition_progress = min(1.0, interval_progress / transition_ratio)
            pose = tuple(
                _transition(start, end, transition_progress)
                for start, end in zip(start_pose, end_pose)
            )
        frames.append(list(pose))
    frames[0] = [0.0, 0.0, 0.0]
    return frames


def _generate_gesture_modulation(
    plan: MotionPlan,
    frame_count: int,
    config: TrajectoryGeneratorConfig,
) -> list[list[float]]:
    """Generate sparse local excursions around, but independent of, the carrier."""
    modulation = [[0.0, 0.0, 0.0] for _ in range(frame_count)]
    for segment in plan.segments:
        start_frame = min(
            frame_count - 2,
            max(0, int(round(segment.start_sec * MOTION_FPS))),
        )
        end_frame = min(
            frame_count - 1,
            max(start_frame + 1, int(round(segment.end_sec * MOTION_FPS))),
        )
        span = end_frame - start_frame
        amplitude_rad = math.radians(segment.amplitude_deg)
        axis = _AXIS_INDEX[segment.primary_axis]
        for frame_index in range(start_frame, end_frame + 1):
            progress = (frame_index - start_frame) / span
            scale = _primitive_modulation_scale(segment, progress, config)
            modulation[frame_index][axis] += amplitude_rad * scale
    return modulation


def _compose_with_safety(
    base: list[list[float]],
    modulation: list[list[float]],
    max_offset_deg: float,
) -> list[list[float]]:
    """Compose layers while attenuating carrier first at the offset limit."""
    limit = math.radians(max_offset_deg)
    frames: list[list[float]] = []
    for base_frame, modulation_frame in zip(base, modulation):
        composed: list[float] = []
        for carrier, excursion in zip(base_frame, modulation_frame):
            # A custom lower limit may be below a validated gesture amplitude.
            # Clamp that excursion first; normally only the carrier is attenuated.
            safe_excursion = min(limit, max(-limit, excursion))
            safe_carrier = min(
                limit - safe_excursion,
                max(-limit - safe_excursion, carrier),
            )
            composed.append(
                min(limit, max(-limit, safe_carrier + safe_excursion))
            )
        frames.append(composed)
    return frames


class TrajectoryGenerator:
    """Compose a validated MotionPlan into continuous 30 Hz relative RPY."""

    def __init__(self, config: TrajectoryGeneratorConfig | None = None) -> None:
        self.config = config or TrajectoryGeneratorConfig()
        if not isinstance(self.config, TrajectoryGeneratorConfig):
            raise TypeError("config must be a TrajectoryGeneratorConfig")

    def generate(self, plan: MotionPlan) -> MotionOutput:
        validate_motion_plan(plan)
        frame_count = max(2, int(round(plan.duration_sec * MOTION_FPS)))
        base = _generate_base_flow(plan.duration_sec, frame_count, self.config)
        modulation = _generate_gesture_modulation(plan, frame_count, self.config)
        frames = _compose_with_safety(
            base,
            modulation,
            self.config.max_composed_offset_deg,
        )
        frames[0] = [0.0, 0.0, 0.0]
        return MotionOutput(
            rpy_offset=tuple(tuple(frame) for frame in frames),
            fps=MOTION_FPS,
        )
