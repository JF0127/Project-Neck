"""Deterministic Continuous Motion Generator V4 for speaking MotionPlans."""
from __future__ import annotations

from dataclasses import dataclass
import math

from ..contracts import MotionOutput
from .base import MOTION_FPS
from .motion_plan import MotionPlan, MotionSegment
from .motion_plan_validator import validate_motion_plan
from .prosody import ProsodyAnalysis

# Legacy V3 base settings retained for reproducible software comparison;
# shake and composed limit remain shared engineering defaults.
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

# Each axis has its own non-uniform clock, target series and MOVE/HOLD ratios.
# The co-prime series lengths avoid a short repeated three-axis pose cycle.
_POSTURAL_INTERVALS = (
    (2.6, 3.0, 2.4, 2.8, 3.1, 2.5, 2.9, 2.7, 3.2),  # roll
    (2.2, 2.7, 2.4, 2.9, 2.3, 2.6, 3.0, 2.5, 2.8, 2.4, 2.7),  # pitch
    (1.9, 2.3, 2.0, 2.6, 2.2, 2.5, 1.8, 2.4, 2.1, 2.7, 2.0, 2.3, 2.5),  # yaw
)
_POSTURAL_TARGETS = (
    (0.45, 0.15, -0.55, -0.25, 0.75, 0.2, -0.8, -0.8, 0.35),
    (-0.5, -0.9, -0.2, 0.55, 0.35, -0.65, -0.3, 0.7, 0.1, -0.8, 0.4),
    (0.6, 0.95, 0.35, -0.45, -0.85, -0.15, 0.5, 0.8, 0.2, -0.7, -0.3, 0.65, 0.1),
)
_POSTURAL_MOVE_RATIOS = (
    (0.60, 0.72, 0.64, 0.78, 0.62),
    (0.68, 0.60, 0.74, 0.66, 0.70),
    (0.72, 0.66, 0.76, 0.62, 0.70),
)
_POSTURAL_AMPLITUDES_DEG = (0.9, 1.4, 1.9)


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


def _postural_flow(
    duration_sec: float,
    frame_count: int,
    prosody: ProsodyAnalysis | None,
) -> list[list[float]]:
    """Sample independent minimum-jerk axis transitions, with a soft HOLD after each MOVE.

    Anchors extend beyond the utterance; a short utterance never snaps to a
    truncated endpoint. Speech rate changes clock speed by at most 6%.
    """
    clock = 1.0
    if prosody is not None and prosody.segments:
        levels = [segment.rate_level for segment in prosody.segments]
        clock = 0.94 if levels.count("fast") > len(levels) / 2 else (
            1.06 if levels.count("slow") > len(levels) / 2 else 1.0
        )
    frames = [[0.0, 0.0, 0.0] for _ in range(frame_count)]
    for axis in range(3):
        anchors = [(0.0, 0.0, 1.0)]
        index = 0
        while anchors[-1][0] < duration_sec:
            interval = _POSTURAL_INTERVALS[axis][index % len(_POSTURAL_INTERVALS[axis])] * clock
            target = math.radians(
                _POSTURAL_AMPLITUDES_DEG[axis]
                * _POSTURAL_TARGETS[axis][index % len(_POSTURAL_TARGETS[axis])]
            )
            ratio = _POSTURAL_MOVE_RATIOS[axis][index % len(_POSTURAL_MOVE_RATIOS[axis])]
            anchors.append((anchors[-1][0] + interval, target, ratio))
            index += 1
        anchor = 1
        for frame in range(frame_count):
            timestamp = frame / MOTION_FPS
            while anchor < len(anchors) - 1 and timestamp > anchors[anchor][0]:
                anchor += 1
            start_time, start_value, _ = anchors[anchor - 1]
            end_time, end_value, ratio = anchors[anchor]
            progress = min(1.0, (timestamp - start_time) / ((end_time - start_time) * ratio))
            frames[frame][axis] = _transition(start_value, end_value, progress)
    return frames


def _prosodic_accents(
    prosody: ProsodyAnalysis | None,
    frame_count: int,
) -> list[list[float]]:
    """Sparse pitch emphasis around measured energy peaks; zero is the carrier."""
    frames = [[0.0, 0.0, 0.0] for _ in range(frame_count)]
    if prosody is None:
        return frames
    for index, segment in enumerate(prosody.segments):
        if segment.energy_level != "high" and segment.relative_energy <= 1.15:
            continue
        center = min(segment.end_sec, max(segment.start_sec, segment.energy_peak_time_sec))
        # Avoid very short excursions for tiny alignment segments.
        width = min(0.65, max(0.35, 0.56 - 0.035 * (segment.speech_rate_chars_sec - 5.0)))
        width = min(width, segment.end_sec - segment.start_sec)
        if width < 0.25:
            continue
        half = width / 2
        center = min(segment.end_sec - half, max(segment.start_sec + half, center))
        energy = min(1.0, max(0.0, (segment.relative_energy - 1.15) / 0.85))
        amplitude = math.radians(min(0.75, max(0.3, 0.38 + 0.30 * energy)))
        direction = -1 if index % 3 == 2 else 1
        for frame in range(1, frame_count):
            distance = abs(frame / MOTION_FPS - center) / half
            if distance < 1.0:
                frames[frame][1] += direction * amplitude * _minimum_jerk(1.0 - distance)
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


def _compose_v4(
    postural: list[list[float]],
    prosodic: list[list[float]],
    semantic: list[list[float]],
    max_offset_deg: float,
) -> list[list[float]]:
    """Reduce same-direction overflow: postural first, accent next, gesture last.

    Never invert a layer to compensate for another layer. The final clamp is
    only a numerical guard; validated gestures are already within the limit.
    """
    limit = math.radians(max_offset_deg)
    output = []
    for p, a, s in zip(postural, prosodic, semantic):
        row = []
        for components in zip(p, a, s):
            values = list(components)
            for layer in range(3):
                total = sum(values)
                if abs(total) <= limit:
                    break
                sign = 1 if total > 0 else -1
                if values[layer] * sign > 0:
                    values[layer] -= sign * min(abs(values[layer]), abs(total) - limit)
            row.append(min(limit, max(-limit, sum(values))))
        output.append(row)
    return output


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


@dataclass(frozen=True)
class MotionLayers:
    """Unclipped component layers and safe composed output, all radians RPY."""

    postural_flow: tuple[tuple[float, float, float], ...]
    prosodic_accent: tuple[tuple[float, float, float], ...]
    semantic_gesture: tuple[tuple[float, float, float], ...]
    composed_raw: tuple[tuple[float, float, float], ...]


class TrajectoryGenerator:
    """Compose a validated MotionPlan into V4 continuous 30 Hz relative RPY."""

    def __init__(self, config: TrajectoryGeneratorConfig | None = None) -> None:
        self.config = config or TrajectoryGeneratorConfig()
        if not isinstance(self.config, TrajectoryGeneratorConfig):
            raise TypeError("config must be a TrajectoryGeneratorConfig")

    def generate_layers(
        self, plan: MotionPlan, prosody: ProsodyAnalysis | None = None,
        *, include_prosodic: bool = True, include_semantic: bool = True,
    ) -> MotionLayers:
        validate_motion_plan(plan)
        if prosody is not None and not isinstance(prosody, ProsodyAnalysis):
            raise TypeError("prosody must be a ProsodyAnalysis")
        frame_count = max(2, int(round(plan.duration_sec * MOTION_FPS)))
        postural = _postural_flow(plan.duration_sec, frame_count, prosody)
        accent = (
            _prosodic_accents(prosody, frame_count)
            if include_prosodic else [[0.0] * 3 for _ in range(frame_count)]
        )
        semantic = (
            _generate_gesture_modulation(plan, frame_count, self.config)
            if include_semantic else [[0.0] * 3 for _ in range(frame_count)]
        )
        # A segment beginning at t=0 must not displace the initial relative pose.
        semantic[0] = [0.0, 0.0, 0.0]
        composed = _compose_v4(postural, accent, semantic, self.config.max_composed_offset_deg)
        return MotionLayers(
            *(tuple(tuple(frame) for frame in layer)
              for layer in (postural, accent, semantic, composed))
        )

    def generate(self, plan: MotionPlan, prosody: ProsodyAnalysis | None = None) -> MotionOutput:
        layers = self.generate_layers(plan, prosody)
        return MotionOutput(rpy_offset=layers.composed_raw, fps=MOTION_FPS)
