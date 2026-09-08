"""Small shared data types for the new robot Runtime."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import ClassVar, Sequence


@dataclass(frozen=True)
class WordTimestamp:
    """One word interval in seconds relative to its audio stream."""

    text: str
    start_sec: float
    end_sec: float


@dataclass(frozen=True)
class UserAudio:
    """Stable user PCM captured for one turn."""

    pcm_s16le: bytes
    sample_rate: int
    channels: int
    duration_sec: float


@dataclass(frozen=True)
class UserSpeech:
    text: str
    words: Sequence[WordTimestamp]
    language: str | None


@dataclass(frozen=True)
class RobotSpeech:
    """Robot text and its standard raw PCM s16le bytes."""

    text: str
    pcm_s16le: bytes
    words: Sequence[WordTimestamp]
    duration_sec: float


@dataclass(frozen=True)
class MotionOutput:
    """Fixed backend output: 30 fps radian RPY offsets in roll/pitch/yaw order.

    Each frame is relative to the measured pose at motion start:
    ``absolute_rpy[t] = motion_start_rpy + rpy_offset[t]``. It is not an
    increment that should be integrated over time.
    """

    rpy_offset: Sequence[Sequence[float]]
    fps: float = 30.0

    representation: ClassVar[str] = "rpy_offset"
    unit: ClassVar[str] = "radian"
    order: ClassVar[tuple[str, str, str]] = ("roll", "pitch", "yaw")


@dataclass(frozen=True)
class FinalTrajectory:
    """Runtime-internal absolute RPY trajectory; this is not Motor JSON."""

    rpy: Sequence[Sequence[float]]
    fps: float
    states: Sequence[str]
    duration_sec: float

    unit: ClassVar[str] = "radian"
    order: ClassVar[tuple[str, str, str]] = ("roll", "pitch", "yaw")


@dataclass(frozen=True)
class TurnSummary:
    turn_id: str
    user_text: str
    robot_text: str
    status: str


@dataclass
class SessionContext:
    session_id: str
    recent_turns: list[TurnSummary] = field(default_factory=list)

    MAX_RECENT_TURNS: ClassVar[int] = 10

    def add_turn(self, summary: TurnSummary) -> None:
        self.recent_turns.append(summary)
        if len(self.recent_turns) > self.MAX_RECENT_TURNS:
            del self.recent_turns[:-self.MAX_RECENT_TURNS]


@dataclass(frozen=True)
class DialogueRequest:
    """One stateless Dialogue call with a lightweight Session snapshot."""

    user_text: str
    history: tuple[TurnSummary, ...]

    @classmethod
    def from_session(
        cls, user_text: str, session: SessionContext
    ) -> "DialogueRequest":
        return cls(user_text=user_text, history=tuple(session.recent_turns[-10:]))


@dataclass
class RobotState:
    """Latest known physical robot state; head_rpy is radian roll/pitch/yaw."""

    head_rpy: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    head_rpy_timestamp: float | None = None
    head_rpy_valid: bool = False
    motor_available: bool = False
    motion_executing: bool = False


@dataclass
class TurnContext:
    """Mutable workspace for the one active turn."""

    turn_id: str
    user_audio: UserAudio | None = None
    user_speech: UserSpeech | None = None
    robot_text: str | None = None
    robot_speech: RobotSpeech | None = None
    motion_output: MotionOutput | None = None
    final_trajectory: FinalTrajectory | None = None
    status: str = "collecting"


@dataclass(frozen=True)
class MotionRequest:
    """Independent snapshot passed to a Motion backend."""

    current_turn: TurnContext
    recent_turns: tuple[TurnSummary, ...]
    robot_state: RobotState

    @classmethod
    def snapshot(
        cls,
        current_turn: TurnContext,
        session: SessionContext,
        robot_state: RobotState,
    ) -> "MotionRequest":
        return cls(
            current_turn=copy.deepcopy(current_turn),
            recent_turns=tuple(copy.deepcopy(session.recent_turns[-10:])),
            robot_state=copy.deepcopy(robot_state),
        )
