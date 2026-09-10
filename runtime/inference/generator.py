"""Generate and persist one turn: robot audio + postprocessed trajectory."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from ..contracts import MotionRequest, RobotSpeech
from .artifacts import GenerationArtifacts, save_generation
from .base import MOTION_FPS, MotionBackend
from .default_motion import default_motion_output
from .motor_json import final_trajectory_to_motor_document
from .processor import MotionProcessor


class PoseUnavailableError(RuntimeError):
    """No valid measured head pose, so no absolute trajectory can be built."""


@dataclass(frozen=True)
class GeneratedTurn:
    speech: RobotSpeech
    document: dict | None
    artifacts: GenerationArtifacts
    source: str


class TurnGenerator:
    """Model output and postprocessing for one turn, persisted to disk."""

    def __init__(
        self,
        backend: MotionBackend,
        processor: MotionProcessor,
        output_dir: str | Path,
        model_label: str = "",
        fps: float = MOTION_FPS,
    ) -> None:
        if not isinstance(backend, MotionBackend):
            raise TypeError("backend must implement MotionBackend")
        if not isinstance(processor, MotionProcessor):
            raise TypeError("processor must be a MotionProcessor")
        self.backend = backend
        self.processor = processor
        self.output_dir = Path(output_dir)
        self.model_label = model_label
        self.fps = float(fps)

    @staticmethod
    def _speech(request: MotionRequest) -> RobotSpeech:
        if not isinstance(request, MotionRequest):
            raise TypeError("generation requires a MotionRequest snapshot")
        speech = request.current_turn.robot_speech
        if not isinstance(speech, RobotSpeech):
            raise ValueError("generation requires RobotSpeech on the active turn")
        return speech

    def _metadata(
        self,
        speech: RobotSpeech,
        document: dict | None,
        source: str,
        name: str,
        reason: str | None,
    ) -> dict:
        return {
            "turn": name,
            "source": source,
            "reason": reason,
            "model": self.model_label,
            "fps": self.fps,
            "robot_text": speech.text,
            "audio_duration_sec": speech.duration_sec,
            "word_count": len(speech.words),
            "frame_count": len(document["trajectory"]) if document else 0,
            "created_unix_sec": time.time(),
        }

    def _save(
        self,
        speech: RobotSpeech,
        document: dict | None,
        source: str,
        name: str,
        reason: str | None = None,
    ) -> GenerationArtifacts:
        return save_generation(
            self.output_dir,
            speech,
            document,
            self._metadata(speech, document, source, name, reason),
        )

    def generate(self, request: MotionRequest, name: str) -> GeneratedTurn:
        """Model inference + postprocessing; raises when no valid start pose."""
        speech = self._speech(request)
        if not request.robot_state.head_rpy_valid:
            raise PoseUnavailableError("measured head pose is not valid")
        output = self.backend.infer(request)
        final = self.processor.process(output, request.robot_state.head_rpy)
        document = final_trajectory_to_motor_document(final, name)
        artifacts = self._save(speech, document, "model", name)
        return GeneratedTurn(speech, document, artifacts, "model")

    def generate_fallback(
        self, request: MotionRequest, name: str, reason: str
    ) -> GeneratedTurn:
        """Default shake motion; used when model generation failed."""
        speech = self._speech(request)
        document = None
        if request.robot_state.head_rpy_valid:
            output = default_motion_output(speech.duration_sec, self.fps)
            final = self.processor.process(output, request.robot_state.head_rpy)
            document = final_trajectory_to_motor_document(final, name)
        artifacts = self._save(speech, document, "fallback", name, reason)
        return GeneratedTurn(speech, document, artifacts, "fallback")

    def generate_audio_only(
        self, request: MotionRequest, name: str, reason: str
    ) -> GeneratedTurn:
        """Keep the speech and its audio file, but send no trajectory."""
        speech = self._speech(request)
        artifacts = self._save(speech, None, "pose_unavailable", name, reason)
        return GeneratedTurn(speech, None, artifacts, "pose_unavailable")
