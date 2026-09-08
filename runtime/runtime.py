"""Algorithm Runtime V1 orchestration with explicit coarse-grained state."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .asr import WhisperASR
from .dialogue import DialogueError, DialoguePolicy, FixedDialogue
from .experiment_logger import ExperimentLogger
from .motion import MotionTurn, SpeakerMotionPipeline
from .neck_client import NeckClient
from .tts import EdgeTTS, TTSResult


class RuntimeState(str, Enum):
    IDLE = "IDLE"
    RECEIVING_USER = "RECEIVING_USER"
    TRANSCRIBING = "TRANSCRIBING"
    THINKING = "THINKING"
    SYNTHESIZING = "SYNTHESIZING"
    SPEAKING = "SPEAKING"
    ERROR = "ERROR"


@dataclass(frozen=True)
class TurnResult:
    turn_number: int
    experiment_turn_id: str | None
    user_text: str
    user_words: list[dict]
    robot_text: str
    robot_audio: TTSResult
    motion: MotionTurn


class AlgorithmRuntime:
    """One process-wide runtime owning resident ASR and motion models."""

    def __init__(
        self,
        baseline_checkpoint: str | Path,
        whisper_model: str | Path,
        dialogue: DialoguePolicy | None = None,
        tts_voice: str = "en-US-GuyNeural",
        asr_device: str = "cpu",
        motion_device: str = "auto",
        asr_language: str | None = "en",
        neck_socket: str = "/tmp/neck_model.sock",
        neck_measurement_socket: str = "/tmp/neck_measurement.sock",
        mock_neck: bool = False,
        experiment_logger: ExperimentLogger | None = None,
    ):
        self.state = RuntimeState.IDLE
        self.turn_count = 0
        self._turn_lock = asyncio.Lock()
        self.experiment_logger = experiment_logger
        self._active_experiment_turn_id: str | None = None
        self.asr = WhisperASR(str(whisper_model), device=asr_device, language=asr_language)
        self.dialogue = dialogue or FixedDialogue()
        self.tts = EdgeTTS(tts_voice)
        self.motion = SpeakerMotionPipeline(
            baseline_checkpoint, device=motion_device
        )
        self.neck = NeckClient(
            neck_socket,
            mock=mock_neck,
            measurement_socket_path=neck_measurement_socket,
        )
        print(f"[runtime] ready; state={self.state.value}")

    def _set_state(self, state: RuntimeState) -> None:
        previous = self.state
        self.state = state
        print(f"[runtime][state] {previous.value} -> {state.value}")

    def begin_user_stream(self) -> None:
        if self.state != RuntimeState.IDLE:
            raise RuntimeError(f"runtime is busy: {self.state.value}")
        self._set_state(RuntimeState.RECEIVING_USER)

    def fail(self, error: Exception) -> None:
        self.abort_active_turn(error)
        self._set_state(RuntimeState.ERROR)
        print(f"[runtime][error] {type(error).__name__}: {error}")

    def recover(self) -> None:
        if self.state == RuntimeState.ERROR:
            self._set_state(RuntimeState.IDLE)

    def finish_speaking(self, turn_id: str | None = None) -> None:
        completed_turn = turn_id or self._active_experiment_turn_id
        if self.experiment_logger is not None and completed_turn is not None:
            self.experiment_logger.end_turn(completed_turn)
        if completed_turn == self._active_experiment_turn_id:
            self._active_experiment_turn_id = None
        if self.state == RuntimeState.SPEAKING:
            self._set_state(RuntimeState.IDLE)

    def abort_active_turn(self, error: Exception) -> None:
        if self.experiment_logger is not None and self._active_experiment_turn_id is not None:
            self.experiment_logger.end_turn(
                self._active_experiment_turn_id,
                status="error",
                error=f"{type(error).__name__}: {error}",
            )
        self._active_experiment_turn_id = None

    def record_robot_audio_start(self, turn_id: str | None, stream_id: str) -> None:
        if self.experiment_logger is None or turn_id is None:
            return
        self.experiment_logger.record_dialogue(turn_id, robot_stream_id=stream_id)
        self.experiment_logger.record_event(turn_id, "robot_audio_start", stream_id=stream_id)

    def record_robot_audio_end(self, turn_id: str | None, stream_id: str) -> None:
        if self.experiment_logger is not None and turn_id is not None:
            self.experiment_logger.record_event(turn_id, "robot_audio_end", stream_id=stream_id)

    async def process_user_stream(
        self, pcm_s16le: bytes, user_stream_id: str | None = None
    ) -> TurnResult:
        if self.state != RuntimeState.RECEIVING_USER:
            raise RuntimeError(f"unexpected stream_end in state {self.state.value}")
        if not pcm_s16le:
            raise ValueError("empty user PCM stream")

        async with self._turn_lock:
            self.turn_count += 1
            turn_number = self.turn_count
            turn_id = None
            if self.experiment_logger is not None:
                turn_id = self.experiment_logger.start_turn(turn_number, user_stream_id)
                self._active_experiment_turn_id = turn_id
                self.experiment_logger.record_event(
                    turn_id,
                    "user_audio_end",
                    stream_id=user_stream_id,
                    pcm_bytes=len(pcm_s16le),
                )

            try:
                self._set_state(RuntimeState.TRANSCRIBING)
                transcription = await asyncio.to_thread(self.asr.transcribe_pcm, pcm_s16le)
                user_words = [word.as_dict() for word in transcription.words]
                user_text = transcription.text.strip()
                if self.experiment_logger is not None and turn_id is not None:
                    self.experiment_logger.record_event(
                        turn_id, "asr_complete", word_count=len(user_words)
                    )
                if not user_text:
                    raise DialogueError("ASR returned empty user text")

                self._set_state(RuntimeState.THINKING)
                dialogue_backend = getattr(
                    self.dialogue, "backend", type(self.dialogue).__name__
                )
                dialogue_model = getattr(self.dialogue, "model", None)
                if self.experiment_logger is not None and turn_id is not None:
                    self.experiment_logger.record_event(turn_id, "thinking_start")
                    self.experiment_logger.record_event(
                        turn_id,
                        "dialogue_start",
                        backend=dialogue_backend,
                        model=dialogue_model,
                    )
                try:
                    robot_text = await asyncio.to_thread(self.dialogue.reply, user_text)
                except Exception:
                    metadata = getattr(self.dialogue, "metadata", {})
                    if self.experiment_logger is not None and turn_id is not None:
                        self.experiment_logger.record_event(
                            turn_id,
                            "dialogue_failed",
                            backend=metadata.get("backend", dialogue_backend),
                            model=metadata.get("model", dialogue_model),
                            latency_sec=metadata.get("latency_sec"),
                        )
                    raise
                if not isinstance(robot_text, str) or not robot_text.strip():
                    raise DialogueError("dialogue backend returned empty robot text")
                robot_text = robot_text.strip()
                metadata = getattr(self.dialogue, "metadata", {})
                print(f"[runtime][dialogue] user={user_text!r} -> robot={robot_text!r}")
                if self.experiment_logger is not None and turn_id is not None:
                    self.experiment_logger.record_event(
                        turn_id,
                        "dialogue_complete",
                        backend=metadata.get("backend", dialogue_backend),
                        model=metadata.get("model", dialogue_model),
                        latency_sec=metadata.get("latency_sec"),
                    )
                    self.experiment_logger.record_dialogue(
                        turn_id,
                        user_text=user_text,
                        robot_text=robot_text,
                        user_words=user_words,
                        user_stream_id=user_stream_id,
                        dialogue_backend=metadata.get("backend", dialogue_backend),
                        dialogue_model=metadata.get("model", dialogue_model),
                        dialogue_latency_sec=metadata.get("latency_sec"),
                        dialogue_usage=metadata.get("usage"),
                    )

                self._set_state(RuntimeState.SYNTHESIZING)
                robot_audio = await self.tts.synthesize(robot_text)
                if self.experiment_logger is not None and turn_id is not None:
                    self.experiment_logger.record_event(
                        turn_id,
                        "tts_complete",
                        duration_sec=robot_audio.duration_sec,
                    )
                # Edge normally supplies WordBoundary metadata. Keep one resident ASR
                # fallback for voices/services that don't return it.
                if robot_audio.words:
                    robot_words = [word.as_dict() for word in robot_audio.words]
                else:
                    robot_transcription = await asyncio.to_thread(
                        self.asr.transcribe_waveform, robot_audio.waveform
                    )
                    robot_words = [word.as_dict() for word in robot_transcription.words]

                if self.experiment_logger is not None and turn_id is not None:
                    await asyncio.to_thread(
                        self.experiment_logger.record_robot_motion_inputs,
                        turn_id,
                        robot_audio.pcm_s16le,
                        robot_words,
                        robot_audio.duration_sec,
                    )

                def record_generation(role, trajectory, metadata) -> None:
                    if self.experiment_logger is not None and turn_id is not None:
                        self.experiment_logger.record_generation(
                            turn_id,
                            role,
                            trajectory,
                            fps=metadata["fps"],
                            model=metadata.get("model"),
                            checkpoint=metadata.get("checkpoint"),
                            query_timestamps_sec=metadata.get("query_timestamps_sec"),
                            representation=metadata.get("representation"),
                        )

                motion = await asyncio.to_thread(
                    self.motion.generate_turn,
                    robot_audio.pcm_s16le,
                    robot_words,
                    robot_audio.duration_sec,
                    turn_number,
                    record_generation,
                )
                if self.experiment_logger is not None and turn_id is not None:
                    self.experiment_logger.record_event(
                        turn_id,
                        "neck_trajectory_ready",
                        num_frames=len(motion.document["trajectory"]),
                    )
                    await asyncio.to_thread(
                        self.experiment_logger.record_final_neck_trajectory,
                        turn_id,
                        motion.document,
                    )

                measured_output_path = None
                turn_origin_unix_sec = None
                if self.experiment_logger is not None and turn_id is not None:
                    measured_output_path, turn_origin_unix_sec = (
                        self.experiment_logger.measured_rpy_target(turn_id)
                    )
                await asyncio.to_thread(
                    self.neck.send,
                    motion.document,
                    measured_output_path,
                    turn_origin_unix_sec,
                )
                if self.experiment_logger is not None and turn_id is not None:
                    self.experiment_logger.record_event(
                        turn_id, "neck_trajectory_sent", mock=self.neck.mock
                    )
                self._set_state(RuntimeState.SPEAKING)
                return TurnResult(
                    turn_number=turn_number,
                    experiment_turn_id=turn_id,
                    user_text=user_text,
                    user_words=user_words,
                    robot_text=robot_text,
                    robot_audio=robot_audio,
                    motion=motion,
                )
            except Exception as exc:
                self.abort_active_turn(exc)
                raise
