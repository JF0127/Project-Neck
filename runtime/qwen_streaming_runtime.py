"""Production Silero/Qwen streaming voice Runtime."""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from .audio_server import AudioWebSocketServer
from .contracts import (
    DialogueRequest,
    MotionRequest,
    RobotSpeech,
    RobotState,
    SessionContext,
    TurnContext,
    TurnSummary,
)
from .dialogue import DeepSeekDialogue, DialogueError
from .doubao_tts import DoubaoTTS
from .feedback import MotorFeedbackMonitor
from .inference import MotionProcessor, TrajectoryOptimizer, TurnGenerator
from .logging_utils import log
from .neck_client import NeckClient
from .tts import TTS
from .vad import SileroVAD

SAMPLE_RATE = 16_000
PRE_ROLL_FRAMES = 20  # 400 ms, covering Silero's speech confirmation delay.
PLAYBACK_START_TIMEOUT_SEC = 30.0
SYSTEM_PROMPT = "你是一个机器人助手，请使用自然、简洁的中文进行对话。"


def _server_log(event: str) -> None:
    log(f"[SERVER] {event} mono={time.monotonic():.6f}")


class QwenStreamingRuntime:
    """Production half-duplex Qwen streaming adapter for AudioWebSocketServer."""

    def __init__(
        self,
        model_path: Path,
        vad_model_path: Path,
        dialogue: DeepSeekDialogue,
        tts: TTS,
        robot_state: RobotState | None = None,
        turn_generator: Any | None = None,
        neck_sender: Any | None = None,
        motion_sync_offset_ms: int = 0,
        motion_send_to_motor: bool = True,
        *,
        vad_threshold: float = 0.5,
        vad_min_speech_ms: int = 250,
        vad_min_silence_ms: int = 500,
        asr_language: str = "Chinese",
        asr_chunk_size_sec: float = 1.0,
        asr_unfixed_chunk_num: int = 4,
        asr_unfixed_token_num: int = 5,
        asr_gpu_memory_utilization: float = 0.7,
        asr_max_inference_batch_size: int = 1,
        asr_max_new_tokens: int = 4096,
    ) -> None:
        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3-ASR dependencies are missing; install runtime/requirements.txt"
            ) from exc

        resolved_model_path = model_path.expanduser().resolve()
        if not (resolved_model_path / "config.json").is_file():
            raise FileNotFoundError(
                f"Qwen3-ASR model does not exist: {resolved_model_path}"
            )
        if not asr_language.strip():
            raise ValueError("asr.language must not be empty")
        if not math.isfinite(asr_chunk_size_sec) or asr_chunk_size_sec <= 0.0:
            raise ValueError("asr.chunk_size_sec must be positive")
        if asr_unfixed_chunk_num < 0 or asr_unfixed_token_num < 0:
            raise ValueError("ASR unfixed values must be non-negative")

        self.vad = SileroVAD(
            model_path=vad_model_path,
            threshold=float(vad_threshold),
            min_speech_ms=int(vad_min_speech_ms),
            min_silence_ms=int(vad_min_silence_ms),
        )
        log(f"[runtime][asr] loading Qwen3-ASR streaming: {resolved_model_path}")
        self.model = Qwen3ASRModel.LLM(
            model=str(resolved_model_path),
            gpu_memory_utilization=float(asr_gpu_memory_utilization),
            max_inference_batch_size=int(asr_max_inference_batch_size),
            max_new_tokens=int(asr_max_new_tokens),
        )
        self.asr_language = asr_language.strip()
        self.asr_chunk_size_sec = float(asr_chunk_size_sec)
        self.asr_unfixed_chunk_num = int(asr_unfixed_chunk_num)
        self.asr_unfixed_token_num = int(asr_unfixed_token_num)
        log(
            "[runtime][asr] Qwen3-ASR streaming ready: "
            f"language={self.asr_language} chunk={self.asr_chunk_size_sec:g}s"
        )
        self.dialogue = dialogue
        self.tts = tts
        self.robot_state = robot_state or RobotState()
        self.turn_generator = turn_generator
        self.neck_sender = neck_sender
        self.motion_sync_offset_ms = int(motion_sync_offset_ms)
        self.motion_send_to_motor = bool(motion_send_to_motor)
        self._send_robot_audio: Any | None = None
        self._output_task: asyncio.Task[None] | None = None
        self._session = SessionContext(session_id="qwen_streaming_dialogue")
        self._turn_number = 0
        self._state: Any | None = None
        self._previous_text = ""
        self._speech_started_at: float | None = None
        self._speech_ended_at: float | None = None
        self._pre_roll: deque[bytes] = deque(maxlen=PRE_ROLL_FRAMES)
        self._expected_robot_stream_id: str | None = None
        self._playback_started_future: asyncio.Future[None] | None = None
        self._motion_send_tasks: set[asyncio.Task[None]] = set()
        self._connected = False

    def connection_opened(self, send_robot_audio: Any) -> None:
        if self._connected:
            raise RuntimeError("only one microphone connection is supported")
        self._connected = True
        self._send_robot_audio = send_robot_audio
        self._session = SessionContext(session_id="qwen_streaming_dialogue")
        self._turn_number = 0
        self._expected_robot_stream_id = None
        self._playback_started_future = None
        self._reset_stream()
        log("waiting for speech")

    async def connection_closed(self) -> None:
        self._finish_speech(request_dialogue=False)
        task = self._output_task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._output_task = None
        future, self._playback_started_future = self._playback_started_future, None
        if future is not None and not future.done():
            future.cancel()
        self._expected_robot_stream_id = None
        self._send_robot_audio = None
        self.vad.reset()
        self._pre_roll.clear()
        self._connected = False

    def audio_stream_started(self) -> None:
        self._finish_speech(request_dialogue=False)
        self.vad.reset()
        self._pre_roll.clear()

    def robot_audio_stream_started(self, stream_id: str) -> None:
        self._expected_robot_stream_id = stream_id

    def robot_audio_send_started(self) -> None:
        log("audio send_start")

    def robot_playback_started(self, stream_id: str) -> None:
        if stream_id != self._expected_robot_stream_id:
            log(
                "[SERVER] ignored playback_started for unexpected "
                f"stream_id={stream_id} mono={time.monotonic():.6f}"
            )
            return
        _server_log(f"playback_started stream_id={stream_id}")
        future = self._playback_started_future
        if future is not None and not future.done():
            future.set_result(None)

    def audio_stream_ended(self) -> None:
        completed = self.vad.end_stream()
        if completed is not None:
            self._mark_speech_end()
        self._finish_speech(request_dialogue=completed is not None)
        self._pre_roll.clear()

    def push_audio_frame(self, pcm_frame: bytes) -> None:
        if self._output_task is not None and not self._output_task.done():
            return

        if self._state is None:
            self._pre_roll.append(pcm_frame)

        was_in_speech = self.vad.in_speech
        completed = self.vad.push(pcm_frame)

        if not was_in_speech and self.vad.in_speech:
            self._start_speech()
            self._push_pcm(b"".join(self._pre_roll))
            self._pre_roll.clear()
        elif self._state is not None:
            self._push_pcm(pcm_frame)

        if completed is not None:
            self._mark_speech_end()
            self._finish_speech(request_dialogue=True)
            self._pre_roll.clear()

    def _start_speech(self) -> None:
        self._state = self.model.init_streaming_state(
            language=self.asr_language,
            chunk_size_sec=self.asr_chunk_size_sec,
            unfixed_chunk_num=self.asr_unfixed_chunk_num,
            unfixed_token_num=self.asr_unfixed_token_num,
        )
        self._previous_text = ""
        self._speech_started_at = time.perf_counter()
        self._speech_ended_at = None
        log("speech_start")
        _server_log("speech_start")

    def _push_pcm(self, pcm: bytes) -> None:
        if self._state is None or not pcm:
            return
        waveform = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        self.model.streaming_transcribe(waveform, self._state)
        if self._state.text != self._previous_text:
            self._previous_text = self._state.text

    def _mark_speech_end(self) -> None:
        self._speech_ended_at = time.perf_counter()
        speech_duration = (
            self._speech_ended_at - self._speech_started_at
            if self._speech_started_at is not None
            else 0.0
        )
        log(f"speech_end: speech_duration={speech_duration:.3f}s")
        _server_log(f"speech_end duration={speech_duration:.3f}s")

    def _finish_speech(self, *, request_dialogue: bool) -> None:
        if self._state is None:
            return
        speech_end = self._speech_ended_at or time.perf_counter()
        speech_duration = (
            speech_end - self._speech_started_at
            if self._speech_started_at is not None
            else 0.0
        )
        self.model.finish_streaming_transcribe(self._state)
        asr_final_at = time.perf_counter()
        final_text = self._state.text.strip()
        log(
            f"ASR FINAL: {final_text} "
            f"(speech_duration={speech_duration:.3f}s, "
            f"asr_finalize_latency={asr_final_at - speech_end:.3f}s)"
        )
        self._state = None
        self._previous_text = ""
        self._speech_started_at = None
        self._speech_ended_at = None
        if request_dialogue and final_text:
            _server_log("dialogue_start")
            assistant_text = self._reply(final_text)
            if assistant_text is not None:
                self._schedule_tts(assistant_text)

    def _reply(self, user_text: str) -> str | None:
        log(f"USER: {user_text}")
        request = DialogueRequest(
            user_text=user_text,
            history=tuple(self._session.recent_turns),
        )
        try:
            for _ in self.dialogue.reply_stream(request):
                pass
        except DialogueError as exc:
            log(f"[deepseek] error: {exc}")
            return None

        assistant_text = self.dialogue.response_text
        dialogue_status = str(self.dialogue.metadata.get("status", "complete"))
        self._turn_number += 1
        self._session.add_turn(
            TurnSummary(
                turn_id=f"turn_{self._turn_number}",
                user_text=user_text,
                robot_text=assistant_text,
                status=dialogue_status,
            )
        )
        if dialogue_status != "complete":
            return None
        return assistant_text

    def _schedule_tts(self, assistant_text: str) -> None:
        if self._send_robot_audio is None:
            log("[tts] robot audio sender is unavailable")
            return
        if self._output_task is not None and not self._output_task.done():
            log("[tts] output already in progress; skipping duplicate")
            return
        task = asyncio.create_task(self._synthesize_and_send(assistant_text))
        self._output_task = task
        task.add_done_callback(self._output_task_done)

    def _output_task_done(self, task: asyncio.Task[None]) -> None:
        if self._output_task is task:
            self._output_task = None
        _server_log("turn_complete")

    async def _generate_motion(
        self, assistant_text: str, speech: RobotSpeech
    ) -> Any | None:
        if self.turn_generator is None:
            return None
        if self.motion_send_to_motor:
            if self.neck_sender is None:
                log("[MOTION] skipped: motor sender is unavailable")
                return None
            if not self.robot_state.motor_available:
                log("[MOTION] skipped: motor unavailable")
                return None
            if not self.robot_state.head_rpy_valid:
                log("[MOTION] skipped: head_rpy invalid")
                return None
            if self.robot_state.motion_executing:
                log("[MOTION] skipped: a neck trajectory is already executing")
                return None

        turn = TurnContext(
            turn_id=f"qwen_turn_{self._turn_number}",
            robot_text=assistant_text,
            robot_speech=speech,
            status="outputting",
        )
        request = MotionRequest.snapshot(turn, self._session, self.robot_state)
        try:
            generation = (
                self.turn_generator.generate
                if self.motion_send_to_motor
                else self.turn_generator.generate_relative
            )
            return await asyncio.to_thread(generation, request, turn.turn_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log(f"[MOTION] skipped: {type(exc).__name__}: {exc}")
            return None

    async def _send_motion(self, document: dict) -> None:
        log(f"[MOTION] send_start mono={time.monotonic():.6f}")
        try:
            await asyncio.to_thread(self.neck_sender.send, document)
        except Exception as exc:
            log(
                "[MOTION] skipped: motor send failed: "
                f"{type(exc).__name__}: {exc}"
            )
        finally:
            log(f"[MOTION] send_return mono={time.monotonic():.6f}")

    def _start_motion_send(self, document: dict) -> None:
        task = asyncio.create_task(self._send_motion(document))
        self._motion_send_tasks.add(task)
        task.add_done_callback(self._motion_send_tasks.discard)

    async def _send_audio_and_motion(
        self, speech: RobotSpeech, generated: Any | None
    ) -> None:
        sender = self._send_robot_audio
        if sender is None:
            return
        should_send_motion = (
            generated is not None
            and generated.document is not None
            and self.neck_sender is not None
        )
        playback_future: asyncio.Future[None] | None = None
        if should_send_motion:
            playback_future = asyncio.get_running_loop().create_future()
            self._playback_started_future = playback_future

        audio_task = asyncio.create_task(sender(speech))
        try:
            if should_send_motion and playback_future is not None:
                try:
                    await asyncio.wait_for(
                        asyncio.shield(playback_future),
                        timeout=PLAYBACK_START_TIMEOUT_SEC,
                    )
                except asyncio.TimeoutError:
                    log("[MOTION] skipped: playback_started timeout")
                else:
                    if self.motion_sync_offset_ms > 0:
                        await asyncio.sleep(self.motion_sync_offset_ms / 1000.0)
                    self._start_motion_send(generated.document)
            await audio_task
            log("audio send_end")
        except asyncio.CancelledError:
            audio_task.cancel()
            raise
        finally:
            if self._playback_started_future is playback_future:
                self._playback_started_future = None
            self._expected_robot_stream_id = None

    async def _synthesize_and_send(self, assistant_text: str) -> None:
        try:
            _server_log("tts_start")
            speech = await self.tts.synthesize(assistant_text)
            _server_log(
                f"tts_ready duration={speech.duration_sec:.3f}s"
            )
            generated = await self._generate_motion(assistant_text, speech)
            await self._send_audio_and_motion(speech, generated)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log(f"[tts] error: {exc}")

    def _reset_stream(self) -> None:
        self._finish_speech(request_dialogue=False)
        self.vad.reset()
        self._pre_roll.clear()


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"config section {name!r} must be a mapping")
    return value


def _required(section: dict[str, Any], name: str, section_name: str) -> Any:
    if name not in section:
        raise ValueError(f"config field {section_name}.{name} is required")
    return section[name]


def _path(config_path: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def build_motion_components(
    config: dict[str, Any], config_path: Path
) -> tuple[
    RobotState,
    TurnGenerator | None,
    NeckClient | None,
    MotorFeedbackMonitor | None,
    int,
    bool,
]:
    """Build the shared production Motion V2 and optional Motor components."""
    dialogue_config = _section(config, "dialogue")
    motion_config = _section(config, "motion")
    motor_config = _section(config, "motor")
    robot_state = RobotState()
    output_dir = _path(
        config_path, str(motion_config.get("generated_dir", "generated"))
    )

    turn_generator: TurnGenerator | None = None
    if bool(motion_config.get("enabled", False)):
        backend_name = str(motion_config.get("backend", "deepseek"))
        if backend_name == "deepseek":
            from .inference.deepseek_motion import DeepSeekMotionBackend

            backend = DeepSeekMotionBackend(
                model=str(
                    motion_config.get(
                        "deepseek_model",
                        dialogue_config.get("model", "deepseek-flash"),
                    )
                ),
                base_url=str(
                    motion_config.get(
                        "deepseek_base_url",
                        dialogue_config.get("base_url", "https://api.deepseek.com"),
                    )
                ),
                timeout_sec=float(motion_config.get("deepseek_timeout_sec", 15.0)),
                temperature=float(motion_config.get("deepseek_temperature", 0.2)),
                max_output_tokens=int(
                    motion_config.get("deepseek_max_output_tokens", 1024)
                ),
                output_dir=output_dir,
            )
            model_label = f"deepseek:{backend.model}"
        elif backend_name == "baseline":
            from .inference.baseline_v1 import BaselineV1Backend

            backend = BaselineV1Backend(
                model_path=_path(
                    config_path, _required(motion_config, "model_path", "motion")
                ),
                vocab_path=_path(
                    config_path, _required(motion_config, "vocab_path", "motion")
                ),
                device=str(motion_config.get("device", "auto")),
            )
            model_label = str(backend.model_path)
        else:
            raise ValueError("motion.backend must be deepseek or baseline")
        turn_generator = TurnGenerator(
            backend=backend,
            processor=MotionProcessor(),
            output_dir=output_dir,
            model_label=model_label,
        )

    send_to_motor = bool(
        turn_generator is not None
        and motion_config.get("send_to_motor", False)
        and motor_config.get("send_enabled", False)
    )
    neck_sender = None
    if send_to_motor:
        neck_sender = NeckClient(
            socket_path=str(motor_config.get("socket_path", "/tmp/neck_model.sock")),
            mock=bool(motor_config.get("mock", False)),
            measurement_socket_path=str(
                motor_config.get("measurement_socket", "/tmp/neck_measurement.sock")
            ),
        )

    monitor = None
    if send_to_motor and bool(motor_config.get("feedback_enabled", False)):
        monitor = MotorFeedbackMonitor(
            robot_state=robot_state,
            socket_path=str(
                motor_config.get("feedback_socket", "/tmp/neck_feedback.sock")
            ),
            stale_sec=float(motor_config.get("feedback_stale_sec", 0.2)),
        )
    return (
        robot_state,
        turn_generator,
        neck_sender,
        monitor,
        int(motion_config.get("sync_offset_ms", 0)),
        send_to_motor,
    )


def build_production_runtime(
    config: dict[str, Any],
    config_path: Path,
    *,
    debug: bool = False,
) -> tuple[QwenStreamingRuntime, MotorFeedbackMonitor | None, str, int]:
    """Construct the one supported production Runtime."""
    audio = _section(config, "audio")
    vad_config = _section(config, "vad")
    asr_config = _section(config, "asr")
    dialogue_config = _section(config, "dialogue")
    tts_config = _section(config, "tts")

    if vad_config.get("backend") != "silero":
        raise ValueError("vad.backend must be silero")
    if asr_config.get("backend") != "qwen3_streaming":
        raise ValueError("asr.backend must be qwen3_streaming")
    if dialogue_config.get("backend") != "deepseek":
        raise ValueError("dialogue.backend must be deepseek")
    if tts_config.get("backend") != "doubao":
        raise ValueError("tts.backend must be doubao")

    dialogue = DeepSeekDialogue(
        model=str(_required(dialogue_config, "model", "dialogue")),
        base_url=str(_required(dialogue_config, "base_url", "dialogue")),
        timeout_sec=float(dialogue_config.get("timeout_sec", 30.0)),
        max_tokens=int(dialogue_config.get("max_output_tokens", 4096)),
        temperature=float(dialogue_config.get("temperature", 0.7)),
        system_prompt=SYSTEM_PROMPT,
        debug=debug,
    )
    tts: TTS = DoubaoTTS()
    (
        robot_state,
        turn_generator,
        neck_sender,
        monitor,
        motion_sync_offset_ms,
        motion_send_to_motor,
    ) = build_motion_components(config, config_path)

    runtime = QwenStreamingRuntime(
        model_path=_path(
            config_path, _required(asr_config, "model_path", "asr")
        ),
        vad_model_path=_path(
            config_path, _required(vad_config, "model_path", "vad")
        ),
        dialogue=dialogue,
        tts=tts,
        robot_state=robot_state,
        turn_generator=turn_generator,
        neck_sender=neck_sender,
        motion_sync_offset_ms=motion_sync_offset_ms,
        motion_send_to_motor=motion_send_to_motor,
        vad_threshold=float(vad_config.get("threshold", 0.5)),
        vad_min_speech_ms=int(vad_config.get("min_speech_ms", 250)),
        vad_min_silence_ms=int(vad_config.get("min_silence_ms", 500)),
        asr_language=str(asr_config.get("language", "Chinese")),
        asr_chunk_size_sec=float(asr_config.get("chunk_size_sec", 1.0)),
        asr_unfixed_chunk_num=int(asr_config.get("unfixed_chunk_num", 4)),
        asr_unfixed_token_num=int(asr_config.get("unfixed_token_num", 5)),
        asr_gpu_memory_utilization=float(
            asr_config.get("gpu_memory_utilization", 0.7)
        ),
        asr_max_inference_batch_size=int(
            asr_config.get("max_inference_batch_size", 1)
        ),
        asr_max_new_tokens=int(asr_config.get("max_new_tokens", 4096)),
    )
    log(f"[runtime][dialogue] DeepSeek ready: model={dialogue.model}")
    log(f"[runtime][tts] Doubao ready: speaker={tts.speaker}")
    if turn_generator is None:
        log("[runtime][motion] disabled")
    else:
        backend = turn_generator.backend
        if backend.__class__.__name__ == "DeepSeekMotionBackend":
            log(
                "[runtime][motion] Motion Planner V2 ready: "
                f"model={backend.model}"
            )
        else:
            log(f"[runtime][motion] {backend.__class__.__name__} ready")
        optimizer: TrajectoryOptimizer = turn_generator.processor.optimizer
        log(
            "[runtime][motion] Trajectory Optimizer ready: "
            f"{optimizer.__class__.__name__}"
        )
    if motion_send_to_motor:
        log("[runtime][motor] enabled; waiting for valid feedback pose")
    else:
        log(
            "[runtime][motor] unavailable: "
            "disabled by motion.send_to_motor=false"
        )
    return runtime, monitor, str(audio.get("host", "0.0.0.0")), int(
        audio.get("port", 8765)
    )


async def serve_production_runtime(
    runtime: QwenStreamingRuntime,
    monitor: MotorFeedbackMonitor | None,
    host: str,
    port: int,
) -> None:
    """Serve the production Audio WebSocket endpoint until cancelled."""
    server = AudioWebSocketServer(runtime, host=host, port=port)
    if monitor is not None:
        monitor.start()
    try:
        await server.serve_forever()
    finally:
        if monitor is not None:
            await monitor.stop()
