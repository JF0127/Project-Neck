#!/usr/bin/env python3
"""Receive Mac PCM and print Silero-gated Qwen3-ASR streaming text."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

AUDIO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = AUDIO_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.__main__ import build_motion_components, load_config  # noqa: E402
from runtime.audio_server import AudioWebSocketServer  # noqa: E402
from runtime.contracts import (  # noqa: E402
    DialogueRequest,
    MotionRequest,
    RobotSpeech,
    RobotState,
    SessionContext,
    TurnContext,
    TurnSummary,
)
from runtime.dialogue import DeepSeekDialogue, DialogueError  # noqa: E402
from runtime.doubao_tts import DoubaoTTS  # noqa: E402
from runtime.inference.artifacts import write_wav_atomic  # noqa: E402
from runtime.logging_utils import log  # noqa: E402
from runtime.tts import EdgeTTS, TTS  # noqa: E402
from runtime.vad import SileroVAD  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "dataset" / "models" / "qwen"
VAD_MODEL_PATH = PROJECT_ROOT / "runtime" / "models" / "silero_vad" / "silero_vad.jit"
RUNTIME_CONFIG_PATH = PROJECT_ROOT / "runtime" / "config.yaml"
ROBOT_WAV_PATH = "/home/jhl/projects/Project-Neck/tmp/robot.wav"
SYSTEM_PROMPT = "你是一个机器人助手，请使用自然、简洁的中文进行对话。"
SAMPLE_RATE = 16_000
PRE_ROLL_FRAMES = 20  # 400 ms, covering Silero's speech confirmation delay.
PLAYBACK_START_TIMEOUT_SEC = 30.0


def _server_log(event: str) -> None:
    log(f"[SERVER] {event} mono={time.monotonic():.6f}")


class QwenStreamingRuntime:
    """Minimal Runtime adapter for AudioWebSocketServer."""

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
    ) -> None:
        from qwen_asr import Qwen3ASRModel

        self.model = Qwen3ASRModel.LLM(
            model=str(model_path.expanduser().resolve()),
            gpu_memory_utilization=0.7,
            max_inference_batch_size=1,
            max_new_tokens=4096,
        )
        self.vad = SileroVAD(
            model_path=vad_model_path,
            threshold=0.5,
            min_speech_ms=250,
            min_silence_ms=500,
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
            language="Chinese",
            chunk_size_sec=1.0,
            unfixed_chunk_num=4,
            unfixed_token_num=5,
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
            await asyncio.to_thread(
                write_wav_atomic,
                ROBOT_WAV_PATH,
                speech.pcm_s16le,
            )
            log(f"[tts] saved: {ROBOT_WAV_PATH}")
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


def build_dialogue(
    config_path: Path,
    *,
    debug: bool,
) -> tuple[DeepSeekDialogue, TTS, dict[str, Any], Path]:
    config, resolved_config_path = load_config(config_path)
    dialogue_config = config.get("dialogue", {})
    if not isinstance(dialogue_config, dict):
        raise ValueError("runtime config section 'dialogue' must be a mapping")
    tts_config = config.get("tts", {})
    if not isinstance(tts_config, dict):
        raise ValueError("runtime config section 'tts' must be a mapping")
    dialogue = DeepSeekDialogue(
        model=str(dialogue_config.get("model", "deepseek-flash")),
        base_url=str(dialogue_config.get("base_url", "https://api.deepseek.com")),
        timeout_sec=float(dialogue_config.get("timeout_sec", 30.0)),
        temperature=float(dialogue_config.get("temperature", 0.7)),
        system_prompt=SYSTEM_PROMPT,
        debug=debug,
    )
    tts_backend = str(tts_config.get("backend", "doubao"))
    if tts_backend == "doubao":
        tts: TTS = DoubaoTTS()
    elif tts_backend == "edge":
        tts = EdgeTTS(
            voice=str(tts_config.get("voice", "zh-CN-XiaoxiaoNeural"))
        )
    else:
        raise ValueError("tts.backend must be doubao or edge")
    return dialogue, tts, config, resolved_config_path


async def serve(
    host: str,
    port: int,
    model: Path,
    vad_model: Path,
    config_path: Path,
    *,
    debug: bool,
) -> None:
    dialogue, tts, config, resolved_config_path = build_dialogue(
        config_path, debug=debug
    )
    (
        robot_state,
        turn_generator,
        neck_sender,
        monitor,
        motion_sync_offset_ms,
        motion_send_to_motor,
    ) = build_motion_components(config, resolved_config_path)
    runtime = QwenStreamingRuntime(
        model,
        vad_model,
        dialogue,
        tts,
        robot_state=robot_state,
        turn_generator=turn_generator,
        neck_sender=neck_sender,
        motion_sync_offset_ms=motion_sync_offset_ms,
        motion_send_to_motor=motion_send_to_motor,
    )
    server = AudioWebSocketServer(runtime, host=host, port=port)
    if monitor is not None:
        monitor.start()
    try:
        await server.serve_forever()
    finally:
        if monitor is not None:
            await monitor.stop()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mac PCM -> Silero VAD -> Qwen3-ASR vLLM streaming"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--vad-model", type=Path, default=VAD_MODEL_PATH)
    parser.add_argument("--config", type=Path, default=RUNTIME_CONFIG_PATH)
    parser.add_argument(
        "--debug-deepseek-events",
        action="store_true",
        help="print every DeepSeek Responses API event type",
    )
    args = parser.parse_args()
    try:
        asyncio.run(
            serve(
                args.host,
                args.port,
                args.model,
                args.vad_model,
                args.config,
                debug=args.debug_deepseek_events,
            )
        )
    except KeyboardInterrupt:
        log("[qwen-streaming] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
