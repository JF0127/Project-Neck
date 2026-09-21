"""Stage 3 single-session voice Runtime orchestration."""
from __future__ import annotations

import asyncio
from enum import Enum
import time
from typing import Awaitable, Callable, Protocol
import uuid

from .contracts import (
    DialogueRequest,
    MotionRequest,
    RobotSpeech,
    RobotState,
    SessionContext,
    TurnContext,
    TurnSummary,
    UserAudio,
)
from .dialogue import DialogueError
from .inference.default_motion import DEFAULT_GENERATION_FALLBACK_TEXT
from .inference.generator import GeneratedTurn, PoseUnavailableError, TurnGenerator

RobotAudioSender = Callable[[RobotSpeech], Awaitable[None]]


class VAD(Protocol):
    in_speech: bool

    def push(self, pcm_frame: bytes) -> UserAudio | None: ...
    def end_stream(self) -> UserAudio | None: ...
    def reset(self) -> None: ...


class RuntimeState(str, Enum):
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    OUTPUTTING = "OUTPUTTING"
    COOLDOWN = "COOLDOWN"


class Runtime:
    """Own one Audio session and run at most one voice Turn at a time."""

    def __init__(
        self,
        vad: VAD,
        asr,
        dialogue,
        tts,
        dialogue_fallback_text: str,
        cooldown_ms: int = 200,
        robot_state: RobotState | None = None,
        turn_generator: TurnGenerator | None = None,
        neck_sender=None,
        motion_sync_offset_ms: int = 0,
        motion_send_to_motor: bool = True,
    ) -> None:
        if cooldown_ms < 0:
            raise ValueError("cooldown_ms must be non-negative")
        if motion_sync_offset_ms < 0:
            raise ValueError("motion_sync_offset_ms must be non-negative")
        if not dialogue_fallback_text.strip():
            raise ValueError("dialogue_fallback_text must not be empty")
        self.vad = vad
        self.asr = asr
        self.dialogue = dialogue
        self.tts = tts
        self.dialogue_fallback_text = dialogue_fallback_text.strip()
        self.cooldown_sec = cooldown_ms / 1000.0
        self.robot_state = robot_state or RobotState()
        self.turn_generator = turn_generator
        self.neck_sender = neck_sender
        self.motion_sync_offset_ms = int(motion_sync_offset_ms)
        self.motion_send_to_motor = bool(motion_send_to_motor)

        self.state = RuntimeState.LISTENING
        self.session: SessionContext | None = None
        self.active_turn: TurnContext | None = None
        self.last_error: Exception | None = None
        self._send_robot_audio: RobotAudioSender | None = None
        self._turn_task: asyncio.Task[None] | None = None
        self._turn_timing: dict[str, float] = {}

    def start_session(self, session_id: str | None = None) -> SessionContext:
        if self.session is not None:
            raise RuntimeError("a Runtime session is already active")
        identifier = session_id or f"session_{uuid.uuid4().hex}"
        if not identifier:
            raise ValueError("session_id must not be empty")
        self.session = SessionContext(session_id=identifier)
        return self.session

    def end_session(self) -> SessionContext:
        if self.session is None:
            raise RuntimeError("no Runtime session is active")
        if self.active_turn is not None:
            raise RuntimeError("cannot end a Runtime session with an active turn")
        completed = self.session
        self.session = None
        return completed

    def start_turn(self, turn_id: str | None = None) -> TurnContext:
        if self.session is None:
            raise RuntimeError("cannot start a turn without an active session")
        if self.active_turn is not None:
            raise RuntimeError("a Runtime turn is already active")
        identifier = turn_id or f"turn_{uuid.uuid4().hex}"
        if not identifier:
            raise ValueError("turn_id must not be empty")
        self.active_turn = TurnContext(turn_id=identifier, status="listening")
        self._turn_timing = {}
        return self.active_turn

    def complete_turn(self, status: str = "complete") -> TurnSummary:
        if self.session is None or self.active_turn is None:
            raise RuntimeError("no Runtime turn is active")
        if not status:
            raise ValueError("turn status must not be empty")

        turn = self.active_turn
        turn.status = status
        user_text = turn.user_speech.text if turn.user_speech is not None else ""
        robot_text = turn.robot_text or (
            turn.robot_speech.text if turn.robot_speech is not None else ""
        )
        summary = TurnSummary(turn.turn_id, user_text, robot_text, status)
        self.session.add_turn(summary)
        self.active_turn = None
        return summary

    def create_motion_request(self) -> MotionRequest:
        """Snapshot the active turn for backend-specific inference."""
        if self.session is None or self.active_turn is None:
            raise RuntimeError("motion inference requires an active session and turn")
        return MotionRequest.snapshot(self.active_turn, self.session, self.robot_state)

    def connection_opened(
        self,
        send_robot_audio: RobotAudioSender,
        session_id: str | None = None,
    ) -> SessionContext:
        session = self.start_session(session_id)
        self._send_robot_audio = send_robot_audio
        self.vad.reset()
        self.state = RuntimeState.LISTENING
        self.last_error = None
        return session

    async def connection_closed(self) -> None:
        task = self._turn_task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._turn_task = None
        if self.active_turn is not None and self.session is not None:
            self.complete_turn("disconnected")
        if self.session is not None:
            self.end_session()
        self._send_robot_audio = None
        self.vad.reset()
        self.state = RuntimeState.LISTENING

    def audio_stream_started(self) -> None:
        if self.state == RuntimeState.LISTENING:
            self.vad.reset()

    def audio_stream_ended(self) -> None:
        if self.state != RuntimeState.LISTENING:
            return
        segment = self.vad.end_stream()
        if segment is not None:
            self._accept_segment(segment)

    def push_audio_frame(self, pcm_frame: bytes) -> None:
        if self.state != RuntimeState.LISTENING:
            return

        was_in_speech = self.vad.in_speech
        segment = self.vad.push(pcm_frame)
        if not was_in_speech and self.vad.in_speech and self.active_turn is None:
            self.start_turn()
        if segment is not None:
            self._accept_segment(segment)

    def _accept_segment(self, user_audio: UserAudio) -> None:
        if self.active_turn is None:
            # Real Silero starts the Turn when speech is confirmed. This fallback
            # keeps an equivalent minimal VAD implementation usable.
            self.start_turn()
        assert self.active_turn is not None
        vad_end_perf = time.perf_counter()
        self._turn_timing["vad_end_perf"] = vad_end_perf
        speech_end_perf = getattr(self.vad, "last_speech_end_perf", None)
        vad_delay = getattr(self.vad, "last_end_delay_sec", None)
        if isinstance(speech_end_perf, float):
            self._turn_timing["speech_end_perf"] = speech_end_perf
        if isinstance(vad_delay, float):
            print(
                f"[runtime][timing] turn={self.active_turn.turn_id} "
                f"vad_end_delay_sec={vad_delay:.6f} "
                f"vad_end_perf={vad_end_perf:.6f}"
            )
        else:
            print(
                f"[runtime][timing] turn={self.active_turn.turn_id} "
                f"vad_end_delay_sec=unavailable vad_end_perf={vad_end_perf:.6f}"
            )
        self.active_turn.user_audio = user_audio
        self.active_turn.status = "processing"
        self.state = RuntimeState.PROCESSING
        self._turn_task = asyncio.create_task(self._process_active_turn())

    async def _process_active_turn(self) -> None:
        turn = self.active_turn
        session = self.session
        sender = self._send_robot_audio
        if turn is None or turn.user_audio is None or session is None or sender is None:
            return

        try:
            stage_started = time.perf_counter()
            try:
                user_speech = await asyncio.to_thread(
                    self.asr.transcribe, turn.user_audio
                )
            finally:
                print(
                    f"[runtime][timing] turn={turn.turn_id} "
                    f"asr_sec={time.perf_counter() - stage_started:.6f}"
                )
            turn.user_speech = user_speech
            if not user_speech.text.strip():
                await self._finish_turn("empty_speech")
                return

            request = DialogueRequest.from_session(user_speech.text, session)
            used_fallback = False
            stage_started = time.perf_counter()
            try:
                robot_text = await asyncio.to_thread(self.dialogue.reply, request)
            except DialogueError as exc:
                self.last_error = exc
                robot_text = self.dialogue_fallback_text
                used_fallback = True
            finally:
                print(
                    f"[runtime][timing] turn={turn.turn_id} "
                    f"deepseek_sec={time.perf_counter() - stage_started:.6f}"
                )
            turn.robot_text = robot_text

            stage_started = time.perf_counter()
            try:
                robot_speech = await self.tts.synthesize(robot_text)
            finally:
                print(
                    f"[runtime][timing] turn={turn.turn_id} "
                    f"tts_sec={time.perf_counter() - stage_started:.6f}"
                )
            if not isinstance(robot_speech, RobotSpeech):
                raise TypeError("TTS must return RobotSpeech")
            turn.robot_speech = robot_speech
            turn.status = "outputting"
            self.state = RuntimeState.OUTPUTTING

            if self.turn_generator is None:
                print(
                    f"[runtime][timing] turn={turn.turn_id} "
                    "motion_sec=0.000000 status=disabled"
                )
                await sender(robot_speech)
            else:
                stage_started = time.perf_counter()
                try:
                    generated, used_fallback = await self._generate_turn(
                        turn, used_fallback
                    )
                finally:
                    print(
                        f"[runtime][timing] turn={turn.turn_id} "
                        f"motion_sec={time.perf_counter() - stage_started:.6f}"
                    )
                turn.robot_speech = generated.speech
                await self._play_turn(generated, sender)
            await self._finish_turn(
                "complete_fallback" if used_fallback else "complete"
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.last_error = exc
            if self.active_turn is turn:
                await self._finish_turn("failed")

    async def _generate_turn(
        self, turn: TurnContext, used_fallback: bool
    ) -> tuple[GeneratedTurn, bool]:
        """Generate files for this turn; fall back to the default reply+shake."""
        assert self.turn_generator is not None
        name = f"turn_{turn.turn_id}"
        request = self.create_motion_request()
        try:
            generation = (
                self.turn_generator.generate
                if self.motion_send_to_motor
                else self.turn_generator.generate_relative
            )
            generated = await asyncio.to_thread(generation, request, name)
            return generated, used_fallback
        except PoseUnavailableError as exc:
            self.last_error = exc
            print(f"[MOTION] skipped: head_rpy invalid: {exc}")
            generated = await asyncio.to_thread(
                self.turn_generator.generate_audio_only, request, name, str(exc)
            )
            return generated, used_fallback
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.last_error = exc
            reason = f"{type(exc).__name__}: {exc}"
            if getattr(self.turn_generator.backend, "failure_mode", "") == "audio_only":
                print(f"[MOTION] skipped: {reason}")
                generated = await asyncio.to_thread(
                    self.turn_generator.generate_audio_only, request, name, reason
                )
                return generated, used_fallback
            print(
                "[runtime][motion] generation failed, default reply + shake: "
                f"{reason}"
            )
            fallback_text = DEFAULT_GENERATION_FALLBACK_TEXT
            fallback_speech = await self.tts.synthesize(fallback_text)
            if not isinstance(fallback_speech, RobotSpeech):
                raise TypeError("TTS must return RobotSpeech")
            turn.robot_text = fallback_text
            turn.robot_speech = fallback_speech
            request = self.create_motion_request()
            generated = await asyncio.to_thread(
                self.turn_generator.generate_fallback,
                request,
                name,
                f"{type(exc).__name__}: {exc}",
            )
            return generated, True

    def robot_audio_send_started(self) -> None:
        """Record the server time immediately before robot stream_start."""
        now = time.perf_counter()
        turn_id = self.active_turn.turn_id if self.active_turn is not None else "unknown"
        speech_end = self._turn_timing.get("speech_end_perf")
        vad_end = self._turn_timing.get("vad_end_perf")
        speech_end_to_send = (
            f"{now - speech_end:.6f}" if speech_end is not None else "unavailable"
        )
        vad_end_to_send = (
            f"{now - vad_end:.6f}" if vad_end is not None else "unavailable"
        )
        print(
            f"[runtime][timing] turn={turn_id} "
            f"robot_audio_send_start_perf={now:.6f} "
            f"speech_end_to_send_sec={speech_end_to_send} "
            f"vad_end_to_send_sec={vad_end_to_send}"
        )

    async def _play_turn(
        self, generated: GeneratedTurn, sender: RobotAudioSender
    ) -> None:
        """Play generated audio while sending the trajectory to the motor."""
        audio_task = asyncio.create_task(sender(generated.speech))
        try:
            if generated.document is not None and self.neck_sender is not None:
                if self.robot_state.motion_executing:
                    print("[MOTION] skipped: a neck trajectory is already executing")
                else:
                    if self.motion_sync_offset_ms > 0:
                        await asyncio.sleep(self.motion_sync_offset_ms / 1000.0)
                    try:
                        print("[MOTION] send_start")
                        await asyncio.to_thread(
                            self.neck_sender.send, generated.document
                        )
                        print(
                            "[runtime][motion] trajectory sent: "
                            f"frames={len(generated.document['trajectory'])} "
                            f"source={generated.source}"
                        )
                    except Exception as exc:
                        self.last_error = exc
                        print(
                            "[runtime][motion][warning] trajectory send failed: "
                            f"{type(exc).__name__}: {exc}"
                        )
            await audio_task
        except asyncio.CancelledError:
            audio_task.cancel()
            raise

    async def _finish_turn(self, status: str) -> None:
        if self.active_turn is not None:
            self.complete_turn(status)
        self.vad.reset()
        self.state = RuntimeState.COOLDOWN
        await asyncio.sleep(self.cooldown_sec)
        if self.session is not None:
            self.state = RuntimeState.LISTENING

    async def wait_for_current_turn(self) -> None:
        """Wait for the current pure-software processing/output task in tests."""
        task = self._turn_task
        if task is not None:
            await asyncio.shield(task)
