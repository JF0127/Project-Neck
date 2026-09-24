"""Opt-in, Ubuntu-only V1: local microphone, streaming speaker, per-turn trajectory."""
from __future__ import annotations

import asyncio
import queue
from pathlib import Path
import time

from .audio.runtime.audio_capture import MicrophoneCapture
from .audio.runtime.audio_playback import AudioPlayback
from .contracts import RobotState
from .doubao_streaming_v1 import pcm_chunks
from .feedback import MotorFeedbackMonitor
from .inference.deepseek_motion import DeepSeekMotionBackend
from .logging_utils import log
from .neck_client import NeckClient
from .qwen_streaming_runtime import build_production_runtime
from .ubuntu_v1_motion import UbuntuV1Mixer, FPS


class UbuntuV1Runtime:
    def __init__(self, config, config_path: Path, *, send_to_motor: bool = False):
        # Reuse exactly the production VAD/ASR/dialogue construction; ignore its
        # legacy complete-PCM TTS/Motion output path in this opt-in mode.
        self.mixer = UbuntuV1Mixer()
        voice_config = {**config, "motion": {"enabled": False, "send_to_motor": False}}
        self.voice, _, _, _ = build_production_runtime(voice_config, config_path)
        self.voice._schedule_tts = self.schedule_reply
        original_reply = self.voice._reply
        def reply_with_log(user_text):
            log(f"asr_final: {user_text}")
            return original_reply(user_text)
        self.voice._reply = reply_with_log
        motor_config = config.get("motor", {})
        if send_to_motor and not (config.get("motion", {}).get("send_to_motor", False)
                                  and motor_config.get("send_enabled", False)
                                  and motor_config.get("feedback_enabled", False)):
            raise ValueError("--motor requires motion.send_to_motor, motor.send_enabled and motor.feedback_enabled")
        self.send_to_motor = send_to_motor
        motion = config["motion"]
        self.planner = DeepSeekMotionBackend(
            model=str(motion.get("deepseek_model", "deepseek-flash")),
            timeout_sec=float(motion.get("deepseek_timeout_sec", 15.0)),
            temperature=float(motion.get("deepseek_temperature", 0.2)),
            max_output_tokens=int(motion.get("deepseek_max_output_tokens", 1024)),
        )
        self.state = RobotState()
        motor = config.get("motor", {})
        self.feedback = (MotorFeedbackMonitor(self.state, socket_path=motor.get("feedback_socket", "/tmp/neck_feedback.sock"))
                         if send_to_motor else None)
        self.neck = (NeckClient(socket_path=motor.get("socket_path", "/tmp/neck_model.sock"))
                     if send_to_motor else None)
        self.loop = None
        self.playback_active = False
        self.reply_number = 0

    def schedule_reply(self, text: str) -> None:
        # _finish_speech runs in the microphone worker; schedule on the event loop.
        assert self.loop is not None
        self.loop.call_soon_threadsafe(self._start_reply, text)

    def _start_reply(self, text):
        if self.voice._output_task is not None and not self.voice._output_task.done():
            log("[ubuntu-v1] output already active; ignoring duplicate")
            return
        task = asyncio.create_task(self._output(text))
        self.voice._output_task = task
        task.add_done_callback(self.voice._output_task_done)

    async def _build_turn_trajectory(self, reply_text: str, number: int):
        """Text Motion Planner -> Trajectory Generator, before any TTS request."""
        if self.planner is None:
            raise RuntimeError("DeepSeek Motion Planner is unavailable")
        _, plan = await asyncio.to_thread(self.planner.plan_text, reply_text)
        log(f"motion_plan: {plan.to_dict()}")
        duration = max(1.5, len(reply_text) / 5.0)
        base_pose = tuple(self.state.head_rpy) if self.state.head_rpy_valid else (0.0, 0.0, 0.0)
        document = await asyncio.to_thread(
            self.mixer.turn_document, plan, duration, base_pose,
            f"ubuntu_v1_turn_{number:04d}",
        )
        return document, duration

    async def _output(self, reply_text):
        started = time.monotonic()
        self.reply_number += 1
        log(f"deepseek_reply: {reply_text} mono={started:.6f}")
        log("motion_generation_start")
        try:
            document, duration = await self._build_turn_trajectory(
                reply_text, self.reply_number
            )
        except Exception as exc:
            log(f"[ubuntu-v1] motion generation failed: {type(exc).__name__}: {exc}")
            log("turn_complete status=motion_failed")
            return
        ready = time.monotonic()
        log(f"trajectory_ready duration={duration:.3f}s "
            f"frames={len(document['trajectory'])} "
            f"latency_ms={(ready - started) * 1000.0:.1f}")
        requested = time.monotonic()
        log(f"tts_request latency_ms={(requested - ready) * 1000.0:.1f}")

        stream: asyncio.Queue[bytes | None] = asyncio.Queue()
        playback_task: asyncio.Task | None = None
        motion_task: asyncio.Task | None = None
        first_audio: float | None = None
        total_bytes = 0
        status = "ok"
        try:
            async for pcm in pcm_chunks(reply_text):
                if first_audio is None:
                    first_audio = time.monotonic()
                    log(f"tts_first_audio latency_ms={(first_audio - requested) * 1000.0:.1f}")
                    # Single synchronized trigger: speaker playback + Motor trajectory.
                    self.playback_active = True
                    playback_task = asyncio.create_task(self._playback(stream))
                    motion_task = asyncio.create_task(self._execute_motion(document))
                    stream.put_nowait(pcm)
                    trigger = time.monotonic()
                    log(f"playback_and_motion_start mono={trigger:.6f} "
                        f"latency_ms={(trigger - first_audio) * 1000.0:.1f}")
                    await asyncio.sleep(0)
                else:
                    await stream.put(pcm)
                total_bytes += len(pcm)
            if first_audio is None:
                raise RuntimeError("Doubao produced no PCM")
        except Exception as exc:
            status = "tts_failed"
            log(f"[ubuntu-v1] tts failed: {type(exc).__name__}: {exc}")
        finally:
            if playback_task is not None:
                stream.put_nowait(None)
                try:
                    total_bytes = await playback_task
                except Exception as exc:
                    status = "playback_failed"
                    log(f"[ubuntu-v1] playback failed: {type(exc).__name__}: {exc}")
                log(f"tts_playback_end duration={total_bytes / 32000.0:.3f}s")
        if motion_task is not None:
            await motion_task
        self.playback_active = False
        log(f"turn_complete status={status}")

    async def _playback(self, stream: asyncio.Queue) -> int:
        process = await asyncio.create_subprocess_exec(
            "paplay", "--raw", "--format=s16le", "--rate=16000",
            "--channels=1", "--latency-msec=60",
            stdin=asyncio.subprocess.PIPE,
        )
        total = 0
        try:
            while True:
                chunk = await stream.get()
                if chunk is None:
                    break
                if process.stdin is None:
                    raise RuntimeError("paplay stdin is unavailable")
                process.stdin.write(chunk)
                await process.stdin.drain()
                total += len(chunk)
        finally:
            process.stdin.close()
            await process.stdin.wait_closed()
            code = await process.wait()
        if code != 0:
            raise RuntimeError(f"paplay exited with {code}")
        return total

    async def _wait_for_motor_feedback(self) -> None:
        """--motor: wait for one valid pose before ready/listening (no race)."""
        assert self.feedback is not None
        while not (self.feedback.connected and self.state.head_rpy_valid
                   and self.state.motor_available):
            age = self.feedback.message_age_sec()
            rate = self.feedback.observed_rate_hz
            log(
                "[motion-gate] waiting_for_valid_feedback\n"
                f"connected={self.feedback.connected}\n"
                f"motor_available={self.state.motor_available}\n"
                f"head_rpy_valid={self.state.head_rpy_valid}\n"
                f"feedback_age_ms={'n/a' if age is None else f'{age * 1000.0:.1f}'}\n"
                f"feedback_rate_hz={'n/a' if rate is None else f'{rate:.1f}'}\n"
                "hint=start master_stack_test and run NeckPoseSet 0 0 0 0"
            )
            await asyncio.sleep(1.0)
        age = self.feedback.message_age_sec()
        rate = self.feedback.observed_rate_hz
        log("[ubuntu-v1] Motor feedback ready before listening "
            f"feedback_age_ms={'n/a' if age is None else f'{age * 1000.0:.1f}'} "
            f"feedback_rate_hz={'n/a' if rate is None else f'{rate:.1f}'}")

    async def _execute_motion(self, document: dict) -> None:
        if self.neck is None or self.feedback is None:
            log("[motion-gate]\nreason=motor_interface_unavailable\n"
                f"neck={'set' if self.neck is not None else 'none'}\n"
                f"feedback={'set' if self.feedback is not None else 'none'}")
            log("motion_end status=skipped")
            return
        age = self.feedback.message_age_sec()
        rate = self.feedback.observed_rate_hz
        age_text = 'n/a' if age is None else f'{age * 1000.0:.1f}'
        rate_text = 'n/a' if rate is None else f'{rate:.1f}'
        if (not self.feedback.connected or not self.state.head_rpy_valid
                or not self.state.motor_available):
            if not self.feedback.connected:
                reason = "feedback_disconnected"
            elif not self.state.head_rpy_valid:
                reason = "head_rpy_invalid"
            else:
                reason = "motor_unavailable"
            log(
                "[motion-gate]\n"
                f"reason={reason}\n"
                f"connected={self.feedback.connected}\n"
                f"motor_available={self.state.motor_available}\n"
                f"head_rpy_valid={self.state.head_rpy_valid}\n"
                f"feedback_age_ms={age_text}\n"
                f"feedback_rate_hz={rate_text}\n"
                f"stale_sec={self.feedback.stale_sec:g}\n"
                f"motion_executing={self.state.motion_executing}"
            )
            log("motion_end status=skipped")
            return
        if self.state.motion_executing:
            log(
                "[motion-gate]\n"
                "reason=trajectory_already_executing\n"
                f"connected={self.feedback.connected}\n"
                f"motor_available={self.state.motor_available}\n"
                f"head_rpy_valid={self.state.head_rpy_valid}\n"
                f"feedback_age_ms={age_text}\n"
                f"feedback_rate_hz={rate_text}\n"
                f"motion_executing={self.state.motion_executing}"
            )
            log("motion_end status=skipped")
            return
        expected = len(document["trajectory"]) / FPS
        try:
            await asyncio.to_thread(self.neck.send, document)
        except (OSError, ValueError, RuntimeError) as exc:
            log(f"[ubuntu-v1] motion send failed: {type(exc).__name__}: {exc}")
            log("motion_end status=send_failed")
            return
        deadline = time.monotonic() + expected + 5.0
        while (self.feedback.connected and not self.state.motion_executing
               and time.monotonic() < deadline):
            await asyncio.sleep(1 / FPS)
        if not self.state.motion_executing:
            log("motion_end status=unconfirmed")
            return
        while self.feedback.connected and self.state.motion_executing and time.monotonic() < deadline:
            await asyncio.sleep(1 / FPS)
        log("motion_end")

    async def run(self):
        self.loop = asyncio.get_running_loop()
        self.voice.connection_opened(lambda speech: None)
        await asyncio.to_thread(AudioPlayback._set_pulse_sink_port)
        if self.feedback is not None:
            self.feedback.start()
            await self._wait_for_motor_feedback()
        log("[ubuntu-v1] ready: local microphone; per-turn trajectory starts on first audio " +
            ("(Motor enabled)" if self.send_to_motor else "(Motor off)"))
        try:
            while True:
                # Fresh capture per turn; no speaker echo in the VAD/ASR input.
                mic = MicrophoneCapture()
                await asyncio.to_thread(mic.start)
                try:
                    while self.voice._output_task is None:
                        try:
                            frame = await asyncio.to_thread(mic.read_frame, .1)
                        except queue.Empty:
                            continue
                        await asyncio.to_thread(self.voice.push_audio_frame, frame)
                        await asyncio.sleep(0)  # allow scheduled dialogue/TTS to start
                finally:
                    await asyncio.to_thread(mic.stop)
                task = self.voice._output_task
                if task is not None:
                    await asyncio.gather(task, return_exceptions=True)
                    while self.voice._output_task is not None:
                        await asyncio.sleep(.01)
                self.voice.audio_stream_started()  # reset VAD for the next turn
                log("[ubuntu-v1] listening for next turn")
        finally:
            if self.send_to_motor and self.state.motion_executing:
                log("[ubuntu-v1] in-flight turn trajectory continues on the Motor")
            await self.voice.connection_closed()
            if self.feedback is not None:
                await self.feedback.stop()
