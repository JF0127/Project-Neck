"""Independent XiaoZhi demo: board microphone/cloud, Ubuntu streaming speaker, fixed neck loop.

No Qwen/DeepSeek/Doubao or per-turn motion generation. Motor is explicitly opt-in.
"""
from __future__ import annotations

import asyncio
import ctypes
import ctypes.util
import json
import math
from pathlib import Path

from .contracts import RobotState
from .feedback import MotorFeedbackMonitor
from .neck_client import NeckClient
from .xiaozhi_adapter import XiaoZhiAdapter

TRAJECTORY_ROOT = Path(__file__).parent / "motor/trajectories"
TRAJECTORY_NAMES = ("natural_speaking_01", "natural_speaking_02", "natural_speaking_03")
TRAILING_AUDIO_SEC = 0.35  # BoardBridge text events can overtake up to 4 queued 60-ms audio packets.


class OpusStreamDecoder:
    """One decoder per robot turn; preserve Opus state across packets."""

    def __init__(self, rate: int, channels: int):
        if rate not in (8000, 12000, 16000, 24000, 48000) or channels != 1:
            raise ValueError(f"unsupported Opus format: {rate} Hz, {channels} channels")
        name = ctypes.util.find_library("opus")
        if not name:
            raise RuntimeError("libopus not found")
        self.lib = ctypes.CDLL(name)
        self.lib.opus_decoder_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self.lib.opus_decoder_create.restype = ctypes.c_void_p
        self.lib.opus_decoder_destroy.argtypes = [ctypes.c_void_p]
        self.lib.opus_decode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32,
                                         ctypes.POINTER(ctypes.c_int16), ctypes.c_int, ctypes.c_int]
        self.lib.opus_decode.restype = ctypes.c_int
        error = ctypes.c_int()
        self.handle = self.lib.opus_decoder_create(rate, channels, ctypes.byref(error))
        if not self.handle or error.value:
            if self.handle:
                self.close()
            raise RuntimeError(f"Opus decoder init failed: {error.value}")
        self.rate = rate
        self.pcm = (ctypes.c_int16 * (rate * 120 // 1000))()

    def decode(self, packet: bytes) -> bytes:
        encoded = (ctypes.c_ubyte * len(packet)).from_buffer_copy(packet)
        count = self.lib.opus_decode(self.handle, encoded, len(packet), self.pcm,
                                     len(self.pcm), 0)
        if count < 0:
            raise RuntimeError(f"Opus decode failed: {count}")
        return ctypes.string_at(self.pcm, count * 2)

    def close(self):
        if self.handle:
            self.lib.opus_decoder_destroy(self.handle)
            self.handle = None


class StreamingSpeaker:
    """Decode incoming packets and feed paplay concurrently; never block the TCP reader on playback.

    A bounded queue favors the board's original cloud/speaker path over Ubuntu playback when
    the Ubuntu sink stalls. TTS/playback control events can precede queued audio on the wire.
    """

    def __init__(self, decoder_factory=OpusStreamDecoder, player=None):
        self.decoder_factory = decoder_factory
        self.player = player or self._paplay
        self.turn = None
        self.decoder = None
        self.format = None
        self.queue = None
        self.task = None
        self.end_task = None
        self.closed = set()
        self.last_sequence = None
        self.pending_end = set()
        self.expiry_tasks = {}

    @staticmethod
    async def _paplay(queue, rate):
        process = await asyncio.create_subprocess_exec(
            "paplay", "--raw", "--format=s16le", f"--rate={rate}", "--channels=1",
            "--latency-msec=60", stdin=asyncio.subprocess.PIPE,
        )
        try:
            while True:
                pcm = await queue.get()
                if pcm is None:
                    break
                process.stdin.write(pcm)
                await process.stdin.drain()
            process.stdin.close()
            await process.stdin.wait_closed()
            if await process.wait() != 0:
                raise RuntimeError(f"paplay exited with {process.returncode}")
        finally:
            if process.returncode is None:
                process.terminate()
                await process.wait()

    def _finished(self, task):
        if task.cancelled():
            return
        try:
            task.result()
        except Exception as exc:
            print(f"[XIAOZHI-DEMO] speaker failed: {type(exc).__name__}: {exc}", flush=True)

    def _cancel_end(self):
        if self.end_task is not None:
            self.end_task.cancel()
            self.end_task = None

    def _stop(self):
        self._cancel_end()
        if self.decoder is not None:
            self.decoder.close()
        if self.task is not None:
            self.task.cancel()
        self.decoder = self.queue = self.task = self.format = self.last_sequence = None
        self.turn = None

    def disconnect(self):
        self._stop()
        for task in self.expiry_tasks.values():
            task.cancel()
        self.expiry_tasks.clear()
        self.closed.clear()
        self.pending_end.clear()

    def abort(self, turn, _reason, _timestamp):
        self.closed.add(turn)
        self.pending_end.discard(turn)
        task = self.expiry_tasks.pop(turn, None)
        if task:
            task.cancel()
        if turn == self.turn:
            self._stop()

    def audio(self, turn, sequence, payload, rate, duration, channels):
        if not isinstance(turn, int) or isinstance(turn, bool) or turn <= 0 or turn in self.closed:
            return
        if not (0 < duration <= 120):
            print(f"[XIAOZHI-DEMO] invalid frame duration turn={turn}", flush=True)
            return
        if self.turn != turn:
            if self.turn is not None:
                self.closed.add(self.turn)
            self._stop()
            task = self.expiry_tasks.pop(turn, None)
            if task:
                task.cancel()
            try:
                self.decoder = self.decoder_factory(rate, channels)
            except (ValueError, RuntimeError) as exc:
                print(f"[XIAOZHI-DEMO] cannot decode turn={turn}: {exc}", flush=True)
                self.closed.add(turn)
                return
            self.turn = turn
            self.format = (rate, duration, channels)
            self.queue = asyncio.Queue(maxsize=16)
            self.task = asyncio.create_task(self.player(self.queue, rate))
            self.task.add_done_callback(self._finished)
            print(f"[XIAOZHI-DEMO] streaming turn={turn} rate={rate}", flush=True)
        if self.format != (rate, duration, channels):
            print(f"[XIAOZHI-DEMO] format changed turn={turn}; stopping playback", flush=True)
            self.abort(turn, "format_changed", None)
            return
        self._cancel_end()  # reset trailing flush if playback_end already arrived
        if turn in self.pending_end:
            self.end_task = asyncio.create_task(self._end_after_quiet(turn))
        if self.queue.full() or self.task.done():
            print(f"[XIAOZHI-DEMO] speaker slow/unavailable; dropped packet turn={turn}", flush=True)
            return
        if self.last_sequence is not None and sequence != ((self.last_sequence + 1) & 0xffffffff):
            print(f"[XIAOZHI-DEMO] audio sequence gap turn={turn}: {self.last_sequence}->{sequence}", flush=True)
        self.last_sequence = sequence
        try:
            self.queue.put_nowait(self.decoder.decode(payload))
        except RuntimeError as exc:
            print(f"[XIAOZHI-DEMO] dropped invalid Opus packet turn={turn}: {exc}", flush=True)

    def end(self, turn, _timestamp):
        if not isinstance(turn, int) or turn <= 0 or turn in self.closed:
            return
        self.pending_end.add(turn)
        if turn == self.turn:
            self._cancel_end()
            self.end_task = asyncio.create_task(self._end_after_quiet(turn))
        elif turn not in self.expiry_tasks:
            # The high-priority end event can arrive before queued audio packets.
            self.expiry_tasks[turn] = asyncio.create_task(self._expire_unheard(turn))

    async def _expire_unheard(self, turn):
        await asyncio.sleep(TRAILING_AUDIO_SEC)
        self.expiry_tasks.pop(turn, None)
        if turn != self.turn and turn in self.pending_end:
            self.pending_end.discard(turn)
            self.closed.add(turn)

    async def _end_after_quiet(self, turn):
        await asyncio.sleep(TRAILING_AUDIO_SEC)
        if turn != self.turn:
            return
        self.closed.add(turn)
        self.pending_end.discard(turn)
        self.decoder.close()
        self.decoder = None
        # The player drains queued PCM before closing its stream.
        if not self.task.done():
            await self.queue.put(None)
        self.turn = None
        self.end_task = None


class NaturalTrajectoryLoop:
    """Play three pre-existing Motor JSONs in order, independently of audio/turn state."""

    def __init__(self, *, enabled=False, motor_config=None, root=TRAJECTORY_ROOT):
        self.documents = []
        for name in TRAJECTORY_NAMES:
            document = json.loads((root / f"{name}.json").read_text(encoding="utf-8"))
            NeckClient.validate(document)
            if (document["name"] != name or any(s != "speaking" for s in document["states"]) or not all(
                abs(v) < 1e-8 for frame in (document["trajectory"][0], document["trajectory"][-1]) for v in frame
            )):
                raise ValueError(f"{name} must be a zero-ended speaking trajectory")
            if not all(math.isfinite(v) for frame in document["trajectory"] for v in frame):
                raise ValueError(f"{name} contains non-finite positions")
            self.documents.append(document)
        self.enabled = enabled
        self.state = RobotState()
        motor = motor_config or {}
        self.feedback = (MotorFeedbackMonitor(self.state, socket_path=motor.get("feedback_socket", "/tmp/neck_feedback.sock"))
                         if enabled else None)
        self.neck = (NeckClient(socket_path=motor.get("socket_path", "/tmp/neck_model.sock"))
                     if enabled else None)

    async def run(self):
        if self.feedback:
            self.feedback.start()
        print(f"[XIAOZHI-DEMO] natural 01 -> 02 -> 03 loop; Motor {'ON' if self.enabled else 'OFF'}", flush=True)
        confirmed_neutral = False
        try:
            while True:
                for document in self.documents:
                    if not self.enabled:
                        print(f"[XIAOZHI-DEMO] preview {document['name']} (no Motor send)", flush=True)
                        await asyncio.sleep(len(document["trajectory"]) / 30)
                        continue
                    # Cold start requires a valid, manually neutralized pose; after an
                    # observed full execution Motor may stop sending position frames.
                    while True:
                        fresh = (self.feedback.connected and self.feedback.message_age_sec() is not None
                                 and self.feedback.message_age_sec() <= self.feedback.stale_sec)
                        if not fresh:
                            confirmed_neutral = False
                        valid = self.state.head_rpy_valid and self.state.motor_available
                        neutral = valid and max(abs(math.degrees(v)) for v in self.state.head_rpy) <= 1.0
                        if valid and not neutral:
                            confirmed_neutral = False
                        if fresh and not self.state.motion_executing and (neutral or (confirmed_neutral and not valid)):
                            break
                        await asyncio.sleep(0.2)
                    try:
                        await asyncio.to_thread(self.neck.send, document)
                    except (OSError, ValueError) as exc:
                        print(f"[XIAOZHI-DEMO] Motor send failed; stopping neck loop: {exc}", flush=True)
                        return
                    # Socket EOF is not an execution ACK: wait for feedback start and finish.
                    deadline = asyncio.get_running_loop().time() + 1.0
                    while self.feedback.connected and not self.state.motion_executing and asyncio.get_running_loop().time() < deadline:
                        await asyncio.sleep(1 / 30)
                    if not self.feedback.connected or not self.state.motion_executing:
                        print("[XIAOZHI-DEMO] execution unconfirmed; stopping neck loop", flush=True)
                        return
                    observed_start = asyncio.get_running_loop().time()
                    while self.feedback.connected and self.state.motion_executing:
                        await asyncio.sleep(1 / 30)
                    elapsed = asyncio.get_running_loop().time() - observed_start
                    if self.feedback.connected and elapsed + 1.2 < len(document["trajectory"]) / 30:
                        print("[XIAOZHI-DEMO] trajectory stopped early; stopping neck loop", flush=True)
                        return
                    confirmed_neutral = self.feedback.connected
                    if not confirmed_neutral:
                        print("[XIAOZHI-DEMO] feedback lost; loop paused", flush=True)
        finally:
            if self.feedback:
                await self.feedback.stop()


class XiaoZhiDemoRuntime:
    def __init__(self, config, *, motor=False, speaker=None, trajectories=None):
        motor_config = config.get("motor", {})
        if motor and not (motor_config.get("send_enabled") and motor_config.get("feedback_enabled")):
            raise ValueError("--motor requires motor.send_enabled and motor.feedback_enabled")
        self.speaker = speaker or StreamingSpeaker()
        self.trajectories = trajectories or NaturalTrajectoryLoop(enabled=motor, motor_config=motor_config)
        self.adapter = XiaoZhiAdapter(
            lambda _state: None, on_robot_audio=self.speaker.audio,
            on_robot_playback_end=self.speaker.end,
            on_robot_playback_abort=self.speaker.abort, on_disconnect=self.speaker.disconnect,
        )
        self.config = config

    async def run(self):
        print("[XIAOZHI-DEMO] board Mic + XiaoZhi cloud -> Ubuntu streaming paplay; no DeepSeek or turn artifacts", flush=True)
        tasks = [asyncio.create_task(self.adapter.serve(self.config["xiaozhi"]["host"], self.config["xiaozhi"]["port"])),
                 asyncio.create_task(self.trajectories.run())]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            for task in done:
                await task
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self.speaker.disconnect()
