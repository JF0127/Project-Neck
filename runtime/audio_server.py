"""Frozen bidirectional PCM WebSocket server for Algorithm Runtime V1."""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .runtime import AlgorithmRuntime
from .tts import BYTES_PER_FRAME, iter_pcm_frames

SAMPLE_RATE = 16_000
CHANNELS = 1
PCM_FORMAT = "pcm_s16le"
FRAME_DURATION_SEC = 0.020
MAX_STREAM_BYTES = SAMPLE_RATE * 2 * 300  # hard safety cap: five minutes


@dataclass
class IncomingStream:
    stream_id: str
    pcm: bytearray = field(default_factory=bytearray)
    frame_count: int = 0


def _parse_control(message: str) -> dict:
    try:
        value = json.loads(message)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("invalid JSON control message") from exc
    if not isinstance(value, dict):
        raise ValueError("control message must be an object")
    return value


class AudioWebSocketServer:
    def __init__(self, runtime: AlgorithmRuntime, host: str = "0.0.0.0", port: int = 8765):
        self.runtime = runtime
        self.host = host
        self.port = port
        self.connection_count = 0

    async def _send_robot_audio(self, websocket: Any, result) -> None:
        # With real Neck execution, align robot audio to the first speaking frame.
        # Mock mode skips this wait so pure-software tests remain fast.
        if not self.runtime.neck.mock:
            delay = result.motion.speaking_start_frame / result.motion.document["fps"]
            if delay > 0:
                print(f"[runtime][audio] waiting {delay:.2f}s for speaking motion boundary")
                await asyncio.sleep(delay)

        stream_id = f"robot_{uuid.uuid4().hex}"
        self.runtime.record_robot_audio_start(result.experiment_turn_id, stream_id)
        await websocket.send(json.dumps({
            "type": "stream_start",
            "stream_id": stream_id,
            "source": "robot",
            "sample_rate": SAMPLE_RATE,
            "channels": CHANNELS,
            "format": PCM_FORMAT,
        }, separators=(",", ":")))

        frame_count = 0
        next_send = time.monotonic()
        for frame in iter_pcm_frames(result.robot_audio.pcm_s16le):
            if len(frame) != BYTES_PER_FRAME:
                raise RuntimeError("outgoing robot PCM frame is not 640 bytes")
            await websocket.send(frame)
            frame_count += 1
            next_send += FRAME_DURATION_SEC
            await asyncio.sleep(max(0.0, next_send - time.monotonic()))

        await websocket.send(json.dumps({
            "type": "stream_end",
            "stream_id": stream_id,
        }, separators=(",", ":")))
        self.runtime.record_robot_audio_end(result.experiment_turn_id, stream_id)
        print(
            f"[runtime][audio] robot stream complete: frames={frame_count}, "
            f"bytes={frame_count * BYTES_PER_FRAME}"
        )

    async def handle_connection(self, websocket: Any) -> None:
        self.connection_count += 1
        connection_number = self.connection_count
        peer = getattr(websocket, "remote_address", None)
        stream: IncomingStream | None = None
        print(f"[runtime][audio] connection #{connection_number}: {peer}")
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    if stream is None:
                        raise ValueError("binary PCM received outside a user stream")
                    if len(message) != BYTES_PER_FRAME:
                        raise ValueError(
                            f"input Binary frame must be 640 bytes, got {len(message)}"
                        )
                    if len(stream.pcm) + len(message) > MAX_STREAM_BYTES:
                        raise ValueError("user PCM stream exceeds five-minute safety limit")
                    stream.pcm.extend(message)
                    stream.frame_count += 1
                    continue

                control = _parse_control(message)
                message_type = control.get("type")
                if message_type == "stream_start":
                    if stream is not None:
                        raise ValueError("stream_start received while a stream is active")
                    if control.get("source") != "user":
                        raise ValueError("incoming stream source must be user")
                    if control.get("sample_rate") != SAMPLE_RATE:
                        raise ValueError("sample_rate must be 16000")
                    if control.get("channels") != CHANNELS:
                        raise ValueError("channels must be 1")
                    if control.get("format") != PCM_FORMAT:
                        raise ValueError("format must be pcm_s16le")
                    stream_id = control.get("stream_id")
                    if not isinstance(stream_id, str) or not stream_id:
                        raise ValueError("stream_id must be a non-empty string")
                    self.runtime.begin_user_stream()
                    stream = IncomingStream(stream_id)
                    print(f"[runtime][audio] user stream_start: {stream_id}")
                elif message_type == "stream_end":
                    if stream is None:
                        raise ValueError("stream_end received without stream_start")
                    if control.get("stream_id") != stream.stream_id:
                        raise ValueError("stream_end stream_id mismatch")
                    completed = stream
                    stream = None
                    print(
                        f"[runtime][audio] user stream_end: {completed.stream_id}, "
                        f"frames={completed.frame_count}, bytes={len(completed.pcm)}"
                    )
                    result = await self.runtime.process_user_stream(
                        bytes(completed.pcm), user_stream_id=completed.stream_id
                    )
                    await self._send_robot_audio(websocket, result)
                    self.runtime.finish_speaking(result.experiment_turn_id)
                else:
                    raise ValueError(f"unsupported control type: {message_type}")
        except Exception as exc:
            self.runtime.fail(exc)
            try:
                await websocket.close(code=1011, reason=str(exc)[:120])
            finally:
                self.runtime.recover()
        finally:
            if stream is not None:
                print(
                    f"[runtime][audio] incomplete stream: {stream.stream_id}, "
                    f"frames={stream.frame_count}"
                )
                if self.runtime.state.value == "RECEIVING_USER":
                    self.runtime.fail(RuntimeError("Audio client disconnected mid-stream"))
                    self.runtime.recover()
            print(f"[runtime][audio] connection #{connection_number} closed")

    async def serve_forever(self) -> None:
        try:
            from websockets.asyncio.server import serve
        except ImportError:
            try:
                from websockets import serve
            except ImportError as exc:
                raise RuntimeError(
                    "websockets is required; install runtime/requirements.txt in the Runtime environment"
                ) from exc

        async with serve(self.handle_connection, self.host, self.port, max_size=64 * 1024):
            print(f"[runtime][audio] listening on ws://{self.host}:{self.port}")
            await asyncio.Future()
