"""Frozen PCM WebSocket transport for the new voice Runtime."""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

from .contracts import RobotSpeech
from .logging_utils import log
from .runtime import Runtime
from .tts import BYTES_PER_FRAME, iter_pcm_frames

SAMPLE_RATE = 16_000
CHANNELS = 1
PCM_FORMAT = "pcm_s16le"


@dataclass
class TransportStream:
    stream_id: str
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
    """Validate Audio transport and pass PCM frames to Runtime unchanged."""

    def __init__(self, runtime: Runtime, host: str = "0.0.0.0", port: int = 8765):
        self.runtime = runtime
        self.host = host
        self.port = port
        self.connection_count = 0

    async def _send_robot_audio(
        self, websocket: Any, speech: RobotSpeech
    ) -> None:
        stream_id = f"robot_{uuid.uuid4().hex}"
        send_started = time.perf_counter()
        self.runtime.robot_audio_send_started()
        await websocket.send(
            json.dumps(
                {
                    "type": "stream_start",
                    "stream_id": stream_id,
                    "source": "robot",
                    "sample_rate": SAMPLE_RATE,
                    "channels": CHANNELS,
                    "format": PCM_FORMAT,
                },
                separators=(",", ":"),
            )
        )

        frame_count = 0
        for frame in iter_pcm_frames(speech.pcm_s16le):
            if len(frame) != BYTES_PER_FRAME:
                raise RuntimeError("outgoing robot PCM frame is not 640 bytes")
            await websocket.send(frame)
            frame_count += 1

        await websocket.send(
            json.dumps(
                {"type": "stream_end", "stream_id": stream_id},
                separators=(",", ":"),
            )
        )
        network_send_sec = time.perf_counter() - send_started
        log(
            f"[runtime][audio] robot stream complete: frames={frame_count}, "
            f"bytes={frame_count * BYTES_PER_FRAME}, "
            f"audio_duration={speech.duration_sec:.3f}s, "
            f"network_send_time={network_send_sec:.3f}s"
        )

    async def handle_connection(self, websocket: Any) -> None:
        self.connection_count += 1
        connection_number = self.connection_count
        peer = getattr(websocket, "remote_address", None)
        stream: TransportStream | None = None
        session_opened = False
        log(f"[runtime][audio] connection #{connection_number}: {peer}")
        try:
            self.runtime.connection_opened(
                lambda speech: self._send_robot_audio(websocket, speech)
            )
            session_opened = True
            async for message in websocket:
                if isinstance(message, bytes):
                    if stream is None:
                        raise ValueError("binary PCM received outside an Audio stream")
                    if len(message) != BYTES_PER_FRAME:
                        raise ValueError(
                            f"input Binary frame must be 640 bytes, got {len(message)}"
                        )
                    stream.frame_count += 1
                    self.runtime.push_audio_frame(message)
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
                    stream = TransportStream(stream_id)
                    self.runtime.audio_stream_started()
                    log(f"[runtime][audio] transport stream_start: {stream_id}")
                elif message_type == "stream_end":
                    if stream is None:
                        raise ValueError("stream_end received without stream_start")
                    if control.get("stream_id") != stream.stream_id:
                        raise ValueError("stream_end stream_id mismatch")
                    completed = stream
                    stream = None
                    self.runtime.audio_stream_ended()
                    log(
                        f"[runtime][audio] transport stream_end: "
                        f"{completed.stream_id}, frames={completed.frame_count}"
                    )
                else:
                    raise ValueError(f"unsupported control type: {message_type}")
        except Exception as exc:
            try:
                await websocket.close(code=1011, reason=str(exc)[:120])
            except Exception:
                pass
        finally:
            if session_opened:
                await self.runtime.connection_closed()
            log(f"[runtime][audio] connection #{connection_number} closed")

    async def serve_forever(self) -> None:
        try:
            from websockets.asyncio.server import serve
        except ImportError:
            try:
                from websockets import serve
            except ImportError as exc:
                raise RuntimeError(
                    "websockets is required; install runtime/requirements.txt"
                ) from exc

        async with serve(self.handle_connection, self.host, self.port, max_size=64 * 1024):
            log(f"[runtime][audio] listening on ws://{self.host}:{self.port}")
            await asyncio.Future()
