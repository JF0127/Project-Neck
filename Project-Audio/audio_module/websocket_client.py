"""Bidirectional microphone and speaker streaming over one WebSocket."""

import asyncio
import queue
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from websockets import connect
from websockets.exceptions import ConnectionClosed

from . import config
from .audio_capture import MicrophoneCapture
from .audio_playback import AudioPlayback, PlaybackStats
from .protocol import ProtocolError, parse_control_message, stream_end_message, stream_start_message


@dataclass
class StreamResult:
    stream_id: str
    frames_sent: int
    total_bytes: int
    elapsed: float
    dropped_frames: int
    invalid_frames: int
    max_queue_depth: int
    playback: Optional[PlaybackStats]


async def _receive_robot_stream(websocket: Any) -> Optional[PlaybackStats]:
    playback: Optional[AudioPlayback] = None
    robot_stream_id: Optional[str] = None
    try:
        async for message in websocket:
            if isinstance(message, bytes):
                if playback is None:
                    raise ProtocolError("binary PCM received outside a robot stream")
                playback.enqueue(message)
                continue

            control = parse_control_message(message)
            if control["type"] == "stream_start":
                if playback is not None:
                    raise ProtocolError("robot stream_start received while a stream is active")
                if control.get("source") != "robot":
                    raise ProtocolError("incoming stream source must be robot")
                if control.get("sample_rate") != config.SAMPLE_RATE:
                    raise ProtocolError("robot sample_rate must be 16000")
                if control.get("channels") != config.CHANNELS:
                    raise ProtocolError("robot channels must be 1")
                if control.get("format") != config.PCM_FORMAT:
                    raise ProtocolError("robot format must be pcm_s16le")
                robot_stream_id = control["stream_id"]
                playback = AudioPlayback()
                await asyncio.to_thread(playback.open)
                print(f"Robot stream started: {robot_stream_id}")
            else:
                if playback is None or control["stream_id"] != robot_stream_id:
                    raise ProtocolError("robot stream_end does not match the active stream")
                stats = await asyncio.to_thread(playback.finish)
                playback = None
                print(f"Robot stream ended: {robot_stream_id}")
                return stats
    finally:
        if playback is not None:
            await asyncio.to_thread(playback.close)
    return None


async def stream_microphone(
    duration: Optional[float],
    websocket_url: str = config.WEBSOCKET_URL,
    wait_for_robot: bool = False,
    robot_timeout: float = 30.0,
) -> StreamResult:
    if duration is not None and duration <= 0:
        raise ValueError("duration must be greater than zero")
    if robot_timeout <= 0:
        raise ValueError("robot_timeout must be greater than zero")

    stream_id = f"user_{uuid.uuid4().hex}"
    capture = MicrophoneCapture()
    frames_sent = 0
    total_bytes = 0
    max_queue_depth = 0
    started = False
    started_at = time.monotonic()
    send_elapsed = 0.0
    playback_stats: Optional[PlaybackStats] = None

    async with connect(websocket_url) as websocket:
        receiver_task = asyncio.create_task(_receive_robot_stream(websocket))
        try:
            try:
                capture.start()
                await websocket.send(stream_start_message(stream_id, "user"))
                started = True
                started_at = time.monotonic()
                deadline = started_at + duration if duration is not None else None

                while deadline is None or time.monotonic() < deadline:
                    max_queue_depth = max(max_queue_depth, capture.frames.qsize())
                    timeout = 0.1
                    if deadline is not None:
                        timeout = max(0.001, min(timeout, deadline - time.monotonic()))
                    try:
                        frame = await asyncio.to_thread(capture.read_frame, timeout)
                    except queue.Empty:
                        continue
                    if len(frame) != config.BYTES_PER_FRAME:
                        raise RuntimeError(f"invalid PCM frame size: {len(frame)} bytes")
                    await websocket.send(frame)
                    frames_sent += 1
                    total_bytes += len(frame)

                capture.stop()
                while True:
                    try:
                        frame = capture.frames.get_nowait()
                    except queue.Empty:
                        break
                    if len(frame) != config.BYTES_PER_FRAME:
                        raise RuntimeError(f"invalid PCM frame size: {len(frame)} bytes")
                    await websocket.send(frame)
                    frames_sent += 1
                    total_bytes += len(frame)
            finally:
                capture.stop()
                send_elapsed = time.monotonic() - started_at
                if started:
                    try:
                        await websocket.send(stream_end_message(stream_id))
                    except ConnectionClosed:
                        pass

            if wait_for_robot:
                try:
                    playback_stats = await asyncio.wait_for(receiver_task, timeout=robot_timeout)
                except asyncio.TimeoutError as exc:
                    raise RuntimeError("timed out waiting for robot audio") from exc
        finally:
            if not receiver_task.done():
                receiver_task.cancel()
            await asyncio.gather(receiver_task, return_exceptions=True)

    return StreamResult(
        stream_id=stream_id,
        frames_sent=frames_sent,
        total_bytes=total_bytes,
        elapsed=send_elapsed,
        dropped_frames=capture.dropped_frames,
        invalid_frames=capture.invalid_frames,
        max_queue_depth=max_queue_depth,
        playback=playback_stats,
    )
