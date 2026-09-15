"""Bidirectional microphone and speaker streaming over one WebSocket."""

import asyncio
import queue
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, List, Optional

from websockets import connect
from websockets.exceptions import ConnectionClosed

from . import config
from .audio_capture import MicrophoneCapture
from .audio_playback import AudioPlayback, PlaybackStats
from .protocol import ProtocolError, parse_control_message, stream_end_message, stream_start_message


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}", flush=True)


@dataclass
class UserStreamResult:
    stream_id: str
    frames_sent: int
    total_bytes: int
    elapsed: float
    dropped_frames: int
    invalid_frames: int
    max_queue_depth: int


@dataclass
class RobotStreamResult:
    stream_id: str
    frames_received: int
    total_bytes: int
    playback: PlaybackStats


@dataclass
class DuplexTurnResult:
    turn_number: int
    user: UserStreamResult
    robot: RobotStreamResult
    connection_maintained: bool


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


async def _send_user_stream(
    websocket: Any,
    duration: Optional[float],
    stop_event: Optional[asyncio.Event] = None,
) -> UserStreamResult:
    stream_id = f"user_{uuid.uuid4().hex}"
    capture = MicrophoneCapture()
    frames_sent = 0
    total_bytes = 0
    max_queue_depth = 0
    started = False
    started_at = time.monotonic()

    try:
        capture.start()
        await websocket.send(stream_start_message(stream_id, "user"))
        started = True
        started_at = time.monotonic()
        deadline = started_at + duration if duration is not None else None

        while (
            (deadline is None or time.monotonic() < deadline)
            and (stop_event is None or not stop_event.is_set())
        ):
            max_queue_depth = max(max_queue_depth, capture.frames.qsize())
            timeout = 0.1
            if deadline is not None:
                timeout = max(0.001, min(timeout, deadline - time.monotonic()))
            try:
                frame = await asyncio.to_thread(capture.read_frame, timeout)
            except queue.Empty:
                continue
            if stop_event is not None and stop_event.is_set():
                break
            if len(frame) != config.BYTES_PER_FRAME:
                raise RuntimeError(f"invalid PCM frame size: {len(frame)} bytes")
            await websocket.send(frame)
            frames_sent += 1
            total_bytes += len(frame)

        capture.stop()
        # Fixed-duration diagnostics preserve all frames captured before their
        # deadline. Natural conversation deliberately discards queued input as
        # soon as the robot starts replying.
        if stop_event is None:
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
        elapsed = time.monotonic() - started_at
        if started:
            try:
                await websocket.send(stream_end_message(stream_id))
            except ConnectionClosed:
                pass

    return UserStreamResult(
        stream_id=stream_id,
        frames_sent=frames_sent,
        total_bytes=total_bytes,
        elapsed=elapsed,
        dropped_frames=capture.dropped_frames,
        invalid_frames=capture.invalid_frames,
        max_queue_depth=max_queue_depth,
    )


async def _receive_robot_stream(
    websocket: Any,
    on_stream_start: Optional[Callable[[], None]] = None,
) -> Optional[RobotStreamResult]:
    playback: Optional[AudioPlayback] = None
    robot_stream_id: Optional[str] = None
    robot_frames: list[bytes] = []
    try:
        async for message in websocket:
            if isinstance(message, bytes):
                if robot_stream_id is None:
                    raise ProtocolError("binary PCM received outside a robot stream")
                if len(message) != config.BYTES_PER_FRAME:
                    raise ProtocolError(
                        f"invalid robot PCM frame: {len(message)} bytes"
                    )
                robot_frames.append(message)
                continue

            control = parse_control_message(message)
            if control["type"] == "stream_start":
                if robot_stream_id is not None:
                    raise ProtocolError(
                        "robot stream_start received while a stream is active"
                    )
                if control.get("source") != "robot":
                    raise ProtocolError("incoming stream source must be robot")
                if control.get("sample_rate") != config.SAMPLE_RATE:
                    raise ProtocolError("robot sample_rate must be 16000")
                if control.get("channels") != config.CHANNELS:
                    raise ProtocolError("robot channels must be 1")
                if control.get("format") != config.PCM_FORMAT:
                    raise ProtocolError("robot format must be pcm_s16le")
                robot_stream_id = control["stream_id"]
                robot_frames = []
                if on_stream_start is not None:
                    on_stream_start()
                _log(f"robot_stream_start: {robot_stream_id}")
                continue

            if robot_stream_id is None or control["stream_id"] != robot_stream_id:
                raise ProtocolError(
                    "robot stream_end does not match the active stream"
                )

            _log(
                f"robot_stream_end: {robot_stream_id}, "
                f"frames={len(robot_frames)}"
            )
            playback = AudioPlayback(
                queue_size=max(1, len(robot_frames)),
                prebuffer_frames=len(robot_frames) + 1,
            )
            await asyncio.to_thread(playback.open)
            for frame in robot_frames:
                playback.enqueue(frame)

            _log(f"playback_start: {robot_stream_id}")
            stats = await asyncio.to_thread(playback.finish)
            _log(f"playback_end: {robot_stream_id}")
            result = RobotStreamResult(
                stream_id=robot_stream_id,
                frames_received=stats.received_frames,
                total_bytes=stats.received_frames * config.BYTES_PER_FRAME,
                playback=stats,
            )
            playback = None
            return result
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

    playback_result: Optional[RobotStreamResult] = None
    async with connect(websocket_url) as websocket:
        receiver_task = asyncio.create_task(_receive_robot_stream(websocket))
        try:
            user = await _send_user_stream(websocket, duration)
            if wait_for_robot:
                try:
                    playback_result = await asyncio.wait_for(receiver_task, timeout=robot_timeout)
                except asyncio.TimeoutError as exc:
                    raise RuntimeError("timed out waiting for robot audio") from exc
        finally:
            if not receiver_task.done():
                receiver_task.cancel()
            await asyncio.gather(receiver_task, return_exceptions=True)

    return StreamResult(
        stream_id=user.stream_id,
        frames_sent=user.frames_sent,
        total_bytes=user.total_bytes,
        elapsed=user.elapsed,
        dropped_frames=user.dropped_frames,
        invalid_frames=user.invalid_frames,
        max_queue_depth=user.max_queue_depth,
        playback=playback_result.playback if playback_result is not None else None,
    )


async def run_conversation(
    websocket_url: str = config.WEBSOCKET_URL,
) -> None:
    """Run natural, repeated half-duplex turns on one WebSocket connection."""
    websocket: Any | None = None
    try:
        async with connect(websocket_url) as connection:
            websocket = connection
            _log(f"connection_established: {websocket_url}")
            turn_number = 0
            while True:
                turn_number += 1
                stop_capture = asyncio.Event()
                user_task = asyncio.create_task(
                    _send_user_stream(
                        websocket,
                        duration=None,
                        stop_event=stop_capture,
                    )
                )
                robot_task = asyncio.create_task(
                    _receive_robot_stream(
                        websocket,
                        on_stream_start=stop_capture.set,
                    )
                )
                _log(f"turn {turn_number}: listening")
                try:
                    user, robot = await asyncio.gather(user_task, robot_task)
                finally:
                    stop_capture.set()
                    for task in (user_task, robot_task):
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(
                        user_task,
                        robot_task,
                        return_exceptions=True,
                    )

                if robot is None:
                    raise RuntimeError("connection closed before robot stream")
                _log(
                    f"turn {turn_number}: complete; "
                    f"user_frames={user.frames_sent}, "
                    f"robot_frames={robot.frames_received}, "
                    f"playback_underruns={robot.playback.underruns}"
                )
                _log("microphone_resume")
    finally:
        if websocket is not None:
            code = getattr(websocket, "close_code", None)
            reason = getattr(websocket, "close_reason", None)
            protocol = getattr(websocket, "protocol", None)
            if reason is None and protocol is not None:
                reason = getattr(protocol, "close_reason", None)
            _log(f"connection_closed: code={code}, reason={reason or 'none'}")


async def run_duplex_test(
    turns: int,
    duration: float,
    websocket_url: str = config.WEBSOCKET_URL,
    robot_timeout: float = 30.0,
) -> List[DuplexTurnResult]:
    if turns < 2:
        raise ValueError("duplex-test requires at least two turns")
    if duration <= 0:
        raise ValueError("duration must be greater than zero")
    if robot_timeout <= 0:
        raise ValueError("robot_timeout must be greater than zero")

    results: List[DuplexTurnResult] = []
    async with connect(websocket_url) as websocket:
        print(f"connection established: {websocket_url}")
        connection_identity = id(websocket)
        for turn_number in range(1, turns + 1):
            user = await _send_user_stream(websocket, duration)
            try:
                robot = await asyncio.wait_for(
                    _receive_robot_stream(websocket), timeout=robot_timeout
                )
            except asyncio.TimeoutError as exc:
                raise RuntimeError(f"turn {turn_number}: timed out waiting for robot audio") from exc
            if robot is None:
                raise RuntimeError(f"turn {turn_number}: connection closed before robot stream")

            maintained = (
                id(websocket) == connection_identity
                and getattr(websocket, "close_code", None) is None
            )
            results.append(
                DuplexTurnResult(
                    turn_number=turn_number,
                    user=user,
                    robot=robot,
                    connection_maintained=maintained,
                )
            )
            print(f"turn {turn_number} complete; connection maintained: {'yes' if maintained else 'no'}")
    return results
