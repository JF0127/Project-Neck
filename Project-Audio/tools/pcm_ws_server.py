#!/usr/bin/env python3
"""Minimal Ubuntu WebSocket server for validating incoming PCM streams."""

import argparse
import asyncio
import json
import re
import time
import uuid
import wave
from pathlib import Path
from typing import Any, Dict, Optional

from websockets import serve
from websockets.exceptions import ConnectionClosed

SAMPLE_RATE = 16_000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2
PCM_FORMAT = "pcm_s16le"
BYTES_PER_FRAME = 640
FRAME_DURATION_SECONDS = 0.02


def parse_control(message: str) -> Dict[str, Any]:
    data = json.loads(message)
    if not isinstance(data, dict):
        raise ValueError("control message must be a JSON object")
    return data


def safe_filename(stream_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", stream_id)


def validate_reply_wav(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"reply WAV does not exist: {path}")
    try:
        with wave.open(str(path), "rb") as wav_file:
            if wav_file.getframerate() != SAMPLE_RATE:
                raise ValueError("reply WAV sample rate must be 16000 Hz")
            if wav_file.getnchannels() != CHANNELS:
                raise ValueError("reply WAV must be mono")
            if wav_file.getsampwidth() != SAMPLE_WIDTH_BYTES:
                raise ValueError("reply WAV must use 16-bit PCM")
            if wav_file.getcomptype() != "NONE":
                raise ValueError("reply WAV must be uncompressed PCM")
    except wave.Error as exc:
        raise ValueError(f"invalid reply WAV: {exc}") from exc


async def send_robot_wav(websocket: Any, path: Path) -> None:
    stream_id = f"robot_{uuid.uuid4().hex}"
    start_message = {
        "type": "stream_start",
        "stream_id": stream_id,
        "source": "robot",
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "format": PCM_FORMAT,
    }
    await websocket.send(json.dumps(start_message, separators=(",", ":")))
    print(f"robot stream_start: {stream_id}, WAV={path}")

    frame_count = 0
    next_send = time.monotonic()
    with wave.open(str(path), "rb") as wav_file:
        while True:
            frame = wav_file.readframes(320)
            if not frame:
                break
            if len(frame) < BYTES_PER_FRAME:
                frame += bytes(BYTES_PER_FRAME - len(frame))
            if len(frame) != BYTES_PER_FRAME:
                raise ValueError(f"invalid robot PCM frame size: {len(frame)}")
            await websocket.send(frame)
            frame_count += 1
            next_send += FRAME_DURATION_SECONDS
            await asyncio.sleep(max(0.0, next_send - time.monotonic()))

    await websocket.send(
        json.dumps({"type": "stream_end", "stream_id": stream_id}, separators=(",", ":"))
    )
    print(
        f"robot stream_end: {stream_id}, frames={frame_count}, "
        f"bytes={frame_count * BYTES_PER_FRAME}, duration={frame_count * FRAME_DURATION_SECONDS:.2f}s"
    )


async def handle_connection(
    websocket: Any, save_dir: Optional[Path], reply_wav: Optional[Path]
) -> None:
    stream: Optional[Dict[str, Any]] = None
    wav_file: Optional[wave.Wave_write] = None
    peer = getattr(websocket, "remote_address", None)
    print(f"client connected: {peer}")

    try:
        async for message in websocket:
            if isinstance(message, bytes):
                if stream is None:
                    raise ValueError("binary PCM received before stream_start")
                if len(message) != BYTES_PER_FRAME:
                    raise ValueError(f"invalid PCM frame: {len(message)} bytes, expected 640")
                stream["frames"] += 1
                stream["bytes"] += len(message)
                if wav_file is not None:
                    wav_file.writeframesraw(message)
                continue

            control = parse_control(message)
            message_type = control.get("type")
            if message_type == "stream_start":
                if stream is not None:
                    raise ValueError("stream_start received while another stream is active")
                stream_id = control.get("stream_id")
                if not isinstance(stream_id, str) or not stream_id:
                    raise ValueError("invalid stream_id")
                if control.get("sample_rate") != SAMPLE_RATE:
                    raise ValueError("sample_rate must be 16000")
                if control.get("channels") != CHANNELS:
                    raise ValueError("channels must be 1")
                if control.get("format") != PCM_FORMAT:
                    raise ValueError("format must be pcm_s16le")

                stream = {
                    "id": stream_id,
                    "source": control.get("source"),
                    "frames": 0,
                    "bytes": 0,
                }
                print("stream_start:")
                print(f"  stream_id: {stream_id}")
                print(f"  source: {control.get('source')}")
                print(f"  sample_rate: {control.get('sample_rate')}")
                print(f"  channels: {control.get('channels')}")
                print(f"  format: {control.get('format')}")

                if save_dir is not None:
                    save_dir.mkdir(parents=True, exist_ok=True)
                    output = save_dir / f"received_{safe_filename(stream_id)}.wav"
                    wav_file = wave.open(str(output), "wb")
                    wav_file.setnchannels(CHANNELS)
                    wav_file.setsampwidth(SAMPLE_WIDTH_BYTES)
                    wav_file.setframerate(SAMPLE_RATE)
                    stream["output"] = output

            elif message_type == "stream_end":
                if stream is None:
                    raise ValueError("stream_end received without an active stream")
                if control.get("stream_id") != stream["id"]:
                    raise ValueError("stream_end stream_id does not match stream_start")
                if wav_file is not None:
                    wav_file.close()
                    wav_file = None
                duration = stream["frames"] * FRAME_DURATION_SECONDS
                print("stream_end:")
                print(f"  stream_id: {stream['id']}")
                print(f"  frames: {stream['frames']}")
                print(f"  bytes: {stream['bytes']}")
                print(f"  duration: {duration:.2f} s")
                if "output" in stream:
                    print(f"  saved: {stream['output']}")
                should_reply = reply_wav is not None and stream["source"] == "user"
                stream = None
                if should_reply:
                    await send_robot_wav(websocket, reply_wav)
            else:
                raise ValueError(f"unsupported control type: {message_type}")
    except ConnectionClosed as exc:
        print(f"client disconnected: code={exc.code} reason={exc.reason}")
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"protocol error: {exc}")
        await websocket.close(code=1008, reason=str(exc)[:120])
    finally:
        if wav_file is not None:
            wav_file.close()
        if stream is not None:
            print(
                f"incomplete stream: {stream['id']}, frames={stream['frames']}, "
                f"bytes={stream['bytes']}"
            )
        print(f"connection closed: {peer}")


async def run_server(
    host: str, port: int, save_dir: Optional[Path], reply_wav: Optional[Path]
) -> None:
    async with serve(lambda ws: handle_connection(ws, save_dir, reply_wav), host, port):
        print(f"PCM WebSocket server listening on ws://{host}:{port}")
        if save_dir is not None:
            print(f"Saving WAV files to: {save_dir.resolve()}")
        if reply_wav is not None:
            print(f"Robot reply WAV: {reply_wav.resolve()}")
        await asyncio.Future()


def main() -> None:
    parser = argparse.ArgumentParser(description="PCM WebSocket validation server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--save-dir", type=Path, help="optional directory for received WAV files")
    parser.add_argument("--reply-wav", type=Path, help="WAV file returned as a robot PCM stream")
    args = parser.parse_args()
    if args.reply_wav is not None:
        try:
            validate_reply_wav(args.reply_wav)
        except ValueError as exc:
            parser.error(str(exc))
    try:
        asyncio.run(run_server(args.host, args.port, args.save_dir, args.reply_wav))
    except KeyboardInterrupt:
        print("Server stopped")


if __name__ == "__main__":
    main()
