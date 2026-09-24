#!/usr/bin/env python3
"""Independent Doubao V3 WebSocket PCM streaming -> Ubuntu PulseAudio check."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import struct
import sys
import time
import uuid
import wave

import websockets

# Reuse the production credential, speaker, resource ID and audio rate; no production calls.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.doubao_tts import DEFAULT_DOUBAO_SPEAKER, DOUBAO_RESOURCE_ID  # noqa: E402
from runtime.tts import SAMPLE_RATE  # noqa: E402

URL = "wss://openspeech.bytedance.com/api/v3/tts/unidirectional/stream"
TEXT = "你好，我现在正在测试流式语音合成。如果你能够很快听到我的声音，就说明我们的实时语音链路效果不错。"
AUDIO_EVENT = 352
DONE_EVENT = 152


def parse_event(message: bytes) -> tuple[int, bytes]:
    """V3 binary frame: 4-byte header, event, session id, payload length, payload."""
    if not isinstance(message, bytes) or len(message) < 16 or message[0] != 0x11:
        raise ValueError("unexpected V3 WebSocket frame")
    header_len = (message[0] & 0x0F) * 4
    if header_len < 4 or len(message) < header_len + 12 or message[1] & 0x0F != 4:
        raise ValueError("unexpected V3 frame header/flags")
    event = struct.unpack_from(">I", message, header_len)[0]
    sid_len = struct.unpack_from(">I", message, header_len + 4)[0]
    pos = header_len + 8 + sid_len
    if pos + 4 > len(message):
        raise ValueError("truncated V3 frame")
    size = struct.unpack_from(">I", message, pos)[0]
    payload = message[pos + 4:]
    if size != len(payload):
        raise ValueError("invalid V3 payload size")
    if message[1] >> 4 == 15:
        raise RuntimeError(f"Doubao V3 error: {payload[:300]!r}")
    return event, payload


async def run(output: Path | None = None) -> None:
    key = os.getenv("VOLCENGINE_TTS_API_KEY", "").strip()
    if not key:
        raise RuntimeError("VOLCENGINE_TTS_API_KEY is required")
    speaker = os.getenv("VOLCENGINE_TTS_SPEAKER", "").strip() or DEFAULT_DOUBAO_SPEAKER
    headers = {
        "X-Api-Key": key,
        "X-Api-Resource-Id": DOUBAO_RESOURCE_ID,
        "X-Api-Request-Id": str(uuid.uuid4()),
    }
    request = {
        "user": {"uid": "project-neck"},
        "req_params": {
            "text": TEXT,
            "speaker": speaker,
            "audio_params": {"format": "pcm", "sample_rate": SAMPLE_RATE,
                             "speech_rate": 0, "loudness_rate": 0},
            "additions": json.dumps({"disable_markdown_filter": True}, ensure_ascii=False),
        },
    }
    times: dict[str, float] = {}
    origin = time.perf_counter()

    def mark(name: str) -> None:
        times[name] = time.perf_counter()
        print(f"{name}: mono={times[name]:.6f} relative={times[name] - origin:.6f}s", flush=True)

    queue: asyncio.Queue[bytes | None] = asyncio.Queue()
    chunks: list[bytes] = []
    gaps = 0

    async def play() -> None:
        nonlocal gaps
        first = await queue.get()
        if first is None:
            return
        process = await asyncio.create_subprocess_exec(
            "paplay", "--raw", f"--rate={SAMPLE_RATE}", "--channels=1",
            "--format=s16le", "--latency-msec=60",
            stdin=asyncio.subprocess.PIPE,
        )
        written = 0
        try:
            assert process.stdin is not None
            chunk = first
            while chunk is not None:
                process.stdin.write(chunk)
                await process.stdin.drain()
                written += len(chunk)
                if "playback_start" not in times:
                    mark("playback_start")  # first PCM handed to live paplay, not a full-file replay
                else:
                    elapsed = time.perf_counter() - times["playback_start"]
                    if written / (SAMPLE_RATE * 2) + 0.06 < elapsed:
                        gaps += 1  # possible starvation; PulseAudio's actual underruns are not exposed here
                chunk = await queue.get()
            process.stdin.close()
            await process.stdin.wait_closed()
            code = await process.wait()
            if code:
                raise RuntimeError(f"paplay exited with {code}")
            mark("playback_end")
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()

    player = asyncio.create_task(play())
    mark("request_start")
    try:
        async with websockets.connect(URL, additional_headers=headers, open_timeout=15,
                                      max_size=None) as ws:
            mark("websocket_connected")
            data = json.dumps(request, ensure_ascii=False).encode("utf-8")
            # V3 full client request (JSON): header + big-endian size + one complete text request.
            await ws.send(b"\x11\x10\x10\x00" + struct.pack(">I", len(data)) + data)
            mark("request_sent")
            while True:
                event, payload = parse_event(await asyncio.wait_for(ws.recv(), timeout=60))
                if event == AUDIO_EVENT:
                    if not payload or len(payload) % 2:
                        raise ValueError("invalid PCM s16le chunk")
                    if not chunks:
                        mark("first_audio_received")
                    chunks.append(payload)
                    times["last_audio_received"] = time.perf_counter()
                    queue.put_nowait(payload)  # playback task runs while more WS chunks arrive
                elif event == DONE_EVENT:
                    if "last_audio_received" not in times:
                        raise RuntimeError("no PCM received")
                    t = times["last_audio_received"]
                    print(f"last_audio_received: mono={t:.6f} relative={t - origin:.6f}s", flush=True)
                    break
                # 350/351 carry text/phoneme metadata, not audio.
        if not chunks:
            raise RuntimeError("no PCM received")
        queue.put_nowait(None)
        await player
        audio_sec = sum(map(len, chunks)) / (2 * SAMPLE_RATE)
        synthesis_sec = times["last_audio_received"] - times["request_start"]
        print(f"request_to_first_audio_sec={times['first_audio_received']-times['request_start']:.6f}")
        print(f"request_to_playback_start_sec={times['playback_start']-times['request_start']:.6f}")
        print(f"synthesis_sec={synthesis_sec:.6f} audio_duration_sec={audio_sec:.6f} "
              f"RTF={synthesis_sec/audio_sec:.6f}")
        print(f"audio_chunks={len(chunks)} chunk_bytes={[len(c) for c in chunks]} "
              f"total_bytes={sum(map(len, chunks))} potential_starvation_events={gaps}")
        if output:
            def save() -> None:
                output.parent.mkdir(parents=True, exist_ok=True)
                with wave.open(str(output), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(SAMPLE_RATE)
                    for chunk in chunks:
                        wav.writeframesraw(chunk)
            await asyncio.to_thread(save)  # optional diagnostic, after live playback
            print(f"saved={output}")
    finally:
        if not player.done():
            player.cancel()
            await asyncio.gather(player, return_exceptions=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-wav", type=Path, help="optional WAV, written after playback")
    args = parser.parse_args()
    asyncio.run(run(args.save_wav))
