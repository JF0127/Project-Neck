"""Doubao V3 unidirectional WebSocket: yield PCM chunks as they arrive."""
from __future__ import annotations

import asyncio
import json
import os
import struct
import uuid

import websockets

from .doubao_tts import DEFAULT_DOUBAO_SPEAKER, DOUBAO_RESOURCE_ID
from .tts import SAMPLE_RATE

URL = "wss://openspeech.bytedance.com/api/v3/tts/unidirectional/stream"


def parse_frame(message: bytes) -> tuple[int, bytes]:
    if not isinstance(message, bytes) or len(message) < 16 or message[0] != 0x11:
        raise ValueError("invalid Doubao V3 WebSocket frame")
    header_len = (message[0] & 15) * 4
    if header_len < 4 or message[1] & 15 != 4 or len(message) < header_len + 12:
        raise ValueError("invalid Doubao V3 event header")
    event = struct.unpack_from(">I", message, header_len)[0]
    sid_len = struct.unpack_from(">I", message, header_len + 4)[0]
    pos = header_len + 8 + sid_len
    if pos + 4 > len(message):
        raise ValueError("truncated Doubao V3 frame")
    size = struct.unpack_from(">I", message, pos)[0]
    payload = message[pos + 4:]
    if len(payload) != size:
        raise ValueError("invalid Doubao V3 payload size")
    if message[1] >> 4 == 15:
        raise RuntimeError(f"Doubao V3 error: {payload[:300]!r}")
    return event, payload


async def pcm_chunks(reply_text: str):
    key = os.environ.get("VOLCENGINE_TTS_API_KEY", "").strip()
    if not key:
        raise ValueError("VOLCENGINE_TTS_API_KEY is required")
    speaker = os.environ.get("VOLCENGINE_TTS_SPEAKER", "").strip() or DEFAULT_DOUBAO_SPEAKER
    headers = {"X-Api-Key": key, "X-Api-Resource-Id": DOUBAO_RESOURCE_ID,
               "X-Api-Request-Id": str(uuid.uuid4())}
    request = {"user": {"uid": "project-neck"}, "req_params": {
        "text": reply_text, "speaker": speaker,
        "audio_params": {"format": "pcm", "sample_rate": SAMPLE_RATE,
                         "speech_rate": 0, "loudness_rate": 0},
        "additions": json.dumps({"disable_markdown_filter": True}, ensure_ascii=False),
    }}
    data = json.dumps(request, ensure_ascii=False).encode()
    async with websockets.connect(URL, additional_headers=headers, open_timeout=15, max_size=None) as ws:
        await ws.send(b"\x11\x10\x10\x00" + struct.pack(">I", len(data)) + data)
        while True:
            event, payload = parse_frame(await asyncio.wait_for(ws.recv(), 60))
            if event == 152:  # request completed
                break
            if event == 352:  # PCM s16le
                if not payload or len(payload) % 2:
                    raise ValueError("invalid Doubao PCM chunk")
                yield payload
