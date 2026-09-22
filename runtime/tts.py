"""Shared production TTS contract and frozen PCM framing helpers."""
from __future__ import annotations

from typing import Protocol

from .contracts import RobotSpeech

SAMPLE_RATE = 16_000
SAMPLES_PER_FRAME = 320
BYTES_PER_FRAME = 640


class TTS(Protocol):
    provider: str

    async def synthesize(self, robot_text: str) -> RobotSpeech: ...


def iter_pcm_frames(pcm_s16le: bytes):
    """Yield exactly 640-byte frames; pad the final frame with zero samples."""
    if len(pcm_s16le) % 2:
        raise ValueError("pcm_s16le byte length must be even")
    for start in range(0, len(pcm_s16le), BYTES_PER_FRAME):
        frame = pcm_s16le[start : start + BYTES_PER_FRAME]
        if len(frame) < BYTES_PER_FRAME:
            frame += b"\0" * (BYTES_PER_FRAME - len(frame))
        yield frame
