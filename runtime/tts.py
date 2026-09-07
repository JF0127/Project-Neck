"""Edge TTS with in-memory decoding and frozen PCM wire-format conversion."""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np

from .asr import WordTimestamp

SAMPLE_RATE = 16_000
SAMPLES_PER_FRAME = 320
BYTES_PER_FRAME = 640


@dataclass(frozen=True)
class TTSResult:
    pcm_s16le: bytes
    waveform: np.ndarray
    words: list[WordTimestamp]
    duration_sec: float


class EdgeTTS:
    """Produce one quantized PCM buffer shared by WebSocket and motion inference."""

    def __init__(self, voice: str = "en-US-GuyNeural"):
        self.voice = voice
        self.synthesis_count = 0

    async def synthesize(self, text: str) -> TTSResult:
        import edge_tts
        import soundfile as sf
        from scipy.signal import resample_poly

        audio_chunks: list[bytes] = []
        words: list[WordTimestamp] = []
        communicate = edge_tts.Communicate(text, self.voice, boundary="WordBoundary")
        async for chunk in communicate.stream():
            kind = chunk.get("type")
            if kind == "audio":
                audio_chunks.append(chunk["data"])
            elif kind == "WordBoundary":
                # edge-tts offsets and durations use 100 ns ticks.
                start = float(chunk.get("offset", 0)) / 10_000_000.0
                duration = float(chunk.get("duration", 0)) / 10_000_000.0
                token = str(chunk.get("text", "")).strip()
                if token:
                    words.append(WordTimestamp(token, start, start + duration))

        if not audio_chunks:
            raise RuntimeError("edge-tts returned no audio")
        encoded = b"".join(audio_chunks)
        samples, source_rate = sf.read(
            io.BytesIO(encoded), dtype="float32", always_2d=True
        )
        mono = samples.mean(axis=1)
        if source_rate != SAMPLE_RATE:
            # Rates are integral in this pipeline (normally 24 kHz -> 16 kHz).
            gcd = int(np.gcd(source_rate, SAMPLE_RATE))
            mono = resample_poly(mono, SAMPLE_RATE // gcd, source_rate // gcd)
        mono = np.nan_to_num(mono, nan=0.0, posinf=1.0, neginf=-1.0)
        quantized = np.clip(np.rint(np.clip(mono, -1.0, 1.0) * 32767.0), -32768, 32767).astype("<i2")
        pcm = quantized.tobytes()
        # Motion consumes the exact same quantized signal that Audio receives.
        waveform = quantized.astype(np.float32) / 32768.0
        self.synthesis_count += 1
        print(
            f"[runtime][tts] synthesis #{self.synthesis_count}: "
            f"samples={len(quantized)}, words={len(words)}, format=16k/mono/pcm_s16le"
        )
        return TTSResult(
            pcm_s16le=pcm,
            waveform=waveform,
            words=words,
            duration_sec=len(quantized) / SAMPLE_RATE,
        )


def iter_pcm_frames(pcm_s16le: bytes):
    """Yield exactly 640-byte frames; pad the final frame with zero samples."""
    if len(pcm_s16le) % 2:
        raise ValueError("pcm_s16le byte length must be even")
    for offset in range(0, len(pcm_s16le), BYTES_PER_FRAME):
        frame = pcm_s16le[offset : offset + BYTES_PER_FRAME]
        if len(frame) < BYTES_PER_FRAME:
            frame += bytes(BYTES_PER_FRAME - len(frame))
        yield frame
