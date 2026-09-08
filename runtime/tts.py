"""Edge TTS producing the new Runtime RobotSpeech contract."""
from __future__ import annotations

import io
from typing import Any, Callable

import numpy as np

from .contracts import RobotSpeech, WordTimestamp

SAMPLE_RATE = 16_000
SAMPLES_PER_FRAME = 320
BYTES_PER_FRAME = 640


class EdgeTTS:
    """Synthesize text as 16 kHz mono raw PCM s16le with word boundaries."""

    def __init__(
        self,
        voice: str = "en-US-GuyNeural",
        communicate_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.voice = voice
        self._communicate_factory = communicate_factory
        self.synthesis_count = 0

    @staticmethod
    def _decode_audio(encoded: bytes) -> bytes:
        import soundfile as sf
        from scipy.signal import resample_poly

        samples, source_rate = sf.read(
            io.BytesIO(encoded), dtype="float32", always_2d=True
        )
        mono = samples.mean(axis=1)
        if source_rate != SAMPLE_RATE:
            gcd = int(np.gcd(source_rate, SAMPLE_RATE))
            mono = resample_poly(mono, SAMPLE_RATE // gcd, source_rate // gcd)
        mono = np.nan_to_num(mono, nan=0.0, posinf=1.0, neginf=-1.0)
        quantized = np.clip(
            np.rint(np.clip(mono, -1.0, 1.0) * 32767.0), -32768, 32767
        ).astype("<i2")
        return quantized.tobytes()

    async def synthesize(self, robot_text: str) -> RobotSpeech:
        clean_text = robot_text.strip()
        if not clean_text:
            raise ValueError("EdgeTTS requires non-empty robot_text")

        factory = self._communicate_factory
        if factory is None:
            import edge_tts

            factory = edge_tts.Communicate

        audio_chunks: list[bytes] = []
        words: list[WordTimestamp] = []
        communicate = factory(clean_text, self.voice, boundary="WordBoundary")
        async for chunk in communicate.stream():
            kind = chunk.get("type")
            if kind == "audio":
                audio_chunks.append(chunk["data"])
            elif kind == "WordBoundary":
                start = float(chunk.get("offset", 0)) / 10_000_000.0
                duration = float(chunk.get("duration", 0)) / 10_000_000.0
                token = str(chunk.get("text", "")).strip()
                if token:
                    words.append(
                        WordTimestamp(
                            text=token,
                            start_sec=start,
                            end_sec=start + duration,
                        )
                    )

        if not audio_chunks:
            raise RuntimeError("edge-tts returned no audio")
        pcm_s16le = self._decode_audio(b"".join(audio_chunks))
        if not pcm_s16le or len(pcm_s16le) % 2:
            raise RuntimeError("Edge TTS decoder returned invalid pcm_s16le")

        self.synthesis_count += 1
        duration_sec = len(pcm_s16le) / 2 / SAMPLE_RATE
        print(
            f"[runtime][tts] synthesis #{self.synthesis_count}: "
            f"samples={len(pcm_s16le) // 2}, words={len(words)}, "
            "format=16k/mono/pcm_s16le"
        )
        return RobotSpeech(
            text=clean_text,
            pcm_s16le=pcm_s16le,
            words=tuple(words),
            duration_sec=duration_sec,
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
