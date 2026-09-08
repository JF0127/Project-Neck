"""Local Silero VAD with internal buffering for frozen 20 ms PCM frames."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .contracts import UserAudio

SAMPLE_RATE = 16_000
CHANNELS = 1
INPUT_FRAME_BYTES = 640
SILERO_WINDOW_SAMPLES = 512
SILERO_WINDOW_BYTES = SILERO_WINDOW_SAMPLES * 2
RUNTIME_ROOT = Path(__file__).resolve().parent


class SileroVAD:
    """Reuse one local Silero model and emit complete speech segments."""

    def __init__(
        self,
        model_path: str | Path,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 500,
        score_fn: Callable[[np.ndarray], float] | None = None,
    ) -> None:
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("VAD threshold must be between 0 and 1")
        if min_speech_ms <= 0 or min_silence_ms <= 0:
            raise ValueError("VAD speech and silence durations must be positive")

        path = Path(model_path).expanduser()
        if not path.is_absolute():
            path = RUNTIME_ROOT / path
        self.model_path = path.resolve()
        self.threshold = float(threshold)
        self.min_speech_samples = math.ceil(min_speech_ms * SAMPLE_RATE / 1000)
        self.min_silence_samples = math.ceil(min_silence_ms * SAMPLE_RATE / 1000)
        self._score_fn = score_fn
        self.model: Any | None = None
        self._torch: Any | None = None

        if self._score_fn is None:
            if not self.model_path.is_file():
                raise FileNotFoundError(f"local Silero VAD model does not exist: {self.model_path}")
            import torch

            self._torch = torch
            self.model = torch.jit.load(str(self.model_path), map_location="cpu")
            self.model.eval()
            print(f"[runtime][vad] local Silero VAD ready: {self.model_path}")

        self._pending = bytearray()
        self._candidate = bytearray()
        self._segment = bytearray()
        self._candidate_samples = 0
        self._silence_samples = 0
        self.in_speech = False
        self.reset()

    def reset(self) -> None:
        self._pending.clear()
        self._candidate.clear()
        self._segment.clear()
        self._candidate_samples = 0
        self._silence_samples = 0
        self.in_speech = False
        if self.model is not None:
            reset_states = getattr(self.model, "reset_states", None)
            if callable(reset_states):
                reset_states()

    def _score(self, pcm_window: bytes) -> float:
        waveform = (
            np.frombuffer(pcm_window, dtype="<i2").astype(np.float32) / 32768.0
        )
        if self._score_fn is not None:
            value = self._score_fn(waveform)
        else:
            assert self._torch is not None and self.model is not None
            with self._torch.inference_mode():
                value = self.model(self._torch.from_numpy(waveform), SAMPLE_RATE).item()
        probability = float(value)
        if not math.isfinite(probability):
            raise RuntimeError("Silero VAD returned NaN or Inf")
        return probability

    @staticmethod
    def _user_audio(pcm: bytes) -> UserAudio:
        return UserAudio(
            pcm_s16le=pcm,
            sample_rate=SAMPLE_RATE,
            channels=CHANNELS,
            duration_sec=len(pcm) / 2 / SAMPLE_RATE,
        )

    def _consume_window(self, window: bytes) -> UserAudio | None:
        is_speech = self._score(window) >= self.threshold
        if not self.in_speech:
            if not is_speech:
                self._candidate.clear()
                self._candidate_samples = 0
                return None
            self._candidate.extend(window)
            self._candidate_samples += SILERO_WINDOW_SAMPLES
            if self._candidate_samples >= self.min_speech_samples:
                self.in_speech = True
                self._segment.extend(self._candidate)
                self._candidate.clear()
                self._candidate_samples = 0
                self._silence_samples = 0
            return None

        self._segment.extend(window)
        if is_speech:
            self._silence_samples = 0
            return None

        self._silence_samples += SILERO_WINDOW_SAMPLES
        if self._silence_samples < self.min_silence_samples:
            return None

        completed = bytes(self._segment)
        self._segment.clear()
        self._silence_samples = 0
        self.in_speech = False
        return self._user_audio(completed)

    def push(self, pcm_frame: bytes) -> UserAudio | None:
        """Consume one frozen 640-byte Audio frame and maybe finish a segment."""
        if len(pcm_frame) != INPUT_FRAME_BYTES:
            raise ValueError(f"VAD input frame must be 640 bytes, got {len(pcm_frame)}")
        self._pending.extend(pcm_frame)
        while len(self._pending) >= SILERO_WINDOW_BYTES:
            window = bytes(self._pending[:SILERO_WINDOW_BYTES])
            del self._pending[:SILERO_WINDOW_BYTES]
            segment = self._consume_window(window)
            if segment is not None:
                return segment
        return None

    def end_stream(self) -> UserAudio | None:
        """Flush confirmed speech when the transport stream ends, then reset."""
        segment = None
        if self.in_speech:
            self._segment.extend(self._pending)
            segment = self._user_audio(bytes(self._segment))
        self.reset()
        return segment
