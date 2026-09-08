"""In-memory faster-whisper ASR for the new Runtime data contract."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .contracts import UserAudio, UserSpeech, WordTimestamp

SAMPLE_RATE = 16_000
CHANNELS = 1


class WhisperASR:
    """Load one resident Whisper model and transcribe one stable UserAudio."""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        language: str | None = "en",
        model: Any | None = None,
    ) -> None:
        compute_type = "float16" if device == "cuda" else "int8"
        if model is None:
            from faster_whisper import WhisperModel

            print(
                f"[runtime][asr] loading Whisper once: "
                f"{model_path} ({device}/{compute_type})"
            )
            model = WhisperModel(model_path, device=device, compute_type=compute_type)
            print("[runtime][asr] Whisper ready")
        self.model = model
        self.language = language
        self.transcription_count = 0

    @staticmethod
    def pcm_to_float32(pcm_s16le: bytes) -> np.ndarray:
        if len(pcm_s16le) % 2:
            raise ValueError("pcm_s16le byte length must be even")
        return np.frombuffer(pcm_s16le, dtype="<i2").astype(np.float32) / 32768.0

    @staticmethod
    def _validate_audio(user_audio: UserAudio) -> None:
        if not isinstance(user_audio, UserAudio):
            raise TypeError("WhisperASR.transcribe requires UserAudio")
        if user_audio.sample_rate != SAMPLE_RATE:
            raise ValueError("Whisper ASR requires 16000 Hz UserAudio")
        if user_audio.channels != CHANNELS:
            raise ValueError("Whisper ASR requires mono UserAudio")
        if len(user_audio.pcm_s16le) % 2:
            raise ValueError("UserAudio pcm_s16le byte length must be even")
        if not math.isfinite(user_audio.duration_sec) or user_audio.duration_sec < 0.0:
            raise ValueError("UserAudio duration_sec must be finite and non-negative")
        pcm_duration = len(user_audio.pcm_s16le) / 2 / SAMPLE_RATE
        if not math.isclose(
            user_audio.duration_sec,
            pcm_duration,
            rel_tol=0.0,
            abs_tol=1.0 / SAMPLE_RATE,
        ):
            raise ValueError("UserAudio duration_sec does not match its PCM length")

    def transcribe(self, user_audio: UserAudio) -> UserSpeech:
        self._validate_audio(user_audio)
        waveform = self.pcm_to_float32(user_audio.pcm_s16le)
        if waveform.size == 0:
            return UserSpeech(text="", words=(), language=self.language)

        segments, info = self.model.transcribe(
            waveform,
            language=self.language,
            word_timestamps=True,
            vad_filter=False,
        )
        words: list[WordTimestamp] = []
        segment_text: list[str] = []
        for segment in segments:
            text = (segment.text or "").strip()
            if text:
                segment_text.append(text)
            for word in segment.words or []:
                token = (word.word or "").strip()
                if token and word.start is not None and word.end is not None:
                    words.append(
                        WordTimestamp(
                            text=token,
                            start_sec=float(word.start),
                            end_sec=float(word.end),
                        )
                    )

        self.transcription_count += 1
        text = " ".join(segment_text).strip()
        if not text and words:
            text = " ".join(word.text for word in words)
        language = getattr(info, "language", None) or self.language
        print(
            f"[runtime][asr] transcription #{self.transcription_count}: "
            f"text={text!r}, words={len(words)}"
        )
        return UserSpeech(text=text, words=tuple(words), language=language)
